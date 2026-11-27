"""Regional reductions on the host's local O(2) frames."""

from functools import lru_cache, partial
from math import sqrt

import torch
from e3nn import o3
from e3nn.nn import Gate as HostGate

from equivariantx.co2 import SphericalCoupling
from equivariantx.o2 import O2O3TensorProductConv, O3TensorProduct, WignerD
from equivariantx.o2 import UuO2TensorProductConv as EqxUuConv
from equivariantx.o3 import Gate as EqxGate
from tace.models.mlp import MLPV1, MLPV2
from tace.models.spherical.fused import O2TensorProductConv, UuO2TensorProductConv
from tace.models.spherical.inter import (
    O2CgtpInteraction,
    UuO2Interaction,
    UvO2Interaction,
)
from tace.models.spherical.node import LinearNodeEmbedding
from tace.models.spherical.o2 import UvO2TensorProductConv, uv_convolution
from tace.utils.env import acceleration_enabled, memory_first

from .matching import (
    canonical_component_harmonics,
    canonical_radial,
    custom_forward,
    tracing_rows,
)


def _aligned_programs(layer):
    """The precomputed and streamed reductions reached by native aligned CGTP."""
    if type(layer) is not O2CgtpInteraction or custom_forward(layer, recursive=True):
        return None
    conv = layer.rejector
    if (
        type(conv) is not O2TensorProductConv
        or type(getattr(conv, "tp", None)) is not O3TensorProduct
        or any(
            k in vars(layer)
            for k in ("_compute_messages", "_conv_weights", "_radial_projection")
        )
        or "forward_stream" in vars(conv)
        or "forward_scatter" in vars(conv.tp)
    ):
        return None
    programs = getattr(conv.tp, "convolution", None), getattr(conv, "eqx_tp", None)
    for op in programs:
        if (
            type(op) is not O2O3TensorProductConv
            or type(getattr(op, "frame", None)) is not WignerD
            or type(getattr(op, "transverse_couplings", None))
            is not torch.nn.ModuleDict
            or any(
                type(c) is not SphericalCoupling
                or type(getattr(c, "harmonics", None)) is not o3.SphericalHarmonics
                for c in op.transverse_couplings.values()
            )
            or type(getattr(op, "harmonics", None)) is not torch.nn.ModuleDict
            or any(type(h) is not o3.SphericalHarmonics for h in op.harmonics.values())
            or "forward_packed" in vars(op.frame)
            or any(
                k in vars(op)
                for k in ("forward_transverse", "reference", "scatter_outputs")
            )
            or any(
                getattr(part, "backend", None) not in ("cuda", "torch")
                for part in _aligned_parts(op)
            )
        ):
            return None
    return programs


def _aligned_parts(op):
    # Wigner and transverse Torch implementations can dispatch their own
    # kernels. Setting just the outer switch need not pin the whole reduction.
    return (op, op.frame, *op.transverse_couplings.values())


def _aligned_provider(layer, programs):
    p = next(layer.parameters())
    if p.dtype not in (torch.float32, torch.float64):
        return None
    if p.device.type == "cpu":
        return "eqx_torch"
    if not p.is_cuda or torch.version.hip:
        return None
    op = programs[int(layer.use_eqx)]
    backends = {part.backend for part in _aligned_parts(op)}
    return f"eqx_{op.backend}" if backends == {op.backend} else None


def host_aligned_provider(layer):
    programs = _aligned_programs(layer)
    return _aligned_provider(layer, programs) if programs is not None else None


def _aligned_tables_match(op):
    # The Torch precomputed branch reads live dense CG buffers; the pinned
    # CUDA kernel uses their sparse constants. A user's modified table must
    # not silently disappear on the copy. This check is only at planning time.
    if "aligned_path" in vars(op) or not _canonical_wigner_tables(op.frame):
        return False
    if op.kernel_metadata != repr(
        (
            tuple(op.path_data),
            op.weight_numel,
            op.has_unweighted,
            tuple(op.sparse_paths),
        )
    ) or not (len(op.path_data) == len(op.sparse_paths) == len(op.instructions)):
        return False
    live, expected = [], []
    for i, ((_, path), sparse) in enumerate(zip(op.path_data, op.sparse_paths)):
        table = op._buffers.get(f"cg_{i}")
        if table is None or table.requires_grad or table.shape != path[4:6]:
            return False
        dense = torch.zeros(table.shape, dtype=table.dtype, device="cpu")
        for row, col, coefficient in sparse:
            dense[row, col] = coefficient
        live.append(table.detach().flatten())
        expected.append(dense.flatten())
    return torch.equal(torch.cat(live).cpu(), torch.cat(expected))


def measured_aligned_stack(model):
    """Admit the measured two-layer, precomputed-weight E/F graph on NVIDIA."""
    from equivariantx.ops import config
    from tace.models.adapter import TensorModel
    from tace.models.spherical.paths import generate_paths
    from tace.models.spherical.prod import CgtpACE
    from tace.models.spherical.representation import NodeRepresentation
    from tace.models.spherical.tace import SphericalTACE

    host = model.readout_fn
    rep = host.representation
    if not (
        type(model) is TensorModel
        and type(host) is SphericalTACE
        and type(rep) is NodeRepresentation
        and not model.lmp
        and model.model_dtype == rep.model_dtype == torch.float32
        and not {"prepare_graph", "prepare_property_data"}.intersection(vars(model))
        and not custom_forward(model)
        and not custom_forward(host)
        and not custom_forward(rep)
        and set(host.target_property) == {"energy", "forces"}
        and len(rep.interactions) == len(rep.products) == 2
        and all(type(p) is CgtpACE and p.correlation == 3 for p in rep.products)
        and all(type(i) is O2CgtpInteraction for i in rep.interactions)
        and not rep.use_local_frame
        and torch.get_float32_matmul_precision() == "highest"
        and not torch.is_autocast_enabled("cuda")
        and not torch.version.hip
        and not config.TORCH_REFERENCE
        and memory_first()
    ):
        return False
    # TensorModel prepares FP64 physical directions even for FP32 features:
    # the actual measured branch constructs packed Wigner matrices, not the
    # same-precision transverse algorithm. Preserve alternate provider choices.
    harmonics = o3.Irreps.spherical_harmonics(3)
    output = o3.Irreps("64x0e+64x1o+64x2e+64x3o")
    for layer, inputs in zip(rep.interactions, ("64x0e", "64x0e+64x1o+64x2e")):
        param = next(layer.parameters())
        if not (
            param.is_cuda
            and param.dtype == torch.float32
            and not layer.use_eqx
            and (layer.num_channel, layer.lmax, layer.mmax, layer.Lmax) == (64, 3, 3, 2)
            and layer.irreps_in == o3.Irreps(inputs)
            and layer.irreps_sh == harmonics
            and layer.irreps_out == output
            and layer.l1l2 is None
            and layer.radial_widths is None
        ):
            return False
        programs = _aligned_programs(layer)
        if programs is None:
            return False
        active, alternate = programs
        tp = layer.rejector.tp
        paths, outputs = generate_paths(output, layer.irreps_in, harmonics)
        if not (
            tp.backend == active.backend == active.frame.backend == "torch"
            and not tp.internal_weights
            and not tp.shared_weights
            and layer.rejector.instructions == paths
            and [tuple(ins[:5]) for ins in tp.instructions] == paths
            and tp.irreps_out == outputs
            and all(c.backend == "cuda" for c in active.transverse_couplings.values())
            and all(p.backend == "cuda" for p in _aligned_parts(alternate))
            and (active.radial_mode, alternate.radial_mode) == ("last", "full")
            and all(
                op.method == "auto"
                and not op.channel_mixing
                and op.normalization == "component"
                and op.instructions == tuple(tp.instructions)
                and op.frame.mmax == op.frame.lmax == 3
                and getattr(getattr(op.frame, "cg_2", None), "dtype", None)
                == torch.float64
                for op in programs
            )
            and _aligned_tables_match(active)
        ):
            return False
    return True


def _retarget_aligned(layer, *, backend):
    # Keep the live host flag, radial policy, autotuner and checkpoint state.
    # Pin both existing paths so later flag changes cannot undo the receipt.
    for op in (layer.rejector.tp.convolution, layer.rejector.eqx_tp):
        for part in _aligned_parts(op):
            part.backend = backend
    return layer


def aligned_rule(module, *, backend, task, automatic=False, measured_stack=False):
    programs = _aligned_programs(module)
    if programs is None:
        return None, "preserved unrecognized or customized aligned O(2) CGTP"
    p = next(module.parameters())
    if (
        task not in ("inference", "training")
        or backend not in ("cuda", "torch")
        or p.dtype not in (torch.float32, torch.float64)
        or p.device.type not in ("cpu", "cuda")
        or (backend == "cuda" and (not p.is_cuda or torch.version.hip))
    ):
        return None, "requires a supported Torch or NVIDIA CUDA provider"
    if not getattr(module, "_foundation_regional_provider", None) and (
        (automatic and not measured_stack)
        or p.device.type == "cpu"
        or all(
            part.backend == backend for op in programs for part in _aligned_parts(op)
        )
    ):
        return None, "preserved the host's aligned O(2) CGTP provider"
    detail = (
        f"pin alternate aligned O(2) CGTP branch to EQX {backend}; active provider unchanged"
        if _aligned_provider(module, programs) == f"eqx_{backend}"
        else f"select EQX {backend} for the host's aligned O(2) CGTP reduction"
    )
    return partial(_retarget_aligned, backend=backend), detail


def _eligible_inputs(x, frame, inverse, cutoff, idx):
    return (
        x.is_cuda
        and x.dtype in (torch.float32, torch.float64)
        and not torch.is_autocast_enabled("cuda")
        and frame is not None
        and inverse is not None
        and frame.dtype == x.dtype == inverse.dtype
        and cutoff is not None
        and cutoff.dtype == x.dtype
        and cutoff.ndim == 2
        and cutoff.size(1) == 1
        and idx.size(1) != 0
    )


class _UuFrames(UuO2TensorProductConv):
    supports_receiver_count = True

    def forward(
        self,
        node_feats,
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff,
        *,
        num_nodes=None,
    ):
        if not _eligible_inputs(
            node_feats, wigner, wigner_inv, edge_cutoff, edge_index
        ):
            return super().forward(
                node_feats,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_cutoff,
                num_nodes=num_nodes,
            )
        x = self.reshape_in(node_feats)
        local = self.local_frame_in.gather(x, wigner, edge_index[0], backend="cuda")
        local = self.linear(local, conv_weights)
        global_sum = self.local_frame_out.scatter(
            local * edge_cutoff,
            wigner_inv,
            edge_index[1],
            x.size(0) if num_nodes is None else num_nodes,
            backend="cuda",
        )
        return self.reshape_out.inverse(global_sum)


class _UvFrames(UvO2TensorProductConv):
    supports_receiver_count = True

    def forward(
        self,
        node_feats,
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=None,
        radial_network=None,
        edge_pairs=None,
        *,
        num_nodes=None,
    ):
        if not _eligible_inputs(
            node_feats, wigner, wigner_inv, edge_cutoff, edge_index
        ):
            return super().forward(
                node_feats,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_cutoff,
                radial_network,
                edge_pairs,
                num_nodes=num_nodes,
            )
        return uv_convolution(
            self,
            node_feats,
            None,
            conv_weights,
            edge_index,
            wigner,
            wigner_inv,
            edge_cutoff,
            radial_network=radial_network,
            edge_pairs=edge_pairs,
            num_nodes=num_nodes,
        )


class _CanonicalStream:
    @property
    def use_eqx(self):
        p = next(self.parameters())
        return (
            p.is_cuda
            and p.dtype in (torch.float32, torch.float64)
            and not torch.is_autocast_enabled("cuda")
        )


class _StreamUu(_CanonicalStream, UuO2Interaction):
    pass


class _StreamUv(_CanonicalStream, UvO2Interaction):
    pass


class _SharedUvFrameUuConv(UuO2TensorProductConv):
    """Consume a verified stack's common packed frame on mixed-precision inputs."""

    supports_receiver_count = True

    def _apply(self, fn, recurse=True):
        # The receiving tables may be replaced by .to()/.double(). Never keep
        # old device tensors alive; a fresh plan can verify their replacements.
        vars(self).pop("_foundation_uu_tables", None)
        return super()._apply(fn, recurse=recurse)

    def forward_stream(
        self,
        node_feats,
        radial,
        projection,
        edge_index,
        wigner,
        edge_cutoff,
        graph,
        *,
        radial_network=None,
        edge_pairs=None,
        num_nodes=None,
    ):
        if (
            (self.training or hasattr(self, "_foundation_uu_tables"))
            and node_feats.is_cuda
            and node_feats.dtype == torch.float32
            and not torch.is_autocast_enabled("cuda")
            and graph is not None
            and graph.edge_vector.dtype == torch.float64
            and wigner is not None
            and wigner.ndim == 2
            and wigner.dtype == node_feats.dtype
        ):
            # graph supplies only the optional directions to this host method.
            # With an existing frame no Uu angular basis or rotation is needed.
            graph = None
        return super().forward_stream(
            node_feats,
            radial,
            projection,
            edge_index,
            wigner,
            edge_cutoff,
            graph,
            radial_network=radial_network,
            edge_pairs=edge_pairs,
            num_nodes=num_nodes,
        )


class _StreamUvConv(UvO2TensorProductConv):
    supports_receiver_count = True

    def forward(
        self,
        node_feats,
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=None,
        radial_network=None,
        edge_pairs=None,
        *,
        num_nodes=None,
    ):
        if (
            not node_feats.is_cuda
            or node_feats.dtype not in (torch.float32, torch.float64)
            or torch.is_autocast_enabled("cuda")
        ):
            return super().forward(
                node_feats,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_cutoff,
                radial_network,
                edge_pairs,
                num_nodes=num_nodes,
            )
        return uv_convolution(
            self,
            node_feats,
            None,
            conv_weights,
            edge_index,
            wigner,
            wigner_inv,
            edge_cutoff,
            radial_network,
            edge_pairs,
            num_nodes=num_nodes,
        )


def _stream_eligible(module):
    if type(module) not in (UuO2Interaction, UvO2Interaction, _StreamUu, _StreamUv):
        return False
    conv = module.rejector
    if type(conv) not in (
        (UuO2TensorProductConv, _SharedUvFrameUuConv)
        if isinstance(module, UuO2Interaction)
        else (UvO2TensorProductConv, _StreamUvConv)
    ):
        return False
    fp = next(module.parameters())
    return (
        fp.is_cuda
        and fp.dtype in (torch.float32, torch.float64)
        and type(module.edge_info) in (MLPV1, MLPV2)
        and conv.eqx_tp is not None
        and conv.eqx_tp.backend == "cuda"
        and not any(
            name in module.__dict__
            for name in ("_compute_messages", "_apply_rejector", "_radial_projection")
        )
        and "forward_stream" not in conv.__dict__
        and not any(
            custom_forward(m) or getattr(m, "use_lora", False)
            for m in (module, *conv.modules(), *module.edge_info.modules())
        )
    )


def _local_geometry(rep, layers):
    from tace.models.spherical.representation import NodeRepresentation

    basis = getattr(rep, "o2_angular_basis", None)
    return (
        type(rep) is NodeRepresentation
        and not custom_forward(rep)
        and rep.use_local_frame
        and rep._can_pack_wigner
        and not rep.use_magnetic_interaction
        and canonical_radial(rep)
        and type(basis) is WignerD
        and basis.method == "auto"
        and not custom_forward(basis)
        and "forward_packed" not in basis.__dict__
        and bool(layers)
        and all(
            _stream_eligible(m)
            and basis.lmax == max(m.Lmax, m.lmax)
            and basis.mmax == m.mmax
            for m in layers
        )
    )


def mixed_geometry(rep):
    """O3 and O2 may share the radial graph, but never substitute their angular bases.

    Packed Wigner changes the input signature of every local consumer, and
    streaming layer zero can compact radial rows for every later consumer.
    Admit the whole mixed stack or neither stream, independent of layer order.
    """
    import equivariantx as eqx
    from tace.models.spherical.node import (
        Element2SphericalTensorNodeEmbedding,
        LinearNodeEmbedding,
        SphericalTensorNodeEmbedding,
    )

    from .interactions import _SPHERICAL_TYPES, _mixed_o3_layer

    spherical = [m for m in rep.interactions if type(m) in _SPHERICAL_TYPES]
    local = [
        m
        for m in rep.interactions
        if type(m) in (UuO2Interaction, UvO2Interaction, _StreamUu, _StreamUv)
    ]
    if (
        not spherical
        or not local
        or len(local) + len(spherical) != len(rep.interactions)
        or not _local_geometry(rep, local)
        or not rep.use_o3_angular_basis
        or not rep.use_o2
        or rep.use_so2
        or rep.co3_angular_basis is not None
        or rep._can_skip_wigner
        != (not any(isinstance(m, UvO2Interaction) for m in local))
        or type(rep.node_embedding)
        not in (
            LinearNodeEmbedding,
            SphericalTensorNodeEmbedding,
            Element2SphericalTensorNodeEmbedding,
        )
        or getattr(rep.node_embedding, "use_wigner", False)
        or custom_forward(rep, recursive=True)
        or any(
            "_conv_weights" in vars(m)
            or "stream" in vars(m.edge_info)
            or type(m.rejector.eqx_tp)
            is not (
                eqx.o2.UuO2TensorProductConv
                if isinstance(m, UuO2Interaction)
                else eqx.o2.UvO2TensorProductConv
            )
            for m in local
        )
    ):
        return False
    sh, frame = rep.o3_angular_basis, rep.o2_angular_basis
    if (
        not canonical_component_harmonics(sh, spherical[0].irreps_sh.lmax)
        or not all(_mixed_o3_layer(m, sh) for m in spherical)
        or "from_matrix" in vars(frame)
    ):
        return False
    return _canonical_wigner_tables(frame)


def _canonical_wigner_tables(frame):
    # CUDA packed construction bypasses dense placement, inverse scales and
    # the reference recurrence. Never discard a customized angular table.
    reference = WignerD(frame.mmax, frame.lmax, backend="torch")
    for name, expected in reference.named_buffers():
        value = getattr(frame, name, None)
        if (
            value is None
            or value.requires_grad
            or (
                value.dtype not in (torch.float32, torch.float64)
                if expected.is_floating_point()
                else value.dtype != expected.dtype
            )
        ):
            return False
        value = value.detach().cpu()
        if not torch.equal(value, expected.to(value)) and not (
            value.dtype == torch.float64
            and torch.equal(value, expected.float().to(value))
        ):
            return False
    return True


def local_geometry(rep):
    """Whether the host's local layers can share canonical frames/reciprocal edges."""
    if _local_geometry(rep, rep.interactions):
        return True
    return mixed_geometry(rep)


def _canonical_uu_direction(layer):
    op = getattr(layer.rejector, "eqx_tp", None)
    return (
        type(op) is EqxUuConv
        and op.backend == "cuda"
        and op.with_transverse
        and op.radial_mode == "full"
        and type(op.frame) is WignerD
        and op.frame.lmax == 3
        and op.frame.backend == "cuda"
        and "forward_stream" not in vars(layer.rejector)
        and not {"_compute_messages", "_apply_rejector"}.intersection(vars(layer))
        and "forward_packed" not in vars(op.frame)
        and not {"forward_transverse", "forward_wigner"}.intersection(vars(op))
    )


def _shared_direction_frame(rep, graph):
    from equivariantx.ops.common.cuda.graph import CAPTURE_CONSTANTS
    from equivariantx.utils.autograd import forward_ad_active

    from .matching import tracing_rows

    v = graph.edge_vector
    if (
        tracing_rows(v)
        or torch.jit.is_tracing()
        or not v.is_cuda
        or v.dtype != torch.float64
        or rep.model_dtype != torch.float32
        or torch.is_autocast_enabled("cuda")
        or forward_ad_active()
        # Private replay must warm up and record the same program.
        or CAPTURE_CONSTANTS.get() is not None
        or torch.cuda.is_current_stream_capturing()
        or graph.lmp
        or graph.lmp_data is not None
        or not memory_first()
        or v.size(0) == 0
        or len(rep.interactions) != 2
        or custom_forward(rep)
        or type(rep.node_embedding) is not LinearNodeEmbedding
        or custom_forward(rep.node_embedding)
    ):
        return None
    for layer in rep.interactions:
        if (
            type(layer) is not _StreamUu
            or type(layer.rejector) is not _SharedUvFrameUuConv
        ):
            return None
        conv = layer.rejector
        tables = getattr(conv, "_foundation_uu_tables", None)
        if tables is None or not _canonical_uu_direction(layer):
            return None
        op, frame = conv.eqx_tp, conv.eqx_tp.frame
        if any(custom_forward(m) for m in (layer, conv, op, frame)) or any(
            frame._buffers.get(n) is not original
            or original.requires_grad
            or original._version != version
            for n, original, version in tables
        ):
            return None
    # Keep every directed vector. Reciprocal maps select radial rows only;
    # neither this constructor nor either old constructor indexes by them.
    return rep.interactions[0].rejector.eqx_tp.frame.forward_packed(v, method="auto")


def _configure_shared_uu(rep):
    from tace.models.spherical.representation import NodeRepresentation

    if getattr(rep, "_shared_direction_frame", None) is _shared_direction_frame:
        del rep._shared_direction_frame
    if not (
        type(rep) is NodeRepresentation
        and memory_first()
        and len(rep.interactions) == 2
        and all(type(layer) is _StreamUu for layer in rep.interactions)
        and rep._can_skip_wigner
        and type(rep.node_embedding) is LinearNodeEmbedding
        and not rep.use_o3_angular_basis
        and "_spatial_frames" not in vars(rep)
        and not hasattr(rep, "_shared_direction_frame")
        and _local_geometry(rep, rep.interactions)
        and not custom_forward(rep, recursive=True)
        and all(
            layer.num_channel == 64
            and (layer.Lmax, layer.lmax, layer.mmax) == (2, 3, 3)
            and next(layer.parameters()).dtype == torch.float32
            and _canonical_uu_direction(layer)
            and not (frame := layer.rejector.eqx_tp.frame)._parameters
            and not any(t.is_inference() for t in frame.buffers())
            and _canonical_wigner_tables(frame)
            for layer in rep.interactions
        )
    ):
        return
    for layer in rep.interactions:
        conv = layer.rejector
        conv.__class__ = _SharedUvFrameUuConv
        # Only the per-degree coefficients are read by forward_packed(recursive).
        conv._foundation_uu_tables = tuple(
            (n, t, t._version)
            for n, t in conv.eqx_tp.frame.named_buffers()
            if n.startswith("cg_")
        )
    rep._shared_direction_frame = _shared_direction_frame


def configure_local_geometry(rep, *, explicit_cuda_training=False):
    """Select the CUDA packed Wigner program on an already-converted *copy*."""
    from .magnetic import configure_scalar_spin_frames

    configure_scalar_spin_frames(rep)
    if getattr(rep, "_can_skip_wigner", False):
        _configure_shared_uu(rep)
    if any(isinstance(layer, (_StreamUu, _StreamUv)) for layer in rep.interactions):
        import equivariantx as eqx

        from .interactions import StreamedInteraction, _SharedHarmonicInteraction

        if any(isinstance(m, StreamedInteraction) for m in rep.interactions) and all(
            isinstance(m, (_StreamUu, _StreamUv, StreamedInteraction))
            for m in rep.interactions
        ):
            has_packed_uv = any(type(m) is _StreamUv for m in rep.interactions)
            for layer in rep.interactions:
                if type(layer) is StreamedInteraction:
                    layer.__class__ = _SharedHarmonicInteraction
                elif type(layer) is _StreamUu and has_packed_uv:
                    # Uv already requires packed Wigner. Use its exact tensor
                    # and backward graph for Uu instead of synthesizing a
                    # second direction-only contraction from the edge vectors.
                    conv = layer.rejector
                    if conv.eqx_tp.with_transverse:
                        conv._foundation_restore_transverse = True
                        conv.eqx_tp.with_transverse = False
        elif (
            explicit_cuda_training
            and memory_first()
            and not rep._can_skip_wigner
            and not rep.use_o3_angular_basis
            and any(type(m) is _StreamUv for m in rep.interactions)
            and any(type(m) is _StreamUu for m in rep.interactions)
            and all(
                type(m) in (_StreamUu, _StreamUv)
                and m.num_channel == 32
                and type(m.rejector.eqx_tp)
                is (
                    eqx.o2.UuO2TensorProductConv
                    if type(m) is _StreamUu
                    else eqx.o2.UvO2TensorProductConv
                )
                and not {"forward_wigner", "forward_transverse"}.intersection(
                    vars(m.rejector.eqx_tp)
                )
                and m.rejector.eqx_tp.radial_mode == "full"
                and next(m.parameters()).dtype == torch.float32
                for m in rep.interactions
            )
            and rep.o2_angular_basis.lmax == rep.o2_angular_basis.mmax == 2
            and "from_matrix" not in vars(rep.o2_angular_basis)
            and _local_geometry(rep, rep.interactions)
            and _canonical_wigner_tables(rep.o2_angular_basis)
            and not custom_forward(rep, recursive=True)
        ):
            # Only the measured nonmagnetic, pure O2 mixed stack. Leave a
            # Uu-only representation's direction-only geometry completely alone.
            for m in rep.interactions:
                if type(m) is _StreamUu and type(m.rejector) is UuO2TensorProductConv:
                    m.rejector.__class__ = _SharedUvFrameUuConv
        if not hasattr(rep, "_foundation_wigner_backend"):
            rep._foundation_wigner_backend = rep.o2_angular_basis.backend
        rep.o2_angular_basis.backend = "cuda"
    elif hasattr(rep, "_foundation_wigner_backend"):
        rep.o2_angular_basis.backend = rep._foundation_wigner_backend
        del rep._foundation_wigner_backend


def _restore_frame_backends(conv):
    prior = getattr(conv, "_foundation_frame_backends", None)
    if prior is not None:
        conv.local_frame_in.backend, conv.local_frame_out.backend = prior
        del conv._foundation_frame_backends


def _torch_frames(conv):
    if acceleration_enabled("eqx", kernel="conv"):
        raise ValueError(
            "Cannot retarget local O(2) to Torch while the host's global EQX convolution is active."
        )
    conv._foundation_frame_backends = (
        conv.local_frame_in.backend,
        conv.local_frame_out.backend,
    )
    conv.local_frame_in.backend = conv.local_frame_out.backend = "torch"


class _NodeGate(HostGate):
    """Keep the host gate and its state; select only the measured CUDA expression."""

    def forward(self, features):
        if (
            features.is_cuda
            and features.dtype == torch.float32
            and not torch.is_autocast_enabled("cuda")
            and not tracing_rows(features)
            and not torch.jit.is_tracing()
            and self._node_calls == _node_gate_calls(self)
            and not custom_forward(self, recursive=True)
        ):
            from equivariantx.ops.common.gate import gate

            return gate(features, self._node_expression, self.irreps_out.dim)
        return super().forward(features)


def _node_gate_calls(gate):
    # Capture exactly the callable, width and scalar gain e3nn's live paths use.
    # Module references survive copying/pickling; do not compare tensor values.
    return tuple(
        (
            mul,
            ir,
            act,
            getattr(act, "f", None),
            getattr(getattr(act, "f", None), "inplace", False),
            getattr(act, "_is_id", None),
            act.cst if type(getattr(act, "cst", None)) is float else None,
        )
        for activation in (gate.act_scalars, gate.act_gates)
        for mul, ir, act in activation.paths
    )


def _gate_code(cut, product):
    kernels = (cut._compiled_forward, product._compiled_main_left_right)
    if all(isinstance(k, torch.fx.GraphModule) for k in kernels):
        return tuple(k.code for k in kernels)
    return None


@lru_cache(maxsize=8)
def _canonical_gate_code(scalars, gates, gated, activated_gates):
    from e3nn.nn._gate import _Sortcut

    # These two stateless e3nn modules generate the actual programs read by
    # HostGate.forward. A shape/metadata match alone cannot certify custom FX.
    with torch.device("cpu"):
        cut = _Sortcut(scalars, gates, gated).cut
        product = o3.ElementwiseTensorProduct(gated, activated_gates)
    return _gate_code(cut, product)


def _uv_gate_expression(layer):
    """Match the node gate, independently of edge count, chemistry and radial widths."""
    from e3nn.math import normalize2mom
    from e3nn.nn import Activation
    from e3nn.nn._extract import Extract
    from e3nn.nn._gate import _Sortcut

    gate = layer.nonlinearity
    if (
        type(layer) is not UvO2Interaction
        or next(layer.parameters()).dtype != torch.float32
        or torch.get_float32_matmul_precision() != "highest"
        or not memory_first()
        or type(gate) is not HostGate
        or type(gate.sc) is not _Sortcut
        or type(gate.sc.cut) is not Extract
        or type(gate.mul) is not o3.ElementwiseTensorProduct
        or gate.irreps_in != o3.Irreps("256x0e+64x1o+64x2e+64x3o")
        or gate.irreps_out != o3.Irreps("64x0e+64x1o+64x2e+64x3o")
        or next(gate.parameters(), None) is not None
        or custom_forward(gate, recursive=True)
    ):
        return None
    # e3nn reads Activation.paths, while EQX encodes acts. Never borrow a
    # gate with an independently replaced callable, cut, or tensor product.
    for activations, fn, count in (
        (gate.act_scalars, torch.nn.SiLU, 1),
        (gate.act_gates, torch.nn.Sigmoid, 3),
    ):
        if (
            type(activations) is not Activation
            or len(activations.paths) != len(activations.acts)
            or len(activations.paths) != count
            or any(
                (mul, ir) != (64, (0, 1))
                or act is not called
                or type(act) is not normalize2mom
                or type(act.f) is not fn
                or type(act.cst) is not float
                or getattr(act.f, "inplace", False)
                for (mul, ir, called), act in zip(activations.paths, activations.acts)
            )
        ):
            return None
    if (
        list(gate.sc.cut.instructions) != [(0,), (1,), (2, 3, 4)]
        or len(gate.mul.instructions) != 3
        or any(
            (ins.i_in1, ins.i_in2, ins.i_out) != (i, i, i)
            or ins.connection_mode != "uuu"
            or ins.has_weight
            or ins.path_weight != sqrt(2 * i + 3)
            for i, ins in enumerate(gate.mul.instructions)
        )
        or set(dict(gate.named_buffers())) != {"mul.weight", "mul.output_mask"}
        or gate.mul.weight.numel()
        or any(t.requires_grad for t in gate.buffers())
        or (code := _gate_code(gate.sc.cut, gate.mul)) is None
        or code
        != _canonical_gate_code(
            gate.act_scalars.irreps_in,
            gate.act_gates.irreps_in,
            gate.mul.irreps_in1,
            gate.act_gates.irreps_out,
        )
    ):
        return None
    return EqxGate.expression(gate)


def _apply_stream(module, *, gate_expression=None):
    _restore_frame_backends(module.rejector)
    if type(module) is UvO2Interaction:
        module.__class__ = _StreamUv
        module.rejector.__class__ = _StreamUvConv
        if gate_expression is not None:
            module.nonlinearity.__class__ = _NodeGate
            module.nonlinearity._node_expression = gate_expression
            module.nonlinearity._node_calls = _node_gate_calls(module.nonlinearity)
    else:
        module.__class__ = _StreamUu
    return module


def _restore_stream(module):
    _torch_frames(module.rejector)
    vars(module.rejector).pop("_foundation_uu_tables", None)
    if getattr(module.rejector, "_foundation_restore_transverse", False):
        module.rejector.eqx_tp.with_transverse = True
        del module.rejector._foundation_restore_transverse
    if type(module) is _StreamUv:
        if type(module.nonlinearity) is _NodeGate:
            module.nonlinearity.__class__ = HostGate
            del module.nonlinearity._node_expression
            del module.nonlinearity._node_calls
        module.rejector.__class__ = UvO2TensorProductConv
        module.__class__ = UvO2Interaction
    else:
        if type(module.rejector) is _SharedUvFrameUuConv:
            module.rejector.__class__ = UuO2TensorProductConv
        module.__class__ = UuO2Interaction
    return module


def _apply(module):
    _restore_frame_backends(module.rejector)
    module.rejector.__class__ = (
        _UuFrames if type(module) is UuO2Interaction else _UvFrames
    )
    return module


def _restore(module):
    _torch_frames(module.rejector)
    module.rejector.__class__ = (
        UuO2TensorProductConv
        if type(module) is UuO2Interaction
        else UvO2TensorProductConv
    )
    return module


def local_rule(module, *, backend, task, canonical=False):
    """Match exact host local maps, never magnetic or Cartesian derivatives."""
    if type(module) in (_StreamUu, _StreamUv):
        if backend == "torch":
            return (
                _restore_stream,
                "requested Torch: restored the host local O(2) interaction",
            )
        return None, "already uses the canonical streamed local O(2) convolution"
    if type(module) not in (UuO2Interaction, UvO2Interaction):
        return None, "requires an ordinary spherical Uu or Uv O(2) interaction"
    conv = module.rejector
    expected = (
        UuO2TensorProductConv
        if type(module) is UuO2Interaction
        else UvO2TensorProductConv
    )
    if type(conv) in (_UuFrames, _UvFrames):
        if backend == "torch" and not module.use_eqx:
            return (
                _restore,
                "requested Torch: restored the native local-frame reduction",
            )
        return None, "already scheduled local O(2) frames"
    if type(conv) is not expected:
        return None, "requires the spherical TACE local-frame reduction"
    if module.use_eqx:
        return None, "preserved the host's already-enabled streamed EQX convolution"
    if backend != "cuda" or task not in ("inference", "training") or torch.version.hip:
        return None, "local-frame schedule targets NVIDIA CUDA"
    if canonical and _stream_eligible(module):
        if (
            type(module) is UvO2Interaction
            and (expression := _uv_gate_expression(module)) is not None
        ):
            return partial(
                _apply_stream, gate_expression=expression
            ), "streamed local O(2) convolution and CUDA spherical node gate"
        return (
            _apply_stream,
            "streamed local O(2) convolution with canonical Wigner frames",
        )
    # The extra indexed-launch overhead loses to the host on scalar/low-order
    # layers. Keep unsampled high degrees on the host too. Inspect the actual
    # representations: the first layer can be scalar despite a larger model lmax.
    if not (2 <= conv.irreps_in.lmax <= 4 and 2 <= conv.irreps_out.lmax <= 4):
        return None, "kept host frames outside measured input/output degree range 2..4"
    if conv.eqx_tp is None:
        return None, "unsupported local activation or tensor product"
    if custom_forward(conv, recursive=True):
        return None, "preserved the customized host frame or local map"
    p = next(module.parameters())
    if not p.is_cuda or p.dtype not in (torch.float32, torch.float64):
        return None, "requires CUDA FP32 or FP64"
    return _apply, "EQX local-frame reduction on host rotations and live radial weights"

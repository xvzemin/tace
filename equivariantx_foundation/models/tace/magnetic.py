"""Use the magnetic Uv program with the host's magnetic attributes and frames."""

import torch

from equivariantx import o2
from equivariantx.conv.uv_o2.convolution import (
    UvO2TensorProductConv,
    linear_description,
)
from equivariantx.ops.common.gate import gate_program
from equivariantx.ops.common.wigner import wigner_D
from equivariantx.utils.autograd import forward_ad_active
from equivariantx.utils.frame_program import frame_description
from tace.models.spherical.edge import IdentityEdgeEmbedding
from tace.models.spherical.mtace import (
    MagneticUvO2TensorProductConv,
    O2MagneticInteraction,
)
from tace.models.spherical.node import LinearNodeEmbedding
from tace.models.spherical.o2 import uv_convolution
from tace.models.spherical.representation import NodeRepresentation
from tace.utils.env import memory_first

from .local import _canonical_wigner_tables, _eligible_inputs
from .matching import custom_forward, tracing_rows


class _MagneticUvFrames(MagneticUvO2TensorProductConv):
    def forward(
        self,
        node_feats,
        magnetic_edge_attrs,
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_radial_basis=None,
        edge_cutoff=None,
        radial_network=None,
        edge_pairs=None,
    ):
        # The scalar-spin program uses precomputed weights. A shared packed
        # frame is also valid on the verified three-layer full-order stack.
        scalar_spin = self.magnetic_edge_irreps.lmax == 0
        packed = (
            scalar_spin
            and getattr(self, "_foundation_packed_spin", False)
            and wigner is not None
            and wigner.ndim == 2
            and wigner_inv is wigner
            and not tracing_rows(edge_index)
            and wigner.shape == (edge_index.size(1), 35)
            and node_feats.dtype == torch.float32
            and wigner.device == edge_index.device == node_feats.device
        )
        scalar_inputs = not scalar_spin or (
            isinstance(conv_weights, torch.Tensor)
            and conv_weights.ndim == 2
            and conv_weights.dtype == node_feats.dtype
            and conv_weights.device == magnetic_edge_attrs.device == node_feats.device
            and radial_network is None
            and edge_pairs is None
            and wigner is not None
            and wigner_inv is not None
            and (packed or (wigner.ndim == 3 and wigner_inv.ndim == 3))
        )
        # Packed frames have implicit inverse by definition in the host forward.
        if wigner is not None and wigner.ndim == 2:
            wigner_inv = wigner
        if not (
            scalar_inputs
            and _eligible_inputs(
                node_feats, wigner, wigner_inv, edge_cutoff, edge_index
            )
            and magnetic_edge_attrs.dtype == node_feats.dtype
            and (node_feats.dtype == torch.float32 or not self.use_time_reversal)
            and self.eqx_tp.backend == "cuda"
        ):
            return super().forward(
                node_feats,
                magnetic_edge_attrs,
                conv_weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_radial_basis,
                edge_cutoff,
                radial_network,
                edge_pairs,
            )
        return uv_convolution(
            self,
            node_feats,
            magnetic_edge_attrs,
            conv_weights,
            edge_index,
            wigner,
            wigner_inv,
            edge_cutoff,
            radial_network=radial_network,
            edge_pairs=edge_pairs,
        )


def _lower(module):
    module.rejector.__class__ = _MagneticUvFrames
    return module


def _restore(module):
    vars(module.rejector).pop("_foundation_packed_spin", None)
    module.rejector.__class__ = MagneticUvO2TensorProductConv
    return module


class _ScalarSpinRepresentation(NodeRepresentation):
    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        # Do not retain old device buffers across .to()/.double(). A fresh
        # plan can verify replacements; until then the host reads them live.
        if any(
            self.o2_angular_basis._buffers.get(n) is not value
            for n, value, _ in self._foundation_wigner_tables
        ):
            self._foundation_wigner_tables = ()
        return self

    def _spatial_frames(self, data, graph):
        idx, v, frame = data["edge_index"], graph.edge_vector, self.o2_angular_basis
        if (
            not tracing_rows(idx)
            and not torch.jit.is_tracing()
            and v.is_cuda
            and self.model_dtype == torch.float32
            and not torch.is_autocast_enabled("cuda")
            and not forward_ad_active()
            and not torch.cuda.is_current_stream_capturing()
            and memory_first()
            and not graph.lmp
            and graph.lmp_data is None
            and "edge_pair_rows" not in data
            and idx.dtype == torch.long
            and idx.ndim == 2
            and idx.shape[0] == 2
            and idx.shape[1] > 0
            and v.shape == (idx.shape[1], 3)
            and idx.device == v.device
            and self.use_local_frame
            and not self.use_so2
            and type(self.node_embedding) is LinearNodeEmbedding
            and not custom_forward(self.node_embedding)
            and type(frame) is o2.WignerD
            and frame.lmax == frame.mmax == 2
            and frame.backend == "torch"
            and frame.method == "auto"
            and not custom_forward(frame)
            and not {"forward_packed", "from_matrix"}.intersection(vars(frame))
            and len(self.interactions) == 3
            and all(
                type(layer) is O2MagneticInteraction
                and not layer.use_eqx
                and type(layer.rejector) is _MagneticUvFrames
                and getattr(layer.rejector, "_foundation_packed_spin", False)
                and layer.rejector.eqx_tp.backend == "cuda"
                and not any(
                    custom_forward(m)
                    for m in (layer, layer.rejector, layer.rejector.eqx_tp)
                )
                for layer in self.interactions
            )
            # Five tiny, plan-validated angular tables. Versions/identity
            # are CPU metadata: no GPU sync or comparison in the hot path.
            and not frame._parameters
            and self._foundation_wigner_tables
            and all(
                frame._buffers.get(name) is original
                and not original.requires_grad
                and original._version == version
                for name, original, version in self._foundation_wigner_tables
            )
        ):
            packed = wigner_D(frame, v, backend="cuda")
            return packed, packed
        return super()._spatial_frames(data, graph)


def configure_scalar_spin_frames(rep):
    """Publish packed frames only when every spatial-frame consumer accepts them."""
    if type(rep) is _ScalarSpinRepresentation:
        rep.__class__ = NodeRepresentation
        del rep._foundation_wigner_tables
        for layer in rep.interactions:
            if type(layer) is O2MagneticInteraction:
                vars(layer.rejector).pop("_foundation_packed_spin", None)
    if not (
        type(rep) is NodeRepresentation
        and getattr(rep, "model_dtype", None) == torch.float32
        and memory_first()
        and len(rep.interactions) == 3
        and type(rep.node_embedding) is LinearNodeEmbedding
        and type(rep.edge_embedding) is IdentityEdgeEmbedding
        and getattr(rep, "_can_pack_wigner", False)
        and not getattr(rep, "_can_skip_wigner", True)
        and getattr(rep, "use_local_frame", False)
        and not rep.use_so2
        and all(
            type(layer) is O2MagneticInteraction
            and not layer.use_eqx
            and type(layer.rejector) is _MagneticUvFrames
            and layer.num_channel == 48
            and next(layer.parameters()).is_cuda
            and _scalar_spin_contract(layer, layer.rejector)
            for layer in rep.interactions
        )
        and not custom_forward(rep, recursive=True)
        and "_spatial_frames" not in vars(rep)
    ):
        return
    frame = rep.o2_angular_basis
    if not (
        type(frame) is o2.WignerD
        and frame.lmax == frame.mmax == 2
        and frame.backend == "torch"
        and frame.method == "auto"
        and not {"forward_packed", "from_matrix"}.intersection(vars(frame))
        and not frame._parameters
        and all(not value.is_inference() for value in frame.buffers())
        and _canonical_wigner_tables(frame)
    ):
        return
    rep._foundation_wigner_tables = tuple(
        (name, value, value._version) for name, value in frame.named_buffers()
    )
    for layer in rep.interactions:
        layer.rejector._foundation_packed_spin = True
    rep.__class__ = _ScalarSpinRepresentation


def _same_program(conv):
    kernel = conv.eqx_tp
    frames = (conv.local_frame_in, conv.magnetic_frame, conv.local_frame_out)
    maps = (conv.linear_up, conv.linear_down)
    if not (
        type(kernel) is UvO2TensorProductConv
        and all(type(frame) is o2.LocalFrame for frame in frames)
        and all(type(linear) is o2.Linear for linear in maps)
        and type(conv.nonlinearity) is o2.Gate
        and not any(
            name in conv.__dict__
            for name in ("_to_local", "_convolution", "_to_global")
        )
        and all(
            not {"to_local", "to_global"}.intersection(frame.__dict__)
            for frame in frames
        )
    ):
        return False
    try:
        return (
            kernel.frames == tuple(map(frame_description, frames))
            and kernel.linears == tuple(map(linear_description, maps))
            and kernel.gates == (gate_program(conv.nonlinearity, blocks=True),)
            and kernel.edge_irreps == conv.local_magnetic_irreps
            and kernel.irreps_in == conv.local_irreps_in
            and kernel.irreps_out == conv.local_irreps_out
            and kernel.degree == max(frame.lmax for frame in frames)
            and kernel.node_irreps == conv.node_irreps
        )
    except NotImplementedError:
        return False


def _measured_time_odd_gates(conv, prototype):
    """The full-order FP32 SOC program tested with explicit T-e3nn time irreps."""
    from e3nn.math import normalize2mom

    gate = conv.nonlinearity
    if not (
        conv.use_time_reversal
        and prototype.dtype == torch.float32
        and conv.magnetic_edge_irreps.lmax == 2
        and memory_first()
        and type(gate) is o2.Gate
        and any(p.ir_gate.t == -1 and p.ir_gated.t == -1 for p in gate._paths)
    ):
        return False
    # Tanh is odd on signed scalars. Only the host's measured normalized
    # activations and odd-gate × odd-feature → time-even paths are admitted.
    for act, invariant_fn in (
        (gate.act_scalars, torch.nn.SiLU),
        (gate.act_gates, torch.nn.Sigmoid),
    ):
        if type(act) is not o2.Activation or len(act.irreps_in) != len(act.acts):
            return False
        if not all(
            type(fn) is normalize2mom
            and type(getattr(fn, "f", None))
            is (invariant_fn if ir.is_invariant_scalar() else torch.nn.Tanh)
            for (ir, _), fn in zip(act.irreps_in, act.acts, strict=True)
        ):
            return False
    return all(
        p.ir_gate.t == 1
        or (p.ir_gate.p == 1 and p.ir_gated.t == -1 and p.ir_out.t == 1)
        for p in gate._paths
    )


def _scalar_spin_contract(module, conv):
    """Three even spin contractions projected by the host to one channel bank."""

    def even_scalars(irreps, width):
        return (
            irreps is not None
            and len(irreps) == 1
            and (
                irreps[0].mul,
                irreps[0].ir.l,
                irreps[0].ir.p,
                getattr(irreps[0].ir, "t", 1),
            )
            == (width, 0, 1, 1)
        )

    return (
        even_scalars(getattr(module, "magnetic_edge_irreps", None), 3)
        and even_scalars(conv.magnetic_edge_irreps, module.num_channel)
        and conv.magnetic_edge_irreps == module.magnetic_edge_irreps_out
        and conv.irreps_out.lmax == 2
        and conv.irreps_in.lmax in (0, 2)
        and conv.mmax == conv.irreps_in.lmax
        and not conv.use_time_reversal
        and getattr(conv.eqx_tp, "with_transverse", None) is False
        and not custom_forward(module, recursive=True)
        and not {
            "_compute_messages",
            "_apply_rejector",
            "_magnetic_weights",
        }.intersection(vars(module))
    )


def magnetic_rule(module, *, backend, task, automatic=False):
    """Select the magnetic Uv reduction for inference or training."""
    if type(module) is not O2MagneticInteraction:
        return None, "requires the ordinary spherical magnetic Uv interaction"
    conv = module.rejector
    if type(conv) is _MagneticUvFrames:
        if backend == "torch" and not module.use_eqx:
            return _restore, "restore the host's magnetic Uv convolution"
        return None, "already uses the scheduled magnetic Uv convolution"
    if type(conv) is not MagneticUvO2TensorProductConv:
        return None, "preserved the host's custom magnetic convolution"
    if backend != "cuda" or task not in ("inference", "training") or torch.version.hip:
        return None, "magnetic Uv kernel requires NVIDIA CUDA training or inference"
    if module.use_eqx:
        return None, "preserved the host's enabled EQX magnetic convolution"
    if conv.eqx_tp is None or conv.eqx_tp.backend != "cuda":
        return None, "no CUDA program for this magnetic activation or coupling"
    p = next(module.parameters())
    if (
        conv.use_time_reversal
        or any(ir.t == -1 for ir, _ in conv.local_irreps_in + conv.local_irreps_out)
    ) and not _measured_time_odd_gates(conv, p):
        return None, "preserved unevaluated explicitly time-odd gate paths"
    rank = conv.magnetic_edge_irreps.lmax
    # Ranks one and three reduce training peaks on the full magnetic model.
    # They do not consistently reduce latency. Keep the previous speed-first
    # policy for these ranks rather than using inference timing to select training.
    if rank not in (0, 2) and not memory_first():
        return None, "kept the host magnetic convolution in the speed-first plan"
    if rank == 0:
        if automatic and not (
            memory_first() and module.num_channel == 48 and p.dtype == torch.float32
        ):
            return (
                None,
                "kept the host outside the FP32 C48 memory-first scalar-spin plan",
            )
        if not _scalar_spin_contract(module, conv):
            return None, "preserved the host's scalar-spin magnetic interaction"
    elif not (
        rank in (1, 2, 3)
        and conv.mmax == conv.irreps_out.lmax == rank
        and conv.irreps_in.lmax <= rank
    ):
        return None, "kept the host outside the measured full-order spin-orbit schedule"
    if any(
        custom_forward(child) or getattr(child, "use_lora", False)
        for child in conv.modules()
    ):
        return None, "preserved customized local magnetic frames, gate or channel maps"
    if not p.is_cuda or p.dtype not in (torch.float32, torch.float64):
        return None, "requires CUDA FP32 or FP64"
    if not _same_program(conv):
        return (
            None,
            "preserved the host's modified magnetic frame, gate or channel paths",
        )
    if rank == 0:
        return (
            _lower,
            "EQX local Uv on host-projected scalar spin edges and spatial frames",
        )
    return (
        _lower,
        "EQX magnetic Uv contraction/receiver sum on host spin features and frames",
    )

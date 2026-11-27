"""Local schedules for the weighted spherical TACE message reduction."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from sys import modules

import torch
from e3nn import o3

from equivariantx.o3 import O3TensorProductConv as EQXConvolution
from equivariantx.ops.precision import cpu_matmul_allowed
from tace.models.mlp import MLPV1, MLPV2
from tace.models.spherical.fused import O3TensorProductConv
from tace.models.spherical.inter import O3CgtpInteraction
from tace.utils.env import acceleration_enabled, memory_first

from .matching import (
    canonical_component_harmonics,
    canonical_radial,
    custom_forward,
    directed_radial,
    mapped_radial_inputs,
    streamable_radial_mlp,
    tracing_rows,
)


class _ScheduledConvolution(O3TensorProductConv):
    """Consume *the host's* angular features, radial weights and edge ordering.

    Only the gather/product/sum is replaced. In particular we never synthesize
    harmonics from coordinates: a layer's angular basis and cutoff conventions
    need not be those of a released checkpoint.
    """

    def forward(self, x, y, w, edge_index, *, num_nodes=None):
        if (
            x.device.type not in ("cpu", "cuda")
            or "forward_stream" in vars(self)
            or x.dtype not in (torch.float32, torch.float64)
            or y.dtype != x.dtype
            or w.dtype != x.dtype
            or torch.is_autocast_enabled(x.device.type)
            or (
                x.device.type == "cpu"
                and x.dtype == torch.float32
                and not cpu_matmul_allowed()
            )
            or edge_index.size(1) == 0
        ):
            return super().forward(x, y, w, edge_index, num_nodes=num_nodes)
        return self.forward_stream(
            x,
            y,
            w,
            w.new_empty((0, self.weight_numel)),
            edge_index,
            None,
            num_nodes=num_nodes,
        )


class StreamedInteraction(O3CgtpInteraction):
    """Use the host's streamed radial convolution for canonical spherical harmonics."""

    @property
    def use_eqx(self):
        p = next(self.parameters())
        return (
            p.is_cuda
            and p.dtype in (torch.float32, torch.float64)
            and not torch.is_autocast_enabled("cuda")
        )


class _SharedHarmonicInteraction(StreamedInteraction):
    def _compute_messages(self, *args, graph=None, **kwargs):
        # In O3CgtpInteraction this private argument is used only to synthesize
        # harmonics from vectors. NodeRepresentation already supplies those
        # harmonics to each layer, in node dtype and with autograd intact.
        return super()._compute_messages(*args, graph=None, **kwargs)


class _HostFrameRadialInteraction(O3CgtpInteraction):
    """Stream directed radial features without advertising a new frame format."""

    LAST_PROJECTION_BYTES = 128 << 20
    LAST_PROJECTION_EXPANSION = 8

    def _stream_wide_projection(self, edge_index, dtype):
        if dtype != torch.float32 or tracing_rows(edge_index):
            return False
        last = self.edge_info.mlp[-1]
        return (
            last.out_dim >= self.LAST_PROJECTION_EXPANSION * last.in_dim
            and edge_index.size(1) * last.out_dim * 4 >= self.LAST_PROJECTION_BYTES
        )

    def _compute_messages(
        self,
        node_feats,
        node_types,
        edge_radial_basis,
        edge_feats,
        edge_attrs,
        edge_index,
        edge_cutoff,
        edge_wigner=None,
        edge_wigner_inv=None,
        magnetic_node_info=None,
        magnetic_edge_attrs=None,
        graph=None,
        element=None,
        edge_pairs=None,
    ):
        # Keep host use_eqx: NodeRepresentation also consults that property
        # when deciding whether other layers must accept packed/indexed inputs.
        distributed = getattr(graph, "lmp_data", None) is not None
        if (
            node_feats.is_cuda
            and node_feats.dtype in (torch.float32, torch.float64)
            and not torch.is_autocast_enabled("cuda")
            and not self.use_eqx
            and element is None
            and edge_pairs is None
            and (not getattr(graph, "lmp", False) or distributed)
            and edge_index.shape[1] != 0
            and edge_attrs is not None
            and edge_attrs.dtype == node_feats.dtype
            and (
                edge_cutoff is None
                or (
                    edge_cutoff.shape == (edge_index.shape[1], 1)
                    and edge_cutoff.dtype == node_feats.dtype
                )
            )
            # The stream calls forward_stream and reads the MLP's live
            # weights directly; newly attached host hooks must still fire.
            and not custom_forward(self.rejector)
            and not custom_forward(self.edge_info, recursive=True)
            and "_conv_weights" not in vars(self)
            and "_radial_projection" not in vars(self)
            and "forward_stream" not in vars(self.rejector)
            and (
                memory_first()
                or (
                    distributed
                    and self._stream_wide_projection(edge_index, node_feats.dtype)
                )
            )
        ):
            explicit = isinstance(edge_feats, tuple) and tracing_rows(edge_index)
            if explicit or directed_radial(edge_feats, edge_index):
                radial, projection, network = self._radial_projection(edge_feats)
                if explicit:
                    # Row maps are runtime operands, not alias guards on edge_index.
                    radial, network = mapped_radial_inputs(
                        edge_feats, network, edge_index, node_feats
                    )
                return self.rejector.forward_stream(
                    node_feats,
                    edge_attrs,
                    radial,
                    projection,
                    edge_index,
                    edge_cutoff,
                    radial_network=network,
                    num_nodes=graph.lmp_natoms[0] if distributed else None,
                )
        return super()._compute_messages(
            node_feats,
            node_types,
            edge_radial_basis,
            edge_feats,
            edge_attrs,
            edge_index,
            edge_cutoff,
            edge_wigner,
            edge_wigner_inv,
            magnetic_node_info,
            magnetic_edge_attrs,
            graph=graph,
            element=element,
            edge_pairs=edge_pairs,
        )


_SPHERICAL_TYPES = (
    O3CgtpInteraction,
    StreamedInteraction,
    _SharedHarmonicInteraction,
    _HostFrameRadialInteraction,
)


@dataclass(frozen=True, eq=False)
class ComponentPolynomial:
    """Evaluate the degree-three e3nn component polynomials on CUDA rows."""

    reference: Callable
    program: str

    def __deepcopy__(self, memo):
        # The original e3nn evaluator may be a ScriptFunction; neither
        # evaluator carries tensors or mutable model state.
        return self

    def __call__(self, degree, x, y, z):
        if degree != 3 or not x.is_cuda or x.dtype != torch.float64 or x.ndim != 1:
            return self.reference(degree, x, y, z)
        from equivariantx.ops.common.edge import edge_program

        xyz = torch.stack((x, y, z), dim=-1)
        rows = torch.arange(x.size(0), device=x.device)
        return edge_program("cuda", self.program, 0, rows, rows, xyz, inline=False)[0]


def schedule_polynomial(basis):
    from equivariantx.utils.harmonics import harmonic_row_program

    basis.sph_func = ComponentPolynomial(basis.sph_func, harmonic_row_program(3))
    return basis


def restore_polynomial(basis):
    basis.sph_func = basis.sph_func.reference
    return basis


def harmonic_region(rep, backend, *, automatic):
    """Select the shared component harmonics for a spherical CGTP stack."""
    from tace.models.spherical.representation import NodeRepresentation

    from .planner import Region

    if type(rep) is not NodeRepresentation:
        return None
    basis = rep.o3_angular_basis
    evaluator = getattr(basis, "sph_func", None)
    path = "o3_angular_basis"
    enabled = (
        backend == "cuda"
        and not torch.version.hip
        and memory_first()
        and (not automatic or torch.get_float32_matmul_precision() == "highest")
    )
    if type(evaluator) is ComponentPolynomial:
        if not enabled:
            return Region(
                path,
                "torch",
                "restore the host harmonic evaluator",
                restore_polynomial,
                "regional",
                "e3nn",
            )
        return Region(
            path,
            "cuda",
            "shared component polynomial",
            None,
            "already_regional",
            "eqx_cuda",
        )
    if (
        not enabled
        or evaluator is None
        or rep.use_local_frame
        or not canonical_component_harmonics(basis, 3)
        or len(rep.interactions) != 2
        or not all(
            type(m) in (O3CgtpInteraction, StreamedInteraction)
            and (m.num_channel, m.Lmax, m.lmax) == (64, 2, 3)
            and not hasattr(m.rejector, "fused_tp")
            and (type(m) is StreamedInteraction or not m.use_eqx)
            for m in rep.interactions
        )
        or next(rep.parameters()).dtype != torch.float32
    ):
        return None
    return Region(
        path,
        "cuda",
        "shared component polynomial for FP64 edge directions",
        schedule_polynomial,
        "regional",
        "eqx_cuda",
    )


def _weighted_uvu(tp):
    return (
        bool(tp.instructions)
        and all(p.connection_mode == "uvu" and p.has_weight for p in tp.instructions)
        and not tp.internal_weights
        and not tp.shared_weights
    )


def host_interaction_provider(layer):
    """Identify the unchanged convolution selected by the host on this device."""
    import equivariantx as eqx
    from tace.models.spherical import fused, inter, mtace, o2

    types = {
        inter.O3CgtpInteraction: (fused.O3TensorProductConv, EQXConvolution),
        inter.O3IctdInteraction: (
            fused.O3CartesianTensorProductConv,
            eqx.co3.CartesianTensorProductConv,
        ),
        inter.O2CgtpInteraction: (
            fused.O2TensorProductConv,
            eqx.o2.O2O3TensorProductConv,
        ),
        inter.UuO2Interaction: (
            fused.UuO2TensorProductConv,
            eqx.o2.UuO2TensorProductConv,
        ),
        inter.UvO2Interaction: (
            o2.UvO2TensorProductConv,
            eqx.o2.UvO2TensorProductConv,
        ),
        mtace.O2MagneticInteraction: (
            mtace.MagneticUvO2TensorProductConv,
            eqx.o2.UvO2TensorProductConv,
        ),
    }
    expected = types.get(type(layer))
    if expected is None or type(layer.rejector) is not expected[0]:
        return None
    conv = layer.rejector
    param = next(layer.parameters())
    if (
        param.dtype not in (torch.float32, torch.float64)
        or param.device.type not in ("cuda", "cpu")
        or custom_forward(layer, recursive=True)
        or any(k in vars(layer) for k in ("_compute_messages", "_apply_rejector"))
        or "forward_stream" in vars(conv)
    ):
        return None
    if layer.use_eqx:
        kernel = getattr(conv, "eqx_tp", None)
        if type(kernel) is not expected[1] or kernel.backend not in ("torch", "cuda"):
            return None
        if param.is_cuda:
            return f"eqx_{kernel.backend}"
        # Ordinary spherical CGTP dispatches its EQX operation on CPU too.
        # A Uv model on CPU merely owning a CUDA program is not a CUDA hit.
        return "eqx_torch" if type(layer) is inter.O3CgtpInteraction else None
    if type(layer) is inter.O3CgtpInteraction and param.is_cuda:
        fused_tp = getattr(conv, "fused_tp", None)
        for package, name, provider in (
            ("lunaequivariance", "LunaTensorProductConv", "luna"),
            ("openequivariance", "OeqTensorProductConv", "oeq"),
            ("cuequivariance", "CueTensorProductConv", "cueq"),
        ):
            wrapper = modules.get(f"tace.models.backends.{package}")
            if type(fused_tp) is getattr(wrapper, name, None):
                return provider
    return None


def canonical_geometry(rep):
    """Whether the host constructs component-normalized natural spherical harmonics."""
    from tace.models.spherical.representation import NodeRepresentation

    if type(rep) is NodeRepresentation and rep.use_local_frame:
        from .local import mixed_geometry

        return mixed_geometry(rep)
    basis = getattr(rep, "o3_angular_basis", None)
    return (
        type(rep) is NodeRepresentation
        and not rep.use_local_frame
        and bool(rep.interactions)
        and all(
            type(layer) in (O3CgtpInteraction, StreamedInteraction)
            for layer in rep.interactions
        )
        and canonical_component_harmonics(basis, rep.interactions[0].irreps_sh.lmax)
        and all(layer.irreps_sh == basis.irreps_out for layer in rep.interactions)
        and not custom_forward(rep)
        and (not rep.reciprocal_radial or canonical_radial(rep))
    )


def grouped_radial_geometry(rep):
    """Whether every layer can consume the host's reciprocal degree-wise radial MLP.

    The first interaction enables reciprocal inputs for the whole representation.
    Match every consumer, not just the first compatible layer.
    """
    from itertools import accumulate

    from tace.models.mlp import BatchMLPV1, BatchMLPV2
    from tace.models.physical_basis import (
        ExpWeightedjnSphericalBesselBasis,
        SpacePhysicalBasis,
    )
    from tace.models.radial import C2PolynomialCutoff, jnSphericalBesselBasis
    from tace.models.spherical.edge import IdentityEdgeEmbedding, IdentityEdgeUpdate

    if not (
        rep.reciprocal_radial
        and type(rep.radial_basis.radial_fn)
        in (
            jnSphericalBesselBasis,
            ExpWeightedjnSphericalBesselBasis,
            SpacePhysicalBasis,
        )
        and type(rep.radial_basis.cutoff_fn) is C2PolynomialCutoff
        and not rep.radial_basis.use_distance_transform
        and type(rep.edge_embedding) is IdentityEdgeEmbedding
        and all(type(m) is IdentityEdgeUpdate for m in rep.edge_updates)
        and not custom_forward(rep, recursive=True)
    ):
        return False
    for layer in rep.interactions:
        radial = layer.edge_info
        if not (
            type(layer.rejector) in (O3TensorProductConv, _ScheduledConvolution)
            and not hasattr(layer.rejector, "fused_tp")
            and type(layer.rejector.tp) is o3.TensorProduct
            and _weighted_uvu(layer.rejector.tp)
            and type(radial) in (BatchMLPV1, BatchMLPV2)
            and radial.context_features == 0
            and radial.chain_steps
            and all(streamable_radial_mlp(branch) for branch in radial.branches)
            and not any(
                name in vars(radial)
                for name in (
                    "stream",
                    "packed",
                    "evaluate_groups",
                    "to_edge_program",
                    "stream_rows",
                )
            )
            and not any(
                name in vars(layer)
                for name in ("_radial_projection", "_conv_weights", "_compute_messages")
            )
            and "forward_stream" not in vars(layer.rejector)
        ):
            return False
        # The materialized path uses these live permutation buffers, whereas
        # the edge program uses path slices. Do not silently replace a custom
        # permutation with the layout originally inferred from the CGTP paths.
        offsets = tuple(accumulate(radial.branch_widths, initial=0))
        columns = [
            offsets[i] + j
            for i, start, width in radial.outputs
            for j in range(start, start + width)
        ]
        if columns == list(range(radial.out_features)):
            if radial.output_order is not None or radial.inverse_order is not None:
                return False
        else:
            order = torch.tensor(columns, device="cpu")
            if not (
                radial.output_order is not None
                and radial.inverse_order is not None
                and radial.output_order.dtype
                == radial.inverse_order.dtype
                == torch.int64
                and torch.equal(radial.output_order.detach().cpu(), order)
                and torch.equal(radial.inverse_order.detach().cpu(), order.argsort())
            ):
                return False
    return True


def _mixed_o3_layer(layer, basis):
    """The mixed contract may enable reciprocal weights for *every* layer."""
    if type(layer) not in _SPHERICAL_TYPES:
        return False
    conv = layer.rejector
    return (
        type(conv) in (O3TensorProductConv, _ScheduledConvolution)
        and type(getattr(conv, "eqx_tp", None)) is EQXConvolution
        and type(conv.tp) is o3.TensorProduct
        and conv.tp.irreps_in2 == layer.irreps_sh == basis.irreps_out
        and _weighted_uvu(conv.tp)
        and not hasattr(conv, "fused_tp")
        and type(layer.edge_info) in (MLPV1, MLPV2)
        and not any(
            name in vars(layer)
            for name in ("_compute_messages", "_conv_weights", "_radial_projection")
        )
        and "forward_stream" not in vars(conv)
        and "stream" not in vars(layer.edge_info)
        and not any(getattr(m, "use_lora", False) for m in layer.edge_info.modules())
    )


def rotary_radial_geometry(rep):
    """Directed O3/legacy rotary stack; only the O3 consumers are changed."""
    from tace.models.spherical.edge import (
        Element2EdgeUpdate,
        ElementEdgeEmbedding,
        IdentityEdgeUpdate,
    )
    from tace.models.spherical.representation import NodeRepresentation
    from tace.models.spherical.tece_oam_rra import UvSO2Interaction

    from .tece_oam_rra.regional import StreamedRotaryInteraction

    if (
        type(rep) is not NodeRepresentation
        or not rep.use_so2
        or not rep.use_local_frame
        or not rep.use_o3_angular_basis
        or rep.reciprocal_radial
        or rep.use_magnetic_interaction
        or isinstance(rep.edge_embedding, ElementEdgeEmbedding)
        or acceleration_enabled("eqx", kernel="conv")
    ):
        return False
    o3_layers = [
        m
        for m in rep.interactions
        if type(m) in (O3CgtpInteraction, _HostFrameRadialInteraction)
    ]
    rotary = (UvSO2Interaction, StreamedRotaryInteraction)
    return bool(
        o3_layers
        and len(o3_layers) < len(rep.interactions)
        and all(
            type(m) in (O3CgtpInteraction, _HostFrameRadialInteraction, *rotary)
            for m in rep.interactions
        )
        and len(rep.edge_updates) == len(rep.interactions)
        and all(
            type(m) in (IdentityEdgeUpdate, Element2EdgeUpdate)
            for m in rep.edge_updates
        )
        and canonical_component_harmonics(
            rep.o3_angular_basis, o3_layers[0].irreps_sh.lmax
        )
        and all(_mixed_o3_layer(m, rep.o3_angular_basis) for m in o3_layers)
        and not custom_forward(rep, recursive=True)
    )


def _lower(module, *, backend, stream=False, shared_radial=False):
    conv = module.rejector
    if getattr(getattr(conv, "eqx_tp", None), "backend", None) != backend:
        prototype = next(module.parameters())
        conv.eqx_tp = EQXConvolution(
            conv.tp,
            backend=backend,
            batch_paths=True,
            normalize=False,
            radial_mode="full" if memory_first() else "last",
        ).to(prototype)
    conv.__class__ = _ScheduledConvolution
    module.__class__ = (
        _HostFrameRadialInteraction
        if shared_radial
        else StreamedInteraction
        if stream
        else O3CgtpInteraction
    )
    return module


def interaction_rule(
    module, *, backend, task, canonical=False, shared_radial=False, grouped_radial=False
):
    """Return an in-place transform for a *copy* of an eligible host layer."""
    if type(module) not in _SPHERICAL_TYPES:
        return None, "requires an ordinary spherical O3CgtpInteraction"
    shared_radial = (
        shared_radial
        and backend == "cuda"
        and module.radial_widths is None
        and streamable_radial_mlp(module.edge_info)
        and "_radial_projection" not in vars(module)
        and "forward_stream" not in vars(module.rejector)
    )
    conv = module.rejector
    if type(conv) is _ScheduledConvolution:
        if type(conv.eqx_tp) is not EQXConvolution:
            raise ValueError(
                "Cannot replace a customized EQX convolution; use the original host model."
            )
        if module.use_eqx and not isinstance(module, StreamedInteraction):
            return None, "preserved the host's already-enabled streamed EQX convolution"
        if conv.eqx_tp.backend == backend:
            return None, f"already uses the scheduled {backend} EQX convolution"
        if backend in ("torch", "cuda") and task in ("training", "inference"):
            if custom_forward(conv.eqx_tp, recursive=True) or conv.eqx_tp.state_dict():
                raise ValueError(
                    "Cannot replace a customized EQX convolution; use the original host model."
                )
            stream = (
                canonical
                and backend == "cuda"
                and (grouped_radial or type(module.edge_info) in (MLPV1, MLPV2))
                and not custom_forward(conv)
                and not custom_forward(module.edge_info, recursive=True)
            )
            return partial(
                _lower, backend=backend, stream=stream, shared_radial=shared_radial
            ), f"switch scheduled EQX convolution to {backend}"
    if type(conv) is not O3TensorProductConv or type(conv.tp) is not o3.TensorProduct:
        return None, "requires the native e3nn spherical tensor product"
    if custom_forward(conv.tp, recursive=True):
        return None, "preserved the host's customized tensor product"
    if module.use_eqx:
        return None, "preserved the host's already-enabled streamed EQX convolution"
    if hasattr(conv, "fused_tp"):
        return None, "preserved the host's already-enabled convolution provider"
    if not _weighted_uvu(conv.tp):
        return None, "requires externally weighted per-edge uvu paths"
    if backend not in ("torch", "cuda") or task not in ("training", "inference"):
        return None, "requires torch/cuda eager training or inference"
    if next(module.parameters()).dtype not in (torch.float32, torch.float64):
        return None, "requires FP32 or FP64"
    stream = (
        canonical
        and backend == "cuda"
        and (grouped_radial or type(module.edge_info) in (MLPV1, MLPV2))
        and not custom_forward(conv)
        and not custom_forward(module.edge_info, recursive=True)
    )
    if shared_radial:
        return partial(_lower, backend=backend, shared_radial=True), (
            "streamed spherical CGTP and directed radial MLP on host harmonics"
        )
    if stream and grouped_radial:
        return partial(_lower, backend=backend, stream=True), (
            "streamed degree-wise CGTP radial MLP with reciprocal radial features"
        )
    detail = (
        "streamed spherical CGTP and radial projection"
        if stream
        else "EQX gather/CGTP/receiver reduction with host radial weights"
    )
    return partial(_lower, backend=backend, stream=stream), detail

"""Regional schedules for native Cartesian TACE (no change of angular basis)."""

from functools import partial

import torch

from equivariantx import co2, co3
from equivariantx.co3 import IctdUuuFullyConnectedSymmetricContraction
from equivariantx.co3.conv import CartesianTensorProductConv
from equivariantx.ops.common.cuda.graph import CAPTURE_CONSTANTS
from equivariantx.utils.autograd import forward_ad_active
from tace.models.adapter import TensorModel
from tace.models.cartesian.layers import CartesianTensorProductConvLayer
from tace.models.cartesian.local import (
    UuConvolution,
    UvConvolution,
    use_streamed_convolution,
)
from tace.models.cartesian.prod import NonlinearIctdACE
from tace.models.linear import e3nnElementLinear, e3nnLinear, has_lora
from tace.models.spherical.fused import UuuTensorProduct
from tace.models.spherical.inter import (
    O3CgtpInteraction,
    UuO2Interaction,
    UvO2Interaction,
)
from tace.models.spherical.prod import CgtpACE
from tace.utils.env import acceleration_enabled, memory_first

from .matching import (
    canonical_radial,
    custom_forward,
    streamable_radial_mlp,
    tracing_rows,
)
from .products import _ScheduledCgtpACE, _supported_precision, node_local_path_gates


class _ScheduledCartesianUu(UuConvolution):
    """Planar Cartesian restriction, channelwise UU, synthesis and receiver sum."""

    def forward(self, features, weights, edge_index, rotation, rotation_inv, cutoff):
        if (
            use_streamed_convolution(self.eqx_tp, features, edge_index)
            and rotation is not None
            and rotation is rotation_inv
            and cutoff is not None
            and features.dtype == weights.dtype == rotation.dtype == cutoff.dtype
            and not torch.is_autocast_enabled("cuda")
            and cutoff.ndim == 2
            and cutoff.shape[-1] == 1
        ):
            return (
                self.eqx_tp(features, weights, edge_index, rotation, cutoff)
                * self.scale
            )
        return super().forward(
            features, weights, edge_index, rotation, rotation_inv, cutoff
        )


def _frame_parts(conv):
    return tuple(
        (frame, frame.basis_change)
        for frame in (conv.local_frame_in, conv.local_frame_out)
    )


def _lower_local_uu(module):
    conv = module.rejector
    # Register only the parameter-free CO2 execution plan. Host frame/linear
    # modules remain registered exactly once at their canonical paths.
    conv.eqx_tp = (
        co2.UuTensorProductConv(
            conv.local_frame_in,
            conv.linear,
            conv.local_frame_out,
            backend="cuda",
            radial_mode="last",
        )
        .plan.to(next(module.parameters()))
        .train(module.training)
    )
    # These are aliases, not a second registered module tree. Bucket dispatch
    # can cross from native frames to the compiled program when padded; reject
    # a program (or fallback map) that was replaced independently of the other.
    conv._foundation_uu_parts = (_frame_parts(conv), conv.linear, conv.eqx_tp)
    conv.__class__ = _ScheduledCartesianUu
    return module


def _direct_frames(module, ranks):
    conv = module.rejector
    frames = (conv.local_frame_in, conv.local_frame_out)
    return (
        module.num_channel == 64
        and all(type(f) is co2.LocalFrame and f.method == "direct" for f in frames)
        and tuple((f.lmax, f.mmax) for f in frames) == ranks
    )


def _direct_uu_geometry(module, *, ranks=((2, 2), (2, 2))):
    conv = module.rejector
    return (
        _direct_frames(module, ranks)
        and type(conv.linear) is co2.UuLinear
        and conv.linear.path_mode == "sum"
    )


def _direct_uv_geometry(module, ranks):
    from e3nn.math import normalize2mom

    from equivariantx import o2

    c = module.rejector
    up, gate, down = c.linear_up, c.nonlinearity, c.linear_down
    return (
        _direct_frames(module, ranks)
        and c.magnetic_frame is None
        and "convolution" not in vars(c)
        and "to_local" not in vars(c.local_frame_in)
        and "to_global" not in vars(c.local_frame_out)
        and type(gate) is co2.Gate
        and all(
            type(layer) is co2.Linear
            and layer.internal_weights
            and layer.shared_weights
            and type(layer.projection) is torch.nn.Identity
            and layer.output_basis == "cartesian"
            for layer in (up, down)
        )
        and c.node_irreps == c.local_frame_in.irreps_out
        and list(up.irreps_in) == [(2 * mul, ir) for mul, ir in c.node_irreps]
        and c.local_irreps_in == up.irreps_in
        and up.irreps_out == gate.irreps_in
        and gate.irreps_out == down.irreps_in
        and down.irreps_out == c.local_frame_out.irreps_out
        and c.weight_numel == sum(mul for mul, _ in up.irreps_in)
        and all(
            type(block) is o2.Activation
            and all(
                act is None
                or (
                    type(act) is normalize2mom
                    and type(act.f)
                    in (
                        torch.nn.SiLU,
                        torch.nn.Tanh,
                        torch.nn.Sigmoid,
                        torch.nn.Identity,
                    )
                    and not getattr(act.f, "inplace", False)
                )
                for act in block.acts
            )
            for block in (gate.act_scalars, gate.act_gates)
        )
    )


def _uv_program_parts(c):
    gate = c.nonlinearity
    return (
        _frame_parts(c),
        c.eqx_tp,
        c.eqx_tp.metadata,
        c.linear_up,
        gate,
        c.linear_down,
        c.linear_up.layout,
        c.linear_down.layout,
        gate._paths,
        gate._scalar_locations,
        gate._gate_locations,
        gate._gated_locations,
        tuple(
            None if act is None else (act.f, act.cst, act._is_id)
            for block in (gate.act_scalars, gate.act_gates)
            for act in block.acts
        ),
    )


def _lower_local_uv(module):
    c = module.rejector
    c.eqx_tp = (
        co2.UvTensorProductConv(
            c.local_frame_in,
            c.linear_up,
            c.nonlinearity,
            c.linear_down,
            c.local_frame_out,
            backend="cuda",
            radial_mode="last",
        )
        .plan.to(next(module.parameters()))
        .train(module.training)
    )
    c._foundation_uv_parts = _uv_program_parts(c)
    return module


def _local_uv_rule(module, *, backend, task, automatic):
    c = module.rejector
    if type(c) is not UvConvolution or type(c.local_frame_in) is not co2.LocalFrame:
        return None, "retained the host Cartesian UV convolution"
    ref = next(module.parameters())
    scalar = c.local_frame_in.lmax == 0
    ranks = ((0, 0), (2, 0)) if scalar else ((2, 2), (2, 2))
    if (
        module.use_eqx
        or c.eqx_tp is not None
        or backend != "cuda"
        or task not in ("training", "inference")
        or not ref.is_cuda
        or torch.version.hip
        or ref.dtype != torch.float32
        or custom_forward(c, recursive=True)
        or not _direct_uv_geometry(module, ranks)
    ):
        return None, "retained the host Cartesian UV convolution"
    if automatic:
        measured_layout = (
            c.weight_numel == 128
            and c.linear_up.irreps_out.dim == c.linear_down.irreps_out.dim == 192
            and not c.nonlinearity._paths
            if scalar
            else c.weight_numel == 1408
        )
        if not (
            measured_layout
            and memory_first()
            and torch.get_float32_matmul_precision() == "highest"
        ):
            return (
                None,
                "retained the host outside the measured IEEE FP32 memory-first UV layout",
            )
    return (
        _lower_local_uv,
        "direct Cartesian UV Linear-Gate-Linear receiver for >=8192 edges with shared rotations",
    )


def cartesian_local_bucket_geometry(rep):
    """Check the planar receiver programs and any native scalar input layer."""
    from equivariantx.ops.common.cartesian_frame import CartesianEdgeProgram
    from tace.models.cartesian.local import RotationMatrices

    if not (
        rep.use_local_frame
        and rep.use_o2
        and not rep.use_so2
        and not rep.use_magnetic_interaction
        and not rep._can_pack_wigner
        and not rep._can_skip_wigner
        and type(rep.o2_angular_basis) is RotationMatrices
        and not rep.use_o3_angular_basis
        and rep.o3_angular_basis is None
        and rep.co3_angular_basis is None
    ):
        return False
    installed = False
    for index, m in enumerate(rep.interactions):
        if (
            type(m) not in (UuO2Interaction, UvO2Interaction)
            or m.use_eqx
            or "_compute_messages" in vars(m)
            or "_conv_weights" in vars(m)
            or "_apply_rejector" in vars(m)
        ):
            return False
        c = m.rejector
        if type(m) is UvO2Interaction:
            if (
                type(c) is not UvConvolution
                or type(c.local_frame_in) is not co2.LocalFrame
            ):
                return False
            scalar = index == 0 and c.local_frame_in.lmax == 0
            ranks = ((0, 0), (2, 0)) if scalar else ((2, 2), (2, 2))
            if not _direct_uv_geometry(m, ranks):
                return False
            if c.eqx_tp is None:
                if not scalar:
                    return False
            elif (
                type(c.eqx_tp) is CartesianEdgeProgram
                and c.eqx_tp.backend == "cuda"
                and getattr(c, "_foundation_uv_parts", ()) == _uv_program_parts(c)
            ):
                installed = True
            else:
                return False
            continue
        if type(c) not in (UuConvolution, _ScheduledCartesianUu) or (
            "to_global" in vars(c.local_frame_out)
            or "to_local" in vars(c.local_frame_in)
        ):
            return False
        if type(c) is _ScheduledCartesianUu:
            if not (
                _direct_uu_geometry(m)
                and type(c.eqx_tp) is CartesianEdgeProgram
                and c.eqx_tp.backend == "cuda"
                and getattr(c, "_foundation_uu_parts", ())
                == (_frame_parts(c), c.linear, c.eqx_tp)
            ):
                return False
            installed = True
        elif not (
            index == 0
            and type(c) is UuConvolution
            and c.eqx_tp is None
            and _direct_uu_geometry(m, ranks=((0, 0), (2, 0)))
        ):
            return False
    return installed


def _local_uu_rule(module, backend, task):
    conv = module.rejector
    if type(conv) is not UuConvolution or custom_forward(conv, recursive=True):
        return None, "requires the host's ordinary Cartesian UU frame map"
    if conv.eqx_tp is not None or module.use_eqx:
        return None, "preserved the host's existing Cartesian UU provider"
    ref = next(module.parameters())
    if (
        task not in ("training", "inference")
        or backend != "cuda"
        or not ref.is_cuda
        or torch.version.hip
        or ref.dtype != torch.float32
        or not _direct_uu_geometry(module)
    ):
        return (
            None,
            "Cartesian UU fused schedule requires C=64, rank/order 2, direct frames and FP32 CUDA",
        )
    return (
        _lower_local_uu,
        "CO2 direct UU receiver program for >=8192 edges and shared rotations; otherwise host",
    )


class _ScheduledIctdMessages(CartesianTensorProductConvLayer):
    supports_radial_stream = False

    def forward(self, features, edge_attrs, weights, edge_index, num_nodes=None):
        if (
            not _supported_precision(features)
            or features.dtype != edge_attrs.dtype
            or features.dtype != weights.dtype
            or edge_index.shape[1] == 0
        ):
            return super().forward(
                features, edge_attrs, weights, edge_index, num_nodes=num_nodes
            )
        # Even for non-STF inputs the result is exactly the host's. Do not
        # replace supplied attributes by computed harmonics or symmetrize
        # free tensor components before differentiating.
        result = self.scheduled_convolution(
            features,
            edge_attrs,
            weights,
            weights.new_empty((0, self.weight_numel)),
            edge_index,
            num_nodes=num_nodes,
        )
        return (
            result.index_select(-1, self.eqx_tp.output_pack_index)
            if self.compact_output
            else result
        )


def paired_cartesian_stack(model):
    """Measured memory-first E/F stack, with scalar, orientation-independent radial inputs."""
    from equivariantx.ops import config
    from tace.models.cartesian.tace import CartesianTACE
    from tace.models.radial import C2PolynomialCutoff, j0SphericalBesselBasis
    from tace.models.spherical.edge import IdentityEdgeEmbedding, IdentityEdgeUpdate
    from tace.models.spherical.representation import NodeRepresentation
    from tace.models.spherical.scatter_norm import AvgNumNeighborsScatterNorm

    rep = model.readout_fn.representation
    return (
        type(model) in (TensorModel, _PairedCartesianModel)
        and type(model.readout_fn) is CartesianTACE
        and type(rep) is NodeRepresentation
        and not {
            "prepare_graph",
            "prepare_property_data",
            "uses_reciprocal_weights",
        }.intersection(vars(model))
        and memory_first()
        and not model.lmp
        and not config.TORCH_REFERENCE
        and not torch.backends.cuda.matmul.allow_tf32
        and not torch.is_autocast_enabled("cuda")
        and model.model_dtype == rep.model_dtype == torch.float32
        and set(model.get_target_property()) <= {"energy", "forces", "node_energy"}
        and not (
            model.flags.compute_edge_forces
            or model.flags.compute_atomic_virials
            or model.flags.compute_atomic_stresses
        )
        and not any(custom_forward(m) for m in (model, model.readout_fn, rep))
        and rep.reciprocal_radial
        and rep.num_channel == 64
        and len(rep.interactions) == len(rep.products) == len(rep.edge_updates) == 2
        and (
            all(
                type(p) in (CgtpACE, _ScheduledIctdACE) and p.correlation == 3
                for p in rep.products
            )
            or all(
                type(p) is NonlinearIctdACE
                and p.correlation == 3
                # Check the measured native factor geometry only at selection.
                # Installing their independent product plan changes backends;
                # no gate, projection or product participates in radial sharing.
                and (type(model) is _PairedCartesianModel or _measured_ictd_product(p))
                for p in rep.products
            )
        )
        and type(rep.edge_embedding) is IdentityEdgeEmbedding
        and all(type(m) is IdentityEdgeUpdate for m in rep.edge_updates)
        and canonical_radial(rep)
        and type(rep.radial_basis.radial_fn) is j0SphericalBesselBasis
        and type(rep.radial_basis.cutoff_fn) is C2PolynomialCutoff
        and not rep.radial_basis.use_distance_transform
        and all(
            type(m)
            is (
                _PairedIctdInteraction
                if type(model) is _PairedCartesianModel
                else O3CgtpInteraction
            )
            and type(m.rejector)
            in (CartesianTensorProductConvLayer, _ScheduledIctdMessages)
            and not m.use_eqx
            and (m.Lmax, m.lmax, m.mmax) == (2, 3, 3)
            and type(m.scatter_norm) is AvgNumNeighborsScatterNorm
            and m.radial_widths is None
            and streamable_radial_mlp(m.edge_info)
            and m.edge_info.mlp[-1].in_dim >= 1024
            and m.edge_info.mlp[-1].bias is None
            and not any(
                k in vars(m)
                for k in ("_conv_weights", "_compute_messages", "_radial_projection")
            )
            and not any(custom_forward(x) for x in (m, m.rejector, m.scatter_norm))
            and not custom_forward(m.edge_info, recursive=True)
            for m in rep.interactions
        )
    )


class _PairedCartesianModel(TensorModel):
    _foundation_paired_ictd = True

    def _directed_replay_view(self):
        # Share the entire live readout (and all parameters/buffers). The bare
        # host drops pair maps before its forward: the two interactions then
        # take their original directed branch, even during warm-up.
        from copy import copy

        view = copy(self)
        view.__class__ = TensorModel
        return view

    def prepare_property_data(self, data):
        from tace.models.utils import prepare_edge_pairs

        # Preserve host preparation (including its independent generic EQX
        # policy). Only add a map for ordinary position-derived eager geometry.
        data = super().prepare_property_data(data)
        index = data.get("edge_index")
        shifts = data.get("edge_shifts")
        if (
            "edge_pair_rows" not in data
            and not self.lmp
            and index is not None
            and index.is_cuda
            and not tracing_rows(index)
            and not torch.jit.is_tracing()
            and not forward_ad_active()
            and CAPTURE_CONSTANTS.get() is None
            and not torch.cuda.is_current_stream_capturing()
            and index.shape[1] > 0
            and "edge_vector" not in data
            and shifts is not None
            and shifts.dtype in (torch.int32, torch.int64)
            and paired_cartesian_stack(self)
        ):
            prepare_edge_pairs(data)
        return data


def configure_paired_cartesian(model):
    # Called on the newly copied host, not the model originally passed to plan().
    if all(
        type(m) is _PairedIctdInteraction
        for m in model.readout_fn.representation.interactions
    ):
        model.__class__ = _PairedCartesianModel


class _PairedIctdInteraction(O3CgtpInteraction):
    _foundation_paired_ictd = True

    def _compute_messages(
        self,
        node_feats,
        node_types,
        edge_radial_basis,
        edge_feats,
        edge_attrs,
        edge_index,
        edge_cutoff,
        *args,
        graph=None,
        element=None,
        edge_pairs=None,
    ):
        if (
            edge_pairs is not None
            and graph is not None
            and graph.lmp_data is None
            and element is None
            and not self.use_eqx
            and memory_first()
            and not tracing_rows(edge_index)
            and not torch.jit.is_tracing()
            and not forward_ad_active()
            and CAPTURE_CONSTANTS.get() is None
            and isinstance(edge_feats, torch.Tensor)
            and node_feats.is_cuda
            and node_feats.dtype
            == edge_feats.dtype
            == edge_attrs.dtype
            == torch.float32
            and not torch.is_autocast_enabled("cuda")
        ):
            radial, projection, network = self._radial_projection(edge_feats)
            conv = self.rejector
            # Keep every supplied directed raw Cartesian component. Only the
            # scalar weights are shared; their live MLP is evaluated by tile.
            attrs = edge_attrs if edge_cutoff is None else edge_attrs * edge_cutoff
            out = conv.scheduled_convolution(
                node_feats,
                attrs,
                radial,
                projection,
                edge_index,
                radial_network=network,
                edge_pairs=edge_pairs,
            )
            return (
                out.index_select(-1, conv.eqx_tp.output_pack_index)
                if conv.compact_output
                else out
            )
        return super()._compute_messages(
            node_feats,
            node_types,
            edge_radial_basis,
            edge_feats,
            edge_attrs,
            edge_index,
            edge_cutoff,
            *args,
            graph=graph,
            element=element,
            edge_pairs=edge_pairs,
        )


def _lower_interaction(module, *, backend, paired_radial=False):
    conv = module.rejector
    schedule = CartesianTensorProductConv(
        conv.tp,
        backend=backend,
        normalize=False,
        symmetric_inputs=False,
        compact_output=False,
    )
    # The original TP is already registered on the host. Reference exactly
    # that module without duplicating its checkpoint keys in the new plan.
    del schedule._modules["tp"]
    object.__setattr__(schedule, "tp", conv.tp)
    for child in schedule.modules():
        child._non_persistent_buffers_set.update(child._buffers)
    conv.scheduled_convolution = schedule.to(next(module.parameters())).train(
        module.training
    )
    conv.__class__ = _ScheduledIctdMessages
    if paired_radial:
        module.__class__ = _PairedIctdInteraction
    return module


def cartesian_interaction_rule(
    module, *, backend, task, paired_radial=False, automatic=False
):
    """Select an ICTD contraction or a direct planar Cartesian convolution."""
    if type(module) is UuO2Interaction:
        return _local_uu_rule(module, backend, task)
    if type(module) is UvO2Interaction:
        return _local_uv_rule(module, backend=backend, task=task, automatic=automatic)
    if (
        type(module) is not O3CgtpInteraction
        or type(module.rejector) is not CartesianTensorProductConvLayer
    ):
        return (
            None,
            "requires a native Cartesian TACE O(3) convolution; local O(2) retains its own frame schedule",
        )
    conv = module.rejector
    if type(conv.tp) is not co3.ictd.TensorProduct or conv.tp.project:
        return None, "requires raw native ICTD tensor-product components"
    if type(conv.eqx_tp) is not CartesianTensorProductConv or custom_forward(
        conv, recursive=True
    ):
        return None, "preserved customized Cartesian tensor-product execution"
    if module.use_eqx:
        return None, "preserved the host's existing streamed Cartesian EQX convolution"
    if task not in ("training", "inference") or backend not in ("torch", "cuda"):
        return None, "requires ordinary Torch or CUDA training/inference"
    if next(module.parameters()).dtype not in (torch.float32, torch.float64):
        return None, "requires FP32 or FP64"
    if (
        not conv.tp.instructions
        or any(
            p.connection_mode != "uvu" or not p.has_weight for p in conv.tp.instructions
        )
        or conv.tp.internal_weights
        or conv.tp.shared_weights
    ):
        return None, "requires externally weighted per-edge Cartesian uvu couplings"
    reason = (
        "indexed raw ICTD contraction/receiver sum with the original output packing"
    )
    if paired_radial:
        reason += "; paired scalar radial MLP for eligible eager CUDA E/F inputs"
    return partial(
        _lower_interaction, backend=backend, paired_radial=paired_radial
    ), reason


class _ScheduledIctdACE(_ScheduledCgtpACE):
    """The weight/bias assembly is basis independent; the contraction is ICTD."""

    @property
    def scheduled_contraction(self):
        return self.eqx_ace


def _lower_product(module, *, backend):
    if module.eqx_ace.backend != backend:
        module.eqx_ace = (
            IctdUuuFullyConnectedSymmetricContraction(
                [ace.tp for ace in module.aces],
                [coef.linear for coef in module.coefs],
                backend=backend,
            )
            .to(next(module.parameters()))
            .train(module.training)
        )
    module.__class__ = _ScheduledIctdACE
    return module


def _select_ictd_factors(module, programs, *, backend):
    for factor, program in zip(module.aces, programs, strict=True):
        if program is None:
            continue
        tp = factor.tp
        tp.kernel_metadata = program
        tp.backend = tp.projection.backend = backend
        for basis in tp.projection.symmetric_bases.values():
            basis.backend = backend
    return module


def _cartesian_linear(op):
    return type(op) is e3nnLinear and type(op.linear) is co3.Linear and not has_lora(op)


def _measured_ictd_product(module):
    """The two FP32 path-gated layouts timed with the complete 3BPA E/F step."""
    return (
        not torch.version.hip
        and next(module.parameters()).dtype == torch.float32
        and memory_first()
        and not acceleration_enabled("eqx", kernel="product")
        and torch.get_float32_matmul_precision() == "highest"
        and module.num_channel == module.num_hidden_channel == 64
        and (module.Lmax, module.lmax, module.correlation) == (2, 3, 3)
        and module.num_layers == 2
        and module.layer in (0, 1)
        and module.asymmetric
        and module.instructions is None
        and module.l1l2 is None
        and module.stochastic_depth_p == 0
        and module.irreps_in == co3.Irreps("64x0e+64x1o+64x2e+64x3o")
        and module.irreps_out
        == co3.Irreps("64x0e" if module.layer == 1 else "64x0e+64x1o+64x2e")
        and node_local_path_gates(module, _cartesian_linear)
        and all(
            type(f.tp.projection) is co3.Projector
            and f.tp.projection.backend == "torch"
            for f in module.aces
        )
    )


def _nonlinear_ictd_rule(module, backend, task, automatic, measured_stack):
    if (
        task not in ("training", "inference")
        or backend not in ("cuda", "torch")
        or next(module.parameters()).dtype not in (torch.float32, torch.float64)
        or module.use_generalized_cg
        or module.use_time_reversal
        or hasattr(module, "eqx_metadata")
        or custom_forward(module, recursive=True)
        or not module.aces
        or any(
            type(factor) is not UuuTensorProduct
            or type(factor.tp) is not co3.ictd.TensorProduct
            or not factor.tp.project
            for factor in module.aces
        )
    ):
        return None, "retained the host nonlinear ICTD product"
    if automatic and not (
        measured_stack and backend == "cuda" and _measured_ictd_product(module)
    ):
        return None, "retained the host nonlinear ICTD product"
    if backend == "torch" and not hasattr(module, "_foundation_regional_provider"):
        return None, "retained the host ICTD factor backends"
    if backend == "cuda" and (
        not next(module.parameters()).is_cuda
        or any(f.tp.backend != "torch" for f in module.aces)
    ):
        return None, "requires native Torch ICTD factors on a CUDA model"
    # The same bounded sparse coefficients used by the native CUDA TP. Each factor
    # retains its own projector, so the next order and its gate see STF features.
    programs = tuple(
        f.tp.kernel_metadata or f.tp.channel_metadata(f.tp.couplings)
        for f in module.aces
    )
    if automatic and tuple(p is not None for p in programs) != (
        True,
        module.layer == 1,
    ):
        return None, "retained the host outside the measured ICTD factor budget"
    count = sum(p is not None for p in programs)
    if not count:
        return None, "no nonlinear ICTD factor fits the channel-program budget"
    return partial(_select_ictd_factors, programs=programs, backend=backend), (
        f"{backend} projected ICTD channel products for {count}/{len(programs)} factors; native gates and coefficient maps"
    )


def cartesian_bucket_product(module):
    """Whether the installed Cartesian contraction and gates are node-local."""
    if type(module) is _ScheduledIctdACE:
        op = module.scheduled_contraction
        return (
            type(op) is IctdUuuFullyConnectedSymmetricContraction
            and op.backend == "cuda"
            and len(op.weight_numels) == module.correlation
        )
    if type(module) is not NonlinearIctdACE:
        return False

    return node_local_path_gates(module, _cartesian_linear) and all(
        type(f) is UuuTensorProduct
        and type(f.tp) is co3.ictd.TensorProduct
        and f.tp.project
        and f.tp.backend == f.tp.projection.backend == "cuda"
        and f.tp.kernel_metadata is not None
        for f in module.aces
    )


def cartesian_product_rule(
    module, *, backend, task, automatic=False, measured_stack=False
):
    """Use the independent, projected Cartesian coefficient and coupling program."""
    if type(module) is NonlinearIctdACE:
        return _nonlinear_ictd_rule(module, backend, task, automatic, measured_stack)
    if (
        type(module) is not CgtpACE
        or type(getattr(module, "eqx_ace", None))
        is not IctdUuuFullyConnectedSymmetricContraction
    ):
        return (
            None,
            "requires ordinary native ICTD ACE; retain other product algorithms",
        )
    if task not in ("training", "inference") or backend not in ("torch", "cuda"):
        return None, "requires ordinary Torch or CUDA training/inference"
    ref = next(module.parameters())
    if ref.dtype not in (torch.float32, torch.float64):
        return None, "requires FP32 or FP64"
    if module.use_time_reversal or module.use_generalized_cg or not module.aces:
        return (
            None,
            "preserved the host's generalized or time-reversal Cartesian product",
        )
    if custom_forward(module, recursive=True) or any(
        has_lora(m) for m in module.modules()
    ):
        return None, "preserved customized Cartesian product maps"
    if ref.is_cuda and (
        module.eqx_ace.backend == backend or module.eqx_ace.backend == "cuda"
    ):
        return None, "preserved the host's already-enabled Cartesian product provider"
    if any(
        type(ace.tp) is not co3.ictd.TensorProduct or not ace.tp.project
        for ace in module.aces
    ):
        return None, "requires projected ICTD factors at each degree"
    expected = e3nnLinear if module.agnostic else e3nnElementLinear
    if any(
        type(coef) is not expected or type(coef.linear) is not co3.Linear
        for coef in module.coefs
    ):
        return None, "requires native Cartesian coefficient maps"
    if module.eqx_ace.backend != backend and module.eqx_ace.state_dict():
        return (
            None,
            "preserved a Cartesian product with persistent provider-specific state",
        )
    return partial(
        _lower_product, backend=backend
    ), "ICTD contractions and live coefficient maps with STF projection"

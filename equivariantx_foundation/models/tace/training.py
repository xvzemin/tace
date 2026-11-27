"""CUDA replay of energy/force/stress and selected magnetic training steps."""

from contextlib import contextmanager

import torch

from equivariantx.ops.common.cuda.graph import retain_constants

_SPIN = "initial_noncollinear_magmoms"
_MAG_FORCE = "noncollinear_magnetic_forces"


def _magnetic_bucket_stages(rep):
    """Rank-two spin-orbit Uv with polynomial spin features and local updates."""
    from e3nn import o3

    from equivariantx.o2 import WignerD
    from equivariantx.o3.cgtp import TensorProduct as EqxTensorProduct
    from tace.models.angular import SolidHarmonics
    from tace.models.linear import e3nnElementLinear, e3nnLinear
    from tace.models.spherical.fused import UuuTensorProduct
    from tace.models.spherical.mtace import (
        MagneticBasis,
        MagneticChebyshevBasis,
        O2MagneticInteraction,
    )
    from tace.models.spherical.node import ElementNodeUpdate

    from .magnetic import _MagneticUvFrames, _same_program
    from .matching import canonical_component_harmonics, streamable_radial_mlp
    from .products import _ScheduledCgtpACE

    basis = rep.magnetic_basis
    return (
        not rep.use_time_reversal  # Explicit T-e3nn products need separate capture validation.
        and rep.use_local_frame
        and not rep.use_packed_wigner
        and not rep.reciprocal_radial
        and not rep.use_magnetic_wigner6j
        and rep.use_magnetic_edge_attrs
        and rep.magnetic_edge_updates is None
        and type(rep.o2_angular_basis) is WignerD
        and rep.o2_angular_basis.backend == "torch"
        and canonical_component_harmonics(rep.o3_angular_basis, 2)
        and type(basis) is MagneticBasis
        and basis.Lmax == 2
        and basis.use_spin_orbit_coupling
        and basis.magnetic_scale.ndim == 2
        and basis.magnetic_scale.shape[0] == 1
        and all(type(p) is _ScheduledCgtpACE for p in rep.products)
        # The clamped radial coordinate uses |m|²; regular solid harmonics are
        # polynomials (never normalized by |m|). Zero dummy spins are nonsingular
        # for both magnetic forces and their parameter adjoints.
        and type(basis.radial_basis) is MagneticChebyshevBasis
        and type(basis.angular_basis) is SolidHarmonics
        and not basis.angular_basis.normalize
        and type(basis.magnetic_edge_tensor_product) is UuuTensorProduct
        and type(basis.magnetic_edge_tensor_product.tp)
        in (o3.TensorProduct, EqxTensorProduct)
        and type(rep.node_updates) is torch.nn.ModuleList
        and len(rep.node_updates) == len(rep.interactions)
        and all(
            type(m) is ElementNodeUpdate
            and type(m.embedding) is e3nnElementLinear
            and type(m.embedding.linear) is o3.Linear
            for m in rep.node_updates
        )
        and all(
            type(m) is O2MagneticInteraction
            and type(m.rejector) is _MagneticUvFrames
            and not any(
                key in vars(m)
                for key in (
                    "_magnetic_weights",
                    "_apply_rejector",
                    "_compute_messages",
                    "_conv_weights",
                    "_radial_projection",
                )
            )
            and not m.rejector.use_time_reversal
            and _same_program(m.rejector)
            and m.rejector.mmax == m.rejector.eqx_tp.degree == 2
            and m.rejector.eqx_tp.backend == "cuda"
            and type(m.magnetic_linear) is e3nnLinear
            and type(m.magnetic_linear.linear) is o3.Linear
            and streamable_radial_mlp(m.source_magnetic_info)
            and streamable_radial_mlp(m.target_magnetic_info)
            for m in rep.interactions
        )
    )


def _rotary_bucket_stages(rep):
    """Require the installed dense-host-frame O3/rotary CUDA programs."""
    from equivariantx.o2 import WignerD
    from tace.models.spherical.tece_oam_rra import Convolution
    from tace.utils.env import memory_first

    from .interactions import _HostFrameRadialInteraction, rotary_radial_geometry
    from .matching import streamable_radial_mlp
    from .tece_oam_rra.program import metadata
    from .tece_oam_rra.regional import StreamedRotaryInteraction

    basis = getattr(rep, "o2_angular_basis", None)
    if not (
        memory_first()
        and not rep.reciprocal_radial
        and not rep.use_packed_wigner
        and type(basis) is WignerD
        and basis.method == "auto"
        and basis.backend == "torch"
        and "forward_packed" not in vars(basis)
        and rotary_radial_geometry(rep)
    ):
        return False
    for layer in rep.interactions:
        if not streamable_radial_mlp(layer.edge_info):
            return False
        conv = layer.rejector
        if type(layer) is _HostFrameRadialInteraction:
            if conv.eqx_tp.backend != "cuda" or conv.eqx_tp.radial_mode != "full":
                return False
        elif not (
            type(layer) is StreamedRotaryInteraction
            and type(conv) is Convolution
            and not any(
                k in vars(layer)
                for k in ("_compute_messages", "_rotary_radial", "_convolution_radial")
            )
            and not any(hasattr(conv, k) for k in ("prepare_stream", "finish_stream"))
            and not getattr(conv, "save_force_primals", False)
            # Max and denominator are reduced per receiver and head. Keep the
            # host epsilon for a finite denominator even for zero-cutoff images.
            and conv.attention_eps == 1e-16
            and conv._eqx_metadata is not None
            and conv._eqx_metadata == metadata(conv)
        ):
            return False
    return True


def _regional_bucket_stages(host, rep):
    """Accept only the installed, component-local message and product programs."""
    from tace.models.cartesian.tace import CartesianTACE
    from tace.models.spherical.tace import SphericalTACE

    if type(host) is SphericalTACE:
        from .interactions import (
            StreamedInteraction,
            _SharedHarmonicInteraction,
            canonical_geometry,
        )
        from .local import _StreamUu, _StreamUv, local_geometry, mixed_geometry
        from .products import spherical_bucket_product

        messages = (
            _magnetic_bucket_stages(rep)
            if rep.use_magnetic_interaction
            else (
                (
                    all(type(m) is StreamedInteraction for m in rep.interactions)
                    and canonical_geometry(rep)
                )
                or (
                    all(type(m) in (_StreamUu, _StreamUv) for m in rep.interactions)
                    and local_geometry(rep)
                )
                or (
                    # Mixed O3/O2 shares radial rows, but SH and Wigner are separate
                    # edge-local functions. Require the installed mixed program, not
                    # native layers merely eligible for it. _pack supplies independent
                    # directed radial rows and cannot couple a dummy to a real edge.
                    all(
                        type(m) in (_SharedHarmonicInteraction, _StreamUu, _StreamUv)
                        and m.use_eqx
                        and m.rejector.eqx_tp.backend == "cuda"
                        for m in rep.interactions
                    )
                    and mixed_geometry(rep)
                    and rep.o2_angular_basis.backend == "cuda"
                )
                or _rotary_bucket_stages(rep)
            )
        )
        return messages and all(spherical_bucket_product(m) for m in rep.products)
    elif type(host) is CartesianTACE:
        from equivariantx import co3
        from equivariantx.co3.conv import CartesianTensorProductConv
        from tace.models.spherical.inter import O3CgtpInteraction

        from .cartesian import (
            _PairedIctdInteraction,
            _ScheduledIctdMessages,
            cartesian_bucket_product,
            cartesian_local_bucket_geometry,
        )

        # Each harmonic depends on one edge vector. The unprojected uvu
        # convolution only gathers its senders and sums at its receivers;
        # the output packing, STF projections, and ACE coefficients act per node.
        messages = (
            not rep.use_local_frame
            and rep.use_o3_angular_basis
            and type(rep.o3_angular_basis) is co3.CartesianHarmonics
            and getattr(rep, "co3_angular_basis", None) is None
            and all(
                type(m) in (O3CgtpInteraction, _PairedIctdInteraction)
                and not m.use_eqx
                and type(m.rejector) is _ScheduledIctdMessages
                and type(m.rejector.scheduled_convolution) is CartesianTensorProductConv
                and m.rejector.scheduled_convolution.backend == "cuda"
                and m.rejector.scheduled_convolution.tp is m.rejector.tp
                for m in rep.interactions
            )
        )
        if rep.use_local_frame:
            messages = cartesian_local_bucket_geometry(rep)
        return messages and all(cartesian_bucket_product(m) for m in rep.products)
    return False


def _can_capture_parameters(model):
    params = tuple(model.parameters())
    return (
        bool(params)
        and params[0].is_cuda
        and any(p.requires_grad for p in params)
        and all(
            p.device == params[0].device
            and not p._backward_hooks
            and not p._post_accumulate_grad_hooks
            for p in params
        )
    )


def _regional_bucket_model(model):
    """Match models closed under disconnected graph padding, not shape presets.

    Each accepted interaction and node product preserves graph membership.
    Inductively the real energy and its coordinate/strain/spin derivatives are
    independent of the padding component at every layer. Masking its objective
    also removes its contribution to shared-parameter gradients. As in eager
    training this assumes finite intermediates; GPU capture still needs to fit.
    """
    host = getattr(model, "readout_fn", None)
    rep = getattr(host, "representation", None)
    if not getattr(rep, "_foundation_regional", False):
        return False
    from equivariantx.nn import MergeLayerNorm, MergeRmsNorm, SoftCapRmsNorm
    from tace.models.adapter import TensorModel
    from tace.models.blocks import AtomicEnergy, ScaleShift
    from tace.models.mlp import ACTIVATION
    from tace.models.radial import (
        C2PolynomialCutoff,
        RadialBasis,
        ZBLBasis,
        j0SphericalBesselBasis,
    )
    from tace.models.spherical.edge import (
        Element2EdgeEmbedding,
        Element2EdgeUpdate,
        ElementEdgeEmbedding,
        IdentityEdgeEmbedding,
        IdentityEdgeUpdate,
        NonLinearEdgeEmbedding,
    )
    from tace.models.spherical.node import LinearNodeEmbedding
    from tace.models.spherical.readout import ScalarReadOut
    from tace.models.spherical.representation import NodeRepresentation
    from tace.models.spherical.scatter_norm import (
        AvgNumNeighborsScatterNorm,
        CutoffScatterNorm,
        DensityScatterNorm,
        IdentityScatterNorm,
        SqrtAvgNumNeighborsScatterNorm,
    )

    from .matching import custom_forward, streamable_radial_mlp

    if type(model) is not TensorModel:
        if not getattr(model, "_foundation_paired_ictd", False):
            return False
        from .cartesian import _PairedCartesianModel

        if type(model) is not _PairedCartesianModel:
            return False
    if not (
        model.training
        and type(rep) is NodeRepresentation
        and not rep.use_dens
        and not hasattr(rep, "uie_embedding")
        and rep.uee_embeddings is None
        # ScaleShift counts atoms/receiver edges by graph (including legacy
        # isolated-atom behavior). ZBL uses only each bonded pair and scatters
        # to its receiver. The dummy cannot enter either real contribution,
        # regardless of whether ZBL is applied before or after scaling.
        and set(host._modules) - {"scale_shift", "zbl"}
        == {"representation", "atomic_energy_layer", "energy_readouts"}
        and ("scale_shift" not in host._modules or type(host.scale_shift) is ScaleShift)
        and (
            "zbl" not in host._modules
            or (
                type(host.zbl) is ZBLBasis
                and host.zbl.is_polynomial_cutoff
                and host.zbl.cutoff_fn is C2PolynomialCutoff.calculate_envelope
            )
        )
        and type(host.atomic_energy_layer) is AtomicEnergy
        and 0
        < model.num_fidelities
        == host.num_fidelities
        == host.atomic_energy_layer.atomic_energy.shape[0]
        and type(host.energy_readouts) is torch.nn.ModuleList
        and all(
            type(m) is ScalarReadOut and m.num_fidelities == model.num_fidelities
            for m in host.energy_readouts
        )
        and type(rep.node_embedding) is LinearNodeEmbedding
        and type(rep.edge_embedding)
        in (
            IdentityEdgeEmbedding,
            ElementEdgeEmbedding,
            Element2EdgeEmbedding,
            NonLinearEdgeEmbedding,
        )
        and type(rep.radial_basis) is RadialBasis
        and type(rep.radial_basis.radial_fn) is j0SphericalBesselBasis
        and type(rep.radial_basis.cutoff_fn) is C2PolynomialCutoff
        and not rep.radial_basis.use_distance_transform
        and 0 < len(rep.interactions) == len(rep.products) == len(rep.edge_updates)
        and all(
            type(m) is IdentityEdgeUpdate
            or (not rep.reciprocal_radial and type(m) is Element2EdgeUpdate)
            for m in rep.edge_updates
        )
        # A message reads only its source and this edge's geometry, then sums
        # at its destination. Disjoint graphs therefore stay disjoint under
        # composition, including mixed UU/UV layers sharing compatible frames.
        and _regional_bucket_stages(host, rep)
        and all(
            (
                type(m.scatter_norm)
                in (
                    AvgNumNeighborsScatterNorm,
                    SqrtAvgNumNeighborsScatterNorm,
                    IdentityScatterNorm,
                    CutoffScatterNorm,
                )
                or (
                    type(m.scatter_norm) is DensityScatterNorm
                    and streamable_radial_mlp(m.scatter_norm.edge_density)
                )
            )
            and not hasattr(m, "stochastic_depth")
            for m in rep.interactions
        )
        and all(
            type(norm) in (MergeLayerNorm, MergeRmsNorm, SoftCapRmsNorm)
            for norm in (
                getattr(rep, "final_norm", None),
                *(
                    getattr(m, name, None)
                    for m in rep.interactions
                    for name in ("norm1", "norm2")
                ),
            )
            if norm is not None
        )
        and not any(hasattr(m, "stochastic_depth") for m in rep.products)
        and not custom_forward(model, recursive=True)
        and not model.lmp
        and (
            model.num_fidelities == 1
            and set(model.get_embedding_property()) == {_SPIN}
            and set(model.get_target_property())
            == {"energy", "forces", "stress", _MAG_FORCE}
            if rep.use_magnetic_interaction
            else not model.get_embedding_property()
            and set(model.get_target_property())
            in ({"energy", "forces"}, {"energy", "forces", "stress"})
        )
        and 0 < model.get_cutoff() < float("inf")
        and model.get_model_dtype() == torch.float32
    ):
        return False
    # Neither the spherical nor Cartesian path may silently accept a batch
    # reduction nested inside an ordinary block (e.g. Softmax over nodes).
    nn = torch.nn
    local_primitives = {
        nn.Identity,
        nn.Embedding,
        nn.Linear,
        nn.ModuleList,
        nn.ModuleDict,
        nn.ParameterList,
        nn.Sequential,
        *ACTIVATION.values(),
    }
    for m in model.modules():
        cls = type(m)
        if cls.__module__.split(".")[0] not in (
            "tace",
            "e3nn",
            "torch",
            "equivariantx",
            "equivariantx_foundation",
        ) or isinstance(
            m, (nn.modules.dropout._DropoutNd, nn.modules.batchnorm._BatchNorm)
        ):
            return False
        if cls in (nn.LayerNorm, nn.RMSNorm):
            if len(m.normalized_shape) != 1:  # Only normalize the feature axis.
                return False
        elif cls.__module__.startswith("torch.nn.") and cls not in local_primitives:
            return False
        if cls is nn.Embedding and (m.scale_grad_by_freq or m.max_norm is not None):
            # Masked dummy indices still affect frequency scaling/renormalization.
            return False
    return _can_capture_parameters(model)


def _released_bucket_model(model):
    from .planner import release_schedule

    if not model.training or getattr(model, "lmp", True):
        return False
    if not _can_capture_parameters(model):
        return False
    release, _ = release_schedule(model, task="training_buckets")
    return (
        release is not None
        and model.get_cutoff() == 6.0
        and not model.get_embedding_property()
        and set(model.get_target_property())
        in ({"energy", "forces"}, {"energy", "forces", "stress"})
        and not any(
            hasattr(product, "stochastic_depth")
            for product in model.readout_fn.representation.products
        )
    )


def supports_training_buckets(model):
    """Whether the model accepts its fully labelled, unweighted E/F(/S) or EFSM bucket objective."""
    return _regional_bucket_model(model) or _released_bucket_model(model)


def prepare_training_data(model, batch, *, get_fidelity_idx=None):
    """Prepare physical inputs and a runtime head index for each structure."""
    data = model.prepare_property_data(batch)
    if getattr(model, "num_fidelities", 1) > 1:
        ptr = data["ptr"]
        graphs = ptr.numel() - 1
        ids = data.get("fidelity_idx")
        if ids is None:
            # Treat the default as an input rather than a Python constant in a captured graph.
            head = (
                model.get_fidelity_idx()
                if get_fidelity_idx is None
                else get_fidelity_idx()
            )
            ids = ptr.new_full((graphs,), head, dtype=torch.int64)
        elif (
            not torch.is_tensor(ids)
            or ids.shape != (graphs,)
            or ids.dtype not in (torch.int32, torch.int64)
            or ids.device != ptr.device
        ):
            raise ValueError(
                "fidelity_idx must contain one integer per structure on the graph device"
            )
        data["fidelity_idx"] = ids.long()
    return data


@contextmanager
def _preserve_gradients(params):
    previous = tuple(p.grad for p in params)
    try:
        yield
    finally:
        for p, grad in zip(params, previous, strict=True):
            p.grad = grad


def _replay_ddp_group():
    dist = torch.distributed
    if not (
        dist.is_available() and dist.is_initialized() and dist.get_backend() == "nccl"
    ):
        raise ValueError("DDP replay requires an initialized default NCCL group")
    return dist.group.WORLD


class _LocalParameterVJP:
    """Evaluate a VJP on capture-local leaves sharing the optimizer's storage."""

    def __init__(self, model, params):
        self.model = model
        self.leaves = tuple(p.detach().requires_grad_() for p in params)
        by_id = {id(p): leaf for p, leaf in zip(params, self.leaves, strict=True)}
        self.state = {
            name: by_id[id(p)]
            for name, p in model.named_parameters()
            if p.requires_grad
        }

    def __call__(self, batch, objective):
        # tie_weights=True replaces every alias, restoring the original modules
        # even on exceptions. This Python binding only runs at warmup/record.
        outputs = torch.func.functional_call(self.model, self.state, (batch,))
        loss = objective(outputs, batch)
        grads = torch.autograd.grad(loss, self.leaves, allow_unused=True)
        return outputs, loss, grads


class UnsupportedReplayBatch(ValueError):
    """The input fields do not match the training graph contract."""


class FixedShapeTrainingStep:
    """Replay forward and parameter backward, then update live optimizer state.

    This single-GPU helper owns one fixed-shape input/gradient workspace. Use
    one instance for one model, optimizer and batch signature; it accepts
    changes to coordinates, topology, elements and labels of the same shape.
    The supplied objective takes (predictions, prepared_batch) and returns a
    scalar. With ``joint=True``, supply TACE's compile adapter and its
    supported loss module; the explicit joint-derivative program is run under
    capture instead. Warmup and capture preserve the caller's parameter gradients
    and do not update parameters. Only each explicit call advances the optimizer.
    The caller must not replace parameters, change
    training/precision policy, or use this concurrently with other gradient
    consumers. The returned metrics alias the workspace until the next call.
    No gradient accumulation, distributed collectives, AMP or dropout.
    """

    def __init__(
        self,
        model,
        optimizer,
        batch,
        objective,
        targets,
        *,
        joint=False,
        _workspace=None,
        _recordings=None,
        _get_fidelity_idx=None,
        _ddp=False,
        _private_leaves=False,
    ):
        if not model.training or not torch.cuda.is_available():
            raise ValueError("A training-mode CUDA model is required")
        if _ddp:
            _replay_ddp_group()
        local_vjp = _ddp or _private_leaves
        if local_vjp and (
            joint
            or _workspace is not None
            or _recordings is not None
            or not (type(objective) is PaddedEFSLoss and objective._ddp == _ddp)
        ):
            raise ValueError("Local VJP replay requires a private E/F or E/F/S graph")
        if torch.is_autocast_enabled("cuda") or (
            not _ddp
            and torch.distributed.is_initialized()
            and torch.distributed.get_world_size() > 1
        ):
            raise ValueError(
                "This single-GPU replay does not support AMP or distributed training"
            )
        self.tf32 = torch.backends.cuda.matmul.allow_tf32
        self.model, self.optimizer = model, optimizer
        # Capture references the objective's CUDA buffers by address.
        self.objective = objective
        self.get_fidelity_idx = _get_fidelity_idx
        if getattr(model, "_foundation_paired_ictd", False):
            from .cartesian import _PairedCartesianModel

            if type(model) is _PairedCartesianModel:
                self.get_fidelity_idx = _get_fidelity_idx or model.get_fidelity_idx
                model = model._directed_replay_view()
        self._replay_model = model
        params = tuple(p for p in model.parameters() if p.requires_grad)
        if not params or any(
            p.device != params[0].device or not p.is_cuda for p in params
        ):
            raise ValueError("All parameters must reside on one CUDA device")
        if set(params) != {
            p for group in optimizer.param_groups for p in group["params"]
        }:
            raise ValueError("The optimizer must own all trainable model parameters")
        self.params = params
        self.device = params[0].device
        batch = prepare_training_data(
            model, batch, get_fidelity_idx=self.get_fidelity_idx
        )
        self.static = {
            k: v.detach().clone() if torch.is_tensor(v) else v for k, v in batch.items()
        }
        self.graph = torch.cuda.CUDAGraph()
        self.constants = {}
        self.stream = (
            torch.cuda.Stream(device=self.device)
            if _workspace is None
            else _workspace.capture_stream
        )
        self.stream.wait_stream(torch.cuda.current_stream(self.device))
        direct = (
            joint
            and type(optimizer) is torch.optim.AdamW
            and all(group.get("fused") for group in optimizer.param_groups)
            and not any(
                p._backward_hooks or p._post_accumulate_grad_hooks for p in params
            )
            and not any(p.requires_grad for p in objective.parameters())
        )

        def evaluate():
            # Grad buffers are allocated inside the graph pool by backward.
            # Replay therefore writes fresh gradients without accumulation.
            for parameter in params:
                parameter.grad = None
            if direct:
                from tace.models.compile.joint import joint_training

                outputs, loss, gradients = joint_training(
                    model, objective, self.static, return_gradients=True
                )
                # Preserve independent buffers if a compiled output is a view
                # of another gradient, a parameter, a prediction, or an input.
                owned = {
                    v.untyped_storage().data_ptr()
                    for v in (
                        *self.model.parameters(),
                        *self.model.buffers(),
                        *objective.buffers(),
                        *self.static.values(),
                        *outputs.values(),
                        loss,
                    )
                    if torch.is_tensor(v)
                }
                for parameter, gradient in gradients:
                    if gradient is not None:
                        ptr = gradient.untyped_storage().data_ptr()
                        if ptr in owned or not gradient.is_contiguous():
                            gradient = gradient.clone(
                                memory_format=torch.contiguous_format
                            )
                        owned.add(ptr)
                    parameter.grad = gradient
            elif joint:
                outputs, loss = model.forward_loss(self.static, objective)
                loss.backward(inputs=params)
            else:
                outputs = model(self.static)
                loss = objective(outputs, self.static)
                loss.backward(inputs=params)
            return {k: outputs[k].detach() for k in targets} | {"loss": loss.detach()}

        if local_vjp:
            self._local_vjp = _LocalParameterVJP(model, params)

            def evaluate_local():
                outputs, loss, self.gradients = self._local_vjp(self.static, objective)
                return {k: outputs[k].detach() for k in targets} | {
                    "loss": loss.detach()
                }

            evaluate = evaluate_local
        preserve = () if local_vjp else params
        with (
            _preserve_gradients(preserve),
            torch.cuda.device(self.device),
            torch.cuda.stream(self.stream),
            retain_constants(self.constants),
        ):
            for _ in range(2):
                evaluate()
                if local_vjp:
                    del self.gradients  # Warmup VJPs are not replay outputs.
        self.stream.synchronize()
        warmed = dict(self.constants) if _workspace is not None else None

        # A deferred recording must preserve gradients at call time, not at warmup.
        @_preserve_gradients(preserve)
        def record():
            with (
                torch.cuda.device(self.device),
                retain_constants(self.constants),
                torch.cuda.graph(
                    self.graph,
                    stream=self.stream,
                    capture_error_mode="thread_local",
                    pool=None if _workspace is None else _workspace.pool,
                ),
            ):
                self.outputs = evaluate()
            if warmed is not None and (
                warmed.keys() != self.constants.keys()
                or any(t is not self.constants[k] for k, t in warmed.items())
            ):
                raise RuntimeError(
                    "A shared training graph created a persistent constant during capture"
                )
            if not local_vjp:
                self.gradients = tuple(p.grad for p in params)

        if _recordings is None:
            record()
        else:
            _recordings.append(record)

    def replay_outputs(self, batch):
        """Refresh workspace outputs and ``gradients`` without assigning ``p.grad``.

        Outputs and gradients are reused, including any gradients still
        attached to parameters by a previous call to ``replay``.
        """
        if (
            not self.model.training
            or torch.is_autocast_enabled("cuda")
            or torch.backends.cuda.matmul.allow_tf32 != self.tf32
        ):
            raise ValueError("Training mode or numerical policy changed since capture")
        batch = prepare_training_data(
            self._replay_model, batch, get_fidelity_idx=self.get_fidelity_idx
        )
        if batch.keys() != self.static.keys():
            raise UnsupportedReplayBatch("The training batch field set changed")
        with torch.no_grad():
            for k, dst in self.static.items():
                src = batch[k]
                if torch.is_tensor(dst):
                    if (
                        not torch.is_tensor(src)
                        or src.shape != dst.shape
                        or src.dtype != dst.dtype
                        or src.device != dst.device
                    ):
                        raise UnsupportedReplayBatch(
                            f"Training input signature changed at {k}"
                        )
                    dst.copy_(src)
                elif src != dst:
                    raise UnsupportedReplayBatch(
                        f"Static non-tensor input changed at {k}"
                    )
        self.graph.replay()
        return self.outputs

    def replay(self, batch):
        """Assign the fresh gradients to parameters without an optimizer update."""
        outputs = self.replay_outputs(batch)
        for p, grad in zip(self.params, self.gradients, strict=True):
            p.grad = grad
        return outputs

    def __call__(self, batch):
        outputs = self.replay(batch)
        self.optimizer.step()
        return outputs


def _padding_cell(lattice, cutoff):
    # Distinct axes avoid frame chart boundaries; every dummy image lies outside cutoff.
    scale = cutoff / 6.0
    scale = scale.clamp_min(1.0) if torch.is_tensor(scale) else max(1.0, scale)
    axes = torch.arange(5, 8, dtype=lattice.dtype, device=lattice.device)
    return torch.diag(axes * scale).unsqueeze(0)


def _pack(
    batch,
    atoms,
    edges,
    graphs,
    stress,
    joint=False,
    *,
    microbatches=1,
    total_atoms=None,
    edge_layout="native",
    dummy_cell=None,
    magnetic=False,
    max_graphs=None,
):
    """Append one disconnected padding graph and optional empty graph slots."""
    n, e = len(batch["positions"]), batch["edge_index"].shape[1]
    if n < 1 or len(batch["ptr"]) != graphs + 1 or n > atoms or e > edges:
        raise ValueError("Real minibatch exceeds this (graphs, atoms, edges) bucket")
    if max_graphs is not None and (type(max_graphs) is not int or max_graphs < graphs):
        raise ValueError("Real graph count exceeds max_graphs or max_graphs is invalid")
    tail = 1 if max_graphs is None else max_graphs + 1 - graphs
    if edge_layout not in ("native", "identity_pairs", "directed"):
        raise ValueError("Unknown bucket edge layout")
    if magnetic and (joint or edge_layout != "directed" or not stress):
        raise ValueError("Magnetic buckets require directed, non-joint EFSM replay")
    if magnetic and any(
        not torch.is_tensor(batch.get(key))
        or batch[key].shape != (n, 3)
        or not batch[key].is_floating_point()
        or batch[key].device != batch["positions"].device
        for key in (_SPIN, _MAG_FORCE)
    ):
        raise ValueError(
            "Magnetic buckets require a spin and a labelled magnetic force (N, 3) per atom"
        )
    if edge_layout != "native":
        if "lmp_data" in batch or "natoms" in batch:
            raise ValueError(
                "Regional buckets accept complete graphs, not local/ghost batches"
            )
        if "edge_pair_rows" in batch or "edge_pair_inverse" in batch:
            raise ValueError(
                "Regional buckets require raw edges, not a supplied reciprocal map"
            )
        ptr = batch["ptr"]
        if (
            graphs < 1
            or batch["batch"].shape != (n,)
            or batch["forces"].shape != (n, 3)
            or batch["energy"].numel() != graphs
            or (stress and batch["stress"].shape != (graphs, 3, 3))
            or not ("atomic_numbers" in batch or "node_type" in batch)
        ):
            raise ValueError(
                "Regional buckets require fully labelled real structures and atoms"
            )
        torch._assert_async(
            (ptr[0] == 0) & (ptr[-1] == n) & (ptr.diff() > 0).all(),
            "Bucket ptr must span nonempty real structures",
        )
        edge = batch["edge_index"]
        torch._assert_async(
            ((edge >= 0) & (edge < n)).all(),
            "Bucket cannot bind a ghost edge to a padding atom",
        )
        # Mislabelled real atoms could otherwise be assigned the dummy's graph
        # ID and silently excluded by the force mask.
        membership = batch["batch"]
        torch._assert_async(
            (
                membership
                == torch.bucketize(
                    torch.arange(n, device=ptr.device),
                    ptr[1:-1].contiguous(),
                    right=True,
                )
            ).all(),
            "Bucket batch membership must match ptr",
        )
        torch._assert_async(
            (membership[edge[0]] == membership[edge[1]]).all(),
            "Bucket edges must stay within their structure",
        )
    # TACE represents absent labels with zero weights and a numeric placeholder.
    # Refuse those (and non-unit user weights) under this unweighted objective.
    for key in (
        "energy_weight",
        "forces_weight",
        *(("stress_weight",) if stress else ()),
        # Graph readers also use a zero-weight placeholder for a missing
        # embedding property. A genuine zero spin is fine; an absent one is not.
        *((f"{_SPIN}_weight", f"{_MAG_FORCE}_weight") if magnetic else ()),
        "entropy",
    ):
        if key in batch:
            if magnetic and (
                not torch.is_tensor(batch[key]) or batch[key].shape != (graphs,)
            ):
                raise ValueError(f"Magnetic buckets require one {key} per structure")
            torch._assert_async((batch[key] == 1).all(), f"Bucket requires unit {key}")
    extra = atoms + 1 - n  # At least one real index for the dummy self-image edges.
    dest = batch["edge_index"]
    # Even at the largest real edge count *every* padding atom must have a
    # nonzero neighborhood. The OAM-L radial/harmonic derivatives at empty
    # neighborhoods can be singular in force-loss double backward. Reserve
    # one image slot per possible dummy atom, independent of real topology.
    extra_edges = edges + atoms - e
    pad_nodes = (
        dest.new_full((extra_edges,), n)
        if extra == 1
        else torch.arange(extra_edges, device=dest.device, dtype=dest.dtype)
        .remainder_(extra)
        .add_(n)
    )
    shifts = batch["edge_shifts"]
    if shifts.shape != (e, 3):
        raise ValueError("Expected one three-component shift per directed edge")
    edge_shifts = shifts.new_empty((e + extra_edges, 3))
    edge_shifts[:e].copy_(shifts)
    edge_shifts[e:].fill_(1)
    lattice = batch["lattice"]
    if dummy_cell is None:
        dummy_cell = _padding_cell(lattice, 6.0)

    def cat(key, tail):
        return torch.cat((batch[key], tail))

    result = dict(
        positions=cat("positions", batch["positions"].new_zeros(extra, 3)),
        batch=cat("batch", batch["batch"].new_full((extra,), graphs)),
        ptr=cat("ptr", batch["ptr"].new_full((tail,), atoms + 1)),
        edge_index=torch.cat((dest, pad_nodes.expand(2, -1)), dim=1),
        edge_shifts=edge_shifts,
        lattice=cat("lattice", dummy_cell.expand(tail, -1, -1)),
        pbc=cat("pbc", batch["pbc"].new_ones(tail, 3)),
        energy=cat(
            "energy", batch["energy"].new_zeros((tail, *batch["energy"].shape[1:]))
        ),
        forces=cat("forces", batch["forces"].new_zeros(extra, 3)),
    )
    for key in ("atomic_numbers", "node_type"):
        if key in batch:
            result[key] = cat(key, batch[key][:1].expand(extra))
    if "fidelity_idx" in batch:
        result["fidelity_idx"] = cat(
            "fidelity_idx", batch["fidelity_idx"].new_zeros(tail)
        )
    if edge_layout == "identity_pairs":
        # A physical reciprocal/duplicate edge count depends on topology. Each
        # directed edge as its own representative is always exact and avoids
        # dynamic unique() during capture, including on the dummy component.
        identity = torch.arange(e + extra_edges, device=dest.device)
        result["edge_pair_rows"] = result["edge_pair_inverse"] = identity
    if stress:
        result["stress"] = cat(
            "stress", batch["stress"].new_zeros((tail, *batch["stress"].shape[1:]))
        )
    if magnetic:
        for key in (_SPIN, _MAG_FORCE):
            result[key] = cat(key, batch[key].new_zeros(extra, 3))
    if joint:
        # Native NormalLoss averages E and F over the padded lengths while
        # S is normalized by the number of positive-weight real structures.
        # Undo only those two denominator changes, keeping the dummy at zero.
        real = lattice.new_ones(graphs + 1)
        real[-1] = 0
        result["energy_weight"] = real * ((graphs + 1) / graphs)
        result["forces_weight"] = real * ((atoms + 1) / n)
        result["entropy"] = torch.ones_like(real)
        if stress:
            result["stress_weight"] = real
    if microbatches > 1:
        if joint:
            result["energy_weight"].div_(microbatches)
            result["forces_weight"].mul_(n / total_atoms)
            if stress:
                result["stress_weight"] = real / microbatches
        else:
            result["macro_atoms"] = batch["ptr"].new_full((), total_atoms)
    return result


_EF_WEIGHTS = ("entropy", "energy_weight", "forces_weight")
_EF_GEOMETRY = {
    "positions",
    "ptr",
    "batch",
    "edge_index",
    "edge_shifts",
    "pbc",
    "lattice",
    "atomic_numbers",
    "node_type",
    "fidelity_idx",
    "energy",
    "forces",
}
# PyG bookkeeping or derivable from edge_index, unused by the admitted models.
_EFS_METADATA = {"num_neighbors", "num_nodes", "_num_nodes"}
_GLOBAL_STRESS = ("_valid_stress_count", "_loss_world_size")


def _directed_edge_layout(model):
    rep = getattr(model.readout_fn, "representation", None)
    if getattr(rep, "_foundation_regional", False) and model.uses_reciprocal_weights():
        return "identity_pairs"
    return "directed"


def pack_efs_training_batch(
    model,
    batch,
    max_atoms,
    max_edges,
    max_graphs=None,
    *,
    stress=True,
    _global_stress=False,
):
    """Pack E/F or E/F/S graphs and sample weights for a fixed capacity.

    Preserve real weights and labels verbatim (including native NaN semantics).
    Use finite E/F placeholders for absent labels: their native loss does not
    mask NaNs before the residual. Optional max_graphs fixes graph-level
    shapes without adding atoms or edges. With ``stress=False``, cached stress
    labels and their weights are ignored. Feed the result to PaddedEFSLoss only.
    """
    fields = set(batch.keys())
    if _global_stress:
        device = batch["positions"].device
        if not stress or any(
            not torch.is_tensor(count := batch.get(k))
            or count.ndim != 0
            or count.dtype != torch.int64
            or count.device != device
            for k in _GLOBAL_STRESS
        ):
            raise UnsupportedReplayBatch("Expected scalar global stress counts")
        fields.difference_update(_GLOBAL_STRESS)
    if stress:
        geometry_keys = _EF_GEOMETRY | {"stress"}
        weights = (*_EF_WEIGHTS, "stress_weight")
    else:
        geometry_keys, weights = _EF_GEOMETRY, _EF_WEIGHTS
        fields.difference_update(("stress", "stress_weight"))
    unknown = fields - geometry_keys - _EFS_METADATA - set(weights)
    if unknown:
        raise UnsupportedReplayBatch(
            f"Unsupported E/F/S replay fields: {sorted(unknown)}"
        )
    required = geometry_keys - {"atomic_numbers", "node_type", "fidelity_idx"}
    if not required.union(weights) <= fields or not fields.intersection(
        {"atomic_numbers", "node_type"}
    ):
        raise UnsupportedReplayBatch(
            "Expected complete geometry, requested labels and sample weights"
        )
    geometry = {k: batch[k] for k in fields & geometry_keys}
    device = model.readout_fn.cutoff.device
    if any(
        not torch.is_tensor(batch[k]) or batch[k].device != device
        for k in fields - _EFS_METADATA
    ):
        raise UnsupportedReplayBatch(
            "E/F/S replay inputs and model must share one device"
        )
    n, e, g = (
        len(batch["positions"]),
        batch["edge_index"].size(-1),
        len(batch["ptr"]) - 1,
    )
    shapes = {
        "positions": (n, 3),
        "edge_index": (2, e),
        "edge_shifts": (e, 3),
        "ptr": (g + 1,),
        "energy": (g,),
        "pbc": (g, 3),
        "lattice": (g, 3, 3),
        "atomic_numbers": (n,),
        "node_type": (n,),
        "fidelity_idx": (g,),
    }
    if any(k in fields and batch[k].shape != shape for k, shape in shapes.items()):
        raise UnsupportedReplayBatch("Invalid E/F/S replay geometry or label shape")
    if any(batch[k].shape != (g,) or batch[k].is_complex() for k in weights):
        raise UnsupportedReplayBatch(
            "Expected one real E/F/S weight and entropy per structure"
        )
    packed = _pack(
        geometry,
        max_atoms,
        max_edges,
        g,
        stress,
        edge_layout=_directed_edge_layout(model),
        dummy_cell=_padding_cell(batch["lattice"], model.readout_fn.cutoff),
        max_graphs=max_graphs,
    )
    slots = packed["ptr"].numel() - 1
    if max_graphs is not None:
        # Empty slots belong in the geometry ptr, but must not divide per-atom
        # losses by zero. Give only the loss a count of one for each empty slot.
        ptr = batch["ptr"]
        end = torch.arange(
            max_atoms + 1, max_atoms + 1 + slots - g, dtype=ptr.dtype, device=device
        )
        packed["_efs_loss_ptr"] = torch.cat((ptr, end))
    for k in weights:
        packed[k] = torch.cat((batch[k], batch[k].new_zeros(slots - g)))
    if _global_stress:
        packed.update((k, batch[k]) for k in _GLOBAL_STRESS)
    # Runtime inputs: an identical bucket shape need not have the same real N.
    # Fill on-device. new_tensor([python scalars]) would block the submitting
    # thread on a pageable H2D copy, even with substantial training work queued.
    scales = batch["lattice"].new_ones(3 if stress else 2, dtype=torch.float64)
    scales[0].fill_(slots / g)
    scales[1].fill_((max_atoms + 1) / n)
    packed["_efs_term_scales"] = scales
    return packed


class PaddedEFSLoss(torch.nn.Module):
    """Apply the original NormalLoss terms, correcting only their denominators."""

    def __init__(self, objective, *, _ddp=False):
        from importlib import import_module

        from omegaconf import ListConfig

        from tace.utils.loss.normal import NormalLoss
        from tace.utils.loss.registry import LOSS_FN

        from .matching import custom_forward

        super().__init__()
        self._ddp = _ddp
        if _ddp:
            _replay_ddp_group()
        if (
            type(objective) is not NormalLoss
            or set(objective.loss_property)
            not in ({"energy", "forces"}, {"energy", "forces", "stress"})
            or custom_forward(objective)
            # Even a world-size-one process group runs all_reduce inside
            # TACE's tensor_loss. Do not record rank collectives in this graph.
            or (
                not _ddp
                and torch.distributed.is_available()
                and torch.distributed.is_initialized()
            )
        ):
            raise ValueError(
                "Replay requires an unmodified E/F or E/F/S NormalLoss and single-device or explicitly bound NCCL execution"
            )
        terms = []
        axes = {"energy": 0, "energy_per_atom": 0, "forces": 1}
        stress = "stress" in objective.loss_property
        if stress:
            axes.update(stress=2, voigt_stress=2)
        self._ddp_stress = _ddp and stress
        for i, name in enumerate(objective.loss_function_name):
            kind, _, field = name.partition("_")
            if kind not in ("mse", "mae", "huber", "l2mae") or field not in axes:
                raise ValueError(f"Unsupported E/F/S replay loss: {name}")
            native = getattr(import_module(f"tace.utils.loss.{kind}_fn"), name)
            if LOSS_FN.get(name) is not native:
                raise ValueError(f"Replaced E/F/S replay loss: {name}")
            kwargs = dict(objective.loss_function_kwargs[i])
            weights = kwargs.pop("element_weights", None)
            table = None
            if weights is not None:
                if not isinstance(weights, (list, tuple, ListConfig)) or not all(
                    type(v) in (int, float) for v in weights
                ):
                    raise ValueError(
                        "Replay element_weights must be a constant numeric sequence"
                    )
                # Keep configuration precision; the native term converts to
                # its normal weight dtype on-device. Move this module to the
                # model's device once before warmup, never copy a CPU list in capture.
                table = f"element_weights_{i}"
                self.register_buffer(
                    table,
                    torch.tensor(list(weights), dtype=torch.float64, device="cpu"),
                    persistent=False,
                )
            axis = axes[field]
            target = ("energy", "forces", "stress")[axis]
            label_keys = (target, "entropy", f"{target}_weight")
            if field == "energy_per_atom":
                label_keys += ("ptr",)
            if target == "forces":
                label_keys += ("batch",)
                if kwargs.get("padded"):
                    label_keys += ("ptr",)
                if table is not None:
                    label_keys += ("node_type",)
            terms.append((native, axis, target, label_keys, kwargs, table))
        if {axis for _, axis, *_ in terms} != set(axes.values()):
            raise ValueError(
                "The replay loss functions must cover every requested property."
            )
        self.terms = tuple(terms)
        self.objective = objective

    def forward(self, pred, data):
        if self._ddp_stress and not (
            "_valid_stress_count" in data and "_loss_world_size" in data
        ):
            raise ValueError("DDP stress counts must be prepared outside CUDA capture")
        scales = data["_efs_term_scales"]
        loss = 0.0
        for i, (fn, axis, target, keys, kwargs, table) in enumerate(self.terms):
            labels = {k: data[k] for k in keys}
            if target == "energy" and "ptr" in labels and "_efs_loss_ptr" in data:
                labels["ptr"] = data["_efs_loss_ptr"]
            if target == "stress" and "_valid_stress_count" in data:
                labels["_valid_stress_count"] = data["_valid_stress_count"]
                labels["_loss_world_size"] = data["_loss_world_size"]
            if table is not None:
                kwargs = {**kwargs, "element_weights": getattr(self, table)}
            term = fn({target: pred[target]}, labels, **kwargs)
            # Huber E/F applies sample weights *inside* its transition. Alter
            # the final mean, never its inputs. Positive stress counts are unchanged.
            loss += (
                term
                * self.objective.loss_property_weights[i]
                * scales[axis].to(term.dtype)
            )
        return loss


def _efs_loss(pred, data, microbatches=1):
    counts = data["ptr"].diff()[:-1]
    energy = (
        pred["energy"].reshape(-1)[:-1] - data["energy"].reshape(-1)[:-1]
    ) / counts
    valid = data["batch"] != len(counts)
    force = (pred["forces"] - data["forces"]).square() * valid[:, None]
    energy = energy.square().mean()
    atoms = data["ptr"][-2]
    if microbatches > 1:
        energy = energy / microbatches
        atoms = data["macro_atoms"]
    loss = energy + force.sum() / (3 * atoms)
    if _MAG_FORCE in data:
        residual = pred[_MAG_FORCE] - data[_MAG_FORCE]
        loss = loss + (residual.square() * valid[:, None]).sum() / (3 * atoms)
    if "stress" in data:
        stress = (pred["stress"][:-1] - data["stress"][:-1]).square().mean()
        loss = loss + (stress / microbatches if microbatches > 1 else stress)
    return loss


class BucketedEFSTrainingStep:
    """Train variable-size E/F(/S) minibatches in a CUDA Graph bucket.

    The objective sums unweighted energy-per-atom, force-component and optional
    stress-component MSE, excluding the disconnected padding graph. For a
    supported magnetic model it also includes the noncollinear magnetic-force
    MSE over the three components of every real atom. Both its spin input and
    its magnetic-force label are required on every call.

    Parameters
    ----------
    model : torch.nn.Module
        Training-mode CUDA model accepted by :func:`supports_training_buckets`.
    optimizer : torch.optim.Optimizer
        Optimizer over the model's live trainable parameters.
    batch : dict
        Fully labelled example batch. Subsequent batches may change geometry,
        edges, species and targets but must keep the same number of structures.
    max_atoms, max_edges : int
        Maximum real atoms and directed edges in one microbatch.
    joint : bool, optional
        Use the released model's joint derivative program. Not supported by
        regional models.
    microbatches : int, optional
        Batches per optimizer update. With more than one, pass a sequence.
        Energy and stress average over all structures and forces over all atoms.

    Examples
    --------
    >>> step = BucketedEFSTrainingStep(model, optimizer, batch, max_atoms=256,
    ...                               max_edges=4608, microbatches=2)  # doctest: +SKIP
    >>> result = step([batch, other])  # doctest: +SKIP
    """

    def __init__(
        self,
        model,
        optimizer,
        batch,
        *,
        max_atoms,
        max_edges,
        joint=False,
        microbatches=1,
        _workspace=None,
        _recordings=None,
    ):
        regional = _regional_bucket_model(model)
        if not regional and not _released_bucket_model(model):
            raise ValueError("Bucket padding requires a supported foundation model")
        if regional and (joint or _workspace is not None):
            raise ValueError(
                "Regional buckets support non-joint, private CUDA replay only"
            )
        self.targets = set(model.get_target_property())
        if type(microbatches) is not int or microbatches < 1:
            raise ValueError("microbatches must be a positive integer")
        if microbatches > 1 and any(
            p._backward_hooks or p._post_accumulate_grad_hooks
            for p in model.parameters()
        ):
            raise ValueError(
                "Microbatch accumulation requires parameters without gradient hooks"
            )
        self.microbatches = microbatches
        self.edge_layout = _directed_edge_layout(model) if regional else "native"
        self.dummy_cell = _padding_cell(batch["lattice"], model.get_cutoff())
        self.cap = (
            max_atoms,
            max_edges,
            len(batch["ptr"]) - 1,
            "stress" in self.targets,
            joint,
        )
        objective = _efs_loss
        get_fidelity_idx = model.get_fidelity_idx
        if microbatches > 1 and not joint:
            from functools import partial

            objective = partial(_efs_loss, microbatches=microbatches)
        if joint:
            from tace.models.compile.wrapper import CompileTensorModel
            from tace.utils.loss.normal import NormalLoss

            model = CompileTensorModel(model.readout_fn).train()
            targets = ["energy", "forces"] + (
                ["stress"] if "stress" in self.targets else []
            )
            names = ["mse_energy_per_atom", "mse_forces"] + (
                ["mse_stress"] if "stress" in self.targets else []
            )
            objective = NormalLoss(targets, names, [1.0] * len(targets)).train()
        self.step = FixedShapeTrainingStep(
            model,
            optimizer,
            self._pack_batch(batch, total_atoms=len(batch["positions"]) * microbatches),
            objective,
            sorted(self.targets),
            joint=joint,
            _workspace=_workspace,
            _recordings=_recordings,
            _get_fidelity_idx=get_fidelity_idx,
        )
        if microbatches > 1:
            self.active = [
                (p, g)
                for p, g in zip(self.step.params, self.step.gradients, strict=True)
                if g is not None
            ]
            self.accumulated = [torch.empty_like(g) for _, g in self.active]

    def _pack_batch(self, batch, *, total_atoms=None):
        return _pack(
            batch,
            *self.cap,
            microbatches=self.microbatches,
            total_atoms=total_atoms,
            edge_layout=self.edge_layout,
            dummy_cell=self.dummy_cell,
            magnetic=_MAG_FORCE in self.targets,
        )

    @staticmethod
    def trim(outputs, batch):
        """Select predictions for real structures and atoms.

        >>> batch = {"positions": torch.zeros(2, 3)}
        >>> outputs = {"energy": torch.tensor([1.5, 0.0]), "forces": torch.ones(4, 3)}
        >>> real = BucketedEFSTrainingStep.trim(outputs, batch)
        >>> real["energy"].tolist(), tuple(real["forces"].shape)
        ([1.5], (2, 3))
        """
        return {
            k: v[: len(batch["positions"])]
            if k in {"forces", _MAG_FORCE}
            else v[:-1]
            if k in {"energy", "stress"}
            else v
            for k, v in outputs.items()
        }

    def __call__(self, batch):
        if self.microbatches == 1:
            with torch.no_grad():
                packed = self._pack_batch(batch)
            return self.trim(self.step(packed), batch)
        if not isinstance(batch, (tuple, list)) or len(batch) != self.microbatches:
            raise ValueError(f"Expected {self.microbatches} structure microbatches")
        total_atoms = sum(len(b["positions"]) for b in batch)
        gradients = [g for _, g in self.active]
        outputs = []
        with torch.no_grad():
            for i, b in enumerate(batch):
                packed = self._pack_batch(b, total_atoms=total_atoms)
                pred = self.trim(self.step.replay(packed), b)
                outputs.append({key: value.clone() for key, value in pred.items()})
                if i == 0:
                    torch._foreach_copy_(self.accumulated, gradients)
                else:
                    torch._foreach_add_(self.accumulated, gradients)
            for (parameter, _), grad in zip(self.active, self.accumulated, strict=True):
                parameter.grad = grad
        self.step.optimizer.step()
        return {
            key: sum(p[key] for p in outputs)
            if key == "loss"
            else torch.cat([p[key] for p in outputs])
            for key in outputs[0]
        }


class _SerialTrainingWorkspace:
    def __init__(self, model, optimizer, joint):
        from threading import get_ident

        from torch.nn.modules import module as module_hooks
        from torch.optim import optimizer as optimizer_hooks

        if (
            joint
            or type(optimizer) is not torch.optim.AdamW
            or not all(group.get("fused") for group in optimizer.param_groups)
            or not _released_bucket_model(model)
        ):
            raise ValueError(
                "Shared workspace requires a supported foundation model with non-joint fused AdamW"
            )
        self.params = tuple(p for p in model.parameters() if p.requires_grad)
        if not self.params or not self.params[0].is_cuda:
            raise ValueError("Shared training workspaces require CUDA parameters")
        if (
            any(p._backward_hooks or p._post_accumulate_grad_hooks for p in self.params)
            or any(
                m._forward_hooks
                or m._forward_pre_hooks
                or m._backward_hooks
                or m._backward_pre_hooks
                for m in model.modules()
            )
            or optimizer._optimizer_step_pre_hooks
            or optimizer._optimizer_step_post_hooks
            or any(
                getattr(module_hooks, name, None)
                for name in (
                    "_global_forward_hooks",
                    "_global_forward_pre_hooks",
                    "_global_backward_hooks",
                    "_global_backward_pre_hooks",
                )
            )
            or optimizer_hooks._global_optimizer_pre_hooks
            or optimizer_hooks._global_optimizer_post_hooks
        ):
            raise ValueError("Shared workspace does not support training hooks")
        device = self.params[0].device
        self.pool = torch.cuda.graph_pool_handle()
        self.capture_stream = torch.cuda.Stream(device=device)
        self.caller_stream = torch.cuda.current_stream(device)
        self.thread = get_ident()
        self.gradients = [torch.empty_like(p) for p in self.params]

    def check(self):
        from threading import get_ident

        if (
            self.thread != get_ident()
            or self.caller_stream != torch.cuda.current_stream(self.params[0].device)
        ):
            raise RuntimeError(
                "Shared workspaces must replay on their original thread and stream"
            )

    def publish(self, outputs):
        # No graph-private prediction or gradient is carried across replays.
        gradients = [p.grad for p in self.params]
        if any(g is None for g in gradients):
            raise RuntimeError(
                "Shared training workspaces expect all parameter gradients"
            )
        torch._foreach_copy_(self.gradients, gradients)
        for p, g in zip(self.params, self.gradients, strict=True):
            p.grad = g
        return {k: v.clone() for k, v in outputs.items()}


class MultiBucketEFSTrainingStep:
    """Train a supported foundation model with the smallest CUDA Graph that fits a batch.

    Parameters
    ----------
    model : torch.nn.Module
        Training-mode L, 7M or TECE-RRA adapter with energy and force targets,
        optionally stress; selected regional spherical or Cartesian ICTD models support
        non-joint private buckets. OAM and OMat24 checkpoints share architecture.
    optimizer : torch.optim.Optimizer
        Optimizer shared by all buckets. Each call performs one update.
    capacities : iterable of (int, int)
        Maximum real atom and directed-edge counts per bucket. All batches
        must contain the same number of structures. Among matching capacities,
        the one with the fewest edges, then fewest atoms, is selected.
    joint : bool, optional
        Use the joint derivative program in each bucket.
    share_workspace : bool, optional
        Share temporary storage across serial, non-joint fused AdamW steps.
        Use one calling thread and CUDA stream. Model, gradients and optimizer
        state remain outside the pool. Hooks and concurrent use are not supported.
    native_product : bool, optional
        Use native-layout quadratic products and their mixed derivatives for a
        specialized FP32 CUDA OAM-L model.

    Examples
    --------
    Given a supported CUDA model, its fused AdamW optimizer and example
    batches, prepare the shared workspace before the first update::

        caps = [(288, 10240), (448, 21760)]
        step = MultiBucketEFSTrainingStep(model, optimizer, caps, share_workspace=True)
        step.prepare_all(example_batches)
        predictions = step(batch)
    """

    def __init__(
        self,
        model,
        optimizer,
        capacities,
        *,
        joint=False,
        share_workspace=False,
        native_product=False,
    ):
        self.capacities = sorted(
            {(n, e) for n, e in capacities}, key=lambda c: (c[1], c[0])
        )
        if not self.capacities or any(
            type(n) is not int or type(e) is not int or n < 1 or e < 0
            for n, e in self.capacities
        ):
            raise ValueError(
                "Expected one or more (positive atoms, nonnegative edges) capacities"
            )
        self.model, self.optimizer, self.joint = model, optimizer, joint
        self._steps = {}
        self.graphs = None
        self.workspace = (
            _SerialTrainingWorkspace(model, optimizer, joint)
            if share_workspace
            else None
        )
        self._native_products = ()
        if native_product:
            from .tace_oam_l.model import _OAMLRepresentation
            from .tace_oam_l.product_kernels import square_metadata

            rep = getattr(getattr(model, "readout_fn", None), "representation", None)
            if (
                not isinstance(rep, _OAMLRepresentation)
                or not model.training
                or rep.model_dtype != torch.float32
                or any(
                    not p.is_cuda or p.dtype != torch.float32
                    for p in rep.products.parameters()
                )
            ):
                raise ValueError(
                    "native_product requires a training-mode FP32 CUDA OAM-L model"
                )
            self._native_products = tuple(
                (product, square_metadata(product))
                for product in rep.products
                if not product.scalar_norms
            )

    def prepare(self, batch):
        """Capture the matching workspace on first use, without an optimizer update."""
        step = self._select(batch)
        return step if self.workspace is None else self

    def prepare_all(self, batches):
        """Prepare matching buckets without updating parameters.

        Parameters
        ----------
        batches : iterable of batches
            Example batches with the same number of structures. With a shared
            workspace all new buckets warm up before their graphs are recorded.

        Returns
        -------
        MultiBucketEFSTrainingStep
            This step. Unseen buckets are still prepared on first use.
        """
        if self.workspace is None:
            for batch in batches:
                self.prepare(batch)
            return self
        self.workspace.check()
        previous, graphs = self._steps.copy(), self.graphs
        recordings = []
        try:
            for batch in batches:
                self._select(batch, recordings)
            for record in recordings:
                with self._capture_state():
                    record()
        except BaseException:
            self._steps, self.graphs = previous, graphs
            raise
        return self

    @contextmanager
    def _capture_state(self):
        params = self.workspace.params if self.workspace is not None else ()
        previous = [p.native_square for p, _ in self._native_products]
        with _preserve_gradients(params):
            try:
                for product, meta in self._native_products:
                    product.native_square = meta
                yield
            finally:
                for (product, _), old in zip(self._native_products, previous):
                    product.native_square = old

    def _select(self, batch, recordings=None):
        if self.workspace is not None:
            self.workspace.check()
        graphs = len(batch["ptr"]) - 1
        if self.graphs is not None and graphs != self.graphs:
            raise ValueError("The number of real structures per training batch changed")
        shape = (len(batch["positions"]), batch["edge_index"].shape[1])
        for cap in self.capacities:
            if shape[0] <= cap[0] and shape[1] <= cap[1]:
                if cap not in self._steps:
                    options = (
                        {} if self.workspace is None else {"_workspace": self.workspace}
                    )
                    if recordings is not None:
                        options["_recordings"] = recordings
                    with self._capture_state():
                        self._steps[cap] = BucketedEFSTrainingStep(
                            self.model,
                            self.optimizer,
                            batch,
                            max_atoms=cap[0],
                            max_edges=cap[1],
                            joint=self.joint,
                            **options,
                        )
                    self.graphs = graphs
                return self._steps[cap]
        raise ValueError(f"Batch {shape} exceeds configured training buckets")

    def __call__(self, batch):
        result = self._select(batch)(batch)
        return result if self.workspace is None else self.workspace.publish(result)

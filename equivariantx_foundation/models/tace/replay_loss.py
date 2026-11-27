"""Device-local energy and force replay with optional stress."""

from collections import OrderedDict
from threading import get_ident

import torch

from equivariantx_foundation.execution.replay import replay_capacity

try:
    from torch._C._dynamo.guards import dict_version as _dict_version
except ImportError:
    _dict_version = None

from .matching import custom_forward
from .training import (
    FixedShapeTrainingStep,
    PaddedEFSLoss,
    _regional_bucket_model,
    _released_bucket_model,
    _replay_ddp_group,
    pack_efs_training_batch,
)


def _local_readout(head, linear_type):
    from e3nn.math import normalize2mom
    from e3nn.nn import Activation

    from tace.models.linear import e3nnLinear
    from tace.models.spherical.readout import ScalarReadOut

    if type(head) is not ScalarReadOut or "_apply_linear" in vars(head):
        return False
    if head.last_layer:
        if not (
            set(head._modules) == {"linear2", "acts"}
            and type(head.linear2) is type(head.acts) is torch.nn.ModuleList
            and len(head.linear2) == len(head.acts) + 1
            and all(
                type(act) is Activation
                and all(
                    type(norm) is normalize2mom and type(norm.f) is torch.nn.SiLU
                    for norm in act.acts
                )
                for act in head.acts
            )
        ):
            return False
        linears = head.linear2
    else:
        if not (
            set(head._modules) == {"linear1"}
            and type(head.linear1) is torch.nn.ModuleList
            and len(head.linear1) == 1
        ):
            return False
        linears = head.linear1
    return all(
        type(layer) is e3nnLinear and type(layer.linear) is linear_type
        for layer in linears
    )


def _validate_model(model, *, properties=None):
    from e3nn import o3
    from e3nn.nn import BatchNorm

    from equivariantx import co3
    from tace.models.adapter import TensorModel
    from tace.models.blocks import AtomicEnergy, ScaleShift
    from tace.models.cartesian.tace import CartesianTACE
    from tace.models.mlp import ACTIVATION
    from tace.models.radial import C2PolynomialCutoff, ZBLBasis

    from .heads import _standard_readout
    from .interactions import StreamedInteraction
    from .local import _StreamUu, _StreamUv
    from .tace_oam_7m.model import SevenRepresentation
    from .tace_oam_l.model import _OAMLRepresentation
    from .tece_oam_rra.radial import RRARepresentation

    host = getattr(model, "readout_fn", None)
    rep = getattr(host, "representation", None)
    regional = getattr(rep, "_foundation_regional", False)
    cartesian = regional and type(host) is CartesianTACE
    if regional:
        supported = (
            _regional_bucket_model(model)
            and model.num_fidelities == 1
            and (
                cartesian
                or (
                    type(model) is TensorModel
                    and all(
                        type(m) in (StreamedInteraction, _StreamUu, _StreamUv)
                        for m in rep.interactions
                    )
                )
            )
        )
    else:
        supported = (
            type(model) is TensorModel
            and type(rep)
            in (SevenRepresentation, _OAMLRepresentation, RRARepresentation)
            and _released_bucket_model(model)
        )
    nn = torch.nn
    local = {
        nn.ModuleList,
        nn.ModuleDict,
        nn.ParameterList,
        nn.Sequential,
        nn.Embedding,
        nn.Linear,
        *ACTIVATION.values(),
    }
    targets = set(model.get_target_property()) if supported else set()
    if not (
        supported
        and _standard_readout(host)
        and targets in ({"energy", "forces"}, {"energy", "forces", "stress"})
        and model.get_model_dtype() == torch.float32
        and not {
            "prepare_property_data",
            "prepare_graph",
            "first_derivative_fn",
            "second_derivative_fn",
            "_directed_replay_view",
        }.intersection(vars(model))
        and set(host._modules) - {"zbl", "scale_shift"}
        == {"representation", "energy_readouts", "atomic_energy_layer"}
        and type(host.atomic_energy_layer) is AtomicEnergy
        and type(host.energy_readouts) is nn.ModuleList
        and len(host.energy_readouts) > 0
        and all(
            _local_readout(h, co3.Linear if cartesian else o3.Linear)
            for h in host.energy_readouts
        )
        and (not hasattr(host, "scale_shift") or type(host.scale_shift) is ScaleShift)
        and (
            not hasattr(host, "zbl")
            or (
                type(host.zbl) is ZBLBasis
                and host.zbl.is_polynomial_cutoff
                and host.zbl.cutoff_fn is C2PolynomialCutoff.calculate_envelope
            )
        )
        and not custom_forward(model, recursive=True)
        and all(
            m.training
            and not hasattr(m, "stochastic_depth")
            and not isinstance(m, BatchNorm)
            and type(m).__module__.split(".")[0]
            in {"tace", "e3nn", "torch", "equivariantx", "equivariantx_foundation"}
            and (
                not type(m).__module__.startswith("torch.nn.")
                or type(m) in local
                or (regional and type(m) is nn.Identity)
                or (
                    type(m) in (nn.LayerNorm, nn.RMSNorm)
                    and len(m.normalized_shape) == 1
                )
            )
            and not (
                type(m) is nn.Embedding
                and (m.max_norm is not None or m.scale_grad_by_freq or m.sparse)
            )
            for m in model.modules()
        )
    ):
        raise ValueError(
            "Bucketed loss requires a supported FP32 CUDA whole training plan or a "
            "supported spherical or Cartesian regional plan, with standard "
            "graph-local energy/force heads, optional stress, and no training hooks or stochastic layers."
        )
    if properties is not None and targets != set(properties):
        raise ValueError(
            "Replay requires the model and loss to request the same properties."
        )
    return (
        ("energy", "forces", "stress") if "stress" in targets else ("energy", "forces")
    )


def _loss_signature(loss):
    from omegaconf import ListConfig

    from tace.utils.loss.registry import LOSS_FN

    if (
        not loss.training
        or custom_forward(loss)
        or len(loss._modules)
        or len(loss._parameters)
        or len(loss._buffers)
    ):
        raise ValueError(
            "Bucketed loss requires the original stateless NormalLoss in training mode."
        )
    names = tuple(loss.loss_function_name)
    weights = tuple(loss.loss_property_weights)

    def freeze(k, v):
        if type(v) in (bool, int, float, str, type(None)):
            return v
        if (
            k == "element_weights"
            and isinstance(v, (list, tuple, ListConfig))
            and all(type(x) in (int, float) for x in v)
        ):
            return tuple(v)
        raise ValueError(f"Unsupported bucketed loss configuration: {k}.")

    kwargs = tuple(
        tuple(sorted((k, freeze(k, v)) for k, v in kw.items()))
        for kw in loss.loss_function_kwargs
    )
    if not (len(names) == len(weights) == len(kwargs)) or any(
        type(x) not in (int, float) for x in weights
    ):
        raise ValueError(
            "Bucketed loss requires scalar configuration and one coefficient per term."
        )
    return (
        tuple(loss.loss_property),
        names,
        weights,
        kwargs,
        tuple(LOSS_FN.get(n) for n in names),
    )


_SCALARS = frozenset((int, bool, float, str, type(None), torch.dtype))
_FIXED_CALLABILITY = _SCALARS | {dict, list, set, tuple, OrderedDict}


def _module_bindings(m):
    return (
        tuple(
            (k, id(v) if callable(v) else v)
            for k, v in vars(m).items()
            if type(v) in _SCALARS or callable(v)
        ),
        *(
            tuple((k, id(v)) for k, v in d.items())
            for d in (m._modules, m._parameters, m._buffers)
        ),
    )


def _location(t):
    return t._cdata, t.data_ptr(), t.shape, t.stride(), t.dtype, t.device


def _parameter_binding(p):
    return (
        _location(p),
        p.requires_grad,
        bool(p._backward_hooks),
        bool(p._post_accumulate_grad_hooks),
    )


def _policy(model):
    return (
        frozenset(model.get_target_property()),
        frozenset(model.get_embedding_property()),
        tuple(vars(model.flags).items()),
        torch.get_default_dtype(),
        torch.get_float32_matmul_precision(),
        torch.backends.cudnn.allow_tf32,
        torch.are_deterministic_algorithms_enabled(),
    )


class _Bindings:
    """Track local registrations, tensor storage and mutable capture policy."""

    def __init__(self, model):
        self.nodes = tuple((m, type(m), custom_forward(m)) for m in model.modules())
        self.structure = tuple(_module_bindings(m) for m, _, _ in self.nodes)

        def registered(name):
            return {
                id(t): t
                for m, _, _ in self.nodes
                for t in getattr(m, name).values()
                if t is not None
            }.values()

        self.parameters = tuple(
            (p, _parameter_binding(p)) for p in registered("_parameters")
        )
        self.buffers = tuple(
            (b, _location(b), b._version) for b in registered("_buffers")
        )
        self._watch()
        self.policy = _policy(model)

    def _watch(self):
        # ScriptModule registries need not be Python dictionaries.
        self.tables = ()
        self.registries = tuple(
            (m, vars(m), m._modules, m._parameters, m._buffers)
            for m, _, _ in self.nodes
        )
        tables = tuple(d for _, *ds in self.registries for d in ds)
        if _dict_version is not None and all(type(d) is dict for d in tables):
            self.tables = tuple((d, _dict_version(d)) for d in tables)
            # A user object's class can gain or lose __call__ in place.
            self.callability = tuple(
                (v, callable(v))
                for _, d, *_ in self.registries
                for v in d.values()
                if type(v) not in _FIXED_CALLABILITY
            )

    def matches(self, model):
        if model is not self.nodes[0][0] or any(
            type(m) is not cls or custom_forward(m) != hooks
            for m, cls, hooks in self.nodes
        ):
            return False
        if not self.tables or (
            any(
                vars(m) is not d
                or m._modules is not children
                or m._parameters is not params
                or m._buffers is not bufs
                for m, d, children, params, bufs in self.registries
            )
            or any(_dict_version(d) != stamp for d, stamp in self.tables)
            or any(callable(v) != was for v, was in self.callability)
        ):
            if any(
                _module_bindings(m) != s
                for (m, _, _), s in zip(self.nodes, self.structure)
            ):
                return False
            if self.tables:
                self._watch()
        return (
            all(_parameter_binding(p) == s for p, s in self.parameters)
            and all(b._version == v and _location(b) == s for b, s, v in self.buffers)
            and self.policy == _policy(model)
        )


class _ParameterLossChunk(torch.autograd.Function):
    """Scale one block of saved derivatives and pass the cotangent upstream."""

    @staticmethod
    def forward(ctx, loss, gradients, *parameters):
        ctx.save_for_backward(*gradients)
        return loss

    @staticmethod
    def backward(ctx, scale):
        from tace.models.compile.joint import ParameterLoss

        return (scale, *ParameterLoss.backward(ctx, scale)[1:])


_VJP_CHUNK_BYTES = 128 * 1024**2


def _bind_parameter_loss(parameters, gradients):
    """Bind captured VJPs to parameters in native order, skipping unused leaves."""
    from tace.models.compile.joint import ParameterLoss

    owners, groups = [], {}
    chunks, start, size = [], 0, 0
    for p, g in zip(parameters, gradients, strict=True):
        if g is not None:
            nbytes = g.numel() * g.element_size()
            if size and size + nbytes > _VJP_CHUNK_BYTES:
                chunks.append(slice(start, len(owners)))
                start, size = len(owners), 0
            size += nbytes
            slots, tensors = groups.setdefault((g.device, g.dtype), ([], []))
            slots.append(len(owners))
            tensors.append(g)
            owners.append(p)
    chunks.append(slice(start, len(owners)))
    clone = getattr(torch, "_foreach_clone", None)

    def attach(loss):
        # Each loss owns its VJP, including when two forwards precede backward.
        copies = [None] * len(owners)
        for slots, tensors in groups.values():
            if clone is None:
                values = [torch.empty_like(g) for g in tensors]
                torch._foreach_copy_(values, tensors)
            else:
                values = clone(tensors)
            for i, value in zip(slots, values, strict=True):
                copies[i] = value
        if len(chunks) == 1:
            return ParameterLoss.apply(loss, tuple(copies), *owners)
        token = loss.detach()
        for block in chunks:
            token = _ParameterLossChunk.apply(
                token, tuple(copies[block]), *owners[block]
            )
        return token.clone()

    return attach


class BucketedTrainingLoss:
    """Bind atom/edge/structure replay capacities to the live training model.

    Parameters
    ----------
    model : torch.nn.Module
        Training-mode Foundation model on one CUDA device.
    optimizer : torch.optim.Optimizer
        Optimizer owning the trainable parameters.
    objective : NormalLoss
        Training loss over energy and forces, optionally stress.
    capacity : sequence
        Maximum real atoms, edges and optionally structures, or a sequence
        of fixed atom/edge/structure triples.
    ddp : bool, optional
        Use the default NCCL group for distributed training.

    Notes
    -----
    Lightning owns backward, loss scaling, accumulation and all updates. Treat
    architecture, kernel policy, unregistered constants and tensor layouts as
    immutable throughout a binding. Call ``clear`` before deliberately changing
    the model. A different graph count triggers capture only without a graph
    capacity. Changing the supported loss configuration always requires capture.
    A list of fixed capacities selects the smallest fitting padded edge count
    and lazily records each shape once. Its losses own their predictions and
    parameter derivatives even across captures. With a single capacity on one
    GPU, finish and release earlier autograd graphs before a new capture.

    With standard NCCL DDP, all ranks must supply the same number of
    microbatches. The samples and graph, atom and edge counts may differ.
    The only collective
    owned by this adapter is the global count of valid stress labels when
    stress is requested. DDP owns parameter reductions, including no_sync.
    """

    def __init__(self, model, optimizer, objective, capacity, *, ddp=False):
        if type(ddp) is not bool:
            raise ValueError("ddp must be a boolean")
        self.capacity = replay_capacity(capacity)
        self._multiple = isinstance(self.capacity[0], tuple)
        self._steps = {}
        self._group = _replay_ddp_group() if ddp else None
        self.targets = _validate_model(
            model, properties=getattr(objective, "loss_property", ())
        )
        PaddedEFSLoss(objective, _ddp=ddp)
        self.model, self.optimizer, self.objective = model, optimizer, objective
        self.captures = self.replays = 0
        self.step = None
        self.clear()

    def clear(self):
        """Release all CUDA graphs without touching optimizer or caller gradients."""
        try:
            if self.step is not None:
                # Complete queued reads from the private pool before it is released.
                self._stream.synchronize()
        finally:
            self._steps.clear()
            self.step = self._state = self._signature = self._stream = self._thread = (
                None
            )

    def __call__(self, batch, loss_fn):
        if (
            loss_fn is not self.objective
            or not torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or (
                self._group is None
                and torch.distributed.is_initialized()
                and torch.distributed.get_world_size() > 1
            )
        ):
            raise ValueError(
                "Replay requires the original loss, enabled gradients, no AMP and a matching single-device or DDP binding."
            )
        if self.step is not None and (
            self._thread != get_ident()
            or self._stream != torch.cuda.current_stream(self.step.device)
        ):
            raise ValueError(
                "A training replay belongs to its original CUDA stream and thread."
            )
        if self.step is not None and not self._state.matches(self.model):
            raise ValueError(
                "Model state, hooks, configuration or precision changed since bucket capture; clear or rebuild the training plan."
            )
        graphs = len(batch["ptr"]) - 1
        capacity = self.capacity
        if self._multiple:
            n, e = len(batch["positions"]), batch["edge_index"].shape[1]
            capacity = next(
                (c for c in capacity if n <= c[0] and e <= c[1] and graphs <= c[2]),
                None,
            )
            if capacity is None:
                raise ValueError(
                    f"No replay bucket fits {n} atoms, {e} edges, {graphs} structures. "
                    f"Configured capacities: {self.capacity}."
                )
        # A two-entry capacity retains only the current structure count.
        signature = (_loss_signature(loss_fn), graphs if len(capacity) == 2 else None)
        if self.step is not None and signature != self._signature:
            self.clear()
        cached = self._steps.get(capacity)
        ddp = self._group is not None
        if cached is None:
            targets = _validate_model(self.model, properties=loss_fn.loss_property)
            objective = PaddedEFSLoss(loss_fn, _ddp=ddp)
        else:
            targets = self.targets
        stress = "stress" in targets
        with torch.no_grad():
            packed = pack_efs_training_batch(
                self.model, batch, *capacity, stress=stress
            )
            if ddp:
                if self._group is not _replay_ddp_group():
                    raise ValueError("The replay DDP process group changed")
                if stress:
                    from tace.utils.loss.common import prepare_tensor_loss_counts

                    packed = prepare_tensor_loss_counts(packed, ("stress",))
        if cached is None:
            step = FixedShapeTrainingStep(
                self.model,
                self.optimizer,
                packed,
                objective.to(device=packed["positions"].device),
                targets,
                _ddp=ddp,
                _private_leaves=self._multiple,
            )
            if ddp and any(g is None for g in step.gradients):
                raise ValueError(
                    "DDP bucket replay requires all trainable parameters to participate "
                    "in the training loss. Use ordinary training for disconnected parameters."
                )
            if self._state is None:
                self._state = _Bindings(self.model)
                self._stream, self._thread = (
                    torch.cuda.current_stream(step.device),
                    get_ident(),
                )
            elif not self._state.matches(self.model):
                self.clear()
                raise ValueError(
                    "Model state changed while recording another replay bucket."
                )
            cached = self._steps[capacity] = (
                step,
                _bind_parameter_loss(step.params, step.gradients),
            )
            self.targets = targets
            self._signature = signature
            self.captures += 1
        step, attach_loss = cached
        self.step = step
        out = step.replay_outputs(packed)
        predictions = {
            k: out[k][: (len(batch["positions"]) if k == "forces" else graphs)].clone()
            for k in targets
        }
        loss = attach_loss(out["loss"])
        self.replays += 1
        return predictions, loss

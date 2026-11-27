"""Device-local energy and force replay with optional stress."""

import logging
from collections import OrderedDict
from math import ceil
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
    UnsupportedReplayBatch,
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


class _ReplayBindingChanged(ValueError):
    pass


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

    _trim_before_capture = False

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

    def __call__(self, batch, loss_fn, *, _global_stress=False):
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
            raise _ReplayBindingChanged(
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
        if _global_stress and not (ddp and stress):
            raise ValueError("Global stress counts require a DDP E/F/S binding.")
        with torch.no_grad():
            packed = pack_efs_training_batch(
                self.model,
                batch,
                *capacity,
                stress=stress,
                _global_stress=_global_stress,
            )
            if ddp:
                if self._group is not _replay_ddp_group():
                    raise ValueError("The replay DDP process group changed")
                if stress and not _global_stress:
                    from tace.utils.loss.common import prepare_tensor_loss_counts

                    packed = prepare_tensor_loss_counts(packed, ("stress",))
        if cached is None:
            if self._trim_before_capture:
                torch.cuda.empty_cache()
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


def _scaled_workspace(nbytes, source, target):
    n, e, g = source
    m, f, h = target
    # The padded forward has an extra node, an edge slot per node and a graph slot.
    return ceil(
        nbytes * max(1, (m + 1) / (n + 1), (f + m) / (e + n), (h + 1) / (g + 1))
    )


def _replay_headroom(device):
    free, total = torch.cuda.mem_get_info(device)
    fraction = getattr(torch.cuda, "get_per_process_memory_fraction", None)
    if fraction is not None and (limit := fraction(device)) < 1:
        total = int(total * limit)
        free = min(free, total - torch.cuda.memory_reserved(device))
    return max(0, free - total // 10)


class _ReplayCapacities:
    def __init__(self, limits=(256, 8192, 8), *, wide_edges=0, full_batches_only=False):
        self.limits = limits
        self.wide_edges = wide_edges
        self.full_batches_only = full_batches_only
        self.wide = False
        self.pending = OrderedDict()
        self.probe = None
        self.workspace = 0

    def clear_probes(self):
        self.pending.clear()
        self.probe = None
        self.workspace = 0

    def choose(self, shape, live=()):
        self.probe = None
        self.workspace = 0
        nmax, emax, gmax = self.limits
        full_batch = bool(self.wide_edges and shape[2] == gmax)
        if self.full_batches_only and shape[2] != gmax:
            return None
        if min(shape) < 1 or shape[0] > nmax or shape[2] > gmax:
            return None
        if full_batch and not live and emax < shape[1] <= self.wide_edges:
            self.wide = True
        wide = full_batch and (self.wide or self.full_batches_only)
        if wide:
            emax = self.wide_edges
        if shape[1] > emax:
            return None

        def fits(cap):
            return (
                all(x <= y for x, y in zip(shape, cap)) and 2 * cap[1] <= 5 * shape[1]
            )

        def smallest(caps):
            return min(
                (c for c in caps if fits(c)),
                key=lambda c: (sum(c[:2]), c),
                default=None,
            )

        cap = smallest(live)
        if cap is not None or len(live) >= 2:
            return cap
        key = smallest(self.pending) or tuple(1 << (n - 1).bit_length() for n in shape)
        samples = self.pending.pop(key, [])
        if len(samples) < (3 if wide else 2):
            self.probe = [shape, 0]
            samples.append(self.probe)
            self.pending[key] = samples
            if len(self.pending) > 8:
                self.pending.popitem(last=False)
            return None
        n, e, g = map(max, zip(shape, *(s for s, _ in samples), strict=True))
        if wide:
            factor = 1.125 if live else 1
            cap = (
                min(nmax, 16 * ceil(n * factor / 16)),
                min(emax, 512 * ceil(e * factor / 512)),
                g,
            )
            if not fits(cap):
                return None
        else:
            cap = min(n + 1, nmax), min(e + 8, emax), g
            if full_batch and live:
                larger = cap[0], min(emax, ceil(1.125 * cap[1])), g
                if fits(larger):
                    cap = larger
        if all(size > 0 for _, size in samples):
            self.workspace = max(_scaled_workspace(size, s, cap) for s, size in samples)
        return cap


class OnlineTrainingLoss:
    """Compute a Lightning loss and its requested E/F or E/F/S observables."""

    # Maximum atom/edge/structure capacity and free GiB required at admission.
    _families = {
        "tace_oam_7m": ((1536, 65536, 85), 6),
        "tace_oam_l": ((160, 8192, 16), 6),
        "tece_oam_rra": ((144, 6400, 16), 12),
        "regional_cgtp": ((256, 4096, 8), 6),
    }

    @staticmethod
    def _regional_recipe(model, *, scheduled):
        from tace.models.spherical.inter import O3CgtpInteraction
        from tace.models.spherical.prod import CgtpACE

        from .interactions import StreamedInteraction
        from .products import _ScheduledCgtpACE

        host = model.readout_fn
        rep = getattr(host, "representation", None)
        interaction = StreamedInteraction if scheduled else O3CgtpInteraction
        product = _ScheduledCgtpACE if scheduled else CgtpACE
        if (
            len(getattr(rep, "interactions", ())) != 2
            or len(getattr(rep, "products", ())) != 2
            or any(type(m) is not interaction for m in rep.interactions)
            or any(type(m) is not product for m in rep.products)
        ):
            return None
        cfg = host.model_config
        if (cfg["num_channel"], cfg["Lmax"]) != (64, 2):
            return None
        targets = set(model.get_target_property())
        orders = tuple(m.correlation for m in rep.products)
        if targets == {"energy", "forces"} and (cfg["lmax"], orders) == (3, (3, 3)):
            return "ef"
        if (
            targets == {"energy", "forces", "stress"}
            and (cfg["lmax"], orders) == (2, (2, 2))
            and cfg["parity"] is True
            and model.num_fidelities == 1
            and all(m.l1l2 is None for m in (*rep.interactions, *rep.products))
            and cfg["edge_update"]["type"]
            == cfg["edge_embedding"]["type"]
            == "identity"
            and cfg["radial_basis"]["hidden"] in ([64, 64], (64, 64))
            and cfg["radial_basis"]["num_radial_basis"] == 8
            and not cfg["radial_basis"]["apply_cutoff"]
        ):
            return "efs"
        return None

    @classmethod
    def ddp_candidate(cls, model):
        from .planner import release_schedule

        return (
            release_schedule(model, task="training_buckets")[0] is not None
            or cls._regional_recipe(model, scheduled=False) == "efs"
        )

    @classmethod
    def _family(cls, model):
        from .planner import release_schedule

        if _released_bucket_model(model):
            return release_schedule(model, task="training_buckets")[0]
        if (
            torch.get_float32_matmul_precision() == "highest"
            and _regional_bucket_model(model)
            and cls._regional_recipe(model, scheduled=True)
        ):
            return "regional_cgtp"
        return None

    @staticmethod
    def _owns_parameters(model, optimizer):
        return {p for p in model.parameters() if p.requires_grad} == {
            p for group in optimizer.param_groups for p in group["params"]
        }

    @classmethod
    def try_create(cls, model, optimizer, objective, *, ddp=False, batch_size=None):
        if (
            type(optimizer) not in (torch.optim.Adam, torch.optim.AdamW)
            or torch.distributed.is_initialized() != ddp
            or torch.is_autocast_enabled("cuda")
            or not torch.is_grad_enabled()
            or not model.training
            or hasattr(model, "forward_loss")
            or not cls._owns_parameters(model, optimizer)
        ):
            return None
        family = cls._family(model)
        if family not in cls._families or (
            ddp
            and family == "regional_cgtp"
            and (
                torch.distributed.get_world_size() != 2
                or "stress" not in model.get_target_property()
            )
        ):
            return None
        try:
            group = _replay_ddp_group() if ddp else None
            _validate_model(model, properties=getattr(objective, "loss_property", ()))
            signature = _loss_signature(objective)
            PaddedEFSLoss(objective, _ddp=ddp)
        except (AttributeError, ValueError):
            return None
        return cls(
            model, optimizer, objective, signature, family, group, batch_size=batch_size
        )

    def __init__(
        self, model, optimizer, objective, signature, family, group, *, batch_size=None
    ):
        self.model, self.optimizer, self.objective = model, optimizer, objective
        self.signature = signature
        self.family = family
        self._group = group
        self._global_stress = group is not None and "stress" in signature[0]
        limits, free_gib = self._families[family]
        regional_stress = family == "regional_cgtp" and "stress" in signature[0]
        full_batches_only = regional_stress and batch_size == 16
        wide_edges = 6144 if regional_stress else 0
        if full_batches_only:
            limits = (256, 8192, 16)
            wide_edges = limits[1]
        self.policy = _ReplayCapacities(
            limits, wide_edges=wide_edges, full_batches_only=full_batches_only
        )
        self.fast_limits = (
            (512, 32768, 32)
            if family == "tace_oam_7m"
            else (limits[0], wide_edges or limits[1], limits[2])
        )
        self.min_free = free_gib * 2**30
        self._tight_memory_attempted = False
        self.workspaces = {}
        self.runner = None
        self.disabled = False
        self.allow_capture = None

    def clear(self):
        if self.runner is not None:
            self.runner.clear()
            self.runner = None
        self.policy.clear_probes()
        self.policy.wide = False
        self.workspaces.clear()

    def can_retain(self, peak, *, extra_bytes=0):
        """Leave the validation reserve free after an upcoming allocation (e.g. EMA.store)."""
        if not (
            self.family == "regional_cgtp"
            and "stress" in self.signature[0]
            and not self.disabled
            and self.runner is not None
            and self.runner.step is not None
            and self.workspaces
            and peak > 0
        ):
            return False
        reserve = max(self.min_free, 2 * peak + 2**30)
        return _replay_headroom(self.runner.step.device) >= reserve + extra_bytes

    def can_resume(self, model, optimizer, objective):
        """Check a retained recording after validation restores training mode."""
        runner = self.runner
        if not (
            runner is not None
            and runner.step is not None
            and model is self.model
            and model.training
            and optimizer is self.optimizer
            and objective is self.objective
            and self._owns_parameters(model, optimizer)
            and runner._thread == get_ident()
            and runner._stream == torch.cuda.current_stream(runner.step.device)
            and runner._state.matches(model)
        ):
            return False
        try:
            return self.signature == _loss_signature(objective)
        except (AttributeError, ValueError):
            return False

    def _eager(self, batch, objective):
        data = self.model.prepare_property_data(batch)
        predictions = self.model(data)
        return predictions, objective(predictions, data)

    def _can_capture(self, workspace, device):
        if not workspace:
            return False
        free = _replay_headroom(device)
        if free >= max(self.min_free, 4 * workspace + 2**30):
            return True
        if (
            self.family != "tece_oam_rra"
            or self.workspaces
            or self._tight_memory_attempted
        ):
            return False
        self._tight_memory_attempted = True
        required = max(self.min_free, 2 * workspace + 2**30)
        if free < required:
            if (
                torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
                < required - free
            ):
                return False
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            free = _replay_headroom(device)
        return free >= required

    def _unrecorded(self, batch, objective, shape):
        device = batch["positions"].device
        probe = self.policy.probe
        if self.workspaces:
            estimate = min(
                _scaled_workspace(size, cap, shape)
                for cap, size in self.workspaces.items()
            )
            if _replay_headroom(device) < max(self.min_free, 2 * estimate):
                self.clear()
                torch.cuda.empty_cache()
                probe = None
        if probe is None:
            return self._eager(batch, objective)
        before = torch.cuda.memory_allocated(device)
        outputs = self._eager(batch, objective)
        probe[1] = max(0, torch.cuda.memory_allocated(device) - before)
        return outputs

    def __call__(self, batch, objective):
        try:
            unchanged = self.signature == _loss_signature(objective)
        except (AttributeError, ValueError):
            unchanged = False
        if self._group is not None:
            if self._group is not _replay_ddp_group():
                raise ValueError("The automatic replay DDP process group changed")
            if objective is not self.objective or not unchanged:
                raise ValueError(
                    "Distributed automatic replay requires its original loss configuration"
                )
            if self._global_stress:
                from tace.utils.loss.common import prepare_tensor_loss_counts

                # Every rank reduces structure counts before its local route decision.
                batch = prepare_tensor_loss_counts(batch, ("stress",))
        if (
            self.disabled
            or objective is not self.objective
            or not self.model.training
            or not torch.is_grad_enabled()
            or torch.is_autocast_enabled("cuda")
            or (self._group is None and torch.distributed.is_initialized())
            or any(v.requires_grad for v in batch.values() if torch.is_tensor(v))
        ):
            return self._eager(batch, objective)
        if not unchanged:
            self.clear()
            self.disabled = True
            return self._eager(batch, objective)
        shape = (
            len(batch["positions"]),
            batch["edge_index"].size(1),
            len(batch["ptr"]) - 1,
        )
        live = self.runner.capacity if self.runner is not None else ()
        cap = self.policy.choose(shape, live)
        if cap is None:
            return self._unrecorded(batch, objective, shape)
        if cap not in live:
            window = (
                16
                if all(x <= y for x, y in zip(cap, self.fast_limits, strict=True))
                else 128
            )
            if self.allow_capture is not None and not self.allow_capture(window):
                return self._unrecorded(batch, objective, shape)
            if self._family(self.model) != self.family or not self._owns_parameters(
                self.model, self.optimizer
            ):
                self.clear()
                self.disabled = True
                return self._eager(batch, objective)
            workspace = self.policy.workspace
            if not self._can_capture(workspace, batch["positions"].device):
                return self._unrecorded(batch, objective, shape)
            if self.runner is None:
                try:
                    self.runner = BucketedTrainingLoss(
                        self.model,
                        self.optimizer,
                        objective,
                        (cap,),
                        ddp=self._group is not None,
                    )
                except ValueError:
                    self.disabled = True
                    return self._eager(batch, objective)
                self.runner._trim_before_capture = True
            else:
                self.runner.capacity = replay_capacity((*live, cap))
        try:
            outputs = (
                self.runner(batch, objective, _global_stress=True)
                if self._global_stress
                else self.runner(batch, objective)
            )
            if cap not in live:
                self.workspaces[cap] = workspace
                logging.getLogger(__name__).info(
                    "Recorded %s training on %s: %d atoms, %d edges, %d structures",
                    self.family,
                    batch["positions"].device,
                    *cap,
                )
            return outputs
        except UnsupportedReplayBatch:
            # Only input-schema rejection is recoverable here, not a kernel or model error.
            if cap not in live:
                if not live:
                    self.clear()
                else:
                    self.runner._steps.pop(cap, None)
                    self.runner.capacity = live
        except _ReplayBindingChanged:
            self.clear()
            self.disabled = True
        except torch.OutOfMemoryError:
            # The private VJP performs no optimizer update and does not write p.grad.
            self.disabled = True
        if self.disabled and self.runner is not None:
            self.clear()
            torch.cuda.empty_cache()
        return self._unrecorded(batch, objective, shape)

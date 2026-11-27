################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Foundation execution for the ordinary Lightning training loop."""

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from math import isfinite

import torch
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.plugins.precision import DoublePrecision, Precision
from lightning.pytorch.strategies import DDPStrategy, SingleDeviceStrategy

from tace.utils.strategy import SimpleDDPStrategy


@contextmanager
def foundation_source(env=()):
    """Leave provider selection to the Foundation plan while constructing its source."""
    from tace.utils.env import ACCELERATION_ENV, EQX_KERNELS

    flag = ACCELERATION_ENV["eqx"]
    if flag in os.environ or flag in env:
        yield
        return
    previous = EQX_KERNELS.copy()
    EQX_KERNELS.update(dict.fromkeys(previous, False))
    try:
        yield
    finally:
        EQX_KERNELS.update(previous)


def _same_native_state(source, candidate):
    def parameters(m):
        # Include parameter aliases, frozen parameters, and the order used by EMA.
        return [(k, id(p)) for k, p in m.named_parameters(remove_duplicate=False)]

    if parameters(candidate) != parameters(source):
        return False
    native, active = source.state_dict(), candidate.state_dict()
    if native.keys() != active.keys():
        return False
    names = dict(source.named_parameters(remove_duplicate=False))
    for key, a in native.items():
        b = active[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            return False
        if key not in names and not torch.equal(a, b):
            return False
    return True


def _packaged_loss(path, model, objective, device):
    from equivariantx_foundation.targets import aoti
    from tace.models.compile.joint import ParameterLoss

    index = device.index if device.index is not None else -1
    runner = aoti.load_training(path, device_index=index)
    capacity = getattr(runner, "replay_bucket", None)
    padded = signature = None
    parameters = buffers = None
    trainability = ()
    batch_fields = frozenset()

    def fields(data):
        return frozenset(k for k, v in data.items() if torch.is_tensor(v))

    def compute(batch, loss_fn):
        nonlocal runner, padded, signature
        nonlocal parameters, buffers, trainability, batch_fields
        if loss_fn is not objective:
            raise ValueError(
                "A joint training package requires its original loss module."
            )
        if capacity is not None:
            from equivariantx_foundation.models.tace.replay_loss import _loss_signature

            current = _loss_signature(objective)
            if padded is None:
                padded = aoti._bucket_objective(model, objective)
                signature = current
            elif current != signature:
                raise ValueError(
                    "Fixed bucket training objective changed since export."
                )
            data = aoti._bucket_data(model, batch, capacity)
        if parameters is None:
            if capacity is None:
                data, params, state = aoti.training_inputs(model, objective, batch)
            else:
                params, state = aoti._training_state(model, padded)
            aoti.validate_training_model(
                runner, model, objective if padded is None else padded, data
            )
            if capacity is not None:
                binding = runner._host_binding
                runner = None
                runner = aoti.load_training(
                    path, device_index=index, replay_inputs=(data, params, state)
                )
                if runner.replay_bucket != capacity or runner._host_binding != binding:
                    raise ValueError(
                        "Fixed bucket training package changed during binding."
                    )
            parameters, buffers = params, state
            trainability = tuple(p.requires_grad for p in parameters.values())
            batch_fields = fields(data)
        else:
            if capacity is None:
                data = aoti.training_batch(model, objective, batch)
            if fields(data) != batch_fields:
                raise ValueError("Joint training batch tensor fields changed.")
            if any(
                p.requires_grad != flag
                for p, flag in zip(parameters.values(), trainability, strict=True)
            ):
                raise ValueError("Trainable parameters changed since the joint export.")
        predictions, loss, gradients = runner(data, parameters, buffers)
        if capacity is not None:
            atoms, structures = len(batch["positions"]), len(batch["ptr"]) - 1
            predictions = {
                key: value[: atoms if key in ("node_energy", "forces") else structures]
                for key, value in predictions.items()
            }
        # Only connected parameters participate in autograd and DDP's unused
        # parameter detection. Lightning supplies the accumulation scale.
        names = [name for name, value in gradients.items() if value is not None]
        return predictions, ParameterLoss.apply(
            loss, tuple(gradients[n] for n in names), *(parameters[n] for n in names)
        )

    return compute


def _remaining_batches(trainer):
    total = trainer.num_training_batches
    if type(total) not in (int, float) or not total > 0:
        return 0
    completed = trainer.fit_loop.epoch_loop.batch_progress.current.completed
    if total and completed == total:
        completed = 0
    remaining = total - completed
    if trainer.max_steps >= 0:
        remaining = min(remaining, trainer.max_steps - trainer.global_step)
    if trainer.enable_validation:
        interval = trainer.val_check_batch
        if type(interval) not in (int, float) or not interval > 0:
            return 0  # Wall-clock validation does not imply a batch-count window.
        position = (
            trainer.global_step
            if getattr(trainer, "check_val_every_n_epoch", 1) is None
            else completed
        )
        remaining = min(remaining, interval - position % interval)
    return remaining if isfinite(remaining) else 0


def _standard_ddp_options(strategy):
    options = strategy._ddp_kwargs
    return (
        not set(options).difference(
            {
                "find_unused_parameters",
                "gradient_as_bucket_view",
                "bucket_cap_mb",
                "init_sync",
                "broadcast_buffers",
                "forward_sync_buffers",
                "static_graph",
                "process_group",
            }
        )
        and options.get("init_sync") is not False
        and options.get("broadcast_buffers") is not False
        and options.get("forward_sync_buffers") is not False
        and not options.get("static_graph")
        and strategy._ddp_comm_hook is None
        and strategy._ddp_comm_wrapper is None
        and strategy._model_averaging_period is None
    )


def _default_reducer(ddp, module):
    return (
        type(ddp) is torch.nn.parallel.DistributedDataParallel
        and ddp.module is module
        and ddp.process_group is torch.distributed.group.WORLD
        and not ddp.parameters_to_ignore
        and not ddp._comm_hooks
        and not hasattr(ddp, "buffer_hook")
    )


def _agree_ddp_replay(trainer, objective, step, remaining):
    # Vote once, before the first training batch. Every rank must either use
    # the common labelled-count reduction or leave it to the ordinary loss.
    mode = 0 if step is None else 1 + int("stress" in objective.loss_property)
    val = trainer.num_val_batches if trainer.enable_validation else []
    if len(val) > 1:
        mode = 0
    signature = (
        mode,
        trainer.num_training_batches,
        remaining,
        val[0] if val else 0,
        (trainer.val_check_batch or 0) if trainer.enable_validation else 0,
    )
    if any(type(v) not in (int, float) or not isfinite(v) for v in signature):
        signature = (0,) * len(signature)
    device = trainer.strategy.root_device
    vote = torch.tensor(
        [signature, [-v for v in signature]], device=device, dtype=torch.float64
    )
    torch.distributed.all_reduce(vote, op=torch.distributed.ReduceOp.MIN)
    minimum, neg_maximum = vote.tolist()
    return bool(minimum[0]) and minimum == [-v for v in neg_maximum]


@dataclass
class _ValidationProfile:
    loader: object
    count: int = 0
    shape: tuple = (0, 0, 0)
    peak: int = 0
    ema: object = None
    backup_bytes: int = 0

    def observe(self, batch):
        shape = (
            len(batch["positions"]),
            batch["edge_index"].size(1),
            len(batch["ptr"]) - 1,
        )
        self.count += 1
        self.shape = tuple(map(max, self.shape, shape))
        return shape


def _ema_backup_bytes(callback, module, *, initializing=False):
    """Size of the ordinary in-place EMA swap; None if ownership is uncertain."""
    from tace.utils.ema import ExponentialMovingAverage

    if (
        callback.device is not None
        or callback.pending_state is not None
        or any(callable(getattr(type(callback), key, None)) for key in vars(callback))
    ):
        return None
    average = callback.ema
    params = tuple(module.model.parameters())
    if average is None:
        if not initializing or callback.owner is not None:
            return None
    elif (
        type(average) is not ExponentialMovingAverage
        or callback.owner is None
        or callback.owner() is not module
        or average.collected_params is not None
        or any(callable(getattr(type(average), key, None)) for key in vars(average))
        or len(average._params_refs) != len(params)
        or len(average.shadow_params) != len(params)
        or any(
            ref() is not p
            or type(shadow) is not torch.Tensor
            or (shadow.shape, shadow.dtype, shadow.device) != (p.shape, p.dtype, p.device)
            for p, ref, shadow in zip(params, average._params_refs, average.shadow_params)
        )
    ):
        return None
    return sum(p.numel() * p.element_size() for p in params)


def _profile_validation(trainer, planner, module):
    from lightning.pytorch.callbacks import ModelCheckpoint, TQDMProgressBar
    from torch_geometric.loader import DataLoader

    from tace.dataset.datamodule import GraphDataModule
    from tace.utils.callbacks import EMACallback, PrintMetricsCallback

    if (
        type(getattr(trainer, "datamodule", None)) is not GraphDataModule
        or type(trainer.val_dataloaders) is not DataLoader
        or trainer.reload_dataloaders_every_n_epochs
        or len(trainer.num_val_batches) != 1
        or type(trainer.num_val_batches[0]) is not int
        or trainer.num_val_batches[0] <= 0
    ):
        return None
    hooks = tuple(k for k in vars(Callback) if k.startswith("on_validation_"))
    profile = _ValidationProfile(trainer.val_dataloaders)
    before_swap = False
    for cb in trainer.callbacks:
        if any(k in vars(cb) for k in hooks):
            return None
        if cb is planner:
            before_swap = True
        if isinstance(cb, EMACallback):
            if type(cb) is not EMACallback or not before_swap or profile.ema is not None:
                return None
            size = _ema_backup_bytes(cb, module, initializing=trainer.sanity_checking)
            if size is None:
                return None
            profile.ema, profile.backup_bytes = cb, size
        elif not (
            type(cb)
            in (
                FoundationTrainingCallback,
                ModelCheckpoint,
                TQDMProgressBar,
                PrintMetricsCallback,
            )
            or all(
                getattr(getattr(cb, k), "__func__", None) is getattr(Callback, k)
                for k in hooks
            )
        ):
            return None
    return profile


class FoundationTrainingCallback(Callback):
    """Install a state-compatible Foundation training plan for the duration of fit.

    Parameters
    ----------
    backend : {None, "torch", "cuda"}, optional
        Explicit regional provider, or the planner default for the rank's
        device. Model parameters must use FP32 or FP64 without autocast.
    training_package : str or pathlib.Path, optional
        Joint AOTInductor package exported for the same regional model, loss
        and local batch layout. Used only for training microbatches. Parameters
        and buffers must retain their identities and the architecture, objective
        configuration and trainable parameter set must remain fixed during fit.
    replay_bucket : {"auto", None} or sequence, optional
        Maximum real atoms, directed edges and optionally structures per local
        microbatch, or a list of fixed atom/edge/structure capacities for
        whole-model CUDA graph training. The default learns recurring capacities
        for eligible CUDA models on one device or with standard NCCL DDP.
        None disables automatic replay.
        The standard TACE graph loader can
        supply the structure bound from its batch size. Mutually exclusive
        with a package.

    Notes
    -----
    Selected by ``tace-fast-train``. Lightning owns optimization, loss,
    metrics, EMA and DDP.
    Validation during fit uses the registered adapter. Standalone evaluation
    uses the original model and checkpoints load without this callback.

    Examples
    --------
    >>> from tace.utils.callbacks import FoundationTrainingCallback
    >>> callback = FoundationTrainingCallback(backend="torch")
    """

    def __init__(self, backend=None, training_package=None, replay_bucket="auto"):
        self._auto = replay_bucket == "auto" and training_package is None
        self._auto_checked = False
        self._defer_auto_ddp = False
        if replay_bucket == "auto":
            replay_bucket = None
        if replay_bucket is not None:
            if training_package is not None:
                raise ValueError("Choose either replay_bucket or training_package.")
            from equivariantx_foundation.execution.replay import replay_capacity

            replay_bucket = replay_capacity(replay_bucket)
        self.replay_bucket = replay_bucket
        self.backend = backend
        self.training_package = training_package
        self._source = None
        self._load_hooks = ()
        self._step = None
        self._validation_profile = self._validation_pass = None
        self._retained = False

    def setup(self, trainer, pl_module, stage):
        from equivariantx_foundation.registry import require_cuda
        from tace.lightning.lit_model import LightningWrapperModel
        from tace.models.adapter import TensorModel
        from tace.utils.env import acceleration_enabled

        if stage != "fit":
            return
        if self._source is not None or self._step is not None:
            raise RuntimeError("FoundationTrainingCallback already owns an active fit")
        if (
            type(pl_module) is not LightningWrapperModel
            or type(pl_module.model) is not TensorModel
            or not pl_module.automatic_optimization
            or acceleration_enabled("compile")
            or getattr(
                getattr(pl_module.model.readout_fn, "representation", None),
                "_foundation_regional",
                False,
            )
        ):
            raise ValueError(
                "The Foundation plan requires an eager LightningWrapperModel and TensorModel "
                "(not XL-v2 or compile). For generic EQX+AOTInductor use tace-train; "
                "for the dedicated OAM-v2 schedule use tace-fast-train +fast_schedule=oam_v2."
            )
        if self.backend not in (None, "torch", "cuda"):
            raise ValueError(
                "Foundation training backend must be None, 'torch' or 'cuda'"
            )
        strategy = trainer.strategy
        if (
            type(strategy) not in (SingleDeviceStrategy, DDPStrategy, SimpleDDPStrategy)
            or strategy.restore_checkpoint_after_setup
            or getattr(strategy, "_layer_sync", None) is not None
        ):
            raise ValueError(
                "Foundation training requires single-device, DDP or SimpleDDPStrategy with checkpoint restoration before strategy setup"
            )
        device = strategy.root_device
        require_cuda(device)
        ddp_replay = self.replay_bucket is not None and type(strategy) is DDPStrategy
        if self.replay_bucket is not None:
            if (
                type(strategy) not in (SingleDeviceStrategy, DDPStrategy)
                or device.type != "cuda"
                or self.backend == "torch"
            ):
                raise ValueError(
                    "replay_bucket requires single-device CUDA or standard NCCL DDP."
                )
            if ddp_replay:
                if not _standard_ddp_options(strategy):
                    raise ValueError(
                        "Bucket replay requires standard DDP synchronization."
                    )
        precision = trainer.precision_plugin
        if (
            type(precision) not in (Precision, DoublePrecision)
            or precision.precision not in ("32-true", "64-true")
            or (
                type(precision) is DoublePrecision
                and pl_module.model.get_model_dtype() != torch.float64
            )
        ):
            raise ValueError(
                "Foundation training requires FP32 or FP64 true precision without AMP"
            )

        if (
            self._auto
            and type(strategy) is DDPStrategy
            and self.backend != "torch"
            and type(precision) is Precision
            and precision.precision == "32-true"
            and trainer.accumulate_grad_batches == 1
            and not trainer.fast_dev_run
            and not 0 <= trainer.max_steps < 64
            and not (
                type(trainer.limit_train_batches) is int
                and trainer.limit_train_batches < 64
            )
            and not (
                trainer.limit_val_batches != 0
                and type(trainer.val_check_interval) is int
                and trainer.val_check_interval < 64
            )
            and _standard_ddp_options(strategy)
            and not strategy._ddp_kwargs.get("find_unused_parameters")
            and strategy._ddp_kwargs.get("process_group")
            in (None, torch.distributed.group.WORLD)
        ):
            from equivariantx_foundation.models.tace.replay_loss import (
                OnlineTrainingLoss,
            )

            self._defer_auto_ddp = OnlineTrainingLoss.ddp_candidate(pl_module.model)
        if type(strategy) is SimpleDDPStrategy or ddp_replay or self._defer_auto_ddp:
            # Build derived constants only after checkpoint restore and rank sync.
            return

        # setup() runs in the rank's initialized environment, but before the
        # strategy has placed modules or constructed DDP and optimizer groups.
        source = pl_module.model.to(device)
        if not self._install(pl_module, source):
            return
        self._source = source

        # A fit(ckpt_path=...) restores the Lightning state *after* setup().
        # Restore into native modules first, then rebuild any derived plans
        # from the restored buffers before strategy.setup() wraps this tree.
        def before(module, *_):
            module.model = source

        def after(module, incompatible):
            if not (incompatible.missing_keys or incompatible.unexpected_keys):
                self._install(module, source)

        self._load_hooks = (
            pl_module.register_load_state_dict_pre_hook(before),
            pl_module.register_load_state_dict_post_hook(after),
        )

    def _install(self, pl_module, source, *, regional_only=False):
        from equivariantx_foundation.models.tace.planner import _optimize_with_report

        from .optimizer import get_parameter_roles, get_weight_decay_exclusions

        options = dict(
            backend=self.backend,
            task="training",
            regional_only=regional_only or self.training_package is not None,
        )
        with foundation_source():
            report, candidate = _optimize_with_report(source, **options)
        if report.schedule == "regional" and candidate is source:
            if self.training_package is not None:
                return False
            raise ValueError(
                "No Foundation training plan applies to this model. "
                f"{report.reason} Use tace-train for native or generic EQX training."
            )

        def roles():
            # Evaluate on the whole LightningModule (also owns trainable loss).
            return get_weight_decay_exclusions(pl_module), get_parameter_roles(
                pl_module
            )

        if _same_native_state(source, candidate):
            native_roles = roles()
            pl_module.model = candidate
            try:
                compatible = roles() == native_roles
            except BaseException:
                pl_module.model = source
                raise
            if compatible:
                logging.info(
                    "Using Foundation training plan: %s (%s)",
                    report.schedule,
                    report.backend,
                )
                return True
            pl_module.model = source
            reason = "optimizer parameter roles"
        else:
            reason = "parameter identities/order or checkpoint schema"
        if report.schedule != "regional":
            logging.info(
                "Foundation %s changes %s; selecting regional training",
                report.schedule,
                reason,
            )
            del candidate
            return self._install(pl_module, source, regional_only=True)
        raise ValueError(f"Foundation regional plan changes native {reason}")

    def _remove_load_hooks(self):
        for handle in self._load_hooks:
            handle.remove()
        self._load_hooks = ()

    def on_fit_start(self, trainer, pl_module):
        self._remove_load_hooks()
        strategy = getattr(trainer, "strategy", None)
        ddp_replay = self.replay_bucket is not None and type(strategy) is DDPStrategy
        if ddp_replay:
            if not _default_reducer(strategy.model, pl_module):
                raise ValueError("Bucket replay requires the default DDP reducer.")
        if type(strategy) is SimpleDDPStrategy or ddp_replay or self._defer_auto_ddp:
            source = pl_module.model
            if self._install(pl_module, source):
                self._source = source
        if self._source is not None and (
            pl_module.model is self._source
            or any(
                a is not b
                for a, b in zip(
                    self._source.parameters(), pl_module.model.parameters(), strict=True
                )
            )
        ):
            raise RuntimeError(
                "Foundation training plan was not installed after checkpoint restoration"
            )
        if self.training_package is not None or self.replay_bucket is not None:
            model = pl_module.model
            if hasattr(model, "forward_loss") or hasattr(self._source, "forward_loss"):
                raise ValueError("Cannot replace an existing model.forward_loss.")
            if self.replay_bucket is not None:
                from equivariantx_foundation.models.tace.replay_loss import (
                    BucketedTrainingLoss,
                )

                (optimizer,) = trainer.optimizers
                step = BucketedTrainingLoss(
                    model,
                    optimizer,
                    pl_module.loss_fn,
                    self.replay_bucket,
                    ddp=ddp_replay,
                )
            else:
                step = _packaged_loss(
                    self.training_package, model, pl_module.loss_fn, pl_module.device
                )
            model.forward_loss = step
            self._step = (model, step)

    def _try_online(self, trainer, pl_module):
        strategy = trainer.strategy
        dist = torch.distributed
        ddp = type(strategy) is DDPStrategy
        if ddp:
            self._auto_checked = True
            if not (
                dist.is_initialized()
                and dist.get_backend() == "nccl"
                and dist.get_world_size() > 1
                and strategy._ddp_kwargs.get("process_group")
                in (None, dist.group.WORLD)
                and (
                    not isinstance(
                        strategy.model, torch.nn.parallel.DistributedDataParallel
                    )
                    or strategy.model.process_group is dist.group.WORLD
                )
            ):
                return
        elif type(strategy) is not SingleDeviceStrategy or dist.is_initialized():
            return
        remaining = _remaining_batches(trainer)
        model = pl_module.model
        step = None
        if (
            self._source is not None
            and self._step is None
            and self.backend != "torch"
            and strategy.root_device.type == "cuda"
            and type(trainer.precision_plugin) is Precision
            and trainer.precision_plugin.precision == "32-true"
            and trainer.accumulate_grad_batches == 1
            and not trainer.fast_dev_run
            and len(trainer.optimizers) == 1
            and not hasattr(model, "forward_loss")
            and not hasattr(self._source, "forward_loss")
            and remaining >= 64
            and (
                not ddp
                or (
                    self._defer_auto_ddp
                    and _standard_ddp_options(strategy)
                    and _default_reducer(strategy.model, pl_module)
                    and not strategy.model.find_unused_parameters
                    and not strategy.model.static_graph
                    and strategy.model.broadcast_buffers
                    and strategy.model.require_backward_grad_sync
                    # DDP clears require_forward_param_sync after a no-grad
                    # validation forward. The first training probe is eager.
                    and not strategy.model._join_config.enable
                )
            )
        ):
            from equivariantx_foundation.models.tace.replay_loss import (
                OnlineTrainingLoss,
            )

            self._auto_checked = True
            step = OnlineTrainingLoss.try_create(
                model,
                trainer.optimizers[0],
                pl_module.loss_fn,
                ddp=ddp,
                batch_size=getattr(trainer.train_dataloader, "batch_size", None),
            )
        if ddp and not _agree_ddp_replay(trainer, pl_module.loss_fn, step, remaining):
            return
        if step is not None:
            step.allow_capture = lambda steps: _remaining_batches(trainer) >= steps
            model.forward_loss = step
            self._step = model, step

    def on_train_epoch_start(self, trainer, pl_module):
        if self._auto and not self._auto_checked:
            self._try_online(trainer, pl_module)
        capacity = self.replay_bucket
        if (
            self._step is None
            or capacity is None
            or len(capacity) != 2
            or type(capacity[0]) is not int
        ):
            return
        from torch.utils.data import BatchSampler
        from torch_geometric.loader import DataLoader
        from torch_geometric.loader.dataloader import Collater

        from tace.dataset.dataloader import EpochDataLoader
        from tace.dataset.datamodule import GraphDataModule

        loader = trainer.train_dataloader
        if (
            type(trainer.datamodule) is GraphDataModule
            and type(loader) in (DataLoader, EpochDataLoader)
            and type(loader.batch_sampler) is BatchSampler
            and type(loader.collate_fn) is Collater
            and not loader.collate_fn.follow_batch
            and not loader.collate_fn.exclude_keys
            and not loader.drop_last
            and not loader.batch_sampler.drop_last
            and type(loader.batch_size) is int
            and 1 < loader.batch_size <= capacity[0]
            and loader.batch_size == loader.batch_sampler.batch_size
        ):
            capacity = (*capacity, loader.batch_size)
        step = self._step[1]
        if step.capacity != capacity:
            step.clear()
            step.capacity = capacity

    def on_validation_start(self, trainer, pl_module):
        from equivariantx_foundation.models.tace.replay_loss import OnlineTrainingLoss

        self._retained = False
        self._validation_pass = None
        step = self._step[1] if self._step is not None else None
        if (
            self._auto
            and self._source is not None
            and (current := _profile_validation(trainer, self, pl_module)) is not None
            and not trainer.should_stop
            and (
                trainer.max_epochs is None
                or trainer.max_epochs < 0
                or trainer.current_epoch + 1 < trainer.max_epochs
            )
            and (trainer.max_steps < 0 or trainer.global_step < trainer.max_steps)
            and (
                (
                    step is None
                    and not self._auto_checked
                    and OnlineTrainingLoss._regional_recipe(
                        pl_module.model, scheduled=True
                    )
                    == "efs"
                )
                or (
                    type(step) is OnlineTrainingLoss
                    and step.family == "regional_cgtp"
                    and "stress" in step.signature[0]
                )
            )
        ):
            profile = self._validation_profile
            self._validation_pass = current
            self._retained = (
                step is not None
                and profile is not None
                and profile.loader is current.loader
                and profile.ema is current.ema
                and profile.count == trainer.num_val_batches[0]
                and step.can_retain(profile.peak, extra_bytes=current.backup_bytes)
            )
        if self._retained:
            step.policy.clear_probes()
        elif step is not None and self.training_package is None:
            step.clear()

    def on_validation_batch_start(
        self, trainer, pl_module, batch, batch_idx, dataloader_idx=0
    ):
        if self._validation_pass is not None:
            shape = self._validation_pass.observe(batch)
            profile = self._validation_profile
            if self._retained and (
                any(x > y for x, y in zip(shape, profile.shape, strict=True))
                or not self._step[1].can_retain(profile.peak)
            ):
                self._step[1].clear()
                self._retained = False

    def on_validation_end(self, trainer, pl_module):
        profile = self._validation_pass
        if profile is not None and not self._retained:
            profile.peak = torch.cuda.max_memory_allocated(trainer.strategy.root_device)
            self._validation_profile = profile
        self._validation_pass = None

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if self._retained:
            self._retained = False
            _, step = self._step
            if (
                len(trainer.optimizers) != 1
                or trainer.optimizers[0] is not step.optimizer
            ):
                step.disabled = True
            if (
                step.disabled
                or getattr(pl_module.model, "forward_loss", None) is not step
                or (
                    self._validation_profile.ema is not None
                    and _ema_backup_bytes(self._validation_profile.ema, pl_module) is None
                )
                or not step.can_retain(self._validation_profile.peak)
                or not step.can_resume(
                    pl_module.model, trainer.optimizers[0], pl_module.loss_fn
                )
            ):
                step.clear()

    def teardown(self, trainer, pl_module, stage):
        self._retained = False
        self._validation_profile = self._validation_pass = None
        self._remove_load_hooks()
        if self._step is not None:
            model, step = self._step
            if self.training_package is None:
                step.clear()
            if getattr(model, "forward_loss", None) is step:
                del model.forward_loss
            self._step = None
        self._auto_checked = False
        self._defer_auto_ddp = False
        source = self._source
        if source is not None:
            # Parameters are shared, but persistent buffers belong to the
            # running adapter, including any statistics updated during fit.
            active = pl_module.model
            try:
                if active is not source:
                    source.to(next(active.parameters()).device)
                    source.load_state_dict(active.state_dict(), strict=True)
            finally:
                pl_module.model = source.train(active.training)
                self._source = None

    def on_exception(self, trainer, pl_module, exception):
        # Lightning calls strategy.teardown(), not callback.teardown(), on an
        # interrupted fit. Drop deferred state-load hooks and release the adapter.
        try:
            self.teardown(trainer, pl_module, "fit")
        except Exception:
            logging.exception(
                "Could not synchronize all native buffers after the interrupted fit"
            )

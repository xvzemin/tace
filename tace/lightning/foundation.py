################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Foundation execution for the ordinary Lightning training loop."""

import logging
import os
from contextlib import contextmanager

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

    runner = aoti.load_training(
        path, device_index=device.index if device.index is not None else -1
    )
    parameters = buffers = None
    trainability = ()
    batch_fields = frozenset()

    def fields(data):
        return frozenset(k for k, v in data.items() if torch.is_tensor(v))

    def compute(batch, loss_fn):
        nonlocal parameters, buffers, trainability, batch_fields
        if loss_fn is not objective:
            raise ValueError(
                "A joint training package requires its original loss module."
            )
        if parameters is None:
            data, params, state = aoti.training_inputs(model, objective, batch)
            aoti.validate_training_model(runner, model, objective, data)
            parameters, buffers = params, state
            trainability = tuple(p.requires_grad for p in parameters.values())
            batch_fields = fields(data)
        else:
            data = aoti.training_batch(model, objective, batch)
            if fields(data) != batch_fields:
                raise ValueError("Joint training batch tensor fields changed.")
            if any(
                p.requires_grad != flag
                for p, flag in zip(parameters.values(), trainability, strict=True)
            ):
                raise ValueError("Trainable parameters changed since the joint export.")
        predictions, loss, gradients = runner(data, parameters, buffers)
        # Only connected parameters participate in autograd and DDP's unused
        # parameter detection. Lightning supplies the accumulation scale.
        names = [name for name, value in gradients.items() if value is not None]
        return predictions, ParameterLoss.apply(
            loss, tuple(gradients[n] for n in names), *(parameters[n] for n in names)
        )

    return compute


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
    replay_bucket : sequence, optional
        Maximum real atoms, directed edges and optionally structures per local
        microbatch, or a list of fixed atom/edge/structure capacities for
        whole-model CUDA graph training. The standard TACE graph loader can
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

    def __init__(self, backend=None, training_package=None, replay_bucket=None):
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
                options = strategy._ddp_kwargs
                if (
                    set(options)
                    - {
                        "find_unused_parameters",
                        "gradient_as_bucket_view",
                        "bucket_cap_mb",
                        "init_sync",
                        "broadcast_buffers",
                        "forward_sync_buffers",
                        "static_graph",
                        "process_group",
                    }
                    or options.get("init_sync") is False
                    or options.get("broadcast_buffers") is False
                    or options.get("forward_sync_buffers") is False
                    or options.get("static_graph")
                    or strategy._ddp_comm_hook is not None
                    or strategy._ddp_comm_wrapper is not None
                    or strategy._model_averaging_period is not None
                ):
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

        if type(strategy) is SimpleDDPStrategy or ddp_replay:
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
            ddp = strategy.model
            if not (
                type(ddp) is torch.nn.parallel.DistributedDataParallel
                and ddp.module is pl_module
                and ddp.process_group is torch.distributed.group.WORLD
                and not ddp.parameters_to_ignore
                and not ddp._comm_hooks
                and not hasattr(ddp, "buffer_hook")
            ):
                raise ValueError("Bucket replay requires the default DDP reducer.")
        if type(strategy) is SimpleDDPStrategy or ddp_replay:
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

    def on_train_epoch_start(self, trainer, pl_module):
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
        if self.replay_bucket is not None and self._step is not None:
            self._step[1].clear()

    def teardown(self, trainer, pl_module, stage):
        self._remove_load_hooks()
        if self._step is not None:
            model, step = self._step
            if self.replay_bucket is not None:
                step.clear()
            if getattr(model, "forward_loss", None) is step:
                del model.forward_loss
            self._step = None
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

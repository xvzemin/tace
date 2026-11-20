################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging

from lightning.pytorch.callbacks import Callback
from lightning.pytorch.trainer.states import TrainerFn

from .ema import ExponentialMovingAverage


class EpochSamplerCallback(Callback):
    """Commit consumed batches before checkpoint callbacks save loader state."""

    def setup(self, trainer, pl_module, stage):
        if stage != "fit":
            return
        if trainer.overfit_batches:
            raise ValueError("epoch_size cannot be combined with overfit_batches.")
        limit = trainer.limit_train_batches
        if not isinstance(limit, float) or limit != 1.0:
            raise ValueError(
                "Use dataset.epoch_sampler.epoch_size instead of limit_train_batches; "
                "limit_train_batches must remain 1.0."
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        trainer.train_dataloader.mark_batch_complete()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if trainer.state.fn != TrainerFn.FITTING:
            return
        state = checkpoint.get(type(trainer.datamodule).__qualname__, {})
        if state.get("epoch_dataloader") is None:
            raise ValueError(
                "The checkpoint has no epoch sampler progress. Resume with the "
                "original sampling configuration, or start a new training run."
            )


class PrintMetricsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        self._print_metrics(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._print_metrics(trainer, pl_module, "val")

    def _print_metrics(self, trainer, pl_module, prefix):
        current_epoch = pl_module.current_epoch
        total_epoch = trainer.max_epochs - 1
        metrics = getattr(pl_module, f"{prefix}_metrics").compute()
        logging.info("")
        logging.info("")
        logging.info(f"[Epoch {current_epoch}/{total_epoch}]")
        logging.info("(All errors scaled by ×1000)")
        for name, value in metrics.items():
            logging.info(f"{name}: {value.item():.6f}")
        logging.info("")
        logging.info("")


class EMACallback(Callback):
    def __init__(self, decay: float = 0.99, use_num_updates: bool = True, device=None):
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.device = device
        self.ema = None

    def on_fit_start(self, trainer, pl_module):
        self.ema = ExponentialMovingAverage(
            pl_module.model.parameters(),
            self.decay,
            self.use_num_updates,
        )
        self.ema.to(self.device or pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ema.update()

    def on_validation_start(self, trainer, pl_module):
        self.ema.store()
        self.ema.copy_to()

    def on_validation_end(self, trainer, pl_module):
        self.ema.restore()

    def on_test_start(self, trainer, pl_module):
        self.ema.store()
        self.ema.copy_to()

    def on_test_end(self, trainer, pl_module):
        self.ema.restore()

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, trainer, pl_module, checkpoint):
        if self.ema and "ema_state_dict" in checkpoint:
            self.ema.load_state_dict(checkpoint["ema_state_dict"])

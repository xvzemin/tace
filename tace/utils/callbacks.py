################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging


from lightning.pytorch.callbacks import Callback


from .ema import ExponentialMovingAverage


# === Print Metriacs Callback ===
class PrintMetricsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        self._print_metrics(trainer, pl_module, "train")

    def on_validation_epoch_end(self, trainer, pl_module):
        self._print_metrics(trainer, pl_module, "val")

    # def on_test_epoch_end(self, trainer, pl_module):
    #     self._print_metrics(trainer, pl_module, "test")

    def _print_metrics(self, trainer, pl_module, prefix):
        current_epoch = pl_module.current_epoch
        total_epoch = trainer.max_epochs - 1
        metrics = getattr(pl_module, f"{prefix}_metrics").compute()
        logging.info(f"[Epoch {current_epoch}/{total_epoch}], the error is 1000 times")
        for name, value in metrics.items():
            logging.info(f"{name}: {value.item():.6f}")


# === EMA Callback ===
class EMACallback(Callback):
    def __init__(self, decay: float = 0.99, use_num_updates: bool = True, device=None):
        self.decay = decay
        self.use_num_updates = use_num_updates
        self.device = device
        self.ema = None

    def on_fit_start(self, trainer, pl_module):
        self.ema = ExponentialMovingAverage(
            # [p for p in pl_module.parameters() if p.requires_grad],
            [p for p in pl_module.model.parameters() if p.requires_grad],
            self.decay,
            self.use_num_updates,
        )
        self.ema.to(self.device or pl_module.device)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.ema.update()

    def on_validation_start(self, trainer, pl_module):
        # logging.info("[EMA] Applying EMA weights before valid")
        self.ema.store()
        self.ema.copy_to()

    def on_validation_end(self, trainer, pl_module):
        self.ema.restore()

    def on_test_start(self, trainer, pl_module):
        # logging.info("[EMA] Applying EMA weights before test")
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

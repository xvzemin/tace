################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from collections import Counter
from typing import Optional, Dict, Any

import torch
import lightning as L
from hydra.utils import instantiate
from torchmetrics import MetricCollection


from .select_model import select_model
from ..dataset.quantity import get_target_property, get_embedding_property
from ..utils.ema import ExponentialMovingAverage
from ..utils.metrics import build_metrics, update_metrics
from .. utils._global import _DTYPE

# === LightningModule wrap ===
class LightningWrapperModel(L.LightningModule):
    def __init__(self, cfg: Dict, statistics, target_property, model=None):
        super().__init__()
        self.cfg = cfg
        self.statistics = statistics
        self.no_valid_set = cfg.get("dataset", {}).get("no_valid_set", False)
        self.save_hyperparameters(ignore=["model"])
        self.model = model
        self.loss_fn = instantiate(cfg["loss"])  # TODO for xzm
        self.loss_property = target_property
        synth_metric = cfg.get('synth_metric', None)
        if synth_metric:
            total = sum([v for k, v in synth_metric.items() if k != 'monitor_metric_name'])
            if total:
                synth_metric = {
                    k: (v / total if k != 'monitor_metric_name' else v)
                    for k, v in synth_metric.items()
                }
        self.synth_metric = synth_metric
        self._create_metrics("train")
        self._create_metrics("val")
        test_sets = cfg.get("dataset", {}).get("test_files", [])
        if test_sets is None:
            self.num_test_sets = 0
        elif isinstance(test_sets, str):
            self.num_test_sets = 1
            self._create_metrics(f"test_0")
        else:
            self.num_test_sets = len(test_sets)
            for i in range(self.num_test_sets):
                self._create_metrics(f"test_{i}")
        self.force_dtype = _DTYPE[cfg.get("dataset", {}).get("force_dtype", None)]

    def setup(self, stage: Optional[str] = None):
        self.sync_dist = self.trainer.num_devices > 1
        if self.ema is not None:
            self.ema.to(self.device)

    def _create_metrics(self, prefix):
        metric_collection = MetricCollection(build_metrics(prefix, self.loss_property))
        setattr(self, f"{prefix}_metrics", metric_collection)

    def _process_batch(self, batch):
        output = self.model(batch)
        loss = self.loss_fn(output, batch)
        return output, loss

    def _shared_step(self, batch, batch_idx, prefix):
        if self.force_dtype is not None:
            batch = batch.apply(
                lambda x: x.to(self.force_dtype) if x.is_floating_point() else x
            )            
        output, loss = self._process_batch(batch)

        # # for check unused parameters
        # for name, param in self.named_parameters():
        #     if param.requires_grad and param.grad is None:
        #         logging.warning(f"Parameter '{name}' is not used in loss computation.")

        self.log(
            f"{prefix}/loss",
            loss,
            prog_bar=True,
            logger=True,
            on_epoch=True,
            batch_size=len(batch),
            on_step=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )

        metrics = getattr(self, f"{prefix}_metrics")
        update_metrics(metrics, prefix, output, batch, self.loss_property)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        with torch.enable_grad():
            output = self._shared_step(batch, batch_idx, "val")
        return output

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        with torch.enable_grad():
            output = self._shared_step(batch, batch_idx, f"test_{dataloader_idx}")
        return output

    def on_epoch_end(self, prefix):
        metrics = getattr(self, f"{prefix}_metrics").compute()
        for name, metric in metrics.items():
            self.log(
                name,
                metric,
                sync_dist=self.sync_dist,
                reduce_fx="mean",
                on_epoch=True,
                add_dataloader_idx=True,
            )
        logging.info("═" * 50 + "\n")

        if prefix == 'val':
            if self.synth_metric is not None:
                synth_metric = sum([metrics[k] * v * 0.001 for k, v in self.synth_metric.items() if k != 'monitor_metric_name'])
                self.log(
                    f"{prefix}/synth_metric",
                    synth_metric,
                    sync_dist=self.sync_dist,
                    reduce_fx="mean",
                    on_epoch=True,
                    add_dataloader_idx=True,
                )
        getattr(self, f"{prefix}_metrics").reset()
    def on_train_epoch_end(self):
        self.on_epoch_end("train")

    def on_train_epoch_start(self):
        lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        logging.info(f"LR: {lr:.6e}")

    def on_validation_epoch_end(self):
        self.on_epoch_end("val")

    def on_test_epoch_end(self):
        logging.info(f"The error is 1000 times")
        for i in range(self.num_test_sets):
            metrics = getattr(self, f"test_{i}_metrics").compute()
            for name, metric in metrics.items():
                self.log(
                    name,
                    metric,
                    sync_dist=self.sync_dist,
                    reduce_fx="mean",
                    on_epoch=True,
                    add_dataloader_idx=True,
                )
                logging.info(f"{name}: {metric.item():.6f}")
            getattr(self, f"test_{i}_metrics").reset()
            logging.info("═" * 50 + "\n")

    def on_fit_start(self):
        if hasattr(self, "ema") and self.ema is not None:
            self.ema.to(self.device)

    def setup(self, stage=None):
        if self.trainer.world_size > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False

    def configure_optimizers(self):
        optimizer = instantiate(
            {**{k: v for k, v in self.cfg["optimizer"].items() if k != "extra"}},
            params=self.parameters(),
        )
        scheduler = instantiate(
            {**{k: v for k, v in self.cfg["scheduler"].items() if k != "extra"}},
            optimizer=optimizer,
        )
        lr_scheduler_cfg = self.cfg.get("scheduler", {}).get("extra", {})
        monitor = lr_scheduler_cfg.get("monitor", "val/loss")
        interval = lr_scheduler_cfg.get("interval", "epoch")
        frequency = lr_scheduler_cfg.get("frequency", 1)
        if self.no_valid_set:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": interval,
                    "frequency": frequency,
                },
            }
        else:
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": monitor,
                    "interval": interval,
                    "frequency": frequency,
                },
            }

    @classmethod
    def load_from_checkpoint(
        cls,
        ckpt_path: str,
        map_location: str = "cpu",
        strict: Optional[bool] = True,
        use_ema: int = 1,
        **kwargs: Any,
    ) -> Any:

        checkpoint = torch.load(
            ckpt_path, map_location=map_location, weights_only=False
        )
        dtypes = [
            v.dtype for v in checkpoint["state_dict"].values()
            if hasattr(v, "dtype") and torch.is_floating_point(v)
        ]
        counts = Counter(dtypes)
        dominant_dtype = counts.most_common(1)[0][0]
        torch.set_default_dtype(dominant_dtype)
        cfg = checkpoint['hyper_parameters']['cfg']
        target_property = get_target_property(cfg)
        embedding_property = get_embedding_property(cfg)
        statistics = checkpoint['hyper_parameters']['statistics']

        if "cfg" in kwargs:
            cfg = kwargs["cfg"]

        model = select_model(cfg, statistics, target_property, embedding_property)
        raw_sd = checkpoint["state_dict"]

        filtered_state_dict = {
            k[len("model.") :]: v for k, v in raw_sd.items() if k.startswith("model.")
        }
        model.load_state_dict(filtered_state_dict, strict=strict)

        if use_ema == 1 and "ema_state_dict" in checkpoint:
            ema = ExponentialMovingAverage(
                [p for p in model.parameters() if p.requires_grad],
                decay=checkpoint["ema_state_dict"]["decay"],
                use_num_updates=("num_updates" in checkpoint["ema_state_dict"]),
            )
            ema.load_state_dict(checkpoint["ema_state_dict"])
            ema.copy_to([p for p in model.parameters() if p.requires_grad])

        return model


def finetune(cfg: Dict) -> torch.nn.Module:

    ckpt_path: str = cfg.get("finetune_from_model", None)
    assert ckpt_path is not None

    if ckpt_path.endswith(".ckpt"):
        MODEL = LightningWrapperModel.load_from_checkpoint(
            ckpt_path,
            map_location="cpu",
            strict=True,
            use_ema=1,
        )
    elif ckpt_path.endswith(".pt") or ckpt_path.endswith(".pth"):
        MODEL = torch.load(ckpt_path, weights_only=False, map_location="cpu")
    else:
        raise ValueError("❌ Model path must end with '.ckpt', '.pt', or '.pth'")

    logging.info(f"Load model for Fine-tunning")
    return MODEL

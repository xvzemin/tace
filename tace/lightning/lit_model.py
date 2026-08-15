################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import copy
import importlib
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Optional, Union

import lightning as L
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torchmetrics import MetricCollection

from tace.dataset.quantity import get_embedding_property, get_target_property
from tace.models.adapter import TensorModel
from tace.utils._global import DEVICE, DTYPE
from tace.utils.env import get_tace_apply_u_shift, get_tace_use_dens
from tace.utils.loss.uncertainty import UncertaintyLoss
from tace.utils.metrics import build_metrics, update_metrics
from tace.utils.utils import torch_default_dtype

from ..utils.loss.dens import add_gaussian_noise_to_position
from .lora import to_lora_model
from .los_skip import LossSkipController
from .torch_model import create_model
from .u_shift import apply_u_shift


def get_class_from_cfg(cfg):
    if cfg is None:
        raise ValueError("Config is None")

    if not isinstance(cfg, Dict):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    if "_target_" not in cfg:
        raise ValueError("Config must contain '_target_'")

    target = cfg["_target_"]

    module_path, class_name = target.rsplit(".", 1)
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)

    return cls, cfg


class LightningWrapperModel(L.LightningModule):
    def __init__(
        self,
        cfg: Dict,
        model: torch.nn.Module,
        target_property: list[str],
        embedding_property: list[str],
        statistics,
    ):
        super().__init__()
        cfg = copy.deepcopy(cfg)
        self.save_hyperparameters(ignore=["model"])
        self.cfg = cfg
        self.statistics = statistics
        self.model = to_lora_model(cfg.get("finetune", {}), model)
        # self.normalizers = copy.deepcopy(self.model.readout_fn.normalizers)

        # === Loss ===
        loss_cls, loss_cfg = get_class_from_cfg(cfg["loss"])
        self.loss_fn = loss_cls(
            **{k: v for k, v in loss_cfg.items() if k != "_target_"}
        )
        self.loss_property = target_property

        # === Metric ===
        self._create_metrics("train")
        self._create_metrics("val")
        synth_metric = cfg.get("synth_metric", None)
        if synth_metric:
            total = sum(
                [v for k, v in synth_metric.items() if k != "monitor_metric_name"]
            )
            if total:
                synth_metric = {
                    k: (v / total if k != "monitor_metric_name" else v)
                    for k, v in synth_metric.items()
                }
        self.synth_metric = synth_metric
        test_sets = cfg.get("dataset", {}).get("test_files", [])
        if test_sets is None:
            self.num_test_sets = 0
        elif isinstance(test_sets, str):
            self.num_test_sets = 1
            self._create_metrics("test_0")
        else:
            self.num_test_sets = len(test_sets)
            for i in range(self.num_test_sets):
                self._create_metrics(f"test_{i}")

        # === Misc ===
        self.force_dtype = DTYPE[cfg.get("dataset", {}).get("force_dtype", None)]
        self.no_valid_set = cfg.get("dataset", {}).get("no_valid_set", False)

        self.skip_controller = None
        if cfg["misc"].get("LossSkipController", {}).get("enable", False):
            self.skip_controller = LossSkipController(
                manual_threshold=cfg["misc"]["LossSkipController"]["manual_threshold"],
                start_step=cfg["misc"]["LossSkipController"]["start_step"],
                ema_window=cfg["misc"]["LossSkipController"]["ema_window"],
                multiplier=cfg["misc"]["LossSkipController"]["multiplier"],
                skip_nan=cfg["misc"]["LossSkipController"].get("skip_nan", True),
                skip_large=cfg["misc"]["LossSkipController"].get("skip_large", True),
            )

        # from tace.utils.optimizer.hybrid_muon import get_adam_route, get_effective_shape, get_matrix_view_shape
        # for name, param in model.named_parameters():
        #     route = get_adam_route(name)
        #     eff_shape = get_effective_shape(param.shape)
        #     if route == "muon":
        #         mat_view = get_matrix_view_shape(eff_shape, muon_mode="slice")
        #         logging.info(f"{name:40s} shape={str(param.shape):15s} -> Muon {mat_view}")
        #     else:
        #         logging.info(f"{name:40s} shape={str(param.shape):15s} -> {route}")
        # import sys
        # sys.exit()

    def _create_metrics(self, prefix):
        metric_collection = MetricCollection(build_metrics(prefix, self.loss_property))
        setattr(self, f"{prefix}_metrics", metric_collection)

    def _process_batch(self, batch):
        output = self.model(batch)  # normalized predict value

        # # # normalize label value
        # with torch.no_grad():
        #     normalizerd_batch = copy.copy(batch)
        #     for k, v in self.normalizers.items():
        #         if k == 'energy':
        #             normalizerd_batch[k] = v.norm(normalizerd_batch[k] - output['e_base_graph'].detach())
        #         else:
        #             normalizerd_batch[k] = v.norm(normalizerd_batch[k])
        # loss = self.loss_fn(output, normalizerd_batch)
        loss = self.loss_fn(output, batch)
        return output, loss

    # def on_before_optimizer_step(self, optimizer):

    def _shared_step(self, batch, batch_idx, prefix):

        with torch.no_grad():
            if get_tace_apply_u_shift() == "1":
                batch["energy"] = apply_u_shift(batch, batch["energy"])

            if get_tace_use_dens() == "1":
                batch = add_gaussian_noise_to_position(batch)

        if "forces" in batch:
            batch["direct_forces"] = batch["forces"]
            batch["direct_forces_weight"] = batch["forces_weight"]
        if "stress" in batch:
            batch["direct_stress"] = batch["stress"]
            batch["direct_stress_weight"] = batch["stress_weight"]

        if self.force_dtype is not None:
            batch = batch.apply(
                lambda x: x.to(self.force_dtype) if x.is_floating_point() else x
            )

        # always use normalized value to caculate loss
        output, loss = self._process_batch(batch)

        if prefix == "train" and self.skip_controller is not None:
            skip, reason_code, threshold = self.skip_controller.check(
                loss, self.global_step
            )
            # if self.global_step % 5000 == 0:
            #     self.skip_controller.log_status(self.global_step)
            if skip:
                if (not dist.is_initialized()) or dist.get_rank() == 0:
                    if reason_code == 1:
                        reason_str = "non-finite"
                    elif reason_code == 2:
                        reason_str = "large"
                    else:
                        reason_str = "unknown"
                    logging.warning(
                        f"[Skip:{reason_str}] "
                        f"epoch={self.current_epoch}, "
                        f"step={self.global_step}, "
                        f"batch={batch_idx}, "
                        f"loss={loss.detach().item():.4f}, "
                        f"threshold={threshold}"
                    )

                zero = torch.zeros((), device=self.device, dtype=loss.dtype)

                for p in self.parameters():
                    if p.grad is not None:
                        p.grad.detach_()
                        p.grad.zero_()

                for p in self.parameters():
                    if p.requires_grad:
                        zero = zero + p.sum() * 0.0
                loss = zero
                output = {
                    k: torch.zeros_like(v) if torch.is_tensor(v) else v
                    for k, v in output.items()
                }

        self.log(
            f"{prefix}/loss",
            loss.detach(),
            prog_bar=(prefix == "train"),
            batch_size=len(batch),
            on_step=(prefix == "train"),
            on_epoch=True,
            logger=True,
            sync_dist=self.sync_dist,
            reduce_fx="mean",
        )

        if prefix == "train":
            lr = self.optimizers().param_groups[0]["lr"]
            self.log(
                "train/lr",
                lr,
                on_step=True,
                prog_bar=True,
            )

        metrics = getattr(self, f"{prefix}_metrics")
        # with torch.no_grad():
        #     unnormalizerd_output = copy.copy(output)
        #     if self.model.training:
        #         for k, v in self.normalizers.items():
        #             unnormalizerd_output[k] = v.denorm(unnormalizerd_output[k])
        #         if unnormalizerd_output['e_base_graph'] is not None:
        #             unnormalizerd_output['energy'] += unnormalizerd_output['e_base_graph']
        #         update_metrics(metrics, prefix, unnormalizerd_output, batch, self.loss_property)
        #     else:
        #         update_metrics(metrics, prefix, output, batch, self.loss_property)
        update_metrics(metrics, prefix, output, batch, self.loss_property)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, batch_idx, "train")
        return loss

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

        if prefix == "val":
            if self.synth_metric is not None:
                synth_metric = sum(
                    [
                        metrics[k] * v * 0.001
                        for k, v in self.synth_metric.items()
                        if k != "monitor_metric_name"
                    ]
                )
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
        if isinstance(self.loss_fn, UncertaintyLoss):
            with torch.no_grad():
                logging.info("")
                for k, v in self.loss_fn.log_sigmas.items():
                    logging.info(
                        f"{k}: weight(log_sigma)|{0.5 * torch.exp(-v).item():.3f}({v.item():.3f})"
                    )

    def on_train_epoch_start(self):
        dataloader = self.trainer.train_dataloader
        sampler = dataloader.sampler
        if hasattr(sampler, "step_epoch"):
            sampler.step_epoch(self.current_epoch)

    def on_validation_epoch_end(self):
        self.on_epoch_end("val")

    def on_test_epoch_end(self):
        logging.info("The error is 1000 times")
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
        if isinstance(self.loss_fn, UncertaintyLoss):
            with torch.no_grad():
                logging.info("")
                for k, v in self.loss_fn.log_sigmas.items():
                    logging.info(
                        f"{k}: weight(log_sigma)|{0.5 * torch.exp(-v).item():.3f}({v.item():.3f})"
                    )

    # # Used for check unused params
    # def on_after_backward(self):

    #     for name, p in self.named_parameters():
    #         if not p.requires_grad:
    #             continue
    #         if p.grad is None:
    #             logging.warning(f"[NO_GRAD] {name}")
    #         else:
    #             grad_norm = p.grad.norm().item()
    #             if grad_norm == 0:
    #                 logging.warning(f"[ZERO_GRAD] {name}")

    def setup(self, stage: Union[str, None] = None):
        if self.trainer.world_size > 1:
            self.sync_dist = True
        else:
            self.sync_dist = False
        # if self.ema is not None:
        #     self.ema.to(self.device)

    # def on_load_checkpoint(self, checkpoint):
    #     for opt_state in checkpoint["optimizer_states"]:
    #         for group in opt_state["param_groups"]:
    #             print(group["weight_decay"])
    #             if group["weight_decay"] != 0.0:
    #                 group["weight_decay"] = 1e-8

    def configure_optimizers(self):

        optimizer_cfg = copy.deepcopy(self.cfg["optimizer"])

        optimizer_target = optimizer_cfg.pop("_target_")
        weight_decay = float(optimizer_cfg.pop("weight_decay", 0.0))

        optimizer_cfg.pop("extra", None)
        optimizer_cfg.pop("params", None)

        decay_params = []
        no_decay_params = []
        no_decay_names = []

        all_named_params = []

        for module_name, module in self.named_modules():
            for param_name, param in module.named_parameters(recurse=False):
                if not param.requires_grad:
                    continue

                full_name = f"{module_name}.{param_name}" if module_name else param_name
                all_named_params.append((full_name, param))

                if (
                    param_name.endswith("bias")
                    or "_readout" in full_name
                    or "_readouts" in full_name
                    or ".node_embedding." in full_name
                    or ".source_embedding." in full_name
                    or ".target_embedding." in full_name
                    or ".uie." in full_name
                    or ".uee." in full_name
                    or ".embedding." in full_name
                    or ".router." in full_name
                    or ".radial_basis" in full_name
                    or isinstance(module, torch.nn.LayerNorm)
                    or isinstance(module, torch.nn.RMSNorm)
                    or isinstance(module, torch.nn.Embedding)
                ):
                    no_decay_params.append(param)
                    no_decay_names.append(full_name)
                else:
                    decay_params.append(param)

        if weight_decay > 0 and len(no_decay_names) > 0:
            logging.debug("Parameters excluded from weight decay:")
            for name in no_decay_names:
                logging.debug(f"  {name}")
        else:
            logging.debug("All parameters use same weight decay")

        decay_ids = set(map(id, decay_params))
        no_decay_ids = set(map(id, no_decay_params))
        overlap = decay_ids.intersection(no_decay_ids)
        if len(overlap) > 0:
            raise RuntimeError(
                "Some parameters appear in both decay and no_decay groups"
            )

        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        module_path, class_name = optimizer_target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        OptimizerClass = getattr(module, class_name)

        if optimizer_target == "tace.utils.optimizer.HybridMuonOptimizer":
            optimizer = OptimizerClass(
                params=param_groups,
                named_parameters=all_named_params,
                **optimizer_cfg,
            )
        else:
            optimizer = OptimizerClass(
                params=param_groups,
                **optimizer_cfg,
            )

        if "scheduler" not in self.cfg:
            return {"optimizer": optimizer}

        scheduler_cfg = copy.deepcopy(self.cfg["scheduler"])

        scheduler_target = scheduler_cfg.pop("_target_")

        scheduler_extra = scheduler_cfg.pop("extra", {})
        scheduler_cfg.pop("optimizer", None)

        monitor = scheduler_extra.get("monitor", "val/loss")
        interval = scheduler_extra.get("interval", "epoch")
        frequency = scheduler_extra.get("frequency", 1)

        module_path, class_name = scheduler_target.rsplit(".", 1)
        module = importlib.import_module(module_path)
        SchedulerClass = getattr(module, class_name)

        scheduler = SchedulerClass(
            optimizer=optimizer,
            **scheduler_cfg,
        )

        if getattr(self, "no_valid_set", False):
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
        use_ema: Union[int, bool] = 1,
        dtype: Optional[torch.dtype] = None,
    ) -> Any:

        checkpoint = torch.load(
            ckpt_path, map_location=map_location, weights_only=False
        )
        model_dtype = dtype or _dominant_floating_dtype(
            value
            for name, value in checkpoint["state_dict"].items()
            if name.startswith("model.")
        )
        cfg = checkpoint["hyper_parameters"]["cfg"]
        target_property = get_target_property(cfg)
        embedding_property = get_embedding_property(cfg)
        statistics = checkpoint["hyper_parameters"]["statistics"]
        with torch_default_dtype(model_dtype):
            model = create_model(
                cfg,
                statistics,
                target_property,
                embedding_property,
            )
            model = to_lora_model(cfg.get("finetune", {}), model)
        state_dict = {
            k[len("model.") :]: v
            for k, v in checkpoint["state_dict"].items()
            if k.startswith("model.")
        }
        model.load_state_dict(state_dict, strict=strict)

        # # === SWA ===
        # if 'swa' in cfg['callbacks']:
        #     logging.debug("Since swa is enabled, skip trying load ema.")
        #     return model.to(map_location)

        # === EMA ===
        if bool(use_ema) and "ema_state_dict" in checkpoint:
            ema_params = checkpoint["ema_state_dict"]["shadow_params"]
            for idx, (name, _) in enumerate(model.named_parameters()):
                state_dict[name] = ema_params[idx]
            model.load_state_dict(state_dict, strict=strict)

        return model.to(map_location)


def _dominant_floating_dtype(values) -> torch.dtype:
    dtypes = [
        value.dtype
        for value in values
        if isinstance(value, torch.Tensor) and torch.is_floating_point(value)
    ]
    if not dtypes:
        raise ValueError("Cannot infer model dtype: no floating-point tensors found.")
    return Counter(dtypes).most_common(1)[0][0]


def _resolve_model_dtype(
    dtype: Union[str, int, torch.dtype, None],
) -> Optional[torch.dtype]:
    try:
        return DTYPE[dtype]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"Unsupported model dtype: {dtype!r}") from exc


def _load_tace(
    model: Union[str, Path, torch.nn.Module],
    device: Union[str, torch.device, None] = None,
    strict: bool = True,
    use_ema: bool = True,
    dtype: Union[str, int, torch.dtype, None] = None,
    **kwargs: Any,
) -> TensorModel:
    device = DEVICE[device]
    requested_dtype = _resolve_model_dtype(dtype)
    is_aoti = False
    if isinstance(model, (str, Path)):
        model_path = str(model)
        if model_path.endswith(".ckpt"):
            model = LightningWrapperModel.load_from_checkpoint(
                model_path,
                map_location=device,
                strict=strict,
                use_ema=use_ema,
                dtype=requested_dtype,
            )
        elif model_path.endswith(".pt2"):
            from tace.models.compile import load_aotinductor

            model = load_aotinductor(model_path, device)
            is_aoti = True
        elif model_path.endswith(".pt") or model_path.endswith(".pth"):
            obj = torch.load(
                model_path,
                map_location=device,
                weights_only=False,
                **kwargs,
            )
            if isinstance(obj, dict) and "state_dict" in obj:
                state_dict = obj["state_dict"]
                model_dtype = requested_dtype or _dominant_floating_dtype(
                    state_dict.values()
                )
                with torch_default_dtype(model_dtype):
                    model = create_model(
                        **{k: v for k, v in obj.items() if k != "state_dict"},
                        prune_removed_keys=True,
                    )
                model.load_state_dict(state_dict, strict=strict)
            elif isinstance(obj, torch.nn.Module):
                model = obj
            else:
                raise ValueError(
                    "Unsupported .pt/.pth TACE model format. Expected a TACE "
                    "state_dict package or a serialized torch.nn.Module."
                )
        else:
            raise ValueError("Model path must end with .ckpt, .pt2, .pt or .pth")
    elif isinstance(model, torch.nn.Module):
        is_aoti = hasattr(model, "compiled_model") and hasattr(model, "compile_device")
    else:
        raise TypeError("Model must be a path or torch.nn.Module")

    model_dtype = model.get_model_dtype()
    target_dtype = requested_dtype or model_dtype
    if is_aoti and target_dtype != model_dtype:
        raise ValueError(
            f"AOTI package dtype is fixed to {model_dtype}; "
            f"requested {target_dtype}. Re-export the package with the target dtype."
        )

    torch.set_default_dtype(target_dtype)
    if is_aoti:
        return model.to(device=device)
    return model.to(device=device, dtype=target_dtype)


def load_tace(
    model: Union[str, Path, torch.nn.Module],
    device: Union[str, torch.device, None] = None,
    strict: bool = True,
    use_ema: bool = True,
    target_property: Optional[list[str]] = None,
    dtype: Union[str, int, torch.dtype, None] = None,
    **kwargs: Any,
) -> TensorModel:

    model = _load_tace(
        model=model,
        device=device,
        dtype=dtype,
        strict=strict,
        use_ema=use_ema,
        **kwargs,
    )

    if target_property:
        model.reset_target_property(target_property)

    return model


def export_tace(
    model: torch.nn.Module,
    name: str,
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "cfg": model.readout_fn.model_config,
            "target_property": model.get_target_property(),
            "embedding_property": model.get_embedding_property(),
            "statistics": model.readout_fn.statistics,
        },
        name if name.endswith(".pt") else name + ".pt",
    )


def finetune(cfg: Dict) -> torch.nn.Module:

    precision = cfg["trainer"]["precision"]
    dtype = DTYPE.get(precision, torch.float64)
    model = load_tace(
        cfg["finetune_from_model"],
        device="cpu",
        strict=True,
        use_ema=True,
        dtype=dtype,
    )

    logging.info("Load model for Fine-tunning")
    model.train()

    return model

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Dict

import lightning as L
from hydra.utils import instantiate
from lightning.pytorch.callbacks import (
    ModelCheckpoint,
    StochasticWeightAveraging,
    TQDMProgressBar,
)
from lightning.pytorch.plugins.precision import MixedPrecision, Precision
from torch_geometric.loader import DataLoader

# from lightning.pytorch.callbacks.progress.rich_progress import RichProgressBarTheme
from ..utils.callbacks import PrintMetricsCallback
from ..utils.env import acceleration_enabled
from ..utils.logger import log_parameters
from ..utils.precision import resolve_precision
from .foundation import FoundationTrainingCallback
from .lit_model import LightningWrapperModel


def check_training_entry(cfg, *, fast):
    """Validate the selected training route before dataset preparation."""
    if not fast and cfg.get("fast_schedule") is not None:
        raise ValueError("fast_schedule is only supported by tace-fast-train")
    if cfg.get("trainer", {}).get("callbacks"):
        raise ValueError(
            "Configure callbacks under top-level callbacks, not trainer.callbacks."
        )
    for callback in cfg.get("callbacks", {}).values():
        target = callback.get("_target_") if isinstance(callback, Mapping) else None
        if isinstance(callback, Mapping) and target is None:
            for key in ("training_package", "replay_bucket"):
                if key in callback:
                    raise ValueError(
                        f"{key} requires _target_: tace.utils.callbacks.FoundationTrainingCallback"
                    )
        if not fast and (
            isinstance(callback, FoundationTrainingCallback)
            or target
            in (
                "tace.utils.callbacks.FoundationTrainingCallback",
                "tace.lightning.foundation.FoundationTrainingCallback",
                FoundationTrainingCallback,
            )
        ):
            raise ValueError("Use tace-fast-train for a FoundationTrainingCallback.")


def _maybe_start_wsd_decay_on_resume(cfg: Dict, resume_ckpt: str) -> None:
    scheduler_cfg = cfg.get("scheduler", None)
    if not scheduler_cfg:
        return

    scheduler_target = scheduler_cfg.get("_target_", "")
    if not str(scheduler_target).endswith("WarmupStableDecay"):
        return

    scheduler_extra = scheduler_cfg.get("extra", {})
    if not scheduler_extra.get("start_decay_on_resume", False):
        return
    import torch

    checkpoint = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
    global_step = int(checkpoint.get("global_step", 0))
    num_warmup_steps = int(scheduler_cfg.get("num_warmup_steps", 0))
    old_num_stable_steps = int(scheduler_cfg.get("num_stable_steps", 0))
    new_num_stable_steps = max(0, global_step - num_warmup_steps)
    scheduler_cfg["num_stable_steps"] = new_num_stable_steps

    logging.info(
        "Start WSD decay on resume: global_step=%s, num_warmup_steps=%s, "
        "num_stable_steps %s -> %s",
        global_step,
        num_warmup_steps,
        old_num_stable_steps,
        new_num_stable_steps,
    )


def build_trainer(
    cfg: Dict,
    dataloader_valid: DataLoader = None,
    *,
    fast: bool = False,
    module_type: type[LightningWrapperModel] = LightningWrapperModel,
) -> L.Trainer:
    """Build and configure a PyTorch Lightning Trainer.

    Args:
        cfg: Hydra configuration object containing:
            - logger: Configuration for experiment logger
            - callbacks: Dictionary of callback configurations
            - trainer: Base Trainer configuration

    Returns:
        Fully configured PyTorch Lightning Trainer instance

    Raises:
        ValueError: For missing critical configurations
        RuntimeError: For initialization failures in components
    """

    # === Configuration Preprocessing ===
    if not isinstance(cfg, Dict):
        raise TypeError(f"Expected DictConfig type, got {type(cfg)}")
    check_training_entry(cfg, fast=fast)

    # A specialized fast module owns its fit lifecycle (e.g. OAM's capture);
    # it must never also install the generic execution callback.
    generic_fast = fast and module_type is LightningWrapperModel
    if fast and not generic_fast:
        from .oamv2 import OAMLightningModule

        if module_type is not OAMLightningModule:
            raise ValueError("Unsupported specialized tace-fast-train module.")

    # === Logger Configuration ===
    logger_cfg = cfg.get("logger", None)
    logger_instance = None
    if logger_cfg is not None:
        try:
            logger_instance = instantiate(logger_cfg)
        except Exception as e:
            raise RuntimeError(f"Logger initialization failed: {str(e)}") from e

    # === User Callbacks Configuration ===
    initialized_callbacks = []
    for cb_name, cb_config in cfg.get("callbacks", {}).items():
        if isinstance(cb_config, FoundationTrainingCallback):
            initialized_callbacks.append(cb_config)
            continue
        target = cb_config.get("_target_") if isinstance(cb_config, Mapping) else None
        if target is None:
            logging.warning(f"Skipping unconfigured callback: {cb_name}")
            continue
        try:
            callback = instantiate({k: v for k, v in cb_config.items() if k != "extra"})
            initialized_callbacks.append(callback)
            logging.debug(
                f"Successfully loaded callback: {cb_name} ({type(callback).__name__})"
            )
        except Exception as e:
            error_msg = (
                f"Callback '{cb_name}' initialization failed\n"
                f"Config: {cb_config}\n"
                f"Error: {str(e)}"
            )
            # This explicit execution-provider request must not fall back to a
            # different training path if its Hydra construction fails.
            if fast:
                raise RuntimeError(error_msg) from e
            logging.error(error_msg, exc_info=True)

    providers = [
        c for c in initialized_callbacks if isinstance(c, FoundationTrainingCallback)
    ]
    if providers and not fast:
        raise ValueError("Use tace-fast-train for a FoundationTrainingCallback.")
    if providers and not generic_fast:
        raise ValueError(
            "The specialized fast schedule cannot also use FoundationTrainingCallback."
        )
    if len(providers) > 1:
        raise ValueError("Configure at most one FoundationTrainingCallback.")
    if generic_fast and not providers:
        initialized_callbacks.insert(0, FoundationTrainingCallback())

    # === Built-in callbacks ===
    initialized_callbacks += [
        PrintMetricsCallback(),
        TQDMProgressBar(),  # not allow use hydra to build progress_bar to avoid bug
    ]

    # === Disallow SWA callback ===
    swa_cbs = [
        cb for cb in initialized_callbacks if isinstance(cb, StochasticWeightAveraging)
    ]
    if swa_cbs:
        raise RuntimeError(
            "StochasticWeightAveraging (SWA) is currently not allowed in TACE when used "
            "directly via PyTorch Lightning callbacks.\n\n"
            "Reason: During testing, we observed some strange issues "
            "when using Lightning's built-in SWA.\n\n"
            "Recommended alternative:\n"
            "- Save a series of model checkpoints during training.\n"
            "- After training, use the provided `tace-average` command to compute an averaged "
            "model from any set of models with the same architecture.\n\n"
            "This approach is especially useful in the late stage of training, "
            "particularly when increasing the energy loss weight, and can significantly "
            "improve the model's generalization performance."
        )

    # === Put checkpoint at the end ===
    other_cbs = [
        cb for cb in initialized_callbacks if not isinstance(cb, ModelCheckpoint)
    ]
    checkpoint_cbs = [
        cb for cb in initialized_callbacks if isinstance(cb, ModelCheckpoint)
    ]
    initialized_callbacks = other_cbs + checkpoint_cbs
    if checkpoint_cbs:
        for cb in checkpoint_cbs:
            logging.info(f"Model checkpoints will be saved to: {cb.dirpath}")
            if not cb.monitor:
                logging.warning(
                    f"Checkpoint in {cb.dirpath} has no monitor metric specified"
                )
    else:
        raise RuntimeError(
            "No ModelCheckpoint callback configured. "
            "You must provide at least one ModelCheckpoint to save the model checkpoints."
        )

    try:
        trainer_cfg = cfg["trainer"]
        filter_trainer_cfg = {}
        for k, v in trainer_cfg.items():
            filter_keys = ["logger", "callbacks", "precision", "plugins"]
            if k not in filter_keys:
                filter_trainer_cfg.update({k: v})
        if cfg.get("dataset", {}).get("no_valid_set", False):
            filter_trainer_cfg["num_sanity_val_steps"] = 0
        policy = resolve_precision(cfg, training=True)
        policy.apply()
        if policy.autocast is not None:
            if any(
                acceleration_enabled(name) for name in ("eqx", "oeq", "cue", "luna")
            ):
                raise ValueError(
                    "Autocast is not supported by the selected fused backend; disable it or set dtype.autocast=null."
                )
            import torch

            plugin = MixedPrecision(
                "bf16-mixed" if policy.autocast == "bfloat16" else "16-mixed",
                device="cuda"
                if torch.cuda.is_available() and trainer_cfg.get("accelerator") != "cpu"
                else "cpu",
            )
        else:
            plugin = Precision()
            plugin.precision = "64-true" if policy.model == "float64" else "32-true"
        plugins = [
            instantiate(p) if isinstance(p, dict) else p
            for p in trainer_cfg.get("plugins", []) or []
        ]
        trainer = instantiate(
            filter_trainer_cfg,
            logger=logger_instance,
            callbacks=initialized_callbacks,
            plugins=[*plugins, plugin],
            # _convert_="partial" # none, partial, all
        )
    except Exception as e:
        error_detail = (
            "Trainer initialization failed\n"
            f"Configuration: {cfg['trainer']}\n"
            f"Error: {str(e)}"
        )
        raise RuntimeError(error_detail) from e

    logging.info(f"Callbacks: {list(cfg.get('callbacks', {}))}")

    return trainer


# === Training entrypoint ===
def train(
    cfg: Dict,
    statistics,
    model,
    datamodule,
    target_property,
    embedding_property: list[str] = [],
    *,
    module_type: type[LightningWrapperModel] = LightningWrapperModel,
    fast: bool = False,
):
    resolve_precision(cfg, training=True).apply()
    lit_model = module_type(
        cfg,
        model,
        target_property,
        embedding_property,
        statistics,
    )

    # In manual optimization the module applies the configured clipping itself.
    trainer_cfg = (
        cfg
        if lit_model.automatic_optimization
        else dict(cfg, trainer=dict(cfg["trainer"], gradient_clip_val=None))
    )
    trainer = build_trainer(trainer_cfg, fast=fast, module_type=module_type)

    # TRAIN AND VALID
    resume_ckpt = cfg.get("resume_from_model", None)
    if resume_ckpt:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not exists: {ckpt_path}")
        logging.info(f"Resume Training from checkpoint: {resume_ckpt}")
    else:
        logging.debug(lit_model)
        log_parameters(model)
    trainer.fit(lit_model, datamodule=datamodule, ckpt_path=resume_ckpt)

    # TEST
    if cfg["dataset"].get("test_files", None):
        trainer.test(
            lit_model,
            datamodule=datamodule,
            ckpt_path="best",
            verbose=False,
        )
    logging.info("Training completed at %s", datetime.now().strftime("%Y-%m-%d %H:%M"))

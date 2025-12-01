################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict
from pathlib import Path
from datetime import datetime

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from hydra.utils import instantiate
from torch_geometric.loader import DataLoader


from .lit_model import LightningWrapperModel
from ..utils.callbacks import PrintMetricsCallback

def build_trainer(cfg: Dict, dataloader_valid: DataLoader = None) -> L.Trainer:
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

    # === Logger Configuration ===
    logger_cfg = cfg.get("logger", None)
    logger_instance = None
    if logger_cfg is not None:
        try:
            logger_instance = instantiate(logger_cfg)
        except Exception as e:
            raise RuntimeError(f"Logger initialization failed: {str(e)}") from e

    # === Callbacks Configuration ===
    REQUIRED_CALLBACKS = {
        "checkpoint": ModelCheckpoint,
        # "early_stopping": pl.callbacks.EarlyStopping,
    }

    initialized_callbacks = []
    remaining_required = dict(REQUIRED_CALLBACKS)

    for cb_name, cb_config in cfg.get("callbacks", {}).items():
        try:
            if cb_config.get("_target_") is None:
                logging.warning(f"Skipping unconfigured callback: {cb_name}")
                continue

            callback = instantiate({k: v for k, v in cb_config.items() if k != "extra"})
            initialized_callbacks.append(callback)
            logging.debug(
                f"Successfully loaded callback: {cb_name} ({type(callback).__name__})"
            )

            # Track required callbacks, if succeed, delete field
            for req_name, req_type in list(remaining_required.items()):
                if isinstance(callback, req_type):
                    del remaining_required[req_name]

        except Exception as e:
            error_msg = (
                f"Callback '{cb_name}' initialization failed\n"
                f"Config: {cb_config}\n"
                f"Error: {str(e)}"
            )
            if cb_name in REQUIRED_CALLBACKS:
                raise RuntimeError(error_msg) from e
            logging.error(error_msg, exc_info=True)

    # Validate required callbacks
    if remaining_required:
        missing = ", ".join(remaining_required.keys())
        raise ValueError(f"Missing required callbacks: {missing}")

    # === Built-in callbacks ===
    initialized_callbacks += [PrintMetricsCallback()]

    try:
        trainer_cfg = cfg["trainer"]
        filter_trainer_cfg = {}
        for k, v in trainer_cfg.items():
            filter_keys = ["logger", "callbacks"]
            if k not in filter_keys:
                filter_trainer_cfg.update({k: v})
        num_batch = 2 if not cfg.get("dataset", {}).get("no_valid_set", False) else 0
        filter_trainer_cfg.update({"num_sanity_val_steps": num_batch})
        trainer = instantiate(
            filter_trainer_cfg,
            logger=logger_instance,
            callbacks=initialized_callbacks,
            # _convert_="partial" # none, partial, all
        )
    except Exception as e:
        error_detail = (
            "Trainer initialization failed\n"
            f"Configuration: {(cfg["trainer"])}\n"
            f"Error: {str(e)}"
        )
        raise RuntimeError(error_detail) from e

    # === Post-Initialization Checks ===
    # Verify checkpoint configuration
    checkpoint_cb = next(
        (
            cb
            for cb in initialized_callbacks
            if isinstance(cb, ModelCheckpoint)
        ),
        None,
    )

    if checkpoint_cb:
        logging.info(f"Model checkpoints will be saved to: {checkpoint_cb.dirpath}")
        if not checkpoint_cb.monitor:
            logging.warning("ModelCheckpoint has no monitor metric specified")
    else:
        logging.warning(
            "No ModelCheckpoint callback configured - models won't be saved"
        )

    return trainer


# === Training entrypoint ===
def train(
    cfg: Dict,
    statistics,
    target_property,
    model,
    datamodule,
    trainer = None,
):
    # Restart
    resume_ckpt = cfg.get("resume_from_model", None)
    if resume_ckpt is not None:
        ckpt_path = Path(resume_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"Checkpoint not exists: {ckpt_path}")

    lit_model = LightningWrapperModel(cfg, statistics, target_property, model)
    logging.info(lit_model)

    # Trainer
    if trainer is None:
        trainer = build_trainer(cfg)

    # TRAIN AND VALID
    if resume_ckpt is not None:
        logging.info(f"Resume Training")
        logging.info(f"resume_from_model:{resume_ckpt}")
        trainer.fit(
            lit_model,
            datamodule=datamodule,
            ckpt_path=resume_ckpt,
        )
    else:
        trainer.fit(
            lit_model,
            datamodule=datamodule,
        )

    # TEST
    if cfg['dataset'].get('test_files', None) is not None:
        trainer.test(
            lit_model,
            datamodule=datamodule,
            ckpt_path="best",
            verbose=False,
        )
    logging.info("Training completed at %s", datetime.now().strftime("%Y-%m-%d %H:%M"))
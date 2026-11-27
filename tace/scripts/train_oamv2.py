################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Train TACE-OAM-XL-v2 on all visible CUDA devices."""

import os
import sys
from pathlib import Path

EXECUTION = {
    "EQX_XL_CG_REGISTER_LIMIT": "255",
    "EQX_XL_CG_ROW_SIZE": "16",
    "EQX_XL_COTANGENT_RESIDENT": "1",
    "EQX_XL_EDGE_ANGULAR_BLOCK": "1",
    "EQX_XL_LAUNCH_CONTRACT": "1",
    "EQX_XL_LAYOUT_ADJOINTS": "1",
    "EQX_XL_LT_ALGORITHM": "0",
    "EQX_XL_LT_RADIAL": "1",
    "EQX_XL_LT_SM_TARGET": "56",
    "EQX_XL_MIXED_OWNER": "source",
    "EQX_XL_MIXED_REUSE_ANGULAR": "1",
    "EQX_XL_MIXED_ROW_LIMIT": "16",
    "EQX_XL_NODE_SWEEP": "1",
    "EQX_XL_NODE_TANGENT": "1",
    "EQX_XL_ORDERED_REDUCTIONS": "1",
    "EQX_XL_PACKED_EDGES": "1",
    "EQX_XL_PACKED_FRAGMENTS": "1",
    "EQX_XL_PARALLEL_MATRICES": "1",
    "EQX_XL_POLYNOMIAL_PRODUCTS": "auto",
    "EQX_XL_RECEIVER_PIPELINE": "auto",
    "EQX_XL_SKIP_EDGE_ZERO": "1",
    "EQX_XL_SOURCE_ROW_LIMIT": "16",
    "EQX_XL_SOURCE_VJP": "1",
    "EQX_XL_SOURCE_VJP_REGISTER_LIMIT": "192",
    "EQX_XL_STREAM_ANGULAR_OUTPUTS": "auto",
    "OMP_NUM_THREADS": "4",
    "PYTORCH_CUDA_ALLOC_CONF": "roundup_power2_divisions:2",
    "TACE_USE_COMPILE": "1",
    "TACE_USE_CUE": "0",
    "TACE_USE_DENS": "0",
    "TACE_USE_EQX": "1",
    "TACE_USE_LUNA": "0",
    "TACE_USE_MATRIX_WEIGHT": "1",
    "TACE_USE_OEQ": "0",
    "TORCHINDUCTOR_COMPILE_THREADS": "12",
    "TACE_MATMUL_PRECISION": "tf32",
    "TACE_USE_TF32": "1",
}


def configure_environment():
    """Select the FP32/TF32 schedule before importing EQX."""
    os.environ.update(EXECUTION)
    os.environ["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    os.environ.pop("PYTORCH_ALLOC_CONF", None)
    os.environ.setdefault("WANDB_MODE", "offline")


def launch(cfg):
    """Train with the composed Hydra configuration."""
    import torch
    from omegaconf import open_dict

    from equivariantx_foundation.models.tace.tace_oam_xl_v2.precision import (
        validate_environment,
    )
    from tace.lightning.oamv2 import OAMLightningModule
    from tace.scripts.train import train_from_config
    from tace.utils.precision import resolve_precision

    if cfg.trainer.get("accumulate_grad_batches", 1) != 1:
        raise ValueError("Captured training requires accumulate_grad_batches=1")
    strategy = cfg.trainer.get("strategy")
    if (
        isinstance(strategy, str)
        or not strategy
        or strategy.get("_target_") != "tace.utils.strategy.SimpleDDPStrategy"
    ):
        raise ValueError("Captured training requires SimpleDDPStrategy")
    if cfg.trainer.get("inference_mode", False):
        raise ValueError("Force validation requires inference_mode=false")
    with open_dict(cfg):
        policy = resolve_precision(cfg, training=True)
        cfg.trainer.accelerator = "cuda"
        cfg.trainer.devices = "auto"
    if (
        policy.model != "float32"
        or policy.autocast is not None
        or policy.matmul != "tf32"
    ):
        raise ValueError("This training schedule uses FP32 model tensors and TF32 GEMM")
    configured = cfg.misc.get("env") or {}
    validate_environment({**os.environ, **configured})
    for name, value in configured.items():
        if name in EXECUTION and str(value) != EXECUTION[name]:
            raise ValueError(
                f"Config overrides the selected execution schedule: {name}"
            )
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    torch.set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
    train_from_config(cfg, OAMLightningModule, fast=True)

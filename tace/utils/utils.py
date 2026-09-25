################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import contextlib
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import yaml
from omegaconf import DictConfig, ListConfig
from packaging import version
from torch import Tensor

from tace.utils.env import set_tf32


def set_global_seed(cfg: Dict) -> None:
    seed = cfg["misc"].get("global_seed", 42)
    split_seed = cfg["dataset"].get("split_seed", 42)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    logging.info(f"Global seed: {seed}")
    logging.info(f"Split seed: {split_seed}")


def set_precision(cfg: Dict) -> None:
    precision = cfg["trainer"]["precision"]
    FLOAT64 = {"64-true", "64", 64}
    FLOAT32 = {"32-true", "32", 32}
    FLOAT16 = {"16-mixed", "16", 16}
    BFLOAT16 = {"bf16-mixed", "bf16"}
    ALLOWED_PRECISIONS = FLOAT64 | FLOAT32 | FLOAT16 | BFLOAT16
    assert precision is not None and precision in ALLOWED_PRECISIONS, (
        f"Invalid precision setting: {precision!r}. "
        f"Must be one of: {ALLOWED_PRECISIONS}"
    )
    if precision in FLOAT64:
        torch.set_default_dtype(torch.float64)
    elif precision in FLOAT32:
        torch.set_default_dtype(torch.float32)
    elif precision in FLOAT16 or precision in BFLOAT16:
        torch.set_default_dtype(torch.float32)

    set_tf32(training=True)


def num_params(model) -> None:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_parameters(model) -> None:
    logging.debug(f"Total number of parameters in the model: {num_params(model)}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.debug(f"Layer: {name}, Number of parameters: {param.numel()}")


def to_serializable(obj):
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    elif isinstance(obj, (list, tuple)):
        return [to_serializable(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    elif hasattr(obj, "__dict__"):
        return {
            k: to_serializable(v)
            for k, v in vars(obj).items()
            # if not k.startswith("_")
        }
    else:
        return str(obj)


def log_statistics_to_yaml(obj) -> None:
    if obj is not None:
        for idx, stat in enumerate(obj):
            filename = Path(".") / f"statistics_{idx}.yaml"
            temp_filename = filename.with_suffix(".yaml.tmp")
            with open(temp_filename, "w") as f:
                yaml.dump(stat, f, sort_keys=False, allow_unicode=True)
            temp_filename.replace(filename)


@contextlib.contextmanager
def torch_default_dtype(dtype):
    default_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        yield
    finally:
        torch.set_default_dtype(default_dtype)


def is_rank_0():
    if not torch.distributed.is_available():
        return True
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0


def save_full_cfg(cfg: Dict):
    if is_rank_0():
        path = Path(".") / "_full_config.yaml"
        with open(f"{path}", "w") as f:
            yaml.dump(cfg, f, sort_keys=False)


def deep_convert(cfg):
    if isinstance(cfg, DictConfig):
        cfg = dict(cfg)
        for key, value in cfg.items():
            cfg[key] = deep_convert(value)
        return cfg
    elif isinstance(cfg, ListConfig):
        cfg = list(cfg)
        for i, item in enumerate(cfg):
            cfg[i] = deep_convert(item)
        return cfg
    else:
        return cfg


def voigt_to_matrix(t: Tensor, **kwargs):
    """
    Convert voigt notation to matrix notation
    """
    if t.shape == (3, 3):
        return t
    if t.shape == (6,):
        return torch.tensor(
            [
                [t[0], t[5], t[4]],
                [t[5], t[1], t[3]],
                [t[4], t[3], t[2]],
            ],
            dtype=t.dtype,
        )
    if t.shape == (9,):
        return t.view(3, 3)

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), or (9,) but has shape {t.shape}"
    )


def calculate_cps(
    f1: float, kappa_srme: float, rmsd: float, rmsd_baseline: float = 0.15
) -> float:
    """Matbench discovery CPS, using default weight"""
    s_f1 = max(0.0, min(1.0, f1))
    s_kappa = max(0.0, 1.0 - kappa_srme / 2.0)
    if rmsd <= 0.0:
        s_rmsd = 1.0
    elif rmsd >= rmsd_baseline:
        s_rmsd = 0.0
    else:
        s_rmsd = 1.0 - rmsd / rmsd_baseline
    cps = 0.5 * s_f1 + 0.4 * s_kappa + 0.1 * s_rmsd
    return cps

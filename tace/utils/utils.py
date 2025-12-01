################################################################################
# Authors: Zemin Xu
# License: Academic / Non-Commercial Use Only
################################################################################

import pickle
import contextlib
import logging
from typing import Dict, List
import yaml
from pathlib import Path
from packaging import version

from torch import Tensor
from omegaconf import DictConfig, ListConfig


import numpy as np
import torch


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
    try:
        assert precision is not None and precision in ALLOWED_PRECISIONS, (
            f"Invalid precision setting: {precision!r}. "
            f"Must be one of: {ALLOWED_PRECISIONS}"
        )
    except Exception as e:
        raise RuntimeError(f"The cfg.trainer.precision value must be specified.") from e

    if precision in FLOAT64:
        torch.set_default_dtype(torch.float64)
    elif precision in FLOAT32:
        torch.set_default_dtype(torch.float32)
    elif precision in FLOAT16 or precision in BFLOAT16:
        torch.set_default_dtype(torch.float32)

    # === allow_tf32 ===
    allow_tf32 = cfg["misc"].get("allow_tf32", False)
    if torch.cuda.is_available():
        if allow_tf32:
            torch.backends.cuda.matmul.allow_tf32 = allow_tf32
            torch.backends.cudnn.allow_tf32 = allow_tf32

# torch 2.9
# def set_precision(cfg: Dict) -> None:
#     # === init ===
#     global _GLOBAL_STATE_INITIALIZED
#     if not _GLOBAL_STATE_INITIALIZED:
#         torch.set_default_dtype(torch.float64)
#         if torch.cuda.is_available():
#             if _TORCH_GE_2_9:
#                 torch.backends.fp32_precision = "ieee"
#             else:
#                 torch.backends.cuda.matmul.allow_tf32 = False
#                 torch.backends.cudnn.allow_tf32 = False
#         _GLOBAL_STATE_INITIALIZED = True

#     # === set tensor dtype ===
#     precision = cfg["trainer"]["precision"]
#     FLOAT64 = {"64-true", "64", 64}
#     FLOAT32 = {"32-true", "32", 32}
#     FLOAT16 = {"16-mixed", "16", 16}
#     BFLOAT16 = {"bf16-mixed", "bf16"}
#     ALLOWED_PRECISIONS = FLOAT64 | FLOAT32 | FLOAT16 | BFLOAT16
#     try:
#         assert precision is not None and precision in ALLOWED_PRECISIONS, (
#             f"Invalid precision setting: {precision!r}. "
#             f"Must be one of: {ALLOWED_PRECISIONS}"
#         )
#     except Exception as e:
#         raise RuntimeError(f"The cfg.trainer.precision value must be specified.") from e

#     if precision in FLOAT32:
#         torch.set_default_dtype(torch.float32)
#     elif precision in FLOAT16 or precision in BFLOAT16:
#         torch.set_default_dtype(torch.float32)

#     # === allow_tf32 ===
#     allow_tf32 = cfg["misc"].get("allow_tf32", False)
#     if torch.cuda.is_available():
#         if _TORCH_GE_2_9:
#             current_precision = torch.backends.fp32_precision
#             desired_precision = "tf32" if allow_tf32 else "ieee"
#             if current_precision != desired_precision:
#                 torch.backends.fp32_precision = desired_precision
#         else:
#             if torch.backends.cuda.matmul.allow_tf32 is not allow_tf32:
#                 torch.backends.cuda.matmul.allow_tf32 = allow_tf32
#                 torch.backends.cudnn.allow_tf32 = allow_tf32
    
def num_params(model) -> None:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def log_parameters(model) -> None:
    logging.info(f"Total number of parameters in the model: {num_params(model)}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logging.info(f"Layer: {name}, Number of parameters: {param.numel()}")


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
    # def float_representer(dumper, value):
    #     return dumper.represent_scalar('tag:yaml.org,2002:float', f'{value:.15f}')
    # yaml.add_representer(float, float_representer)
    if obj is not None:
        for idx, stat in enumerate(obj):
            filename = Path(".") / f"statistics_{idx}.yaml"
            with open(filename, "w") as f:
                # yaml.safe_dump(obj, f, sort_keys=False, allow_unicode=True)
                yaml.dump(stat, f, sort_keys=False, allow_unicode=True)


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
        path = Path(".") / "_tace.yaml"
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


def expand_dims_to(T: Tensor, n_dim: int, dim: int = -1) -> Tensor:
    '''jit-safe'''
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T
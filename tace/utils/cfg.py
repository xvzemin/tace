################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Configuration conversion, output and Hydra resolvers."""

import math
from pathlib import Path

import yaml
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, ListConfig, OmegaConf

from .strategy import is_rank_0


def deep_convert(cfg):
    """Convert OmegaConf mappings and lists to Python containers."""
    if isinstance(cfg, DictConfig):
        return {key: deep_convert(value) for key, value in cfg.items()}
    if isinstance(cfg, ListConfig):
        return [deep_convert(value) for value in cfg]
    return cfg


def save_full_cfg(cfg: dict):
    """Write the resolved training configuration on rank zero."""
    if not is_rank_0():
        return
    with Path("_full_config.yaml").open("w", encoding="utf-8") as stream:
        yaml.dump(cfg, stream, sort_keys=False)


def to_serializable(obj):
    """Convert Python objects to scalar, list and mapping values for output."""
    if isinstance(obj, (int, float, str, bool)) or obj is None:
        return obj
    if isinstance(obj, (list, tuple)):
        return [to_serializable(item) for item in obj]
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if hasattr(obj, "__dict__"):
        return {key: to_serializable(value) for key, value in vars(obj).items()}
    return str(obj)


def resolve_path(path_str: str, expect: str = "file") -> str:
    """Resolve a Hydra path, checking files or creating missing directories.

    Parameters
    ----------
    path_str : str
        Path relative to the original working directory.
    expect : {"file", "dir"}
        Required path type.

    Examples
    --------
    >>> Path(resolve_path(".", "dir")).is_dir()
    True
    """
    path = Path(to_absolute_path(path_str))
    if expect == "file":
        if not path.is_file():
            raise FileNotFoundError(f"Expected file not found: {path}")
    elif expect == "dir":
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise NotADirectoryError(f"Expected directory but found file: {path}")
    else:
        raise ValueError(f"Unknown expect value: {expect} (should be file or dir)")
    return str(path)


def register_resolvers():
    """Register path and scalar arithmetic resolvers for training YAML."""
    OmegaConf.register_new_resolver("check_file", lambda s: resolve_path(s, "file"))
    OmegaConf.register_new_resolver("check_dir", lambda s: resolve_path(s, "dir"))
    OmegaConf.register_new_resolver("add", lambda a, b: float(a) + float(b))
    OmegaConf.register_new_resolver("sub", lambda a, b: float(a) - float(b))
    OmegaConf.register_new_resolver("mul", lambda a, b: float(a) * float(b))
    OmegaConf.register_new_resolver(
        "div", lambda a, b: float(a) / float(b) if float(b) != 0 else float("inf")
    )
    OmegaConf.register_new_resolver("float", lambda x: float(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(float(x)))
    OmegaConf.register_new_resolver("floor", lambda x: math.floor(float(x)))

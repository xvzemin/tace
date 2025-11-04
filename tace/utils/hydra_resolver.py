################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
import math
from pathlib import Path


from omegaconf import OmegaConf
from hydra.utils import to_absolute_path


def _resolve_path(path_str: str, expect: str = "file") -> str:
    path = Path(to_absolute_path(path_str))

    if expect == "file":
        if not path.is_file():
            raise FileNotFoundError(f"Expected file not found: {path}")
    elif expect == "dir":
        if not path.exists():
            print(f"[INFO] Directory not found, creating: {path}")
            path.mkdir(parents=True, exist_ok=True)
        elif not path.is_dir():
            raise NotADirectoryError(f"Expected directory but found file: {path}")
    else:
        raise ValueError(f"Unknown expect value: {expect} (should be file or dir)")
    return str(path)


def register_resolvers():
    OmegaConf.register_new_resolver("check_file", lambda s: _resolve_path(s, "file"))
    OmegaConf.register_new_resolver("check_dir", lambda s: _resolve_path(s, "dir"))
    OmegaConf.register_new_resolver("add", lambda a, b: float(a) + float(b))
    OmegaConf.register_new_resolver("sub", lambda a, b: float(a) - float(b))
    OmegaConf.register_new_resolver("mul", lambda a, b: float(a) * float(b))
    OmegaConf.register_new_resolver(
        "div", lambda a, b: float(a) / float(b) if float(b) != 0 else float("inf")
    )
    OmegaConf.register_new_resolver("float", lambda x: float(x))
    OmegaConf.register_new_resolver("ceil", lambda x: math.ceil(float(x)))
    OmegaConf.register_new_resolver("floor", lambda x: math.floor(float(x)))

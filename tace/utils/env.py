################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import os
from pathlib import Path
from string import ascii_letters
from typing import Dict, Optional

import torch

CACHE_DIR = Path.home() / ".cache" / "tace"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

BOOL = {
    0: False,
    1: True,
    "f": False,
    "t": True,
    "false": False,
    "true": True,
    "False": False,
    "True": True,
}

DTYPE = {
    None: None,
    16: torch.float16,
    32: torch.float32,
    64: torch.float64,
    "16": torch.float16,
    "32": torch.float32,
    "64": torch.float64,
    "half": torch.float16,
    "single": torch.float32,
    "double": torch.float64,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
    "fp64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
    "float64": torch.float64,
    # Lightning true precision modes: model parameters use this dtype.
    "16-true": torch.float16,
    "bf16-true": torch.bfloat16,
    "32-true": torch.float32,
    "64-true": torch.float64,
    # Lightning mixed precision keeps model parameters in float32.
    "16-mixed": torch.float32,
    "bf16-mixed": torch.float32,
    # Transformer Engine weight/fallback dtypes used by Lightning.
    "transformer-engine": torch.bfloat16,
    "transformer-engine-float16": torch.float16,
    torch.float16: torch.float16,
    torch.bfloat16: torch.bfloat16,
    torch.float32: torch.float32,
    torch.float64: torch.float64,
}

num_gpus = 32
DEVICE = {
    None: None,
    "cpu": torch.device("cpu"),
    "cuda": torch.device("cuda"),
    torch.device("cpu"): torch.device("cpu"),
    torch.device("cuda"): torch.device("cuda"),
    **{i: torch.device(f"cuda:{i}") for i in range(num_gpus)},
    **{f"cuda:{i}": torch.device(f"cuda:{i}") for i in range(num_gpus)},
    **{torch.device(f"cuda:{i}"): torch.device(f"cuda:{i}") for i in range(num_gpus)},
}

LETTERS = list(ascii_letters)[3:]

ACCELERATION_ENV = {
    "eqx": "TACE_USE_EQX",
    "luna": "TACE_USE_LUNA",
    "oeq": "TACE_USE_OEQ",
    "eqt": "TACE_USE_EQT",
    "cue": "TACE_USE_CUE",
    "compile": "TACE_USE_COMPILE",
}

# Internal switches for kernel comparisons; normal use follows TACE_USE_EQX.
EQX_KERNELS = {"conv": True, "product": True, "linear": True}


def set_tf32(*, training: bool = False, matmul: str | None = None) -> None:
    """Configure TF32, defaulting to enabled only for training."""
    from tace.utils.precision import resolve_precision

    resolve_precision({"dtype": {"matmul": matmul}}, training=training).apply()


def set_env(cfg: Dict):
    env = cfg.get("misc", {}).get("env", {})
    for k, v in env.items():
        if not isinstance(v, str):
            raise TypeError(
                f"Environment variable {k!r} must have a string value, "
                f"got {type(v).__name__}. Quote the value in the YAML file."
            )
        os.environ[k] = v


def enable_acceleration(
    *,
    enable_oeq: bool = False,
    enable_cue: bool = False,
    enable_eqt: bool = False,
    enable_compile: bool = False,
    enable_eqx: bool = False,
    enable_luna: bool = False,
    force: bool = False,
) -> None:
    """
    Enable accelerations, preserving existing settings unless forced.
    """
    enabled_accelerations = {
        "oeq": enable_oeq,
        "cue": enable_cue,
        "eqt": enable_eqt,
        "compile": enable_compile,
        "eqx": enable_eqx,
        "luna": enable_luna,
    }
    for name, enabled in enabled_accelerations.items():
        if enabled or force:
            os.environ[ACCELERATION_ENV[name]] = "1" if enabled else "0"


def acceleration_enabled(name: str, *, kernel: Optional[str] = None) -> Optional[bool]:
    try:
        env_name = ACCELERATION_ENV[name.lower()]
    except KeyError as e:
        options = ", ".join(ACCELERATION_ENV)
        raise ValueError(
            f"Unknown TACE acceleration {name!r}; expected one of: {options}"
        ) from e
    value = os.environ.get(env_name)
    if value is None:
        return None
    if name.lower() == "eqx" and kernel is not None:
        return value == "1" and EQX_KERNELS[kernel]
    return value == "1"


def select_acceleration(*backends: str, kernel: Optional[str] = None) -> Optional[str]:
    """Select an enabled, supported backend in EQX, Luna, OEQ, EQT, CUEQ order."""
    return next(
        (
            name
            for name in ACCELERATION_ENV
            if name in backends and acceleration_enabled(name, kernel=kernel)
        ),
        None,
    )


def get_tace_apply_u_shift():
    return os.environ.get("TACE_APPLY_U_SHIFT", "0")


def get_tace_use_dens():
    return os.environ.get("TACE_USE_DENS", "0")


def get_tace_use_matrix_weight():
    return os.environ.get("TACE_USE_MATRIX_WEIGHT", "0")


def memory_first() -> bool:
    """Recompute the complete radial MLP in EQX when TACE_MEMORY_FIRST=1."""
    return os.environ.get("TACE_MEMORY_FIRST", "0") == "1"

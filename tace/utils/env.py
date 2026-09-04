################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import os
from typing import Dict, Optional

ACCELERATION_ENV = {
    "oeq": "TACE_USE_OEQ",
    "cue": "TACE_USE_CUE",
    "eqt": "TACE_USE_EQT",
    "compile": "TACE_USE_COMPILE",
    "eqx": "TACE_USE_EQX",
}


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
    force: bool = False,
) -> None:
    """Enable accelerations, preserving existing settings unless forced.

    ``enable_eqx`` selects the EQX operator path. Until the general EQX O(3),
    O(2), and Cartesian operators are available, supported calls
    are routed to the temporary kernels in :mod:`tace.models.triton_ops`.
    """
    enabled_accelerations = {
        "oeq": enable_oeq,
        "cue": enable_cue,
        "eqt": enable_eqt,
        "compile": enable_compile,
        "eqx": enable_eqx,
    }
    for name, enabled in enabled_accelerations.items():
        if enabled or force:
            os.environ[ACCELERATION_ENV[name]] = "1" if enabled else "0"


def acceleration_enabled(name: str) -> Optional[bool]:
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
    return value == "1"


def get_tace_apply_u_shift():
    return os.environ.get("TACE_APPLY_U_SHIFT", "0")


def get_tace_use_dens():
    return os.environ.get("TACE_USE_DENS", "0")


def get_tace_use_matrix_weight():
    return os.environ.get("TACE_USE_MATRIX_WEIGHT", "0")

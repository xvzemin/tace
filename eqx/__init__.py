"""Spherical and Cartesian O(3)/O(2) operators and fused convolutions."""

from contextlib import nullcontext
from importlib import import_module

import torch

with (
    torch.serialization.safe_globals(
        [] if slice in torch.serialization.get_safe_globals() else [slice]
    )
    if hasattr(torch.serialization, "safe_globals")
    else nullcontext()
):
    from eqx import o2


__all__ = [
    "o3",
    "co3",
    "co2",
    "o2",
    "ace",
    "conv",
    "ops",
    "backends",
    "models",
    "nn",
    "utils",
]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))


import_module(".ops.registration", __name__)

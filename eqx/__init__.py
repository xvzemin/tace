"""Native O(2) operators, O(3) frame conversion, and fused CUDA convolutions."""

from contextlib import nullcontext
from importlib import import_module

import torch

# e3nn 0.4.x stores slices in its packaged CG constants. Keep this allowlist
# scoped to import rather than changing torch.load or its process-wide defaults.
with (
    torch.serialization.safe_globals(
        [] if slice in torch.serialization.get_safe_globals() else [slice]
    )
    if hasattr(torch.serialization, "safe_globals")
    else nullcontext()
):
    from . import o2

__all__ = ["o2", "o3", "ace", "conv", "kernels"]


def __getattr__(name):
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))

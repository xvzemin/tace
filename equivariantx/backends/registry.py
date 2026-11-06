"""Lazy execution-provider registration."""

from functools import lru_cache, wraps
from importlib import import_module

import torch

IMPLEMENTATIONS = {}


def validate_operands(values):
    """Check the scalar ABI of CUDA operands before taking device pointers."""
    tensors = []
    pending = list(values)
    while pending:
        value = pending.pop()
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            tensors.append(value)
        elif isinstance(value, (tuple, list)):
            pending.extend(value)
    if not tensors:
        return
    reference = tensors[0]
    if reference.dtype not in (torch.float32, torch.float64):
        raise TypeError("EQX CUDA kernels require float32 or float64 operands.")
    if any(
        value.dtype != reference.dtype or value.device != reference.device
        for value in tensors
    ):
        raise ValueError("EQX CUDA floating operands must share a dtype and device.")


def register_kernel(name, backend, implementation):
    """Register an execution provider.

    Parameters
    ----------
    name : str
        Operator name without its namespace.
    backend : str
        Execution provider name.
    implementation : callable or str
        Implementation or lazy ``module:function`` import path.
    """
    key = name, backend
    if key in IMPLEMENTATIONS:
        raise ValueError(f"Kernel {name!r} already has a {backend!r} implementation.")
    IMPLEMENTATIONS[key] = implementation
    get_kernel.cache_clear()


@lru_cache(maxsize=None)
def get_kernel(name, backend):
    """Resolve a registered execution provider.

    Parameters
    ----------
    name : str
        Operator name without its namespace.
    backend : str
        Execution provider name.

    Returns
    -------
    callable
        Implementation with the operator's argument and return conventions.
    """
    try:
        implementation = IMPLEMENTATIONS[name, backend]
    except KeyError:
        raise NotImplementedError(
            f"Kernel {name!r} does not support {backend!r}."
        ) from None
    if isinstance(implementation, str):
        module, function = implementation.split(":")
        implementation = getattr(import_module(module), function)
    if backend == "cuda":
        @wraps(implementation)
        def checked(*args, **kwargs):
            validate_operands((*args, *kwargs.values()))
            return implementation(*args, **kwargs)

        return checked
    return implementation

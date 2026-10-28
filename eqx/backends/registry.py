"""Lazy execution-provider registration."""

from functools import lru_cache
from importlib import import_module

IMPLEMENTATIONS = {}


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
        return getattr(import_module(module), function)
    return implementation

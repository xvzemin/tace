"""Lazy execution providers with independent Torch reference implementations."""

from functools import wraps
from importlib import import_module

import torch

from . import config
from .precision import validate_precision, validate_uniform

IMPLEMENTATIONS = {}
REFERENCES = {}
RESOLVED = {}


def resolve_backend(backend=None, device=None):
    """Select the execution backend.

    Parameters
    ----------
    backend : {None, "torch", "cuda"}, optional
        Explicit provider, or ``config.DEFAULT_BACKEND``, initialized at import.
    device : torch.device or str, optional
        Input device for automatic selection. CPU inputs use Torch.

    Returns
    -------
    str
        Selected execution provider.
    """
    if backend is None:
        backend = config.DEFAULT_BACKEND
        if device is not None and torch.device(device).type != "cuda":
            backend = "torch"
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be None, 'torch' or 'cuda'.")
    return backend


def validate_operands(values):
    """Check the scalar ABI of CUDA operands before taking device pointers."""
    validate_uniform(values)


def register_kernel(name, backend, implementation, *, reference=None):
    """Register an execution provider.

    Parameters
    ----------
    name : str
        Operator name without its namespace.
    backend : str
        Execution provider name.
    implementation : callable or str
        Implementation or lazy ``module:function`` import path.
    reference : callable or str, optional
        Ordinary Torch forward program, selected by ``config.TORCH_REFERENCE``.
        Applies only to the Torch provider.
    """
    key = name, backend
    if key in IMPLEMENTATIONS:
        raise ValueError(f"Kernel {name!r} already has a {backend!r} implementation.")
    if reference is not None and backend != "torch":
        raise ValueError("Reference implementations belong to the torch provider.")
    IMPLEMENTATIONS[key] = implementation
    if reference is not None:
        REFERENCES[name] = reference
    if backend == "torch":
        RESOLVED[implementation, backend] = resolve_kernel(implementation, backend)
        if reference is not None:
            RESOLVED[reference, backend] = resolve_kernel(reference, backend)


def get_kernel(name, backend="torch"):
    """Resolve a registered execution provider.

    Parameters
    ----------
    name : str
        Operator name without its namespace.
    backend : str, optional
        Execution provider name. Defaults to ``"torch"``.

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
    if backend == "torch" and config.TORCH_REFERENCE:
        implementation = REFERENCES.get(name, implementation)
    key = implementation, backend
    if key not in RESOLVED:
        RESOLVED[key] = resolve_kernel(implementation, backend)
    return RESOLVED[key]


def resolve_kernel(implementation, backend):
    """Load a provider once without caching the reference switch."""
    if isinstance(implementation, str):
        module, function = implementation.split(":")
        implementation = getattr(import_module(module), function)

    @wraps(implementation)
    def checked(*args, **kwargs):
        values = (*args, *kwargs.values())
        if backend == "cuda":
            validate_operands(values)
        else:
            validate_precision(values)
        return implementation(*args, **kwargs)

    return checked

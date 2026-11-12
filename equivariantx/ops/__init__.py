"""Operator dispatch, differentiation and execution kernels."""

from .registry import get_kernel, register_kernel
from . import kernels  # noqa: F401 - Register execution providers.

__all__ = ["get_kernel", "register_kernel"]

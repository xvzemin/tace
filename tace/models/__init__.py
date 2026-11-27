"""TACE model representations and runtime adapters."""

# Register the operators and load legacy e3nn constants before importing models.
import equivariantx  # noqa: F401

from .adapter import TensorModel
from .cartesian import CartesianTACE
from .compile import CompileTensorModel
from .spherical import SphericalTACE

__all__ = ["CartesianTACE", "SphericalTACE", "TensorModel", "CompileTensorModel"]

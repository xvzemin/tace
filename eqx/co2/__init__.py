"""Cartesian O(2) operators and coordinate-free harmonic tensor products."""

from .basis import ChangeOfBasis, Projector, path_matrix
from .cartesian_harmonics import CartesianHarmonics
from .gate import Activation, Gate
from .irreps import Irrep, Irreps
from .linear import Linear
from .o3_tensor_product import O3TensorProduct
from .projection import PlanarDetracer, PlaneProjector, plane_projector
from .restriction import (
    Restriction,
    TransverseProjector,
    coupling_coefficients,
    restriction_matrix,
    restriction_scale,
)
from .spherical import SphericalCoupling
from .tensor_product import (
    ElementwiseTensorProduct,
    FullyConnectedTensorProduct,
    TensorProduct,
)

__all__ = [
    "Irrep",
    "Irreps",
    "ChangeOfBasis",
    "Projector",
    "path_matrix",
    "CartesianHarmonics",
    "Linear",
    "Activation",
    "Gate",
    "TensorProduct",
    "FullyConnectedTensorProduct",
    "ElementwiseTensorProduct",
    "Restriction",
    "TransverseProjector",
    "PlaneProjector",
    "PlanarDetracer",
    "plane_projector",
    "restriction_scale",
    "restriction_matrix",
    "coupling_coefficients",
    "O3TensorProduct",
    "SphericalCoupling",
]

"""Spherical O(2) operators and O(3)/O(2) frame conversions."""

from .circular_harmonics import CircularHarmonics, circular_harmonics
from .contraction import AsymmetricContraction, SymmetricContraction
from .gate import Activation, Gate
from .ictd import ICTD, path_matrices
from .irreps import Irrep, Irreps
from .layout import TransposeIrrepsLayout
from .linear import ElementLinear, Linear, UuLinear
from .local_frame import LocalFrame
from .o2grid import O2Grid
from .o3_tensor_product import O3TensorProduct
from .rotation_matrix import (
    rotation_matrix_to_x_axis,
    rotation_matrix_to_y_axis,
    rotation_matrix_to_z_axis,
)
from .tensor_product import TensorProduct
from .wigner import WignerD

__all__ = [
    "TransposeIrrepsLayout",
    "O2O3TensorProductConv",
    "UuO2TensorProductConv",
    "UvO2TensorProductConv",
    "ICTD",
    "path_matrices",
    "Irrep",
    "Irreps",
    "Linear",
    "ElementLinear",
    "UuLinear",
    "Activation",
    "Gate",
    "AsymmetricContraction",
    "SymmetricContraction",
    "LocalFrame",
    "CircularHarmonics",
    "O2Grid",
    "TensorProduct",
    "O3TensorProduct",
    "WignerD",
    "circular_harmonics",
    "rotation_matrix_to_x_axis",
    "rotation_matrix_to_y_axis",
    "rotation_matrix_to_z_axis",
]


def __getattr__(name):
    from importlib import import_module

    modules = {
        "O2O3TensorProductConv": "aligned",
        "UuO2TensorProductConv": "uu",
        "UvO2TensorProductConv": "uv",
    }
    if name not in modules:
        raise AttributeError(name)
    value = getattr(import_module(f".{modules[name]}", __name__), name)
    globals()[name] = value
    return value

"""Convolution operations."""

from ..tensor_product import TensorProductConv
from .tensor_product import GauntTensorProduct


class GauntTensorProductConv(TensorProductConv):
    """Gather, multiply spherical signals and sum at destination nodes.

    Parameters
    ----------
    tensor_product : eqx.o3.GauntTensorProduct
        Defines the irreps, factors and normalization.

    Notes
    -----
    Features use ``mul_ir`` layout. PyTorch operations support CPU, CUDA and
    automatic differentiation. Edge factors and messages are materialized.
    """

    def __init__(self, tensor_product):
        if not isinstance(tensor_product, GauntTensorProduct):
            raise TypeError("tensor_product must be a GauntTensorProduct.")
        super().__init__(tensor_product)

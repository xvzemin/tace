"""Convolution operations."""

from ..tensor_product import TensorProductConv
from .tensor_product import MatrixTensorProduct


class MatrixTensorProductConv(TensorProductConv):
    """Gather, multiply matrix representations and sum at destination nodes.

    Parameters
    ----------
    tensor_product : eqx.o3.MatrixTensorProduct
        Defines the irreps, factors, matrix degree and normalization.

    Notes
    -----
    Features use ``mul_ir`` layout. PyTorch operations support CPU, CUDA and
    automatic differentiation. Edge factors and messages are materialized.
    """

    def __init__(self, tensor_product):
        if not isinstance(tensor_product, MatrixTensorProduct):
            raise TypeError("tensor_product must be a MatrixTensorProduct.")
        super().__init__(tensor_product)

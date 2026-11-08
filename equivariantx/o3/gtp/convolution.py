"""Convolution operations."""

from ..tensor_product import TensorProductConv
from .tensor_product import GauntTensorProduct


class GauntTensorProductConv(TensorProductConv):
    """Gather spherical signals in ``mul_ir`` layout and sum their Gaunt products.

    Parameters
    ----------
    tensor_product : equivariantx.o3.GauntTensorProduct
        Defines the irreps, factors and normalization.

    Examples
    --------
    >>> import torch
    >>> import equivariantx as eqx
    >>> tp = eqx.o3.GauntTensorProduct("2x0e+2x1o", "0e+1o", "2x0e+2x1o")
    >>> layer = eqx.conv.GauntTensorProductConv(tp)
    >>> nodes = torch.randn(3, 8)
    >>> attrs = torch.randn(4, 4)
    >>> edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> layer(nodes, attrs, edges).shape
    torch.Size([3, 8])
    """

    def __init__(self, tensor_product):
        if not isinstance(tensor_product, GauntTensorProduct):
            raise TypeError("tensor_product must be a GauntTensorProduct.")
        super().__init__(tensor_product)

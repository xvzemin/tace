"""Convolution operations."""

from ..tensor_product import TensorProductConv
from .tensor_product import MatrixTensorProduct


class MatrixTensorProductConv(TensorProductConv):
    """Gather spherical features in ``mul_ir`` layout and sum their matrix products.

    Parameters
    ----------
    tensor_product : equivariantx.o3.MatrixTensorProduct
        Defines the irreps, factors, matrix degree and normalization.
    backend : {"torch", "cuda"}, optional
        Defaults to the tensor product backend.
    radial_mode : {"last", "full"}, optional
        Fuse the last radial projection, or recompute all radial layers in
        the convolution adjoints. Defaults to ``"last"``.

    Examples
    --------
    >>> import torch
    >>> import equivariantx as eqx
    >>> tp = eqx.o3.MatrixTensorProduct("2x0e+2x1o", "0e+1o", "2x0e+2x1o")
    >>> layer = eqx.conv.MatrixTensorProductConv(tp)
    >>> nodes = torch.randn(3, 8)
    >>> attrs = torch.randn(4, 4)
    >>> edges = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    >>> layer(nodes, attrs, edges).shape
    torch.Size([3, 8])
    """

    def __init__(self, tensor_product, *, backend=None, radial_mode="last"):
        if not isinstance(tensor_product, MatrixTensorProduct):
            raise TypeError("tensor_product must be a MatrixTensorProduct.")
        super().__init__(tensor_product, backend=backend, radial_mode=radial_mode)

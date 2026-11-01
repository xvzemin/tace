"""Gated ICTD products with element and expert coefficients."""

from ..co3 import Projector
from .bilinear import BilinearContraction
from .ictd_semi_coupled import coupling_entries


class IctdBilinearContraction(BilinearContraction):
    """Contract ICTD products with gates and element/expert coefficients.

    Parameters
    ----------
    tensor_product : co3.TensorProduct
        External-weight channel-wise product.
    linear : co3.Linear
        External-weight coefficient map for one expert.
    num_experts : int, optional
        Number of independent channel groups.
    backend : {"cuda", "torch"}, optional
        Execution backend. CPU inputs use PyTorch.
    shared_linear : co3.Linear, optional
        Element-independent coefficient map over all channels.

    Examples
    --------
    >>> import torch
    >>> from eqx import ace, co3
    >>> tp = co3.TensorProduct(
    ...     "2x1o", "2x1o", "2x0e", [(0, 0, 0, "uuu", True)],
    ...     internal_weights=False, shared_weights=False
    ... )
    >>> linear = co3.Linear("2x0e", "3x0e", internal_weights=False,
    ...                     shared_weights=False)
    >>> layer = ace.IctdBilinearContraction(tp, linear, backend="torch")
    >>> features = co3.Irreps("2x1o").randn(3, -1)
    >>> gates = torch.randn(3, tp.weight_numel)
    >>> weight = torch.randn(2, linear.weight_numel)
    >>> node_type = torch.tensor([0, 1, 0])
    >>> layer(features, features, gates, weight, node_type).shape
    torch.Size([3, 3])
    """

    coupling_entries = staticmethod(coupling_entries)

    def __init__(
        self, tensor_product, linear, num_experts=1, backend="cuda", shared_linear=None
    ):
        super().__init__(tensor_product, linear, num_experts, backend, shared_linear)
        self.projection = Projector(
            [(mul * num_experts, ir) for mul, ir in linear.irreps_out]
        )

    def project_output(self, features):
        return self.projection(features)

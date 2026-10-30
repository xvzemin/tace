"""Channel-wise Cartesian cluster expansions."""

import math

from eqx.co3 import Projector
from eqx.conv.co3.polynomials import coupling_coefficients
from eqx.models.tace.tece_oam_rra.product import BilinearACE

from .tace import TACE


def coupling_entries(tensor_product, index):
    """Return delta/epsilon contraction entries with path normalization."""
    path = tensor_product.instructions[index]
    l1 = tensor_product.irreps_in1[path.i_in1].ir.l
    l2 = tensor_product.irreps_in2[path.i_in2].ir.l
    l3 = tensor_product.irreps_out[path.i_out].ir.l
    traces, odd = divmod(l1 + l2 - l3, 2)
    scale = tensor_product.path_factors[index] / math.sqrt(
        3**traces * (2 if odd else 1)
    )
    return [
        (a, b, c, coefficient * scale)
        for a, b, c, coefficient in coupling_coefficients(l1, l2, l3)
    ]


class CartesianTACE(TACE):
    """Evaluate Cartesian products and element-dependent coefficient maps.

    Parameters
    ----------
    tensor_products : sequence of co3.TensorProduct
        Unweighted channel-wise products retaining every coupling path.
    linears : sequence of co3.Linear
        External-weight coefficient maps, including the order-one term.
    backend : {"cuda", "torch"}, optional
        Execution backend. CPU inputs use PyTorch.

    Notes
    -----
    Features use flattened ``mul_ir`` storage with STF Cartesian inputs.
    The caller supplies the element-dependent weights for every order.

    Examples
    --------
    >>> import torch
    >>> from eqx import ace, co3
    >>> irreps = co3.Irreps("2x0e + 2x1o")
    >>> tp = co3.ElementwiseTensorProduct(irreps, irreps)
    >>> linears = [co3.Linear(layout, "2x0e", internal_weights=False,
    ...                       shared_weights=False)
    ...            for layout in (irreps, tp.irreps_out)]
    >>> layer = ace.CartesianTACE([tp], linears, backend="torch")
    >>> weights = [torch.randn(2, linear.weight_numel) for linear in linears]
    >>> node_type = torch.tensor([0, 1, 0])
    >>> layer(irreps.randn(3, -1), weights, node_type).shape
    torch.Size([3, 2])
    """

    coupling_entries = staticmethod(coupling_entries)

    def __init__(self, tensor_products, linears, backend="cuda"):
        super().__init__(tensor_products, linears, backend)
        self.projection = Projector(linears[0].irreps_out)

    def project_product(self, features, stage):
        return self.tensor_products[stage].projection(features)

    def project_output(self, features):
        return self.projection(features)


class CartesianBilinearACE(BilinearACE):
    """Contract Cartesian products with gates and element/expert coefficients.

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

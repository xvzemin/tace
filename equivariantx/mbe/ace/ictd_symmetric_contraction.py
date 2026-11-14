"""ICTD uuu symmetric contractions with fully connected coefficient maps."""

import math

from ...co3 import ChangeOfBasis, Linear, Projector
from ...conv.co3.polynomials import coupling_coefficients
from ...ops import resolve_backend
from ...ops.torch.mbe.ace import ictd_symmetric_contraction as torch_kernel
from .symmetric_contraction import UuuFullyConnectedSymmetricContraction


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


def cartesian_coefficient(linear):
    """Return the same coefficient paths before their output basis change."""
    if linear.output_basis == "cartesian":
        return linear
    if linear.internal_weights or linear.bias_numel:
        raise ValueError("Provide unbiased external coefficient maps.")
    return Linear(
        linear.irreps_in,
        linear.cartesian_out,
        internal_weights=False,
        shared_weights=linear.shared_weights,
        instructions=[(ins.i_in, ins.i_out) for ins in linear.instructions],
        path_normalization=linear.path_normalization,
    )


class IctdUuuFullyConnectedSymmetricContraction(UuuFullyConnectedSymmetricContraction):
    """Evaluate ICTD products and element-dependent coefficient maps.

    Parameters
    ----------
    tensor_products : sequence of co3.TensorProduct
        Unweighted channel-wise products retaining every coupling path.
    linears : sequence of co3.Linear
        External-weight coefficient maps, including the order-one term.
        All maps must use the same ``output_basis``; Cartesian and spherical
        outputs are supported without changing the coefficient weights.
    backend : {None, "cuda", "torch"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CPU inputs use PyTorch.

    Notes
    -----
    Features use flattened ``mul_ir`` storage with STF Cartesian inputs.
    The caller supplies the element-dependent weights for every order.

    Examples
    --------
    >>> import torch
    >>> from equivariantx.mbe import ace
    >>> from equivariantx import co3
    >>> irreps = co3.Irreps("2x0e + 2x1o")
    >>> tp = co3.ElementwiseTensorProduct(irreps, irreps)
    >>> linears = [co3.Linear(layout, "2x2e", internal_weights=False,
    ...                       shared_weights=False, output_basis="spherical")
    ...            for layout in (irreps, tp.irreps_out)]
    >>> layer = ace.IctdUuuFullyConnectedSymmetricContraction(
    ...     [tp], linears, backend="torch"
    ... )
    >>> weights = [torch.randn(2, linear.weight_numel) for linear in linears]
    >>> node_type = torch.tensor([0, 1, 0])
    >>> layer(irreps.randn(3, -1), weights, node_type).shape
    torch.Size([3, 10])
    """

    coupling_entries = staticmethod(coupling_entries)

    def __init__(self, tensor_products, linears, backend=None):
        backend = resolve_backend(backend)
        linears = tuple(linears)
        if len({linear.output_basis for linear in linears}) > 1:
            raise ValueError("Coefficient maps must use the same output basis.")
        super().__init__(
            tensor_products,
            [cartesian_coefficient(linear) for linear in linears],
            backend,
        )
        self.projection = Projector(self.linears[0].irreps_out, backend=backend)
        self.output_transform = (
            ChangeOfBasis(self.linears[0].irreps_out, inverse=True, backend=backend)
            if linears[0].output_basis == "spherical"
            else None
        )
        self.output_dim = linears[0].irreps_out.dim

    def forward(self, features, weights, node_type):
        """Return indexed coefficients in the declared output basis."""
        output = super().forward(features, weights, node_type)
        if self.output_transform is None:
            return output
        if self.backend == "cuda" and output.is_cuda:
            from ...ops.cuda.mbe.ace import ictd_symmetric_contraction as cuda_kernel

            return cuda_kernel.change_basis(self.output_transform, output)
        return torch_kernel.change_basis(self.output_transform, output)

    def project(self, projection, features):
        """Apply the Cartesian projection using this contraction's backend."""
        if self.backend == "cuda" and features.is_cuda:
            from ...ops.cuda.mbe.ace import ictd_symmetric_contraction as cuda_kernel

            return cuda_kernel.project(projection, features)
        return torch_kernel.project(projection, features)

    def project_product(self, features, stage):
        return self.project(self.tensor_products[stage].projection, features)

    def project_output(self, features):
        return self.project(self.projection, features)

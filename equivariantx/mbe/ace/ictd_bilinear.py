"""Gated ICTD products with element and expert coefficients."""

from ...co3 import ChangeOfBasis, Projector
from ...ops import resolve_backend
from ...ops.torch.mbe.ace import ictd_symmetric_contraction as torch_kernel
from .bilinear import BilinearContraction
from .ictd_symmetric_contraction import cartesian_coefficient, coupling_entries


class IctdBilinearContraction(BilinearContraction):
    """Contract ICTD products with gates and element/expert coefficients.

    Parameters
    ----------
    tensor_product : co3.TensorProduct
        External-weight channel-wise product.
    linear : co3.Linear
        External-weight coefficient map for one expert. Its ``output_basis``
        selects Cartesian or spherical output without changing the weights.
    num_experts : int, optional
        Number of independent channel groups.
    backend : {None, "cuda", "torch"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CPU inputs use PyTorch.
    shared_linear : co3.Linear, optional
        Element-independent coefficient map over all channels, with the same
        output basis as ``linear``.

    Examples
    --------
    >>> import torch
    >>> from equivariantx.mbe import ace
    >>> from equivariantx import co3
    >>> tp = co3.TensorProduct(
    ...     "2x1o", "2x1o", "2x2e", [(0, 0, 0, "uuu", True)],
    ...     internal_weights=False, shared_weights=False
    ... )
    >>> linear = co3.Linear("2x2e", "3x2e", internal_weights=False,
    ...                     shared_weights=False, output_basis="spherical")
    >>> layer = ace.IctdBilinearContraction(tp, linear, backend="torch")
    >>> features = co3.Irreps("2x1o").randn(3, -1)
    >>> gates = torch.randn(3, tp.weight_numel)
    >>> weight = torch.randn(2, linear.weight_numel)
    >>> node_type = torch.tensor([0, 1, 0])
    >>> layer(features, features, gates, weight, node_type).shape
    torch.Size([3, 15])
    """

    coupling_entries = staticmethod(coupling_entries)

    def __init__(
        self, tensor_product, linear, num_experts=1, backend=None, shared_linear=None
    ):
        backend = resolve_backend(backend)
        if (
            shared_linear is not None
            and shared_linear.output_basis != linear.output_basis
        ):
            raise ValueError("Coefficient maps must use the same output basis.")
        raw_linear = cartesian_coefficient(linear)
        super().__init__(
            tensor_product,
            raw_linear,
            num_experts,
            backend,
            None if shared_linear is None else cartesian_coefficient(shared_linear),
        )
        output = [(mul * num_experts, ir) for mul, ir in raw_linear.irreps_out]
        self.projection = Projector(output, backend=backend)
        self.output_transform = (
            ChangeOfBasis(output, inverse=True, backend=backend)
            if linear.output_basis == "spherical"
            else None
        )
        self.output_dim = linear.irreps_out.dim * num_experts

    def forward(self, features, base, gates, weight, node_type, shared_weight=None):
        """Return expert and shared coefficients in the declared output basis."""
        output = super().forward(
            features, base, gates, weight, node_type, shared_weight
        )
        if self.output_transform is None:
            return output
        if self.backend == "cuda" and output.is_cuda:
            from ...ops.cuda.mbe.ace import ictd_symmetric_contraction as cuda_kernel

            return cuda_kernel.change_basis(self.output_transform, output)
        return torch_kernel.change_basis(self.output_transform, output)

    def project_output(self, features):
        if self.backend == "cuda" and features.is_cuda:
            from ...ops.cuda.mbe.ace import ictd_symmetric_contraction as cuda_kernel

            return cuda_kernel.project(self.projection, features)
        return torch_kernel.project(self.projection, features)

"""Channel-wise contractions of Cartesian O(3) tensors."""

from ..o3.contraction import AsymmetricContraction as SphericalContraction
from ..o3.paths import symmetric_contraction_paths
from .ictd import TensorProduct
from .irreps import Irreps


class AsymmetricContraction(SphericalContraction):
    """Contract independent Cartesian STF factors with native ICTD tensor products.

    Parameters
    ----------
    irreps_in, irreps_out : co3.Irreps or str
        Cartesian blocks with a common channel count in flat mul_ir layout.
    correlation : int
        Maximum product order.
    path_mode : {"sum", "expand"}, optional
        Sum or keep the independently weighted paths.
    backend : {None, "torch", "cuda"}, optional
        Cartesian tensor product backend.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co3
    >>> op = co3.AsymmetricContraction("2x1o", "2x1e", 2)
    >>> op((torch.randn(3, 6), torch.randn(3, 6)), torch.ones(op.weight_numel)).shape
    torch.Size([3, 6])
    """

    irreps_type = Irreps
    tensor_product = TensorProduct


class SymmetricContraction(AsymmetricContraction):
    """Contract powers of one Cartesian STF feature. Parameters match AsymmetricContraction.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co3
    >>> op = co3.SymmetricContraction("2x1o", "2x0e", 2)
    >>> op(torch.randn(3, 6), torch.ones(op.weight_numel)).shape
    torch.Size([3, 2])
    """

    paths = staticmethod(symmetric_contraction_paths)

    def forward(self, features, weight):
        return super().forward((features,) * self.correlation, weight)

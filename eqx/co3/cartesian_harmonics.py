################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math

import torch

from .irreps import Irrep, IrrepLike, Irreps
from .projector import project


def _normalization(degree: int) -> float:
    if degree == 0:
        return 1.0
    return math.prod(range(1, 2 * degree, 2)) / math.factorial(degree)


class CartesianHarmonics(torch.nn.Module):
    """Cartesian harmonics stored as symmetric traceless tensors.

    Args:
        lmax: Maximum tensor degree.
        irreps_in: Vector type, either polar ``1o`` or axial ``1e``.
        normalize: Normalize each input vector before constructing harmonics.
    """

    def __init__(
        self,
        lmax: int,
        *,
        irreps_in: IrrepLike = "1o",
        normalize: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(lmax, int) or isinstance(lmax, bool) or lmax < 0:
            raise ValueError("lmax must be a non-negative integer.")
        self.lmax = lmax
        self.irreps_in = Irrep(irreps_in)
        if self.irreps_in.l != 1:
            raise ValueError("CartesianHarmonics input must be 1o or 1e.")
        if not isinstance(normalize, bool):
            raise TypeError("normalize must be a bool.")
        self.normalize = normalize
        self.irreps_out = Irreps(
            [
                (1, Irrep(degree, self.irreps_in.p**degree))
                for degree in range(lmax + 1)
            ]
        )

    def forward(self, vectors: torch.Tensor) -> torch.Tensor:
        if vectors.is_complex():
            raise TypeError("CartesianHarmonics supports real inputs only.")
        if vectors.ndim < 1 or vectors.shape[-1] != 3:
            raise ValueError("vectors must have trailing shape (3,).")
        if self.normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=-1)

        tensor = vectors.new_ones(vectors.shape[:-1] + (1, 1))
        outputs = [tensor]
        vector = vectors.unsqueeze(-1)
        for degree in range(1, self.lmax + 1):
            tensor = torch.einsum("...ic,...jc->...ijc", tensor, vector).reshape(
                vectors.shape[:-1] + (3**degree, 1)
            )
            outputs.append(project(tensor, degree) * _normalization(degree))
        return torch.cat(outputs, dim=-2).squeeze(-1)

    def extra_repr(self) -> str:
        return (
            f"irreps_in={self.irreps_in}, irreps_out={self.irreps_out}, "
            f"normalize={self.normalize}"
        )

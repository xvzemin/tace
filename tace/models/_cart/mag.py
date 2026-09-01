################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from collections.abc import Mapping
from numbers import Real

import torch

from eqx import co3

from ..radial import MagneticChebyshevBasis


def _element_scales(
    magmoms_norm_by_element,
    atomic_numbers: list[int],
) -> torch.Tensor:
    if isinstance(magmoms_norm_by_element, Mapping):
        values = []
        for atomic_number in atomic_numbers:
            value = magmoms_norm_by_element.get(
                atomic_number,
                magmoms_norm_by_element.get(str(atomic_number)),
            )
            if value is None:
                raise ValueError(
                    f"magmoms_norm_by_element is missing atomic number {atomic_number}."
                )
            values.append(float(value))
    elif isinstance(magmoms_norm_by_element, Real):
        values = [float(magmoms_norm_by_element)] * len(atomic_numbers)
    elif magmoms_norm_by_element is None:
        raise ValueError("magmoms_norm_by_element is required for magnetic features.")
    else:
        values = [float(value) for value in magmoms_norm_by_element]
    if len(values) != len(atomic_numbers):
        raise ValueError("magmoms_norm_by_element must contain one value per element.")
    values = torch.tensor(values, dtype=torch.get_default_dtype())
    if not torch.isfinite(values).all() or (values < 0).any():
        raise ValueError("Magnetic-moment scales must be finite and non-negative.")
    return values


class MagneticBasis(torch.nn.Module):
    """Build radial and Cartesian angular features of axial magnetic moments."""

    a = 1.2
    b = 0.1

    def __init__(
        self,
        magmoms_norm_by_element,
        *,
        num_basis: int,
        lmax: int,
        atomic_numbers: list[int],
        normalize: bool = False,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        scale = 1.0 / (
            self.a * _element_scales(magmoms_norm_by_element, atomic_numbers) + self.b
        )
        if trainable:
            self.scale = torch.nn.Parameter(scale)
        else:
            self.register_buffer("scale", scale)
        self.num_basis = num_basis
        self.normalize = normalize
        self.radial_basis = MagneticChebyshevBasis(
            num_basis=num_basis,
            include_constant=True,
        )
        self.angular_basis = co3.CartesianHarmonics(
            lmax,
            irreps_in="1e",
            normalize=False,
            normalization="integral",
        )
        self.irreps_out = self.angular_basis.irreps_out

    def forward(
        self,
        magnetic_moments: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = self.scale[node_attrs.argmax(dim=-1)].unsqueeze(-1)
        norm = magnetic_moments.norm(dim=-1, keepdim=True)
        radial_coordinate = (
            1.0 - 2.0 * torch.clamp(norm * scale, min=0.0, max=1.0).square()
        )
        radial = self.radial_basis(radial_coordinate)
        vectors = magnetic_moments * scale if self.normalize else magnetic_moments
        return radial, self.angular_basis(vectors)


__all__ = ["MagneticBasis"]

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from collections.abc import Mapping
from numbers import Real

import torch
from e3nn import o3

from ..time_reversal import spherical_harmonics_irreps
from ..radial import MagneticChebyshevBasis
from ..angular import SolidHarmonics
from .fused import uuuTensorProduct


class MagneticBasis(torch.nn.Module):
    """Construct radial, node, and edge magnetic representations."""

    a = 1.2
    b = 0.1

    def __init__(
        self,
        scale,
        num_mag_radial_basis: int,
        Lmax: int,
        atomic_numbers: list[int],
        time_reversal: bool = False,
        angular_normalization: str = "element",
        radial_normalization: str = "rational",
    ) -> None:
        super().__init__()

        if not isinstance(Lmax, int) or Lmax < 0:
            raise ValueError("Lmax must be a non-negative integer.")
        if not isinstance(num_mag_radial_basis, int) or num_mag_radial_basis < 1:
            raise ValueError("num_mag_radial_basis must be a positive integer.")
        if angular_normalization not in ("integral", "component", "element"):
            raise ValueError(
                "angular_normalization must be 'integral', 'component', or 'element'."
            )
        if radial_normalization not in ("rational", "clamp"):
            raise ValueError("radial_normalization must be 'clamp' or 'rational'.")

        self.Lmax = Lmax
        self.num_mag_radial_basis = num_mag_radial_basis
        self.angular_normalization = angular_normalization
        self.radial_normalization = radial_normalization
        self.register_buffer(
            "scale",
            1.0 / (self.a * self._resolve_scale(scale, atomic_numbers) + self.b),
        )
        self.radial_basis = MagneticChebyshevBasis(
            num_basis=num_mag_radial_basis,
            include_constant=False,
        )
        self.magnetic_node_irreps_out = spherical_harmonics_irreps(
            Lmax,
            p=1,
            time_reversal=-1 if time_reversal else 1,
        ).regroup()
        self.angular_basis = SolidHarmonics(
            self.magnetic_node_irreps_out,
            normalization=(
                "component"
                if angular_normalization == "element"
                else angular_normalization
            ),
        )

        magnetic_edge_irrep_list = []
        for _, ir1 in self.magnetic_node_irreps_out:
            for _, ir2 in self.magnetic_node_irreps_out:
                for ir_out in ir1 * ir2:
                    if (
                        ir_out.l <= Lmax
                        and ir_out not in magnetic_edge_irrep_list
                    ):
                        magnetic_edge_irrep_list.append(ir_out)
        magnetic_edge_irreps = o3.Irreps(
            [(1, ir) for ir in magnetic_edge_irrep_list]
        ).regroup()
        self.magnetic_edge_tensor_product = uuuTensorProduct(
            self.magnetic_node_irreps_out,
            self.magnetic_node_irreps_out,
            magnetic_edge_irreps,
            trainable=False,
        )
        self.magnetic_edge_irreps_out = (
            self.magnetic_edge_tensor_product.irreps_out.regroup()
        )

    @staticmethod
    def _resolve_scale(scale, atomic_numbers: list[int]) -> torch.Tensor:
        if isinstance(scale, Mapping):
            values = []
            for atomic_number in atomic_numbers:
                value = scale.get(atomic_number, scale.get(str(atomic_number)))
                if value is None:
                    raise ValueError(f"scale is missing atomic number {atomic_number}")
                values.append(float(value))
        elif isinstance(scale, Real):
            values = [float(scale)] * len(atomic_numbers)
        else:
            raise TypeError("scale must be a scalar or an element-dependent mapping")

        scale_tensor = torch.tensor(values, dtype=torch.get_default_dtype())
        if not torch.isfinite(scale_tensor).all() or (scale_tensor < 0.0).any():
            raise ValueError("all scale values must be finite and non-negative")
        return scale_tensor

    def forward(
        self,
        initial_noncollinear_magmoms: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.scale[node_attrs.argmax(dim=-1)].unsqueeze(-1)
        scaled_magmoms = initial_noncollinear_magmoms * scale
        squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
        if self.radial_normalization == "clamp":
            radial_coordinate = 1.0 - 2.0 * torch.clamp(
                squared_magnitude,
                min=0.0,
                max=1.0,
            )
        else:
            radial_coordinate = 2.0 / (1.0 + squared_magnitude) - 1.0
        magnetic_radial_basis = self.radial_basis(radial_coordinate)
        if self.angular_normalization == "element":
            angular_input = scaled_magmoms
            angular_squared_magnitude = squared_magnitude
        else:
            angular_input = initial_noncollinear_magmoms
            angular_squared_magnitude = angular_input.square().sum(
                dim=-1, keepdim=True
            )
        bounded_angular_input = angular_input * torch.rsqrt(
            1.0 + angular_squared_magnitude
        )
        magnetic_node_attrs = self.angular_basis(bounded_angular_input)
        source, target = edge_index
        magnetic_edge_attrs = self.magnetic_edge_tensor_product(
            magnetic_node_attrs[target],
            magnetic_node_attrs[source],
        )

        return (
            magnetic_radial_basis,
            magnetic_node_attrs,
            magnetic_edge_attrs,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  scale={self.scale.tolist()},\n"
            f"  num_mag_radial_basis={self.num_mag_radial_basis},\n"
            f"  Lmax={self.Lmax},\n"
            f"  angular_normalization={self.angular_normalization!r},\n"
            f"  radial_normalization={self.radial_normalization!r},\n"
            f"  magnetic_node_irreps_out={self.magnetic_node_irreps_out},\n"
            f"  magnetic_edge_irreps_out={self.magnetic_edge_irreps_out}\n"
            ")"
        )

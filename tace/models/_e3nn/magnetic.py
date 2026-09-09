################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from collections.abc import Mapping
from numbers import Real

import torch
from e3nn import o3

from ..angular import SolidHarmonics
from ..radial import MagneticChebyshevBasis
from ..time_reversal import make_irrep, spherical_harmonics_irreps
from .fused import uuuTensorProduct


class MagneticBasis(torch.nn.Module):
    """Construct radial, node, and edge magnetic representations."""

    def __init__(
        self,
        magnetic_scale,
        num_mag_radial_basis: int,
        Lmax: int,
        atomic_numbers: list[int],
        time_reversal: bool = False,
        angular_normalization: str = "integral",
        radial_normalization: str = "clamp",
        use_spin_orbit_coupling: bool = True,
        parity: bool = True,
    ) -> None:
        super().__init__()

        if not isinstance(Lmax, int) or Lmax < 0:
            raise ValueError("Lmax must be a non-negative integer.")
        if not isinstance(num_mag_radial_basis, int) or num_mag_radial_basis < 1:
            raise ValueError("num_mag_radial_basis must be a positive integer.")
        if angular_normalization != "integral":
            raise ValueError("angular_normalization currently only supports 'integral'.")
        if radial_normalization != "clamp":
            raise ValueError("radial_normalization currently only supports 'clamp'.")

        self.Lmax = Lmax
        self.num_mag_radial_basis = num_mag_radial_basis
        self.angular_normalization = angular_normalization
        self.radial_normalization = radial_normalization
        self.use_spin_orbit_coupling = use_spin_orbit_coupling
        self.parity = parity
        self.register_buffer(
            "magnetic_scale",
            self._resolve_magnetic_scale(magnetic_scale, atomic_numbers),
        )
        self.radial_basis = MagneticChebyshevBasis(
            num_basis=num_mag_radial_basis,
            include_constant=False,
        )
        self.magnetic_node_irreps_out = spherical_harmonics_irreps(
            Lmax,
            p=1 if parity else -1,
            time_reversal=-1 if time_reversal else 1,
        ).regroup()
        self.angular_basis = SolidHarmonics(
            self.magnetic_node_irreps_out,
            normalization=angular_normalization,
        )

        if use_spin_orbit_coupling:
            magnetic_edge_irrep_list = []
            for _, ir1 in self.magnetic_node_irreps_out:
                for _, ir2 in self.magnetic_node_irreps_out:
                    for ir_out in ir1 * ir2:
                        if (
                            ir_out.l <= Lmax
                            and (parity or ir_out.p == (-1) ** ir_out.l)
                            and ir_out not in magnetic_edge_irrep_list
                        ):
                            magnetic_edge_irrep_list.append(ir_out)
            magnetic_edge_irreps = o3.Irreps(
                [(1, ir) for ir in magnetic_edge_irrep_list]
            ).regroup()
        else:
            magnetic_edge_irreps = o3.Irreps(
                [(1, make_irrep(0, 1, 1))]
            )
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
    def _resolve_magnetic_scale(
        magnetic_scale, atomic_numbers: list[int]
    ) -> torch.Tensor:
        if isinstance(magnetic_scale, Mapping):
            values = []
            for atomic_number in atomic_numbers:
                value = magnetic_scale.get(
                    atomic_number, magnetic_scale.get(str(atomic_number))
                )
                if value is None:
                    raise ValueError(
                        f"magnetic_scale is missing atomic number {atomic_number}"
                    )
                values.append(float(value))
        elif isinstance(magnetic_scale, Real):
            values = [float(magnetic_scale)] * len(atomic_numbers)
        else:
            raise TypeError(
                "magnetic_scale must be a scalar or an element-dependent mapping"
            )

        magnetic_scale = torch.tensor(values, dtype=torch.get_default_dtype())
        if not torch.isfinite(magnetic_scale).all() or (magnetic_scale <= 0.0).any():
            raise ValueError("all magnetic_scale values must be finite and positive")
        return magnetic_scale

    def forward(
        self,
        initial_noncollinear_magmoms: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        magnetic_scale = self.magnetic_scale[node_attrs.argmax(dim=-1)].unsqueeze(-1)
        scaled_magmoms = initial_noncollinear_magmoms / magnetic_scale
        squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
        radial_coordinate = 1.0 - 2.0 * torch.clamp(
            squared_magnitude,
            max=1.0,
        )
        magnetic_radial_basis = self.radial_basis(radial_coordinate)
        magnetic_node_attrs = self.angular_basis(initial_noncollinear_magmoms)
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
            f"  parity={self.parity},\n"
            f"  use_spin_orbit_coupling={self.use_spin_orbit_coupling},\n"
            f"  num_mag_radial_basis={self.num_mag_radial_basis},\n"
            f"  magnetic_node_irreps_out={self.magnetic_node_irreps_out},\n"
            f"  magnetic_edge_irreps_out={self.magnetic_edge_irreps_out}\n"
            f"  magnetic_scale=[{', '.join(f'{x:.4f}' for x in self.magnetic_scale.tolist())}],\n"
            # f"  Lmax={self.Lmax},\n"
            # f"  angular_normalization={self.angular_normalization!r},\n"
            # f"  radial_normalization={self.radial_normalization!r},\n"
            ")"
        )

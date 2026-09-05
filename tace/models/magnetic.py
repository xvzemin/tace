################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from collections.abc import Mapping
from numbers import Real
from typing import Union

import torch
from e3nn import o3

from ._e3nn.fused import uuuTensorProduct
from .angular import SolidHarmonics
from .radial import MagneticChebyshevBasis
from .time_reversal import spherical_harmonics_irreps


def _resolve_magmoms_scale_by_element(
    magmoms_scale_by_element,
    atomic_numbers: Union[list[int], None] = None,
    num_elements: Union[int, None] = None,
) -> torch.Tensor:
    """Resolve magnetic-moment scales to element order."""

    if isinstance(magmoms_scale_by_element, Mapping):
        if atomic_numbers is None:
            raise ValueError(
                "atomic_numbers are required when magmoms_scale_by_element is a mapping"
            )
        values = []
        for atomic_number in atomic_numbers:
            value = magmoms_scale_by_element.get(
                atomic_number,
                magmoms_scale_by_element.get(str(atomic_number)),
            )
            if value is None:
                raise ValueError(
                    f"magmoms_scale_by_element is missing atomic number {atomic_number}"
                )
            values.append(float(value))
    elif isinstance(magmoms_scale_by_element, Real):
        if num_elements is None:
            raise ValueError(
                "num_elements is required when magmoms_scale_by_element is a scalar"
            )
        values = [float(magmoms_scale_by_element)] * num_elements
    elif magmoms_scale_by_element is None:
        raise ValueError(
            "magmoms_scale_by_element must be provided for a magnetic interaction"
        )
    else:
        values = [float(value) for value in magmoms_scale_by_element]

    if num_elements is not None and len(values) != num_elements:
        raise ValueError(
            "magmoms_scale_by_element must contain "
            f"{num_elements} values, got {len(values)}"
        )
    magmoms_scale_by_element_tensor = torch.tensor(
        values, dtype=torch.get_default_dtype()
    )
    if (
        not torch.isfinite(magmoms_scale_by_element_tensor).all()
        or (magmoms_scale_by_element_tensor < 0.0).any()
    ):
        raise ValueError(
            "all magmoms_scale_by_element values must be finite and non-negative"
        )
    return magmoms_scale_by_element_tensor


class MagneticBasis(torch.nn.Module):
    """Construct radial, node, and edge magnetic representations."""

    a = 1.2
    b = 0.1

    def __init__(
        self,
        magmoms_scale_by_element,
        num_basis: int,
        Lmax: int,
        atomic_numbers: list[int],
        num_elements: int,
        time_reversal: bool = False,
        normalization: str = "element",
        radial_normalization: str = "rational",
    ) -> None:
        super().__init__()
        if not isinstance(Lmax, int) or isinstance(Lmax, bool) or Lmax < 0:
            raise ValueError("Lmax must be a non-negative integer.")
        self.num_basis = num_basis
        self.Lmax = Lmax
        if normalization not in ("integral", "component", "element"):
            raise ValueError(
                "normalization must be 'integral', 'component', or 'element'."
            )
        self.normalization = normalization
        if radial_normalization not in ("clamp", "rational"):
            raise ValueError("radial_normalization must be 'clamp' or 'rational'.")
        self.radial_normalization = radial_normalization

        self.register_buffer(
            "scale",
            1.0 / (
                self.a * _resolve_magmoms_scale_by_element(
                    magmoms_scale_by_element,
                    atomic_numbers=atomic_numbers,
                    num_elements=num_elements,
                ) + self.b
            ),
        )
        self.radial_basis = MagneticChebyshevBasis(
            num_basis=num_basis,
            include_constant=True,
        )
        self.magnetic_node_irreps_out = spherical_harmonics_irreps(
            Lmax,
            p=1,
            time_reversal=-1 if time_reversal else 1,
        ).regroup()
        self.angular_basis = SolidHarmonics(
            self.magnetic_node_irreps_out,
            normalization=(
                "integral" if normalization == "element" else normalization
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

    def forward(
        self,
        initial_noncollinear_magmoms: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        scale = self.scale[node_attrs.argmax(dim=-1)].unsqueeze(-1)
        scaled_magmoms = None
        if (
            self.radial_normalization == "rational"
            or self.normalization == "element"
        ):
            scaled_magmoms = initial_noncollinear_magmoms * scale
        if self.radial_normalization == "clamp":
            radial_coordinate = 1.0 - 2.0 * torch.clamp(
                initial_noncollinear_magmoms.norm(dim=-1, keepdim=True) * scale,
                min=0.0,
                max=1.0,
            ).square()
        else:
            assert scaled_magmoms is not None
            squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
            radial_coordinate = (1.0 - squared_magnitude) / (
                1.0 + squared_magnitude
            )
        magnetic_radial_basis = self.radial_basis(radial_coordinate)
        if self.normalization == "element":
            assert scaled_magmoms is not None
            magnetic_node_attrs = self.angular_basis(scaled_magmoms)
        else:
            magnetic_node_attrs = self.angular_basis(initial_noncollinear_magmoms)
        source, target = edge_index
        magnetic_edge_attrs = self.magnetic_edge_tensor_product(
            magnetic_node_attrs[target],
            magnetic_node_attrs[source],
        )

        return magnetic_radial_basis, magnetic_node_attrs, magnetic_edge_attrs

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(\n"
            f"  scale={self.scale.tolist()},\n"
            f"  num_basis={self.num_basis},\n"
            f"  Lmax={self.Lmax},\n"
            f"  normalization={self.normalization!r},\n"
            f"  radial_normalization={self.radial_normalization!r},\n"
            f"  magnetic_node_irreps_out={self.magnetic_node_irreps_out},\n"
            f"  magnetic_edge_irreps_out={self.magnetic_edge_irreps_out}\n"
            ")"
        )

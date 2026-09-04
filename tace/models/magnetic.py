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


def _resolve_magmoms_norm_by_element(
    magmoms_norm_by_element,
    atomic_numbers: Union[list[int], None] = None,
    num_elements: Union[int, None] = None,
) -> torch.Tensor:
    """Resolve magnetic-moment scales to element order."""

    if isinstance(magmoms_norm_by_element, Mapping):
        if atomic_numbers is None:
            raise ValueError(
                "atomic_numbers are required when magmoms_norm_by_element is a mapping"
            )
        values = []
        for atomic_number in atomic_numbers:
            value = magmoms_norm_by_element.get(
                atomic_number,
                magmoms_norm_by_element.get(str(atomic_number)),
            )
            if value is None:
                raise ValueError(
                    f"magmoms_norm_by_element is missing atomic number {atomic_number}"
                )
            values.append(float(value))
    elif isinstance(magmoms_norm_by_element, Real):
        if num_elements is None:
            raise ValueError(
                "num_elements is required when magmoms_norm_by_element is a scalar"
            )
        values = [float(magmoms_norm_by_element)] * num_elements
    elif magmoms_norm_by_element is None:
        raise ValueError(
            "magmoms_norm_by_element must be provided for a magnetic interaction"
        )
    else:
        values = [float(value) for value in magmoms_norm_by_element]

    if num_elements is not None and len(values) != num_elements:
        raise ValueError(
            "magmoms_norm_by_element must contain "
            f"{num_elements} values, got {len(values)}"
        )
    magmoms_norm_by_element_tensor = torch.tensor(
        values, dtype=torch.get_default_dtype()
    )
    if (
        not torch.isfinite(magmoms_norm_by_element_tensor).all()
        or (magmoms_norm_by_element_tensor < 0.0).any()
    ):
        raise ValueError(
            "all magmoms_norm_by_element values must be finite and non-negative"
        )
    return magmoms_norm_by_element_tensor


class MagneticBasis(torch.nn.Module):
    """Construct radial, node, and edge magnetic representations."""

    a = 1.2
    b = 0.1

    def __init__(
        self,
        magmoms_norm_by_element,
        num_basis: int,
        Lmax: int,
        atomic_numbers: list[int],
        num_elements: int,
        time_reversal: bool = False,
        normalize: bool = False,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(Lmax, int) or isinstance(Lmax, bool) or Lmax < 0:
            raise ValueError("Lmax must be a non-negative integer.")
        self.num_basis = num_basis
        self.Lmax = Lmax
        self.normalize = normalize

        if trainable:
            self.scale = torch.nn.Parameter(
                1.0 / (
                    self.a * _resolve_magmoms_norm_by_element(
                        magmoms_norm_by_element,
                        atomic_numbers=atomic_numbers,
                        num_elements=num_elements,
                    ) + self.b
                )
            )
        else:
            self.register_buffer(
                "scale",
                1.0 / (
                    self.a * _resolve_magmoms_norm_by_element(
                        magmoms_norm_by_element,
                        atomic_numbers=atomic_numbers,
                        num_elements=num_elements,
                    ) + self.b
                )
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
            # normalization="component",
            normalization="integral",
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
        magnetic_norm = initial_noncollinear_magmoms.norm(dim=-1, keepdim=True)
        magnetic_norm = (
            1.0
            - 2.0
            * torch.clamp(
                magnetic_norm * scale,
                min=0.0,
                max=1.0,
            ).square()
        )
        magnetic_radial_basis = self.radial_basis(magnetic_norm)
        magnetic_node_attrs = self.angular_basis(
            initial_noncollinear_magmoms * scale
            if self.normalize
            else initial_noncollinear_magmoms
        )
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
            f"  normalize={self.normalize},\n"
            f"  magnetic_node_irreps_out={self.magnetic_node_irreps_out},\n"
            f"  magnetic_edge_irreps_out={self.magnetic_edge_irreps_out}\n"
            ")"
        )

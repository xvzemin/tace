################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from collections.abc import Mapping
from numbers import Real
from typing import Union

import torch


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


class MagmomsNormalizer(torch.nn.Module):
    """
    See https://arxiv.org/abs/2604.08143.
    """

    a = 1.2
    b = 0.1

    def __init__(
        self,
        magmoms_norm_by_element,
        atomic_numbers: Union[list[int], None] = None,
        num_elements: Union[int, None] = None,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "magmoms_norm_by_element",
            _resolve_magmoms_norm_by_element(
                magmoms_norm_by_element,
                atomic_numbers=atomic_numbers,
                num_elements=num_elements,
            ),
        )

    def forward(
        self,
        initial_noncollinear_magmoms: torch.Tensor,
        node_attrs: torch.Tensor,
    ) -> torch.Tensor:
        magnetic_norm = initial_noncollinear_magmoms.norm(dim=-1, keepdim=True)
        effective_max = self.a * self.magmoms_norm_by_element + self.b
        element_max = effective_max[node_attrs.argmax(dim=-1)].unsqueeze(-1)
        return (
            1.0
            - 2.0
            * torch.clamp(
                magnetic_norm / element_max,
                min=0.0,
                max=1.0,
            ).square()
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"magmoms_norm_by_element={self.magmoms_norm_by_element.tolist()}, "
            f"a={self.a}, b={self.b})"
        )

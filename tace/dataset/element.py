################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, Iterable, List

import torch
from ase.data import (
    atomic_masses,
    atomic_names,
    atomic_numbers,
    chemical_symbols,
    covalent_radii,
    ground_state_magnetic_moments,
    vdw_radii,
)
from torch import Tensor

ATOMIC_SYMBOLS: List[str]
ATOMIC_NUMBERS: Dict[str, int]
ATOMIC_NAMES: List[str]
ATOMIC_MASSES: List[float]
ATOMIC_MAGMOM: List[float]
ATOMIC_VDW_RADII: List[float]
ATOMIC_COVALENT_RADII: List[float]

ATOMIC_SYMBOLS = chemical_symbols
ATOMIC_NUMBERS = atomic_numbers
ATOMIC_NAMES = atomic_names
ATOMIC_NAMES[0] = "Dummy"
ATOMIC_MASSES = atomic_masses.tolist()
ATOMIC_VDW_RADII = vdw_radii.tolist()
ATOMIC_COVALENT_RADII = (
    covalent_radii.tolist()
)  # if covalent_radii = 0.2, that means this element missing covalent_radii in ase
ATOMIC_MAGMOM = ground_state_magnetic_moments.tolist()  # ground state


class TorchElement:
    """Map atomic numbers to contiguous element indices.

    Parameters
    ----------
    zs : iterable of int
        Atomic numbers, sorted before assigning indices.

    Examples
    --------
    >>> elements = TorchElement([8, 1])
    >>> elements.z2idx(torch.tensor([8, 1])).tolist()
    [1, 0]
    """

    def __init__(self, zs: Iterable[int]):

        self.zs = sorted(zs)
        self.num_elements = len(self.zs)

        self.lookup_table = torch.full((max(self.zs) + 1,), -1, dtype=torch.int64)
        for idx, z in enumerate(self.zs):
            self.lookup_table[z] = idx

    def z2idx(self, z: Tensor) -> Tensor:
        return self.lookup_table.to(z.device)[z]

    def __len__(self) -> int:
        return len(self.zs)


def build_element_lookup(atomic_numbers: Iterable[int]) -> TorchElement:
    """Build an element table from the dataset's atomic numbers."""
    return TorchElement(atomic_numbers)

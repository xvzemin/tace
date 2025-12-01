################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict, List, Set, Tuple, Union

import torch
from torch import Tensor
import torch.nn.functional as F
from ase.data import (  # reference_states, # stored lattice constant, bond length, etc.
    atomic_masses,
    atomic_names,
    atomic_numbers,
    chemical_symbols,
    covalent_radii,
    ground_state_magnetic_moments,
    vdw_radii,
)

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


class Element:
    def __init__(self, zs: Union[List[int], Tuple[int, ...]]):
        if isinstance(zs, tuple):
            self.zs = list(zs)
        elif isinstance(zs, list):
            self.zs = zs
        else:
            raise TypeError(f"{zs} must be a list or tuple of integers.")
        if not all(isinstance(num, int) for num in self.zs):
            raise ValueError(f"All elements in {zs} must be integers.")
        self.atomic_numbers = sorted(zs)
        self.num_elements = len(zs)
        self.z_to_idx = {z: i for i, z in enumerate(self.atomic_numbers)}

        # # Torch
        # self.lookup_table = torch.full((max(self.atomic_numbers) + 1,), -1, dtype=torch.int64)
        # for idx, z in enumerate(self.atomic_numbers):
        #     self.lookup_table[z] = idx

    def z2idx(self, z: int) -> int:
        return self.z_to_idx[z]

    # def torch_z2idx(self, z: torch.Tensor) -> torch.Tensor:
    #     return self.lookup_table[z]

    # def torch_z2onehot(self, zs: torch.Tensor) -> torch.Tensor:
    #     """
    #     :param zs: shape (N,),
    #     :return: shape (N, num_classes), one-hot
    #     """
    #     idxs = self.lookup_table[zs]  # shape (N,)
    #     return F.one_hot(idxs, num_classes=self.num_elements)

    def idx2symbol(self, idx: int) -> int:
        z = self.idx2z(idx)
        return ATOMIC_SYMBOLS[z]

    def idx2mass(self, idx: int) -> float:
        z = self.idx2z(idx)
        return ATOMIC_MASSES[z]

    def idx2name(self, idx: int) -> str:
        z = self.idx2z(idx)
        return ATOMIC_NAMES[z]

    def idx2covalent_radii(self, idx: int) -> float:
        z = self.idx2z(idx)
        return ATOMIC_COVALENT_RADII[z]

    def idx2vdw_radii(self, idx: int) -> float:
        z = self.idx2z(idx)
        return ATOMIC_VDW_RADII[z]

    def idx2magmom(self, idx: int) -> float:
        z = self.idx2z(idx)
        return ATOMIC_MAGMOM[z]

    def __len__(self) -> int:
        return len(self.atomic_numbers)

    def __repr__(self):
        return f"Element in the model: {(z for z in self.atomic_numbers)}"


class TorchElement(Element):
    def __init__(self, zs: Union[List[int], Tuple[int, ...]]):

        super().__init__(zs)

        self.lookup_table = torch.full(
            (max(self.atomic_numbers) + 1,), -1, dtype=torch.int64
        )
        for idx, z in enumerate(self.atomic_numbers):
            self.lookup_table[z] = idx

    def z2idx(self, z: Tensor) -> Tensor:
        return self.lookup_table[z]

    def z2onehot(self, zs: Tensor) -> Tensor:
        """
        :param zs: shape (N,),
        :return: shape (N, num_classes), one-hot
        """
        idxs = self.lookup_table[zs]  # shape (N,)
        return F.one_hot(idxs, num_classes=self.num_elements)


def build_element_lookup(atomic_numbers: Set[int]) -> TorchElement:
    element = TorchElement(sorted(list(atomic_numbers)))
    return element

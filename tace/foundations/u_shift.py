################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import copy

import ase
import numpy as np
import torch

from tace.utils.env import get_tace_use_u_shift

WTC_U_SHIFT = {
    23: -2.271,  # V
    24: -2.458,  # Cr
    25: -1.602,  # Mn
    26: -1.925,  # Fe
    27: -2.004,  # Co
    28: -2.586,  # Ni
    42: -4.895,  # Mo
    74: -5.875,  # W
}


def apply_wtc_u_shift(atoms: ase.Atoms, energy: float, minus: bool = False) -> float:
    """Apply the energy correction of arXiv:2601.21056, Eq. (4).

    Parameters
    ----------
    atoms : ase.Atoms
        Structure calculated with the Materials Project selective +U scheme.
    energy : float
        Total energy in eV.
    minus : bool, optional
        Undo the correction when True. The default adds the negative offsets.

    Examples
    --------
    >>> apply_wtc_u_shift(ase.Atoms("Fe2O3"), -10.0)
    -13.85
    """
    numbers, counts = np.unique(atoms.numbers, return_counts=True)
    if not np.any((numbers == 8) | (numbers == 9)):
        return energy
    shift = sum(
        count * WTC_U_SHIFT.get(int(z), 0.0) for z, count in zip(numbers, counts)
    )
    return energy - shift if minus else energy + shift


def apply_u_shift(batch, energy):
    """Return corrected energies without changing the input graph or labels.

    Parameters
    ----------
    batch : mapping or torch_geometric.data.Data
        ``atomic_numbers`` and, for batches, the atom-to-graph ``batch`` index.
        O or F anywhere in a structure enables its metal offsets.
    energy : torch.Tensor
        Total energies, with shape ``(n_graphs,)`` or ``(n_graphs, 1)``.

    Examples
    --------
    >>> graph = {"atomic_numbers": torch.tensor([26, 8])}
    >>> apply_u_shift(graph, torch.tensor([-10.0], dtype=torch.float64))
    tensor([-11.9250], dtype=torch.float64)
    """
    numbers = batch["atomic_numbers"].to(device=energy.device, dtype=torch.long)
    indices = batch.get("batch")
    if indices is None:
        indices = torch.zeros_like(numbers)
    else:
        indices = indices.to(device=energy.device, dtype=torch.long)
    table = energy.new_tensor([WTC_U_SHIFT.get(z, 0.0) for z in range(119)])
    count = energy.numel()
    offsets = energy.new_zeros(count).index_add(0, indices, table[numbers])
    enabled = (
        torch.zeros(count, device=energy.device, dtype=torch.long).index_add(
            0, indices, ((numbers == 8) | (numbers == 9)).long()
        )
        > 0
    )
    return energy + (offsets * enabled).reshape_as(energy)


def shift_energy_labels(batch):
    """Return a shallow batch copy with online U-shifted energy labels.

    Parameters
    ----------
    batch : mapping or torch_geometric.data.Data
        Raw labels. The input is returned unchanged unless
        ``TACE_USE_U_SHIFT=1`` and an energy label is present.

    Examples
    --------
    >>> import os
    >>> os.environ["TACE_USE_U_SHIFT"] = "1"
    >>> batch = {"atomic_numbers": torch.tensor([26, 8]), "energy": torch.zeros(1)}
    >>> shifted = shift_energy_labels(batch)
    >>> batch["energy"].item()
    0.0
    >>> del os.environ["TACE_USE_U_SHIFT"]
    """
    if get_tace_use_u_shift() != "1" or batch.get("energy") is None:
        return batch
    shifted = copy.copy(batch)
    shifted["energy"] = apply_u_shift(batch, batch["energy"])
    return shifted

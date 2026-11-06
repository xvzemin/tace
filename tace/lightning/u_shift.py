"""Element reference shifts for oxygen- or fluorine-containing structures."""

import torch

from tace.utils.torch_scatter import scatter_sum

U_SHIFT = {
    23: -2.271, 24: -2.458, 25: -1.602, 26: -1.925,
    27: -2.004, 28: -2.586, 42: -4.895, 74: -5.875,
}


def apply_u_shift(batch, energy):
    """Add element reference shifts when a structure contains oxygen or fluorine."""
    numbers = batch["atomic_numbers"]
    table = energy.new_zeros(119)
    indices = torch.tensor(list(U_SHIFT), device=energy.device)
    table[indices] = energy.new_tensor(list(U_SHIFT.values()))
    count = batch["ptr"].numel() - 1
    shift = scatter_sum(table[numbers], batch["batch"], dim=0, dim_size=count)
    enabled = scatter_sum(
        ((numbers == 8) | (numbers == 9)).long(),
        batch["batch"], dim=0, dim_size=count,
    ) > 0
    return energy + shift * enabled

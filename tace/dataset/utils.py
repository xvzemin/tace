################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import numpy as np
import torch
from torch import Tensor


def default_value_for_rank0_atom(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms,))


def default_value_for_rank1_atom(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms, 3))


def default_value_for_rank1_edge(
    num_atoms: int, class_: str, *, num_edges: int, **kwargs
):
    return np.zeros((num_edges, 3))


def default_value_for_rank2_atom(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms, 3, 3))


def default_value_for_rank3_atom(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms, 3, 3, 3))


def default_value_for_rank4_atom(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms, 3, 3, 3, 3))


def default_value_for_rank0_graph(num_atoms: int, class_: str, **kwargs):
    return np.zeros((1,))


def default_value_for_rank1_graph(num_atoms: int, class_: str, **kwargs):
    return np.zeros((3,))


def default_value_for_rank2_graph(num_atoms: int, class_: str, **kwargs):
    return np.zeros((3, 3))


def default_value_for_rank3_graph(num_atoms: int, class_: str, **kwargs):
    return np.zeros((3, 3, 3))


def default_value_for_rank4_graph(num_atoms: int, class_: str, **kwargs):
    return np.zeros((3, 3, 3, 3))


def default_value_for_rank2_pair(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms * num_atoms, 3, 3))


def shape_fn_for_hessian(tensor: torch.Tensor, num_atoms: int) -> torch.Tensor:
    """Convert a full Hessian to row-major atom-pair blocks.

    Parameters
    ----------
    tensor : torch.Tensor
        Shape ``(3N, 3N)``, ``(9N**2,)``, ``(N, N, 3, 3)``, or ``(N**2, 3, 3)``.
    num_atoms : int
        Number of atoms.

    Returns
    -------
    torch.Tensor
        Contiguous blocks of shape ``(num_atoms**2, 3, 3)``, ordered by
        ``i * num_atoms + j``. Dtype and device are unchanged.
        Flat inputs use the row-major order of the ``(3N, 3N)`` matrix.
    """
    n = num_atoms
    if tensor.shape in {(3 * n, 3 * n), (9 * n * n,)}:
        tensor = tensor.reshape(n, 3, n, 3).permute(0, 2, 1, 3)
    elif tensor.shape not in {(n, n, 3, 3), (n * n, 3, 3)}:
        raise ValueError(
            f"Invalid Hessian shape {tuple(tensor.shape)} for {n} atoms; expected "
            f"{(3 * n, 3 * n)}, {(9 * n * n,)}, {(n, n, 3, 3)}, or {(n * n, 3, 3)}."
        )
    return tensor.reshape(n * n, 3, 3).contiguous()


def hessian_pair_indices(ptr: torch.Tensor) -> torch.Tensor:
    """Return all ordered atom pairs within each structure, including self pairs."""
    counts = ptr[1:] - ptr[:-1]
    sizes = counts.square()
    offsets = sizes.cumsum(0) - sizes
    graph = torch.repeat_interleave(
        torch.arange(counts.numel(), device=ptr.device), sizes
    )
    pair = torch.arange(graph.numel(), device=ptr.device) - offsets[graph]
    source = ptr[graph] + torch.div(pair, counts[graph], rounding_mode="floor")
    target = ptr[graph] + pair.remainder(counts[graph])
    return torch.stack((source, target))


def shape_fn_for_abs_fc_mag(t: Tensor, num_atoms: int, **kwargs):
    return torch.abs(t)


def voigt_to_matrix(t: Tensor, **kwargs):
    """
    Convert voigt notation to matrix notation
    """
    if t.shape == (3, 3):
        return t
    if t.shape == (6,):
        return torch.tensor(
            [
                [t[0], t[5], t[4]],
                [t[5], t[1], t[3]],
                [t[4], t[3], t[2]],
            ],
            dtype=t.dtype,
        )
    if t.shape == (9,):
        return t.view(3, 3)

    raise ValueError(
        f"Stress tensor must be of shape (6,) or (3, 3), or (9,) but has shape {t.shape}"
    )

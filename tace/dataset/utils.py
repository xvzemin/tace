################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import numpy as np
import torch
from torch import Tensor


def default_value_for_rank0_atom(num_atoms: int, class_: str,  **kwargs):
    return np.zeros((num_atoms,))


def default_value_for_rank1_atom(num_atoms: int, class_: str,  **kwargs):
    return np.zeros((num_atoms, 3))


def default_value_for_rank2_atom(num_atoms: int, class_: str,  **kwargs):
    return np.zeros((num_atoms, 3, 3))

def default_value_for_rank3_atom(num_atoms: int, class_: str,  **kwargs):
    return np.zeros((num_atoms, 3, 3, 3))

def default_value_for_rank4_atom(num_atoms: int, class_: str,  **kwargs):
    return np.zeros((num_atoms, 3, 3, 3, 3))

def default_value_for_rank0_graph(num_atoms: int, class_: str, **kwargs):
    return np.zeros((1,))


def default_value_for_rank1_graph(num_atoms: int,  class_: str, **kwargs):
    return np.zeros((3,))


def default_value_for_rank2_graph(num_atoms: int,  class_: str, **kwargs):
    return np.zeros((3, 3))


def default_value_for_rank3_graph(num_atoms: int,  class_: str, **kwargs):
    return np.zeros((3, 3, 3))


def default_value_for_rank4_graph(num_atoms: int,  class_: str, **kwargs):
    return np.zeros((3, 3, 3, 3))


def default_value_for_hessians(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms * num_atoms, 3, 3))


def shape_fn_for_hessians(t: Tensor, num_atoms: int, **kwargs):
    assert t.ndim == num_atoms * num_atoms * 3 * 3, "hessians shape is not correct"
    t = (
        t.reshape(num_atoms, 3, num_atoms, 3)
        .permute(0, 2, 1, 3)
        .reshape(num_atoms * num_atoms, 3, 3)
    )
    return t


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

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


def default_value_for_hessian(num_atoms: int, class_: str, **kwargs):
    return np.zeros((num_atoms * 3 * num_atoms * 3))


def shape_fn_for_hessian(t: Tensor, num_atoms: int, **kwargs):
    # assert np.allclose(t, t.T, atol=1e-6)
    shape1 = (num_atoms * 3, num_atoms * 3)
    shape2 = num_atoms * 3 * num_atoms * 3
    assert t.shape == shape1 or t.shape == shape2, (
        f"hessian shape mismatch: got {tuple(t.shape)}, expected {shape1} or {shape2}"
    )
    return t.view(-1)


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

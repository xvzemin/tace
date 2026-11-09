################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass, field
from typing import Dict, List, Optional

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


# per-edge follows edge_index; per-pair contains all ordered (i, j), including i=j.
PROPERTIES = {
    "fidelity_idx": {
        "ase_name": None,
        "type": "int",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": 1,
        "abbreviation": "F_IDX",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "energy": {
        "ase_name": "energy",
        "type": "float",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": 1,
        "abbreviation": "E",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "forces": {
        "ase_name": "forces",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1o",
        "time_reversal": 1,
        "abbreviation": "F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "requires_periodic_cell": False,
        "must_be_with": ["energy"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["positions"],
    },
    "edge_forces": {
        "ase_name": None,
        "type": "float",
        "scope": "per-edge",
        "rank": 1,
        "irreps": "1x1o",
        "time_reversal": 1,
        "abbreviation": "EDGE_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_edge,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["edge_vector"],
    },
    "direct_forces": {
        "ase_name": "forces",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1o",
        "time_reversal": 1,
        "abbreviation": "D_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "hessian": {
        "ase_name": None,
        "type": "float",
        "scope": "per-pair",
        "rank": 2,
        "irreps": "1x0e+1x1e+1x2e",
        "time_reversal": 1,
        "abbreviation": "HESSIAN",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": shape_fn_for_hessian,
        },
        "default_value_fn": default_value_for_rank2_pair,
        "requires_periodic_cell": False,
        "must_be_with": ["energy", "forces"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ["positions"],
    },
    "direct_hessian": {
        "ase_name": None,
        "type": "float",
        "scope": "per-pair",
        "rank": 2,
        "irreps": "1x0e+1x1e+1x2e",
        "time_reversal": 1,
        "abbreviation": "D_H",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": shape_fn_for_hessian,
        },
        "default_value_fn": default_value_for_rank2_pair,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "stress": {
        "ase_name": "stress",
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "requires_periodic_cell": True,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": [],  # manual set
    },
    "direct_stress": {
        "ase_name": "stress",
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "D_S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "requires_periodic_cell": True,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "virials": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": [],  # manual set
    },
    "direct_virials": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "D_V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "atomic_stresses": {
        "ase_name": "stresses",
        "type": "float",
        "scope": "per-atom",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "A_S",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "requires_periodic_cell": True,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["edge_vector"],
    },
    "atomic_virials": {
        "ase_name": None,
        "type": "float",
        "scope": "per-atom",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "A_V",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["edge_vector"],
    },
    "direct_dipole": {
        "ase_name": "dipole",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "time_reversal": 1,
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "requires_periodic_cell": False,
        "must_be_with": ["charges"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_dipole": {
        "ase_name": "dipole",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "time_reversal": 1,
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["electric_field"],
    },
    "polarization": {
        "ase_name": "polarization",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "time_reversal": 1,
        "abbreviation": "P",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "requires_periodic_cell": True,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["electric_field"],
    },
    "direct_polarizability": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_polarizability": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "requires_periodic_cell": False,
        "must_be_with": ["direct_dipole", "conservative_dipole", "polarization"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ["electric_field"],
    },
    "born_effective_charges": {
        "ase_name": "born_effective_charges",
        "type": "float",
        "scope": "per-atom",
        "rank": 2,
        "irreps": "1x2e",
        "time_reversal": 1,
        "abbreviation": "BEC",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "requires_periodic_cell": False,
        "must_be_with": ["direct_dipole", "conservative_dipole", "polarization"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ["electric_field", "positions"],
    },
    "charges": {
        "ase_name": "charges",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": 1,
        "abbreviation": "C",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "requires_periodic_cell": False,
        "must_be_with": ["total_charge"],
        "enable_prediction": True,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "total_charge": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": 1,
        "abbreviation": "TC",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "initial_collinear_magmoms": {
        "ase_name": "initial_magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": -1,
        "abbreviation": "I_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "final_collinear_magmoms": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": -1,
        "abbreviation": "F_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "abs_final_collinear_magmoms": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": 1,
        "abbreviation": "ABS_F_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": shape_fn_for_abs_fc_mag,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "collinear_magnetic_forces": {
        "ase_name": None,
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": -1,
        "abbreviation": "C_MAG_F",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "requires_periodic_cell": False,
        "must_be_with": ["initial_collinear_magmoms"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["initial_collinear_magmoms"],
    },
    "total_collinear_magmom": {
        "ase_name": "magmom",
        "type": "float",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",
        "time_reversal": -1,
        "abbreviation": "TCM",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "initial_noncollinear_magmoms": {
        "ase_name": "initial_magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1e",
        "time_reversal": -1,
        "abbreviation": "I_NC_MAG",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "final_noncollinear_magmoms": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1e",
        "time_reversal": -1,
        "abbreviation": "F_NC_MAG",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "noncollinear_magnetic_forces": {
        "ase_name": None,
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1e",
        "time_reversal": -1,
        "abbreviation": "NC_MAG_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "requires_periodic_cell": False,
        "must_be_with": ["initial_noncollinear_magmoms"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["initial_noncollinear_magmoms"],
    },
    "total_noncollinear_magmom": {
        "ase_name": "magmom",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1e",
        "time_reversal": -1,
        "abbreviation": "TNCM",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "electric_field": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "time_reversal": 1,
        "abbreviation": "EF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "magnetic_field": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1e",
        "time_reversal": -1,
        "abbreviation": "MF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "requires_periodic_cell": False,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
}

SUPPORTED_PREDICTION_PROPERTIES = tuple(
    name for name, config in PROPERTIES.items() if config["enable_prediction"]
)
SUPPORTED_EMBEDDING_PROPERTIES = tuple(
    name for name, config in PROPERTIES.items() if config["enable_embedding"]
)
DIRECT_PROPERTIES = tuple(
    name
    for name, config in PROPERTIES.items()
    if not (config["first_derivative"] or config["second_derivative"])
)
TIME_ODD_PROPERTIES = tuple(
    name for name, config in PROPERTIES.items() if config["time_reversal"] == -1
)
KEYS = {f"{k}_key": k for k in PROPERTIES}


@dataclass
class KeySpecification:
    """Map model properties to source metadata and array keys.

    Parameters
    ----------
    info_keys : dict
        Graph, edge and pair property names mapped to metadata keys.
    arrays_keys : dict
        Atom property names mapped to array keys.

    Examples
    --------
    >>> keys = KeySpecification.from_defaults()
    >>> keys.update(info_keys={"energy": "dft_energy"}).info_keys["energy"]
    'dft_energy'
    """

    info_keys: Dict[str, str] = field(default_factory=dict)
    arrays_keys: Dict[str, str] = field(default_factory=dict)

    def update(
        self,
        info_keys: Optional[Dict[str, str]] = None,
        arrays_keys: Optional[Dict[str, str]] = None,
    ):
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    @classmethod
    def from_defaults(cls):
        return update_keyspec_from_kwargs(cls(), KEYS)


def update_keyspec_from_kwargs(
    keyspec: KeySpecification, keydict: Dict[str, str]
) -> KeySpecification:
    infos = [f"{k}_key" for k, v in PROPERTIES.items() if (v["scope"] != "per-atom")]
    arrays = [f"{k}_key" for k, v in PROPERTIES.items() if (v["scope"] == "per-atom")]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]
    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec


def get_target_property(cfg: Dict) -> List[str]:
    loss_property = cfg["loss"].get("loss_property", None)
    assert isinstance(loss_property, list)
    assert set(loss_property).issubset(list(PROPERTIES))
    return loss_property


def get_embedding_property(cfg: Dict) -> List[str]:
    embedding_property = []
    for p, v in cfg["model"]["config"].get("universal_embedding", {}).items():
        if v["enable"]:
            embedding_property.append(p)
    for target in get_need_property(get_target_property(cfg), training=True):
        required = (
            PROPERTIES[target]["must_be_with"]
            + PROPERTIES[target]["requires_grad_with"]
        )
        for p in required:
            if (
                p in PROPERTIES
                and PROPERTIES[p]["enable_embedding"]
                and p not in embedding_property
            ):
                embedding_property.append(p)
    atomic_basis_type = cfg["model"]["config"].get("atomic_basis", {}).get("type")
    if isinstance(atomic_basis_type, str):
        atomic_basis_type = [atomic_basis_type]
    if atomic_basis_type is not None and any(
        interaction in ("uv_o2_mag", "o2_mag") for interaction in atomic_basis_type
    ):
        if "initial_noncollinear_magmoms" not in embedding_property:
            embedding_property.append("initial_noncollinear_magmoms")
    return embedding_property


def get_need_property(
    target_property: Optional[List[str]] = None,
    embedding_property: Optional[List[str]] = None,
    training: bool = False,
) -> List[str]:
    """Return the deterministic transitive closure of required properties."""
    requested = list(embedding_property or [])
    if training:
        requested.extend(target_property or [])
    result, active = [], set()

    def visit(name):
        if name in result:
            return
        if name not in PROPERTIES:
            raise ValueError(f"Unknown property: {name}")
        if name in active:
            raise ValueError(f"Cyclic property dependency involving {name}")
        active.add(name)
        config = PROPERTIES[name]
        for dependency in config["must_be_with"] + config["requires_grad_with"]:
            if dependency in PROPERTIES:
                visit(dependency)
        active.remove(name)
        result.append(name)

    for name in requested:
        visit(name)
    return result


# For Metrics

SPECIAL_METRIC_PROPERTIES = (
    "polarization",  # for definition
    "abs_final_collinear_magmoms",  # for absolute value
    "direct_forces",  # for DeNS
    "forces",  # for DeNS
)

# INTENSIVE
EXTENSIVE_PROPERTIES = ("stress", "direct_stress")
MAE_PROPERTIES = tuple(
    p for p in SUPPORTED_PREDICTION_PROPERTIES if p not in SPECIAL_METRIC_PROPERTIES
)
RMSE_PROPERTIES = tuple(
    p for p in SUPPORTED_PREDICTION_PROPERTIES if p not in SPECIAL_METRIC_PROPERTIES
)
MAE_PER_ATOM_PROPERTIES = tuple(
    p
    for p, v in PROPERTIES.items()
    if p not in SPECIAL_METRIC_PROPERTIES
    and v["scope"] == "per-system"
    and p not in EXTENSIVE_PROPERTIES
)
RMSE_PER_ATOM_PROPERTIES = tuple(
    p
    for p, v in PROPERTIES.items()
    if p not in SPECIAL_METRIC_PROPERTIES
    and v["scope"] == "per-system"
    and p not in EXTENSIVE_PROPERTIES
)


fields = {f"compute_{k}": False for k, v in PROPERTIES.items()}


@dataclass
class ComputeFlag:
    __annotations__ = {k: bool for k in fields}
    locals().update(fields)

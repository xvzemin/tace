################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from copy import deepcopy
from typing import List, Optional

import ase
import numpy as np
import torch
from torch_geometric.data import Data

from .element import TorchElement
from .neighbour_list import get_neighborhood
from .quantity import (
    PROPERTY,
    KeySpecification,
    get_need_property,
)


def to_tensor(data):
    if data is None:
        raise ValueError(f"{data} must not be None")
    return torch.tensor(data, dtype=torch.get_default_dtype())


def build_adjacency_matrix_from_edge(edge_src, edge_dst, num_atoms):
    adjacency_matrix = torch.zeros((num_atoms, num_atoms), dtype=torch.int64)
    for source, target in zip(edge_src, edge_dst):
        adjacency_matrix[source, target] = 1
    return adjacency_matrix.reshape(1, num_atoms, num_atoms)


def from_atoms(
    element: TorchElement,
    atoms: ase.Atoms,
    cutoff: float,
    max_neighbors: Optional[int],
    keyspec: KeySpecification,
    target_property: List[str],
    embedding_property: List[str],
    training: bool = True,
    neighborlist_backend: str = "matscipy",
    **kwargs,
):
    # === The basic structure of chemical substances ===
    atomic_numbers = atoms.get_atomic_numbers()
    pbc = tuple(atoms.get_pbc())
    lattice = np.array(atoms.get_cell())
    if not np.isfinite(lattice).all():
        raise ValueError("The cell must contain finite values.")
    volume = abs(np.linalg.det(lattice))
    if all(pbc) and volume == 0:
        raise ValueError("A fully periodic structure requires a non-singular cell.")
    stress_defined = any(pbc) and volume > 0
    positions = atoms.get_positions()
    edge_index, edge_shifts, pbc, lattice = get_neighborhood(
        positions=positions,
        cutoff=cutoff,
        pbc=deepcopy(pbc),
        lattice=deepcopy(lattice),
        max_neighbors=max_neighbors,
        backend=neighborlist_backend,
    )
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
    onehot = element.z2onehot(atomic_numbers).to(dtype=torch.get_default_dtype())
    num_atoms = len(atomic_numbers)

    # === Physical property to predict ===
    properties = {}
    property_weights = {}

    for name in list(keyspec.arrays_keys) + list(keyspec.info_keys):
        property_weights[name] = atoms.info.get(f"{name}_weight", 1.0)

    for name, atoms_key in keyspec.info_keys.items():
        properties[name] = atoms.info.get(atoms_key, None)
        if properties[name] is None:
            property_weights[name] = 0.0

    for name, atoms_key in keyspec.arrays_keys.items():
        properties[name] = atoms.arrays.get(atoms_key, None)
        if properties[name] is None:
            property_weights[name] = 0.0

    lattice = (
        torch.tensor(lattice.reshape((1, 3, 3)), dtype=torch.get_default_dtype())
        if lattice is not None
        else torch.tensor(3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()).view(
            1, 3, 3
        )
    )

    pDict = {}
    wDict = {}
    masks = {}
    need_property = get_need_property(target_property, embedding_property, training)
    for name in need_property:
        in_data = PROPERTY[name]["shape"]["in_data"]
        shape_fn = PROPERTY[name]["shape"].get("shape_fn", None)
        default_value_fn = PROPERTY[name]["default_value_fn"]
        type_ = PROPERTY[name]["type"]
        try:
            p = properties.get(name)
            if p is None:
                if type_ == "float":
                    p = torch.tensor(
                        default_value_fn(num_atoms, type_),
                        dtype=torch.get_default_dtype(),
                    )
                elif type_ == "int":
                    p = torch.tensor(
                        np.round(default_value_fn(num_atoms, type_)),
                        dtype=torch.int64,
                    )
                else:
                    raise TypeError(f"Bug: Check {p}'s type in tace.dataset.quantity")
            else:
                if type_ == "float":
                    p = torch.tensor(p, dtype=torch.get_default_dtype())
                elif type_ == "int":
                    p = torch.tensor(np.round(p), dtype=torch.int64)
                else:
                    raise
                if shape_fn is not None:
                    p = shape_fn(
                        p,
                        num_atoms=num_atoms,
                    )
            p = p.view(*in_data)
            if type_ == "float":
                if torch.isinf(p).any():
                    raise ValueError(f"{name} contains infinite values.")
                valid = torch.isfinite(p) & (properties.get(name) is not None)
                if name in {"stress", "direct_stress", "virials", "direct_virials"}:
                    if name in {"stress", "direct_stress"} and not stress_defined:
                        if valid.any() and property_weights.get(name, 1.0) > 0:
                            raise ValueError(
                                "Stress labels require a periodic direction and "
                                "a non-singular cell; use stress_weight=0 to ignore them."
                            )
                        valid = torch.zeros_like(valid)
                    masks[f"{name}_mask"] = valid
                    if not valid.any():
                        property_weights[name] = 0.0
                elif not valid.all():
                    property_weights[name] = 0.0
                p = torch.where(valid, p, torch.zeros_like(p))
            pDict.update({name: p})
        except Exception as e:
            raise RuntimeError(f"Failed to read property {name}") from e

        try:
            w = (
                torch.tensor(
                    property_weights.get(name), dtype=torch.get_default_dtype()
                ).view(1)  # (1,)
                if property_weights.get(name) is not None
                else torch.tensor(1.0, dtype=torch.get_default_dtype())  # ()
            )
            wDict.update({name: w})
        except Exception as e:
            raise RuntimeError(f"Failed to read property {name}") from e

    data_dict = {
        "entropy": to_tensor(atoms.info.get("entropy", 1.0)),
        "atomic_numbers": atomic_numbers,
        "lattice": lattice,
        "pbc": torch.tensor([pbc], dtype=torch.bool),
        "positions": to_tensor(positions),
        "node_attrs": onehot,
        "edge_index": torch.tensor(edge_index, dtype=torch.int64),
        "edge_shifts": to_tensor(edge_shifts),
        "fidelity_idx": torch.tensor(
            atoms.info.get(keyspec.info_keys["fidelity_idx"], 0), dtype=torch.int64
        ),
    }
    data_dict.update(masks)

    for name in need_property:
        data_dict.update(
            {
                name: pDict[name],
                f"{name}_weight": wDict[name],
            }
        )
    return Data(**data_dict)

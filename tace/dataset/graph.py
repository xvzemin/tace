################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from __future__ import annotations
from copy import deepcopy
from typing import Optional, List, Dict


import ase
import numpy as np
import torch
from torch_geometric.data import Data


from .element import TorchElement
from .neighbour_list import get_neighborhood
from .quantity import (
    KeySpecification,
    PROPERTY,
    UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY,
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
    embedding_property: List[str] = [],
    universal_embedding: Optional[List[Dict[str, Dict[int, str]]]] = None,
    training: bool = True,
):
    # === The basic structure of chemical substances ===
    try:
        atomic_numbers = atoms.get_atomic_numbers()
    except Exception as e:
        raise RuntimeError(f"Failed to get atomic numbers from atoms: {e}")

    try:
        pbc = tuple(atoms.get_pbc())
    except Exception as e:
        raise RuntimeError(f"Failed to get pbc from atoms: {e}")

    try:
        lattice = np.array(atoms.get_cell())
    except Exception as e:
        raise RuntimeError(f"Failed to get cell from atoms: {e}")

    try:
        positions = atoms.get_positions()
    except Exception as e:
        raise RuntimeError(f"Failed to get positions from atoms: {e}")

    try:
        edge_index, edge_shifts, pbc, lattice = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=deepcopy(pbc),
            lattice=deepcopy(lattice),
            max_neighbors=max_neighbors,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compute neighborhood graph: {e}")

    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
    onehot = element.z2onehot(atomic_numbers).to(dtype=torch.get_default_dtype())

    # === Physical property to predict ===
    properties = {}
    property_weights = {}

    for name in list(keyspec.arrays_keys) + list(keyspec.info_keys):
        property_weights[name] = atoms.info.get(f"{name}_weight", 1.0)

    for name, atoms_key in keyspec.info_keys.items():
        properties[name] = atoms.info.get(atoms_key, None)
        if atoms_key not in atoms.info:
            property_weights[name] = 0.0

    for name, atoms_key in keyspec.arrays_keys.items():
        properties[name] = atoms.arrays.get(atoms_key, None)
        if atoms_key not in atoms.arrays:
            property_weights[name] = 0.0

    lattice = (
        torch.tensor(lattice.reshape((1, 3, 3)), dtype=torch.get_default_dtype())
        if lattice is not None
        else torch.tensor(3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()).view(
            1, 3, 3
        )
    )

    num_atoms = len(atomic_numbers)

    # === predict or embedding ===
    if training:
        need_property = set(target_property + embedding_property)
    else:
        need_property = set(embedding_property)

    joint_property = []
    for name in need_property:
        must_be_with = PROPERTY[name]['must_be_with']
        for _, v in must_be_with.items():
            joint_property += v
    need_property.update(list(set(joint_property)))

    pDict = {}
    wDict = {}
    for name in need_property:
        in_data = PROPERTY[name]["shape"]["in_data"]
        shape_fn = PROPERTY[name]["shape"].get("shape_fn", None)
        default_value_fn = PROPERTY[name]["default_value_fn"]
        class_ = PROPERTY[name]["class"]
        try:
            p = properties.get(name)
            if p is None:
                if class_ == 'float':
                    p = torch.tensor(
                        default_value_fn(num_atoms, class_),
                        dtype=torch.get_default_dtype(),
                    )
                elif class_ == 'int':
                    p = torch.tensor(
                        np.round(default_value_fn(num_atoms, class_)),
                        dtype=torch.int64,
                    )
                else:
                    raise
            else:
                if class_ == 'float':
                    p = torch.tensor(p, dtype=torch.get_default_dtype())
                elif class_ == 'int':
                    p = torch.tensor(np.round(p), dtype=torch.int64)
                else:
                    raise
                if shape_fn is not None:
                    p = shape_fn(
                        p,
                        num_aotms=num_atoms,
                    )
            pDict.update({name: p.view(*in_data)})
        except Exception as e:
            raise RuntimeError(f"Failed to read property ``{name}``") from e

        try:
            w = (
                torch.tensor(
                    property_weights.get(name), dtype=torch.get_default_dtype()
                ).view(
                    1
                )  # (1,)
                if property_weights.get(name) is not None
                else torch.tensor(1.0, dtype=torch.get_default_dtype())  # ()
            )
            wDict.update({name: w})
        except Exception as e:
            raise RuntimeError(f"Failed to read property ``{name}``") from e

    if universal_embedding is not None:
        for p in embedding_property:
            assert (
                p in UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY
            ), f"universal_embedding key allowed property are {UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY}, "
            f"if you need more, please contact the author"

    data_dict = {
        "entropy": to_tensor(atoms.info.get('entropy', 1.0)),
        "atomic_numbers": atomic_numbers,
        "lattice": lattice,
        "positions": to_tensor(positions),
        "node_attrs": onehot,
        "edge_index": torch.tensor(edge_index, dtype=torch.int64),
        "edge_shifts": to_tensor(edge_shifts),
        "level": torch.tensor(atoms.info.get(keyspec.info_keys['level'], 0), dtype=torch.int64),
    }

    for name in need_property:
        data_dict.update(
            {
                name: pDict[name],
                f"{name}_weight": wDict[name],
            }
        )
    return Data(**data_dict)

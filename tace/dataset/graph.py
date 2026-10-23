################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import List, Optional

import ase
import numpy as np
import torch
from torch_geometric.data import Data

from .element import TorchElement
from .neighbour_list import get_neighborhood
from .quantity import PROPERTIES, KeySpecification, get_need_property


def read_properties(
    atoms: ase.Atoms,
    keyspec: KeySpecification,
    property_names: List[str],
    dtype: torch.dtype,
    periodic_cell: bool,
    num_edges: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Read property tensors and weights from an ASE structure."""
    values = {}
    weights = {}
    for source, keys in (
        (atoms.info, keyspec.info_keys),
        (atoms.arrays, keyspec.arrays_keys),
    ):
        for name, key in keys.items():
            values[name] = source.get(key)
            weights.setdefault(name, atoms.info.get(f"{name}_weight", 1.0))
            if values[name] is None:
                weights[name] = 0.0

    data = {}
    for name in property_names:
        config = PROPERTIES[name]
        kind = config["type"]
        value = values.get(name)
        present = value is not None
        weight = weights.get(name)
        try:
            scope = config["scope"]
            if scope == "per-edge" and num_edges is None:
                raise ValueError("Reading per-edge properties requires num_edges.")
            if not present:
                value = config["default_value_fn"](
                    len(atoms), kind, num_edges=num_edges
                )
                weight = 0.0
            if kind == "float":
                value = torch.tensor(value, dtype=dtype)
            elif kind == "int":
                value = torch.tensor(np.round(value), dtype=torch.int64)
            else:
                raise TypeError(f"Unsupported property type: {kind}")
            shape_fn = config["shape"]["shape_fn"]
            if present and shape_fn is not None:
                value = shape_fn(value, num_atoms=len(atoms))
            value = value.reshape(config["shape"]["in_data"])
            if scope in {"per-edge", "per-pair"}:
                count = num_edges if scope == "per-edge" else len(atoms) ** 2
                if value.shape[0] != count:
                    raise ValueError(
                        f"{scope} property requires {count} entries, got {value.shape[0]}."
                    )

            if kind == "float":
                if not torch.isfinite(value).all():
                    raise ValueError(f"{name} contains NaN or infinite values.")
                if config["requires_periodic_cell"] and not periodic_cell:
                    if present and (weight is None or weight > 0):
                        raise ValueError(
                            f"{name} labels require a periodic direction and "
                            f"a non-singular cell; use {name}_weight=0 to ignore them."
                        )
                    value = torch.zeros_like(value)

            data[name] = value
            data[f"{name}_weight"] = (
                torch.tensor(weight, dtype=dtype).reshape(1)
                if weight is not None
                else torch.tensor(1.0, dtype=dtype)
            )
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to read property {name}: {exc}") from exc
    return data


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
    dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """Build a neighbor graph with requested labels and embedding properties."""
    dtype = dtype or torch.get_default_dtype()
    atomic_numbers = atoms.get_atomic_numbers()
    pbc = tuple(atoms.get_pbc())
    lattice = np.array(atoms.get_cell())
    if not np.isfinite(lattice).all():
        raise ValueError("The cell must contain finite values.")
    volume = abs(np.linalg.det(lattice))
    if all(pbc) and volume == 0:
        raise ValueError("A fully periodic structure requires a non-singular cell.")
    periodic_cell = any(pbc) and volume > 0
    positions = atoms.get_positions()
    edge_index, edge_shifts, pbc, lattice = get_neighborhood(
        positions=positions,
        cutoff=cutoff,
        pbc=pbc,
        lattice=lattice,
        max_neighbors=max_neighbors,
        backend=neighborlist_backend,
    )
    atomic_numbers = torch.tensor(atomic_numbers, dtype=torch.int64)
    data_dict = {
        "entropy": torch.tensor(atoms.info.get("entropy", 1.0), dtype=dtype),
        "atomic_numbers": atomic_numbers,
        "lattice": torch.tensor(lattice, dtype=dtype).reshape(1, 3, 3),
        "pbc": torch.tensor([pbc], dtype=torch.bool),
        "positions": torch.tensor(positions, dtype=dtype),
        "node_attrs": element.z2onehot(atomic_numbers).to(dtype=dtype),
        "edge_index": torch.tensor(edge_index, dtype=torch.int64),
        "edge_shifts": torch.tensor(edge_shifts, dtype=dtype),
        "fidelity_idx": torch.tensor(
            atoms.info.get(keyspec.info_keys["fidelity_idx"], 0), dtype=torch.int64
        ),
    }
    data_dict.update(
        read_properties(
            atoms,
            keyspec,
            get_need_property(target_property, embedding_property, training),
            dtype,
            periodic_cell,
            num_edges=edge_index.shape[1],
        )
    )

    return Data(**data_dict)

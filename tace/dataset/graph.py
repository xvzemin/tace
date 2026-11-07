"""Validated, float64 graph construction from reader-independent records."""

from __future__ import annotations

from typing import List, Optional

import ase
import numpy as np
import torch
from torch_geometric.data import Data

from .element import TorchElement
from .neighbour_list import get_neighborhood
from .quantity import PROPERTIES, KeySpecification, get_need_property
from .records import (
    AtomicRecord,
    record_from_atoms,
    resolve_property,
    validate_fidelity,
)


def _tensor(value, dtype):
    # Read-only database views cannot safely back writable Torch tensors.
    if isinstance(value, np.ndarray) and not value.flags.writeable:
        value = value.copy()
    return torch.as_tensor(value, dtype=dtype)


def _weight(record, name):
    value = record.info.get(f"{name}_weight", record.data.get(f"{name}_weight", 1.0))
    tensor = _tensor(value, torch.float64)
    if tensor.numel() != 1 or not torch.isfinite(tensor).all() or tensor.item() < 0:
        raise ValueError(f"{name}_weight must be one finite, non-negative scalar")
    return tensor.reshape(1)


def _shape_property(value, name, config, num_atoms, num_edges):
    scope = config["scope"]
    shape = config["shape"]["in_data"]
    counts = {
        "per-system": 1,
        "per-atom": num_atoms,
        "per-edge": num_edges,
        "per-pair": num_atoms**2,
    }
    count = counts[scope]
    if count is None:
        raise ValueError("Reading per-edge properties requires num_edges.")
    expected = (count,) + tuple(shape[1:])
    # ASE Voigt ordering: xx, yy, zz, yz, xz, xy.
    if name == "atomic_stresses" and tuple(value.shape) == (num_atoms, 6):
        order = torch.tensor([[0, 5, 4], [5, 1, 3], [4, 3, 2]])
        value = value[:, order]
    elif config["shape"]["shape_fn"] is not None:
        # Already normalized system tensors need no further Voigt conversion.
        if (
            name not in {"stress", "direct_stress", "virials", "direct_virials"}
            or tuple(value.shape) != expected
        ):
            value = config["shape"]["shape_fn"](value, num_atoms=num_atoms)
    if tuple(value.shape) == expected:
        return value
    if scope == "per-system" and tuple(value.shape) == tuple(shape[1:]):
        return value.reshape(expected)
    if scope in {"per-edge", "per-pair", "per-atom"}:
        if value.ndim > 0 and value.shape[0] != count:
            raise ValueError(
                f"{scope} property requires {count} entries, got {value.shape[0]}."
            )
    raise ValueError(
        f"{scope} property {name} has shape {tuple(value.shape)}; expected {expected}."
    )


def read_properties(
    atoms: ase.Atoms | AtomicRecord,
    keyspec: KeySpecification,
    property_names: List[str],
    periodic_cell: bool,
    num_edges: Optional[int] = None,
) -> dict[str, torch.Tensor]:
    """Resolve labels across ASE namespaces and enforce scope, shape and weights."""
    record = record_from_atoms(atoms)
    data = {}
    for name in property_names:
        try:
            config = PROPERTIES[name]
            value = resolve_property(record, name, keyspec)
            present = value is not None
            weight = _weight(record, name)
            if not present:
                if config["scope"] == "per-edge" and num_edges is None:
                    raise ValueError("Reading per-edge properties requires num_edges.")
                value = config["default_value_fn"](
                    len(record), config["type"], num_edges=num_edges
                )
                weight = torch.zeros(1, dtype=torch.float64)
            if config["type"] == "float":
                value = _tensor(value, torch.float64)
            elif config["type"] == "int":
                if name == "fidelity_idx":
                    value = validate_fidelity(value) if present else 0
                value = _tensor(value, torch.int64)
            else:
                raise TypeError(f"Unsupported property type: {config['type']}")
            value = _shape_property(value, name, config, len(record), num_edges)
            if not torch.isfinite(value).all():
                raise ValueError(f"{name} contains NaN or infinite values.")
            if config["requires_periodic_cell"] and not periodic_cell:
                if present and weight.item() > 0:
                    raise ValueError(
                        f"{name} labels require a periodic direction and a non-singular cell; use {name}_weight=0 to ignore them."
                    )
                value = torch.zeros_like(value)
            data[name] = value
            data[f"{name}_weight"] = weight
        except (TypeError, ValueError, RuntimeError) as exc:
            raise RuntimeError(f"Failed to read property {name}: {exc}") from exc
    return data


def _geometry(record):
    numbers = np.asarray(record.numbers)
    if numbers.ndim != 1 or numbers.dtype.kind not in "iu" or np.any(numbers <= 0):
        raise ValueError(
            "Atomic numbers must be a one-dimensional array of positive integers."
        )
    positions = np.asarray(record.positions, dtype=np.float64)
    lattice = np.asarray(record.cell, dtype=np.float64)
    pbc = np.asarray(record.pbc)
    if positions.shape != (len(numbers), 3) or not np.isfinite(positions).all():
        raise ValueError(
            "Positions must have shape (num_atoms, 3) and contain finite values."
        )
    if lattice.shape != (3, 3) or not np.isfinite(lattice).all():
        raise ValueError("The cell must have shape (3, 3) and contain finite values.")
    if pbc.shape != (3,) or not np.isin(pbc, (False, True)).all():
        raise ValueError("PBC must contain three boolean values.")
    pbc = tuple(bool(value) for value in pbc)
    volume = abs(np.linalg.det(lattice))
    if all(pbc) and volume == 0:
        raise ValueError("A fully periodic structure requires a non-singular cell.")
    return numbers, positions, lattice, pbc, bool(any(pbc) and volume > 0)


def from_record(
    element: TorchElement,
    record: AtomicRecord,
    cutoff: float,
    max_neighbors: Optional[int],
    keyspec: KeySpecification,
    target_property: List[str],
    embedding_property: List[str],
    training: bool = True,
    neighborlist_backend: str = "matscipy",
    *,
    neighborhood=None,
    **kwargs,
):
    """Build one graph; ``neighborhood`` optionally supplies precomputed edges."""
    if "dtype" in kwargs:
        raise TypeError(
            "Graph data always use float64; set the model dtype separately."
        )
    record = record_from_atoms(record)
    numbers, positions, lattice, pbc, periodic_cell = _geometry(record)
    if neighborhood is None:
        neighborhood = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=pbc,
            lattice=lattice,
            max_neighbors=max_neighbors,
            backend=neighborlist_backend,
        )
    edge_index, edge_shifts, pbc, lattice = neighborhood
    fidelity = resolve_property(record, "fidelity_idx", keyspec)
    fidelity = 0 if fidelity is None else validate_fidelity(fidelity)
    entropy = _tensor(
        record.info.get("entropy", record.data.get("entropy", 1.0)), torch.float64
    )
    if entropy.numel() != 1 or not torch.isfinite(entropy).all():
        raise ValueError("entropy must be one finite scalar")
    data = {
        "entropy": entropy.reshape(()),
        "atomic_numbers": _tensor(numbers, torch.int64),
        "lattice": _tensor(lattice, torch.float64).reshape(1, 3, 3),
        "pbc": torch.tensor([pbc], dtype=torch.bool),
        "positions": _tensor(positions, torch.float64),
        "edge_index": _tensor(edge_index, torch.int64),
        "edge_shifts": _tensor(edge_shifts, torch.int64),
        "num_neighbors": torch.as_tensor(
            np.bincount(np.asarray(edge_index)[0], minlength=len(record)),
            dtype=torch.int64,
        ),
        "fidelity_idx": torch.tensor([fidelity], dtype=torch.int64),
    }
    data.update(
        read_properties(
            record,
            keyspec,
            get_need_property(target_property, embedding_property, training),
            periodic_cell,
            num_edges=edge_index.shape[1],
        )
    )
    return Data(num_nodes=len(record), **data)


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
    """ASE compatibility adapter for :func:`from_record`."""
    return from_record(
        element,
        record_from_atoms(atoms),
        cutoff,
        max_neighbors,
        keyspec,
        target_property,
        embedding_property,
        training,
        neighborlist_backend,
        **kwargs,
    )


def batch_geometry(records):
    """Pack validated record geometry for one neighbor-search batch."""
    geometry = [_geometry(record) for record in records]
    ptr = np.concatenate(
        ([0], np.cumsum([len(record) for record in records], dtype=np.int64))
    )
    positions = np.concatenate([item[1] for item in geometry], axis=0)
    cells = np.stack([item[2] for item in geometry])
    pbc = np.asarray([item[3] for item in geometry], dtype=bool)
    return dict(positions=positions, ptr=ptr, pbc=pbc, lattice=cells)


def from_records(element, records, *, device=None, neighborhood=None, **options):
    """Build a graph batch, optionally using precomputed packed neighbors."""
    from .neighbour_list import get_neighborhood_batch

    records = [record_from_atoms(record) for record in records]
    if not records:
        return []
    geometry = batch_geometry(records)
    if neighborhood is None:
        neighborhood = get_neighborhood_batch(
            **geometry,
            cutoff=options["cutoff"],
            max_neighbors=options.get("max_neighbors"),
            backend=options.get("neighborlist_backend", "matscipy"),
            device=device,
        )
    edge_index, shifts, edge_ptr = neighborhood
    ptr, pbc, cells = geometry["ptr"], geometry["pbc"], geometry["lattice"]
    return [
        from_record(
            element,
            record,
            neighborhood=(
                edge_index[:, edge_ptr[i] : edge_ptr[i + 1]] - ptr[i],
                shifts[edge_ptr[i] : edge_ptr[i + 1]],
                tuple(pbc[i]),
                cells[i],
            ),
            **options,
        )
        for i, record in enumerate(records)
    ]

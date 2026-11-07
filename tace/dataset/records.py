"""Small, calculator-independent records shared by readers and graph builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, fields

import numpy as np
from ase import Atoms


@dataclass
class AtomicRecord:
    """Physical arrays plus distinct label namespaces; no calculator is executed."""

    numbers: np.ndarray
    positions: np.ndarray
    cell: np.ndarray
    pbc: np.ndarray
    info: dict = field(default_factory=dict)
    arrays: dict = field(default_factory=dict)
    results: dict = field(default_factory=dict)
    data: dict = field(default_factory=dict)

    def __len__(self):
        return len(self.numbers)

    def to_dict(self):
        """A shallow, typed-array mapping suitable for non-pickle storage."""
        return {item.name: getattr(self, item.name) for item in fields(self)}

    @classmethod
    def from_dict(cls, mapping):
        return cls(**mapping)

    def to_atoms(self):
        """Compatibility adapter; the fast training path consumes records directly."""
        from ase.calculators.calculator import all_properties
        from ase.calculators.singlepoint import SinglePointCalculator

        atoms = Atoms(
            numbers=self.numbers,
            positions=self.positions,
            cell=self.cell,
            pbc=self.pbc,
            info=dict(self.info),
        )
        for key, value in self.arrays.items():
            if key not in {"numbers", "positions"}:
                atoms.arrays[key] = np.asarray(value).copy()
        # Keep arbitrary database data out of ASE's per-atom arrays namespace.
        for key, value in self.data.items():
            atoms.info.setdefault(key, value)
        results = {
            key: value
            for key, value in self.results.items()
            if key in all_properties and value is not None
        }
        if results:
            atoms.calc = SinglePointCalculator(atoms, **results)
        return atoms


def record_from_atoms(atoms: Atoms | AtomicRecord) -> AtomicRecord:
    if isinstance(atoms, AtomicRecord):
        return atoms
    return AtomicRecord(
        numbers=np.asarray(atoms.numbers),
        positions=np.asarray(atoms.positions, dtype=np.float64),
        cell=np.asarray(atoms.cell, dtype=np.float64),
        pbc=np.asarray(atoms.pbc, dtype=bool),
        info=dict(atoms.info),
        arrays={
            key: value
            for key, value in atoms.arrays.items()
            if key not in {"numbers", "positions"}
        },
        results=dict(getattr(atoms.calc, "results", {}) or {}),
    )


def record_from_row(row) -> AtomicRecord:
    """Read an ASE database row without constructing an intermediate Atoms."""
    from ase.calculators.calculator import all_properties

    data = dict(row.data)
    info = dict(row.key_value_pairs)
    arrays = {
        key: row.get(key)
        for key in ("initial_magmoms", "initial_charges", "masses", "tags", "momenta")
        if row.get(key) is not None
    }
    # Some producers put ASE's namespaces inside row.data.
    for key, destination in (("info", info), ("arrays", arrays)):
        nested = data.get(key)
        if isinstance(nested, Mapping):
            for name, value in nested.items():
                destination.setdefault(name, value)
            del data[key]
    return AtomicRecord(
        numbers=np.asarray(row.numbers),
        positions=np.asarray(row.positions, dtype=np.float64),
        cell=np.asarray(row.cell, dtype=np.float64),
        pbc=np.asarray(row.pbc, dtype=bool),
        info=info,
        arrays=arrays,
        results={
            key: row.get(key) for key in all_properties if row.get(key) is not None
        },
        data=data,
    )


def resolve_property(record, name, keyspec):
    """Explicit keys win across namespaces, then canonical/ASE aliases apply."""
    from .quantity import PROPERTIES

    per_atom = PROPERTIES[name]["scope"] == "per-atom"
    preferred = record.arrays if per_atom else record.info
    secondary = record.info if per_atom else record.arrays
    namespaces = (preferred, secondary, record.data, record.results)
    keys = keyspec.arrays_keys if per_atom else keyspec.info_keys
    other_keys = keyspec.info_keys if per_atom else keyspec.arrays_keys
    key = keys.get(name, other_keys.get(name, name))
    candidates = dict.fromkeys((key, name, PROPERTIES[name].get("ase_name")))
    for candidate in candidates:
        if candidate is not None:
            for namespace in namespaces:
                if candidate in namespace:
                    return namespace[candidate]
    return None


def validate_fidelity(value, *, context="fidelity_idx") -> int:
    array = np.asarray(value)
    if array.size != 1 or array.dtype.kind not in "iu":
        raise ValueError(f"{context} must be a non-negative integer, got {value!r}")
    number = int(array.reshape(-1)[0])
    if number < 0 or number > np.iinfo(np.int64).max:
        raise ValueError(f"{context} must be a non-negative int64, got {value!r}")
    return number

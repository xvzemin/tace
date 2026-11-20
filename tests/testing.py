"""Shared TACE test data and numerical checks."""

import hashlib

import lmdb
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.db import connect

from tace.dataset.quantity import KeySpecification


def file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def file_snapshot(directory):
    snapshot = {}
    for path in directory.rglob("*"):
        if path.is_file():
            stat = path.stat()
            snapshot[str(path.relative_to(directory))] = (
                stat.st_ino,
                stat.st_size,
                stat.st_mtime_ns,
                file_digest(path),
            )
    return snapshot


def reader_keys():
    keys = KeySpecification.from_defaults()
    keys.info_keys["energy"] = "reference_energy"
    keys.arrays_keys["forces"] = "reference_forces"
    return keys


def write_reader_database(path):
    open_lmdb = lmdb.open

    def open_small_lmdb(*args, **kwargs):
        kwargs["map_size"] = 1 << 20
        return open_lmdb(*args, **kwargs)

    # ASE-LMDB defaults to a 2 TiB map, which expands the file on Windows.
    # Restrict only fixture creation; readers still use the real lmdb.open.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lmdb, "open", open_small_lmdb)
        database = connect(path)

    atoms_list = []
    with database:
        for index, symbols in enumerate(("H2", "O", "HO", "H2O")):
            atoms = Atoms(symbols)
            atoms.positions = np.arange(len(atoms) * 3).reshape(-1, 3) * 0.2
            atoms.info["reference_energy"] = -float(
                np.where(atoms.numbers == 1, 1, 2).sum()
            )
            atoms.arrays["reference_forces"] = np.full((len(atoms), 3), index * 0.1)
            database.write(
                atoms,
                data={
                    "reference_energy": atoms.info["reference_energy"],
                    "reference_forces": atoms.arrays["reference_forces"],
                },
            )
            atoms_list.append(atoms)
    return atoms_list


def time_reverse_features(features: torch.Tensor, irreps) -> torch.Tensor:
    output = features.clone()
    for ir_mul, ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir_mul.ir.t
    return output

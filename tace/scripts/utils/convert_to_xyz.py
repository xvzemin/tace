"""Convert DeepMD NumPy datasets in a directory to one ASE extxyz file.

Examples
--------
Convert one system::

    python convert_to_xyz.py /data/lcao-datasets/Fe16

Recursively combine every system under a directory into ``train.xyz``::

    python convert_to_xyz.py /data/lcao-datasets
"""

import argparse
from pathlib import Path


import numpy as np
from ase import Atoms
from ase.io.extxyz import output_column_format



FIELD_NAMES = (
    "box",
    "coord",
    "energy",
    "force",
    "force_mag",
    "spin",
    "virial",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="DeepMD system directory or a directory containing systems",
    )
    return parser.parse_args()


def is_system_directory(path: Path) -> bool:
    return (
        (path / "type.raw").is_file()
        and (path / "type_map.raw").is_file()
        and any(candidate.is_dir() for candidate in path.glob("set.*"))
    )


def discover_systems(directory: Path) -> list[Path]:
    if directory.name.startswith("set.") and is_system_directory(directory.parent):
        return [directory.parent]
    if is_system_directory(directory):
        return [directory]

    systems = sorted(
        {
            type_file.parent
            for type_file in directory.rglob("type.raw")
            if is_system_directory(type_file.parent)
        }
    )
    if not systems:
        raise ValueError(
            f"No DeepMD systems found under {directory}. Expected type.raw, "
            "type_map.raw, and set.* directories."
        )
    return systems


def load_symbols(system_directory: Path) -> np.ndarray:
    type_map = (system_directory / "type_map.raw").read_text(
        encoding="utf-8"
    ).split()
    atom_types = np.loadtxt(
        system_directory / "type.raw",
        dtype=np.int64,
        ndmin=1,
    )
    if atom_types.size == 0:
        raise ValueError(f"No atom types found in {system_directory / 'type.raw'}")
    if atom_types.min() < 0 or atom_types.max() >= len(type_map):
        raise ValueError(
            f"Atom type index is outside type_map.raw in {system_directory}"
        )
    return np.asarray(type_map, dtype=object)[atom_types]


def load_set(set_directory: Path) -> dict[str, np.ndarray]:
    missing = [
        name for name in FIELD_NAMES if not (set_directory / f"{name}.npy").is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing fields in {set_directory}: {', '.join(missing)}"
        )
    return {
        name: np.load(set_directory / f"{name}.npy", mmap_mode="r")
        for name in FIELD_NAMES
    }


def frame_count(arrays: dict[str, np.ndarray], set_directory: Path) -> int:
    counts = {name: values.shape[0] for name, values in arrays.items()}
    if len(set(counts.values())) != 1:
        raise ValueError(f"Inconsistent frame counts in {set_directory}: {counts}")
    return next(iter(counts.values()))


def virial_to_stress(virial: np.ndarray, cell: np.ndarray) -> np.ndarray:
    """Convert a DeepMD virial in eV to ASE stress in eV/angstrom**3."""
    volume = abs(np.linalg.det(cell))
    if volume == 0.0:
        raise ValueError("Cannot convert virial for a zero-volume cell")
    return -virial / volume


def write_frame(
    handle,
    symbols: np.ndarray,
    arrays: dict[str, np.ndarray],
    frame: int,
) -> None:
    natoms = len(symbols)
    positions = np.asarray(arrays["coord"][frame]).reshape(natoms, 3)
    magmoms = np.asarray(arrays["spin"][frame]).reshape(natoms, 3)
    magnetic_forces = np.asarray(arrays["force_mag"][frame]).reshape(natoms, 3)
    forces = np.asarray(arrays["force"][frame]).reshape(natoms, 3)
    cell = np.asarray(arrays["box"][frame]).reshape(3, 3)
    virial = np.asarray(arrays["virial"][frame]).reshape(3, 3)

    atoms = Atoms(symbols=symbols, positions=positions, cell=cell, pbc=True)
    atoms.info["energy"] = float(
        np.asarray(arrays["energy"][frame]).reshape(-1)[0]
    )
    atoms.info["stress"] = virial_to_stress(virial, cell)

    columns = (
        "symbols",
        "positions",
        "initial_noncollinear_magmoms",
        "noncollinear_magnetic_forces",
        "forces",
    )
    output_arrays = {
        "symbols": np.asarray(symbols),
        "positions": positions,
        "initial_noncollinear_magmoms": magmoms,
        "noncollinear_magnetic_forces": magnetic_forces,
        "forces": forces,
    }
    comment, _, _, _ = output_column_format(
        atoms,
        list(columns),
        output_arrays,
        write_info=True,
    )

    handle.write(f"{natoms}\n{comment}\n")
    for atom in range(natoms):
        values = np.concatenate(
            (
                positions[atom],
                magmoms[atom],
                magnetic_forces[atom],
                forces[atom],
            )
        )
        serialized = " ".join(f"{value:.17g}" for value in values)
        handle.write(f"{symbols[atom]} {serialized}\n")


def convert(
    directory: Path,
    output: Path,
) -> int:
    systems = discover_systems(directory)

    total_frames = 0
    with output.open("w", encoding="utf-8") as handle:
        for system_directory in systems:
            symbols = load_symbols(system_directory)
            set_directories = sorted(
                path for path in system_directory.glob("set.*") if path.is_dir()
            )
            for set_directory in set_directories:
                arrays = load_set(set_directory)
                frames = frame_count(arrays, set_directory)
                for frame in range(frames):
                    write_frame(handle, symbols, arrays, frame)
                total_frames += frames
                print(
                    f"wrote {system_directory.name}/{set_directory.name}: "
                    f"{frames} frames"
                )
    return total_frames


def main() -> None:
    args = parse_args()
    directory = args.directory.expanduser().resolve()
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    output = directory / "train.xyz"

    total_frames = convert(directory, output)
    print(f"{output}: {total_frames} total frames")


if __name__ == "__main__":
    main()

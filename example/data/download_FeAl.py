"""Download and split the FeAl collinear magnetic dataset."""

import urllib.request
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz

MAGNETIC_AXIS = "x"  # "x", "y", or "z"
SEED = 42

URL = (
    "https://gitlab.com/ivannovikov/datasets_for_magnetic_MTP/-/raw/main/"
    "Fe_Al_fitting_to_magnetic_forces/training_set/"
    "training_set_with_magnetic_forces.cfg"
)
DATA_DIR = Path.home() / "dataset" / "FeAl"
CFG_FILE = DATA_DIR / "training_set_with_magnetic_forces.cfg"
XYZ_FILES = {
    split: DATA_DIR / f"collinear_FeAl_{split}.xyz"
    for split in ("train", "val", "test")
}
SYMBOLS = {0: "Al", 1: "Fe"}


def main():
    if MAGNETIC_AXIS not in "xyz":
        raise ValueError("MAGNETIC_AXIS must be 'x', 'y', or 'z'.")
    magnetic_axis = "xyz".index(MAGNETIC_AXIS)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CFG_FILE.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, CFG_FILE)

    blocks = CFG_FILE.read_text().split("BEGIN_CFG")[1:]
    indices = np.random.default_rng(SEED).permutation(len(blocks))
    num_train = round(0.8 * len(blocks))
    num_val = round(0.1 * len(blocks))
    splits = np.empty(len(blocks), dtype=np.int8)
    splits[indices[:num_train]] = 0
    splits[indices[num_train : num_train + num_val]] = 1
    splits[indices[num_train + num_val :]] = 2
    split_names = tuple(XYZ_FILES)

    with ExitStack() as stack:
        outputs = {
            name: stack.enter_context(path.open("w"))
            for name, path in XYZ_FILES.items()
        }
        for index, block in enumerate(blocks):
            lines = [line.strip() for line in block.splitlines() if line.strip()]

            size_index = lines.index("Size")
            cell_index = lines.index("Supercell")
            atom_index = next(
                i for i, line in enumerate(lines) if line.startswith("AtomData:")
            )
            energy_index = lines.index("Energy")
            virials_index = next(
                i for i, line in enumerate(lines) if line.startswith("PlusStress:")
            )

            size = int(lines[size_index + 1])
            cell = np.array(
                [lines[cell_index + i].split() for i in range(1, 4)], dtype=float
            )
            data = np.array(
                [lines[atom_index + i].split() for i in range(1, size + 1)],
                dtype=float,
            )
            energy = float(lines[energy_index + 1])

            # PlusStress order: xx, yy, zz, yz, xz, xy.
            xx, yy, zz, yz, xz, xy = np.fromstring(lines[virials_index + 1], sep=" ")
            virials = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]], dtype=float)

            atoms = Atoms(
                symbols=[SYMBOLS[int(atom_type)] for atom_type in data[:, 1]],
                positions=data[:, 2:5],
                cell=cell,
                pbc=True,
            )
            atoms.arrays["forces"] = data[:, 5:8]

            magmoms = np.zeros_like(data[:, 8:11])
            magmoms[:, magnetic_axis] = data[:, 8]
            atoms.arrays["initial_noncollinear_magmoms"] = magmoms

            # en_der_m is already -dE/dm, the TACE magnetic-force convention.
            magnetic_forces = np.zeros_like(data[:, 11:14])
            magnetic_forces[:, magnetic_axis] = data[:, 11]
            atoms.arrays["noncollinear_magnetic_forces"] = magnetic_forces
            atoms.info["energy"] = energy

            # MLIP PlusStress is the virial W; TACE uses stress = -W / V.
            atoms.info["virials"] = virials
            atoms.info["stress"] = -virials / atoms.get_volume()

            feature = next(
                (line for line in lines if line.startswith("Feature ")), None
            )
            if feature is not None:
                _, name, value = feature.split(maxsplit=2)
                atoms.info[name] = float(value)

            write_extxyz(
                outputs[split_names[splits[index]]], atoms, write_results=False
            )

    counts = np.bincount(splits, minlength=3)
    for name, count in zip(split_names, counts):
        print(f"Wrote {count} configurations to {XYZ_FILES[name]}")


if __name__ == "__main__":
    main()

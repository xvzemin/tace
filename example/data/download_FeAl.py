"""Download the FeAl collinear magnetic dataset and write one extxyz file."""

import urllib.request
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz

URL = (
    "https://gitlab.com/ivannovikov/datasets_for_magnetic_MTP/-/raw/main/"
    "Fe_Al_fitting_to_magnetic_forces/training_set/"
    "training_set_with_magnetic_forces.cfg"
)
DATA_DIR = Path.home() / "dataset" / "FeAl"
CFG_FILE = DATA_DIR / "training_set_with_magnetic_forces.cfg"
XYZ_FILE = DATA_DIR / "collinear_FeAl.xyz"
SYMBOLS = {0: "Al", 1: "Fe"}


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not CFG_FILE.exists():
        print(f"Downloading {URL}")
        urllib.request.urlretrieve(URL, CFG_FILE)

    blocks = CFG_FILE.read_text().split("BEGIN_CFG")[1:]
    with XYZ_FILE.open("w") as output:
        for block in blocks:
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
            atoms.arrays["initial_noncollinear_magmoms"] = data[:, 8:11]

            # en_der_m is already -dE/dm, the TACE magnetic-force convention.
            atoms.arrays["noncollinear_magnetic_forces"] = data[:, 11:14]
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

            write_extxyz(output, atoms, write_results=False)

    print(f"Wrote {len(blocks)} configurations to {XYZ_FILE}")


if __name__ == "__main__":
    main()

"""Download MatPES PBE/r2SCAN and convert both datasets to TACE extxyz."""

import json
import urllib.request
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers
from ase.io.extxyz import write_extxyz
from ase.stress import voigt_6_to_full_3x3_stress
from ase.units import GPa

VERSION = "2025.1"
BASE_URL = "https://huggingface.co/datasets/materialyze/matpes/resolve/main"
DATA_DIR = Path.home() / "dataset" / "MatPES"
DATASETS = {
    "PBE": f"MatPES-PBE-{VERSION}.jsonl",
    "r2SCAN": f"MatPES-R2SCAN-{VERSION}.jsonl",
}
ATOMIC_REFERENCES = {
    "PBE": "MatPES-PBE-atoms.jsonl",
    "r2SCAN": "MatPES-R2SCAN-atoms.jsonl",
}


def download(filename):
    path = DATA_DIR / filename
    if not path.exists():
        print(f"Downloading {filename}")
        temporary = path.with_suffix(f"{path.suffix}.part")
        urllib.request.urlretrieve(f"{BASE_URL}/{filename}", temporary)
        temporary.replace(path)
    return path


def convert(functional, source):
    target = DATA_DIR / f"MatPES-{functional}-{VERSION}.xyz"
    num_configs = 0
    num_magnetic = 0

    print(f"Converting {source.name} to {target.name}")
    with source.open() as input_file, target.open("w") as output_file:
        for line in input_file:
            record = json.loads(line)
            structure = record["structure"]
            sites = structure["sites"]
            magmoms = np.array(
                [site["properties"].get("magmom", 0.0) for site in sites]
            )
            stress = (
                -0.1
                * GPa
                * voigt_6_to_full_3x3_stress(np.asarray(record["stress"]))
            )

            atoms = Atoms(
                symbols=[site["species"][0]["element"] for site in sites],
                positions=[site["xyz"] for site in sites],
                cell=structure["lattice"]["matrix"],
                pbc=structure["lattice"].get("pbc", True),
            )
            atoms.arrays["forces"] = np.asarray(record["forces"])

            # TACE calls model inputs "initial" properties. These values are
            # nevertheless the final SCF-converged VASP site magnetizations.
            magnetic_moments = np.zeros((len(atoms), 3))
            magnetic_moments[:, 2] = magmoms
            atoms.arrays["initial_noncollinear_magmoms"] = magnetic_moments
            atoms.info["energy"] = record["energy"]
            atoms.info["stress"] = stress
            atoms.info["virials"] = -stress * atoms.get_volume()
            atoms.info["matpes_id"] = record["matpes_id"]
            atoms.info["functional"] = record["functional"]
            write_extxyz(output_file, atoms, write_results=False)

            num_configs += 1
            num_magnetic += bool(np.any(magmoms != 0.0))

    print(
        f"Wrote {num_configs} configurations ({num_magnetic} with nonzero "
        f"magmoms) to {target}"
    )


def write_atomic_energies(sources):
    energies = {}
    for functional, source in sources.items():
        with source.open() as input_file:
            energies[functional] = {
                atomic_numbers[record["elements"][0]]: record["energy"]
                for record in map(json.loads, input_file)
            }

    target = DATA_DIR / "atomic_energies.yaml"
    with target.open("w") as output_file:
        for functional, values in energies.items():
            output_file.write(f"{functional}:\n  atomic_energy:\n")
            for atomic_number, energy in sorted(values.items()):
                output_file.write(f"    {atomic_number}: {energy:.8f}\n")

    print(f"Wrote PBE and r2SCAN isolated-atom energies to {target}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    atomic_sources = {
        functional: download(filename)
        for functional, filename in ATOMIC_REFERENCES.items()
    }
    write_atomic_energies(atomic_sources)

    for functional, filename in DATASETS.items():
        convert(functional, download(filename))

    print(
        "MatPES magmom audit: structure.sites[].properties.magmom contains "
        "the final SCF-converged collinear site magnetization on the sampled "
        "fixed geometry, not the input INCAR MAGMOM. MatPES structures are "
        "static single-point calculations, not DFT-relaxed structures."
    )


if __name__ == "__main__":
    main()

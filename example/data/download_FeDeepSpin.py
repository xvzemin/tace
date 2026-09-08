"""Download and split the Fe-DeepSpin LCAO dataset."""

import urllib.request
import zipfile
from contextlib import ExitStack
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.io.extxyz import write_extxyz

SEED = 42

URL = (
    "https://store.aissquare.com/datasets/"
    "7a5b6de8-9138-42cb-b208-43feacdd5acc/Fe-DeepSpin.zip"
)
DATA_DIR = Path.home() / "dataset" / "Fe-DeepSpin"
ZIP_FILE = DATA_DIR / "Fe-DeepSpin.zip"
FE16_DIR = DATA_DIR / "lcao-datasets" / "Fe16"
FE32_DIR = DATA_DIR / "lcao-datasets" / "Fe32"
XYZ_FILES = {
    split: DATA_DIR / f"lcao_{split}.xyz"
    for split in ("train", "val", "test", "test_ood")
}
FIELDS = ("box", "coord", "energy", "force", "force_mag", "spin", "virial")


def count_configs(source_dir):
    return sum(
        len(np.load(set_dir / "energy.npy", mmap_mode="r"))
        for set_dir in sorted(source_dir.glob("set.*"))
    )


def write_dataset(source_dir, output_paths, splits):
    type_map = (source_dir / "type_map.raw").read_text().split()
    atom_types = np.loadtxt(source_dir / "type.raw", dtype=int, ndmin=1)
    symbols = np.asarray(type_map)[atom_types]
    num_atoms = len(symbols)
    index = 0

    # DeepMD stores these data in eV, Angstrom, and mu_B. 
    with ExitStack() as stack:
        outputs = [stack.enter_context(path.open("w")) for path in output_paths]
        for set_dir in sorted(source_dir.glob("set.*")):
            data = {
                name: np.load(set_dir / f"{name}.npy", mmap_mode="r")
                for name in FIELDS
            }
            num_frames = len(data["energy"])
            if any(len(values) != num_frames for values in data.values()):
                raise ValueError(f"Inconsistent frame counts in {set_dir}")

            for frame in range(num_frames):
                cell = data["box"][frame].reshape(3, 3)
                virials = data["virial"][frame].reshape(3, 3)
                atoms = Atoms(
                    symbols=symbols,
                    positions=data["coord"][frame].reshape(num_atoms, 3),
                    cell=cell,
                    pbc=True,
                )
                atoms.arrays["forces"] = data["force"][frame].reshape(num_atoms, 3)
                atoms.arrays["initial_noncollinear_magmoms"] = data["spin"][
                    frame
                ].reshape(num_atoms, 3)
                atoms.arrays["noncollinear_magnetic_forces"] = data["force_mag"][
                    frame
                ].reshape(num_atoms, 3)
                atoms.info["energy"] = float(data["energy"][frame, 0])
                atoms.info["virials"] = virials
                atoms.info["stress"] = -virials / atoms.get_volume()
                write_extxyz(outputs[splits[index]], atoms, write_results=False)
                index += 1

    if index != len(splits):
        raise ValueError(f"Expected {len(splits)} configurations, found {index}")


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if not ZIP_FILE.exists():
        print(f"Downloading {URL}")
        temporary = ZIP_FILE.with_suffix(".zip.part")
        urllib.request.urlretrieve(URL, temporary)
        temporary.replace(ZIP_FILE)

    if not FE16_DIR.exists() or not FE32_DIR.exists():
        print(f"Extracting {ZIP_FILE}")
        with zipfile.ZipFile(ZIP_FILE) as archive:
            archive.extractall(DATA_DIR.parent)

    num_configs = count_configs(FE16_DIR)
    indices = np.random.default_rng(SEED).permutation(num_configs)
    num_train = round(0.8 * num_configs)
    num_val = round(0.1 * num_configs)
    splits = np.empty(num_configs, dtype=np.int8)
    splits[indices[:num_train]] = 0
    splits[indices[num_train : num_train + num_val]] = 1
    splits[indices[num_train + num_val :]] = 2
    write_dataset(
        FE16_DIR,
        [XYZ_FILES[name] for name in ("train", "val", "test")],
        splits,
    )

    num_ood = count_configs(FE32_DIR)
    write_dataset(
        FE32_DIR,
        [XYZ_FILES["test_ood"]],
        np.zeros(num_ood, dtype=np.int8),
    )

    counts = np.bincount(splits, minlength=3)
    for name, count in zip(("train", "val", "test"), counts):
        print(f"Wrote {count} configurations to {XYZ_FILES[name]}")
    print(f"Wrote {num_ood} configurations to {XYZ_FILES['test_ood']}")


if __name__ == "__main__":
    main()

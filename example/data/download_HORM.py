"""Download the latest official HORM release to ~/dataset/HORM and convert it.

The author's LMDB records contain Python pickles and must be trusted.
"""

import pickle
from pathlib import Path

import kagglehub
import lmdb
import numpy as np
import requests
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from tqdm import tqdm

from tace.dataset.hessian import hessian_blocks

DATASET = "yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm"
OUTPUT = Path.home() / "dataset" / "HORM"
SPLITS = {
    "ts1x_hess_train.lmdb": "train.db",
    "ts1x-val.lmdb": "valid.db",
    "RGD1.lmdb": "test.db",
}


def convert(source, destination, version):
    """Convert trusted HORM records, keeping energies, forces and full Hessians."""
    metadata = {"source": DATASET, "version": version}
    if destination.exists():
        if connect(destination).metadata == metadata:
            print(f"Already converted: {destination}", flush=True)
            return
        raise FileExistsError(f"Refusing to overwrite {destination}")
    temporary = destination.with_name(destination.stem + ".partial.db")
    if temporary.exists():
        raise FileExistsError(f"Unfinished conversion exists: {temporary}")
    environment = lmdb.open(
        str(source), subdir=source.is_dir(), readonly=True, lock=False
    )
    count = 0
    try:
        with environment.begin() as transaction, connect(temporary) as database:
            keys = [
                key
                for key in transaction.cursor().iternext(values=False)
                if key.isdigit()
            ]
            keys.sort(key=int)
            for key in tqdm(keys, desc=destination.name, unit="structures"):
                record = pickle.loads(transaction.get(key))
                numbers = record.get("atomic_numbers", record.get("z"))
                if numbers is None:
                    # The sample calls atomic numbers 'charges'; these are not partial charges.
                    numbers = record.get("charges")
                if numbers is None:
                    elements = torch.tensor([1, 6, 7, 8, 9])
                    numbers = elements[torch.as_tensor(record["one_hot"]).argmax(-1)]
                numbers = torch.as_tensor(numbers).cpu().numpy().reshape(-1)
                if (
                    not np.equal(numbers, numbers.astype(int)).all()
                    or not np.isin(numbers, [1, 6, 7, 8, 9]).all()
                ):
                    raise ValueError(f"Unexpected atomic numbers in record {key!r}")
                positions = (
                    torch.as_tensor(record.get("positions", record.get("pos")))
                    .cpu()
                    .numpy()
                )
                atoms = Atoms(
                    numbers=numbers.astype(int), positions=positions, pbc=False
                )
                raw = torch.as_tensor(record["hessian"]).cpu()
                hessian = hessian_blocks(raw, len(atoms)).numpy()
                if not np.isfinite(hessian).all():
                    raise ValueError(f"Nonfinite Hessian in record {key!r}")
                # HORM already uses eV, angstrom, eV/angstrom and eV/angstrom**2.
                results = {}
                if record.get("energy") is not None:
                    results["energy"] = float(torch.as_tensor(record["energy"]).item())
                if record.get("forces") is not None:
                    results["forces"] = (
                        torch.as_tensor(record["forces"]).cpu().numpy().reshape(-1, 3)
                    )
                atoms.calc = SinglePointCalculator(atoms, **results)
                data = {"direct_hessian": hessian}
                if record.get("ae") is not None:
                    data["atomization_energy"] = float(
                        torch.as_tensor(record["ae"]).item()
                    )
                if record.get("rxn") is not None:
                    data["reaction_id"] = int(torch.as_tensor(record["rxn"]).item())
                database.write(atoms, data=data, source_index=int(key))
                count += 1
            if count == 0:
                raise ValueError(f"No records found in {source}")
            database.metadata = metadata
    finally:
        environment.close()
    temporary.rename(destination)
    print(f"Wrote {count} structures: {destination}", flush=True)


def main():
    response = requests.get(
        f"https://www.kaggle.com/api/v1/datasets/view/{DATASET}", timeout=60
    )
    response.raise_for_status()
    version = response.json()["currentVersionNumber"]
    print(f"HORM v{version} → {OUTPUT}", flush=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    for filename, output in SPLITS.items():
        source = kagglehub.dataset_download(
            f"{DATASET}/versions/{version}",
            path=filename,
            output_dir=str(OUTPUT / "raw" / f"v{version}"),
        )
        convert(Path(source), OUTPUT / output, version)


if __name__ == "__main__":
    main()

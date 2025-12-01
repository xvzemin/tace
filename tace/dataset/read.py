################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from __future__ import annotations
import logging
from typing import List, Dict
from pathlib import Path
import multiprocessing
from concurrent.futures import as_completed
from concurrent.futures import ProcessPoolExecutor


from ase import Atoms
from ase.io import read
from ase.db import connect
from ase.calculators.calculator import all_properties


from .split import random_split
from .quantity import KeySpecification


class DatasetsSplit:
    def __init__(self, train, valid, test):
        self.train = train
        self.valid = valid
        self.test = test

class ThreeDataset:
    def __init__(self, train, valid, test=None):
        self.data = DatasetsSplit(train=train, valid=valid, test=test or [])

    def __getitem__(self, idx):
        return [self.data.train, self.data.valid, self.data.test][idx]

    def __setitem__(self, idx, value):
        if idx == 0:
            self.data.train = value
        elif idx == 1:
            self.data.valid = value
        elif idx == 2:
            self.data.test = value
        else:
            raise IndexError("Index out of range for ThreeDataset")

    def __len__(self):
        return 3

    @property
    def train(self):
        return self.data.train

    @property
    def valid(self):
        return self.data.valid

    @property
    def test(self):
        return self.data.test


def check_keys(
    atomsList: List[Atoms],
    target_property: List[str],
    keyspec: KeySpecification,
    embedding_property: List[str] = [],
    check: bool = True,
):
    for atoms in atomsList:
        if check:
            if atoms.calc is not None:
                for p in (target_property + embedding_property):
                    found = False
                    if p in keyspec.info_keys:
                        key = keyspec.info_keys[p]
                        debug_key = f"{key}s"
                        if key in all_properties:
                            atoms.info[key] = atoms.calc.results[key]
                            found = True
                        if not found:
                            if debug_key in all_properties:
                                atoms.info[key] = atoms.calc.results[debug_key]
                    if p in keyspec.arrays_keys:
                        key = keyspec.arrays_keys[p]
                        debug_key = f"{key}s"
                        if key in all_properties:
                            atoms.arrays[key] = atoms.calc.results[key]
                            found = True
                        if not found:
                            if debug_key in all_properties:
                                atoms.arrays[key] = atoms.calc.results[debug_key]
    return atomsList


def read_file(fpath: str, target_property, keyspec, embedding_property):
    from ase.io import read
    from pathlib import Path

    fpath = Path(fpath)
    try:
        return check_keys(read(fpath, index=":"), target_property, keyspec, embedding_property)
    except Exception as e:
        logging.warning(f"Failed to read {fpath}: {e}, pass")
        return []

def ase_io_read(
    filename: str,
    target_property: List[str],
    keyspec,
    embedding_property: List[str],
    num_workers: int = None,
):
    """
    Behavior
    --------
    - If 'filename' is a file: read it normally using ASE.
    - If 'filename' is a directory: search for all '.xyz' and '.extxyz' files
      (recursively), read them in parallel (using 1/4 of available CPUs),
      and aggregate all structures.

    Parameters
    ----------
    filename : str
        Path to a file or directory.
    target_property : List[str]
        List of target properties to check.
    keyspec :
        Specification of property keys (type depends on your check_keys implementation).
    embedding_property : List[str]
        List of embedding-related properties.

    Returns
    -------
    list
        Aggregated structures passed to check_keys().
    """

    path = Path(filename)

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 4)

    all_structures = []

    if path.is_dir():
        logging.info(f"'{path}' is a directory. Searching for all '.xyz' and '.extxyz' files recursively...")

        xyz_files = list(path.rglob("*.xyz")) + list(path.rglob("*.extxyz"))
        if not xyz_files:
            raise FileNotFoundError(f"No '.xyz' or '.extxyz' files found in directory '{path}'.")

        logging.info(f"Found {len(xyz_files)} files. Reading them in parallel...")
        logging.info(f"Using {num_workers} processes for parallel reading.")

        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {
                executor.submit(read_file, str(f), target_property, keyspec, embedding_property): f
                for f in xyz_files
            }

            # for future in tqdm(as_completed(futures), total=len(futures), desc="Reading files"):
            #     all_structures.extend(future.result())

            for future in as_completed(futures):
                all_structures.extend(future.result())

        logging.info(f"Successfully read {len(all_structures)} structures from {len(xyz_files)} files.")
    else:
        all_structures = check_keys(read(path, index=":"), target_property, keyspec, embedding_property)

    return all_structures
    

def ase_db_connect(
    filename: str,
    target_property: List[str],
    keyspec: KeyboardInterrupt,
    embedding_property: List[str],
    num_workers: int = max(1, multiprocessing.cpu_count() // 4),
):
    return check_keys(
        [row.toatoms() for row in connect(filename).select()],
        target_property,
        keyspec,
        embedding_property,
    )


def fair_lmdb(filename: str):
    raise NotImplementedError("fair_lmdb is not yet implemented")


HOW_TO_READ = {"ase": ase_io_read, "ase_db": ase_db_connect, "lmdb": fair_lmdb}


def _read(
    cfg: Dict,
    target_property: List[str],
    keyspec: KeyboardInterrupt,
    embedding_property: List[str],
    in_datamodule: bool = False,
) -> ThreeDataset:
    file_type = cfg.get("dataset", {}).get("type", "ase")
    no_valid_set = cfg.get("dataset", {}).get("no_valid_set", False)
    num_workers = cfg.get("dataset", {}).get("num_workers", max(1, multiprocessing.cpu_count() // 4))
    try:
        train_file = cfg["dataset"]["train_file"]
    except Exception as e:
        raise RuntimeError(
            "Failed to retrieve 'cfg.dataset.train_file' from configuration"
        ) from e
    valid_file = cfg.get("dataset", {}).get("valid_file", None)
    test_files = cfg.get("dataset", {}).get("test_files", None)
    try:
        tmp_train_atoms_list = HOW_TO_READ[file_type](
            train_file, target_property, keyspec, embedding_property, num_workers
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load training file from cfg.dataset.train_file: {e}"
        )
    try:
        tmp_valid_atoms_list = (
            HOW_TO_READ[file_type](
                valid_file, target_property, keyspec, embedding_property, num_workers
            )
            if valid_file is not None
            else None
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load validation file from cfg.dataset.valid_file: {e}"
        )
    try:
        if test_files is not None:
            if isinstance(test_files, str):
                test_atoms_list = [
                    HOW_TO_READ[file_type](
                        test_files, target_property, keyspec, embedding_property, num_workers
                    )
                ]
            elif isinstance(test_files, list):
                test_atoms_list = [
                    HOW_TO_READ[file_type](
                        f, target_property, keyspec, embedding_property, num_workers
                    )
                    for f in test_files
                ]
            else:
                test_atoms_list = None
        else:
            test_atoms_list = None
    except Exception as e:
        raise RuntimeError(f"Failed to load test file from cfg.dataset.test_files: {e}")
    
    if not in_datamodule:
        if test_atoms_list is None:
            logging.info("No test file is given")

    if tmp_valid_atoms_list is not None:
        train_atoms_list = tmp_train_atoms_list
        valid_atoms_list = tmp_valid_atoms_list
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.info(
                f"Using validation set from: {valid_file}",
            )
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
    elif cfg.get("dataset", {}).get("valid_from_index", False):
        # In the earlier version, the order of the training set was not taken 
        # into account, and therefore only the valid indices were saved.
        train_index_path = Path(".") / "train.index"
        valid_index_path = Path(".") / "valid.index"
        assert (
            valid_index_path.is_file()
        ), f"File does not exist or is not a regular file: {valid_index_path}"
        with valid_index_path.open("r", encoding="utf-8") as f:
            valid_indices = [int(line.strip()) for line in f if line.strip()]
        valid_atoms_list = [tmp_train_atoms_list[i] for i in valid_indices]
        if train_index_path.exists():
            assert (
                train_index_path.is_file()
            ), f"File does not exist or is not a regular file: {train_index_path}"
            with train_index_path.open("r", encoding="utf-8") as f:
                train_indices = [int(line.strip()) for line in f if line.strip()]
            train_atoms_list = [tmp_train_atoms_list[i] for i in train_indices]
        else:
            train_atoms_list = [
                item
                for idx, item in enumerate(tmp_train_atoms_list)
                if idx not in valid_indices
            ]
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.info(f"Using valid set index from: {str(valid_index_path)}")
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
    elif no_valid_set:
        train_atoms_list = tmp_train_atoms_list
        valid_atoms_list = None
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.info(
                f"This training has no validation set, you must use lr_scheduler not depending on validation set",
            )
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
    else:
        try:
            ratio = cfg["dataset"]["valid_ratio"]
        except Exception as e:
            raise RuntimeError(
                "Valid_ratio must be provided if no validation file is given."
            ) from e
        assert isinstance(
            ratio, float
        ), "Valid_ratio must be provided if no validation file is given"
        assert 0.0 < ratio < 1.0, "Valid_ratio must be in the range (0, 1)."
        if not in_datamodule:
            logging.info(
                f"Using training set from: {train_file}",
            )
            logging.info(
                "Using random %s%% of training set for validation",
                100 * ratio,
            )
            if test_atoms_list is not None:
                logging.info(
                    f"Using test set from: {test_files}",
                )
        train_atoms_list, valid_atoms_list = random_split(
            tmp_train_atoms_list,
            ratio,
            cfg.get("dataset", {}).get("split_seed", 1),
        )

    assert len(train_atoms_list) > 0, "Training set is empty !"
    if not no_valid_set:
        assert len(valid_atoms_list) > 0, "Validation set is empty !"

    return ThreeDataset(train_atoms_list, valid_atoms_list, test_atoms_list)

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import multiprocessing
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor
from numbers import Integral
from pathlib import Path
from typing import Dict, List, Union

from ase import Atoms

from .quantity import PROPERTY, KeySpecification, get_need_property
from .split import random_split


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
    training: bool = True,
):
    need_property = get_need_property(target_property, embedding_property, training)
    for atoms in atomsList:
        calc = atoms.calc
        if calc is None:
            continue
        results = getattr(calc, "results", None)
        if not results:
            continue
        for p in need_property:
            if PROPERTY[p]["ase_name"]:
                ase_name = PROPERTY[p]["ase_name"]
            else:
                ase_name = p
            if ase_name not in results:
                continue
            value = results[ase_name]
            if p in list(keyspec.info_keys):
                key = keyspec.info_keys[p]
                atoms.info[key] = value
            elif p in list(keyspec.arrays_keys):
                key = keyspec.arrays_keys[p]
                atoms.arrays[key] = value

    return atomsList


def ase_io_read(filename: str):
    from ase.io import read

    try:
        atoms_list = read(filename, index=":")
        if not atoms_list:
            logging.warning(f"File {filename} is empty")
            return []
        return atoms_list
    except Exception as e:
        logging.error(f"Failed to read {filename}: {e}")
        return []


def ase_db_connect(filename: str):
    from ase.db import connect

    try:
        atoms_list = []
        with connect(filename) as db:
            for row in db.select():
                atoms = row.toatoms()
                atoms.info.update(row.key_value_pairs)
                atoms.info.update(row.data)
                atoms_list.append(atoms)
        if not atoms_list:
            logging.warning(f"Database {filename} is empty")
            return []
    except Exception as e:
        logging.error(f"Failed to read {filename}: {e}")
        return []
    return atoms_list


def fair_aselmdb(filename: str):  # [only test energy, forces, stress]
    from ase.db import connect

    try:
        atoms_list = []
        with connect(filename) as db:
            for row in db.select():
                atoms = row.toatoms()
                atoms.info.update(row.key_value_pairs)
                if hasattr(row, "data"):
                    for k, v in row.data.items():
                        if not k.startswith("_"):
                            atoms.info[k] = v
                            atoms.arrays[k] = v
                atoms_list.append(atoms)
        if not atoms_list:
            logging.warning(f"Database {filename} is empty")
            return []
        return atoms_list
    except Exception as e:
        logging.error(f"Failed to read {filename}: {e}")
        return []


def torchsim_h5(filename: str):  # TODO
    raise NotImplementedError("torchsim_h5 is not yet implemented")


RGLOB = {
    "ase": ["*.xyz", "*.extxyz", "*.traj"],
    "ase_db": ["*.db"],
    # "aqcat25_aselmdb": ["*.aselmdb"],
    "fair_aselmdb": ["*.aselmdb"],
    "torchsim_h5": ["*.h5"],
}


HOW_TO_READ = {
    "ase": ase_io_read,
    "ase_db": ase_db_connect,
    "fair_aselmdb": fair_aselmdb,
    "torchsim_h5": torchsim_h5,
}


def read_single_file(
    fpath: str,
    target_property,
    keyspec,
    embedding_property,
    backend="ase",
    fidelity_idx: int = None,
):
    """Read structures and check an optional source fidelity against their metadata."""
    if fidelity_idx is not None and (
        not isinstance(fidelity_idx, Integral) or fidelity_idx < 0
    ):
        raise ValueError(f"{fpath}: fidelity_idx must be a non-negative integer")
    atomsList = HOW_TO_READ[backend](fpath)
    if fidelity_idx is not None:
        key = keyspec.info_keys["fidelity_idx"]
        for index, atoms in enumerate(atomsList):
            if key in atoms.info:
                value = atoms.info[key]
                if not isinstance(value, Integral) or value != fidelity_idx:
                    if isinstance(value, Integral):
                        value = int(value)
                    raise ValueError(
                        f"{fpath}: structure {index} (zero-based) has "
                        f"atoms.info[{key!r}]={value!r}, but the data source "
                        f"specifies fidelity_idx={fidelity_idx}"
                    )
            atoms.info[key] = int(fidelity_idx)
    try:
        return check_keys(atomsList, target_property, keyspec, embedding_property)
    except Exception as e:
        logging.warning("Failed to read when check_keys for atoms")
        return []


def read_all_files(
    filename: Union[str, Path, Mapping, List[Union[str, Path, Mapping]]],
    target_property: List[str],
    keyspec,
    embedding_property: List[str],
    num_workers: int = None,
    backend="ase",
):
    """Read dataset files and assign optional source fidelities.

    Directories are searched recursively. Each discovered file is read once.

    Parameters
    ----------
    filename : str, pathlib.Path, mapping, or list
        File or directory paths, or entries with ``path`` and an optional
        ``fidelity_idx``. The specified fidelity is applied to every structure
        in that source; conflicting structure metadata raises ``ValueError``.
    target_property : List[str]
        List of target properties to check.
    keyspec :
        Specification of property keys.
    embedding_property : List[str]
        List of embedding-related properties.

    Returns
    -------
    list
        Aggregated structures passed to check_keys().
    """

    if num_workers is None:
        num_workers = max(1, multiprocessing.cpu_count() // 4)

    sources = [filename] if isinstance(filename, (str, Path, Mapping)) else filename
    all_files = {}
    for source in sources:
        if isinstance(source, Mapping):
            if "path" not in source or source.keys() - {"path", "fidelity_idx"}:
                raise ValueError(
                    "Dataset entries require 'path' and optionally 'fidelity_idx'"
                )
            path = Path(source["path"]).resolve()
            fidelity_idx = source.get("fidelity_idx")
        else:
            path = Path(source).resolve()
            fidelity_idx = None
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")

        if path.is_file():
            files = [path]
        elif path.is_dir():
            files = (f for pattern in RGLOB[backend] for f in path.rglob(pattern))
        else:
            raise ValueError(f"Unsupported path type: {path}")
        for file in files:
            file = file.resolve()
            previous = all_files.get(file)
            if previous is not None and fidelity_idx is not None:
                if previous != fidelity_idx:
                    raise ValueError(
                        f"{file}: conflicting source fidelity_idx values "
                        f"{previous} and {fidelity_idx}"
                    )
            if file not in all_files or fidelity_idx is not None:
                all_files[file] = fidelity_idx

    if not all_files:
        raise FileNotFoundError("No dataset files found in the provided paths")

    logging.info(f"Found {len(all_files)} files in total")
    logging.info(f"Using {num_workers} processes for parallel reading")

    all_structures = []
    successful_files = 0
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(
                read_single_file,
                str(f),
                target_property,
                keyspec,
                embedding_property,
                backend,
                all_files[f],
            )
            for f in sorted(all_files)
        ]
        for future in futures:
            structures = future.result()
            if structures:
                successful_files += 1
                all_structures.extend(structures)

    logging.info(
        "Matching files: %d; successful files: %d; loaded structures: %d",
        len(all_files),
        successful_files,
        len(all_structures),
    )

    return all_structures


def tace_read_all_files(
    cfg: Dict,
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeySpecification,
    in_datamodule: bool = False,
) -> ThreeDataset:

    file_type = cfg["dataset"].get("type", "ase")
    no_valid_set = cfg["dataset"].get("no_valid_set", False)
    num_workers = max(1, multiprocessing.cpu_count() // 4)
    num_workers = cfg["dataset"].get("num_workers", num_workers)

    train_file = cfg["dataset"]["train_file"]
    assert train_file, (
        "No valid training dataset provided. Please check cfg.dataset.train_file"
    )
    valid_file = cfg.get("dataset", {}).get("valid_file", None)
    test_files = cfg.get("dataset", {}).get("test_files", None)

    try:
        tmp_train_atoms_list = read_all_files(
            train_file,
            target_property,
            keyspec,
            embedding_property,
            num_workers,
            file_type,
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to load training file from cfg.dataset.train_file: {e}"
        )
    try:
        tmp_valid_atoms_list = (
            read_all_files(
                valid_file,
                target_property,
                keyspec,
                embedding_property,
                num_workers,
                file_type,
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
            if isinstance(test_files, (str, Path, Mapping)):
                test_atoms_list = [
                    read_all_files(
                        test_files,
                        target_property,
                        keyspec,
                        embedding_property,
                        num_workers,
                        file_type,
                    )
                ]
            elif isinstance(test_files, list):
                test_atoms_list = [
                    read_all_files(
                        f,
                        target_property,
                        keyspec,
                        embedding_property,
                        num_workers,
                        file_type,
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
        assert valid_index_path.is_file(), (
            f"File does not exist or is not a regular file: {valid_index_path}"
        )
        with valid_index_path.open("r", encoding="utf-8") as f:
            valid_indices = [int(line.strip()) for line in f if line.strip()]
        valid_atoms_list = [tmp_train_atoms_list[i] for i in valid_indices]
        if train_index_path.exists():
            assert train_index_path.is_file(), (
                f"File does not exist or is not a regular file: {train_index_path}"
            )
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
            logging.warning(
                "This training has no validation set, you must use lr_scheduler not depending on validation set",
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
        assert isinstance(ratio, float), (
            "Valid_ratio must be provided if no validation file is given"
        )
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

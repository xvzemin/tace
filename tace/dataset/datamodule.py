################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import gc
import logging
import os
import pickle
import re
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional

import lmdb
import torch.distributed as dist
from hydra.utils import instantiate
from lightning.pytorch import LightningDataModule
from lightning.pytorch.utilities.rank_zero import rank_zero_info, rank_zero_only
from torch.utils.data import Dataset

from .dataloader import build_statistics_dataloader
from .element import TorchElement, build_element_lookup
from .graph import from_atoms
from .quantity import KeySpecification
from .read import tace_read_all_files


class GraphDatasetLMDB(Dataset):
    """
    Lazy-loading LMDB dataset; each LMDB shard contains local keys 0..(n-1).
    Graphs accessed by global index are mapped to (shard_idx, local_idx).
    """

    def __init__(
        self,
        lmdb_paths: List[Path],
        in_memory: bool = False,
        cache_size: int = 1024,
        shard_lengths: Optional[List[int]] = None,
    ):
        self.lmdb_paths = list(lmdb_paths)
        self.in_memory = in_memory
        self.cache_size = cache_size
        self.cache = {}  # (shard_idx, local_idx) -> graph
        self.envs = [None] * len(self.lmdb_paths)
        self.txns = [None] * len(self.lmdb_paths)
        self.lengths = []

        if in_memory:
            self.data_list = []
            self.length = 0
        else:
            self.length = 0
            if shard_lengths is None:
                self.lengths = [None] * len(self.lmdb_paths)
            elif len(shard_lengths) != len(self.lmdb_paths):
                raise ValueError("shard_lengths must match the number of LMDB paths")
            else:
                self.lengths = list(shard_lengths)

    def __len__(self):
        if self.in_memory:
            return self.length

        for i, l in enumerate(self.lengths):
            if l is None:
                env = lmdb.open(
                    str(self.lmdb_paths[i]),
                    readonly=True,
                    lock=False,
                    readahead=False,
                    max_readers=64,
                )
                with env.begin(write=False) as txn:
                    self.lengths[i] = txn.stat()["entries"]
                env.close()
        return sum(self.lengths)

    def __getitem__(self, idx):
        if self.in_memory:
            return self.data_list[idx]

        if idx < 0:
            idx += len(self)
        total_length = len(self)
        if idx < 0 or idx >= total_length:
            raise IndexError(
                f"Index {idx} out of range for dataset of length {total_length}"
            )

        # map global idx -> shard_idx, local_idx
        shard_idx = 0
        while shard_idx < len(self.lengths) and idx >= self.lengths[shard_idx]:
            idx -= self.lengths[shard_idx]
            shard_idx += 1

        # lazy open env and txn
        if self.envs[shard_idx] is None:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.envs[shard_idx] = lmdb.open(
                        str(self.lmdb_paths[shard_idx]),
                        readonly=True,
                        lock=False,
                        readahead=False,
                        max_readers=64,
                    )
                    self.txns[shard_idx] = self.envs[shard_idx].begin(write=False)
                    break
                except lmdb.Error:
                    if attempt == max_retries - 1:
                        raise
                    time.sleep(1)

        key_tuple = (shard_idx, idx)
        if self.cache_size > 0 and key_tuple in self.cache:
            return self.cache[key_tuple]

        key = f"{idx:09d}".encode("ascii")
        value = self.txns[shard_idx].get(key)
        if value is None:
            raise IndexError(
                f"Index {idx} not found in LMDB shard "
                f"{self.lmdb_paths[shard_idx]} (shard_idx {shard_idx})"
            )

        graph = pickle.loads(value)

        if self.cache_size > 0:
            if len(self.cache) >= self.cache_size:
                self.cache.pop(next(iter(self.cache)))
            self.cache[key_tuple] = graph
        return graph

    def close(self):
        for txn in self.txns:
            if txn is not None:
                txn.abort()
        for env in self.envs:
            if env is not None:
                env.close()
        self.envs = [None] * len(self.lmdb_paths)
        self.txns = [None] * len(self.lmdb_paths)

    def __getstate__(self):
        self.close()
        state = dict(self.__dict__)
        state["envs"] = [None] * len(self.lmdb_paths)
        state["txns"] = [None] * len(self.lmdb_paths)
        state["cache"] = {}
        return state

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def create_graphs(
    atoms_list: Optional[List],
    element,
    for_dataset: dict,
    stage: str,
    shard_dirs: List[Path],
    storage_mode: str = "lmdb",
    shard_size: int = 1000,
    lmdb_wait_timeout: int = 86400,
    avg_graph_size_in_KB: int = 75,
    cache_size: int = 1024,
    existing_dataset: Optional[GraphDatasetLMDB] = None,
):
    """
    Create graphs in memory (memory mode) or write/read LMDB shards (lmdb mode).

    atoms_list may be None for ranks that don't hold the raw atoms (e.g., non-rank0 in lmdb mode).
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if storage_mode not in ["lmdb", "memory"]:
        raise ValueError(f"Unknown storage_mode: {storage_mode}")

    # === MEMORY MODE ===
    if storage_mode == "memory":
        if rank == 0:
            if existing_dataset is not None:
                if not existing_dataset.in_memory:
                    raise ValueError("existing_dataset must be an in-memory dataset")
                graphs = existing_dataset.data_list
            else:
                graphs = [
                    from_atoms(element, atoms, **for_dataset)
                    for atoms in (atoms_list or [])
                ]
        else:
            graphs = []

        if world_size > 1 and dist.is_initialized():
            # broadcast number then chunks via broadcast_object_list in batches
            if rank == 0:
                n_obj = [len(graphs)]
            else:
                n_obj = [None]
            dist.broadcast_object_list(n_obj, src=0)
            n = int(n_obj[0])

            # initialize empty list on non-zero ranks
            if rank != 0:
                graphs = [None] * n

            # ===== broadcast in batches =====
            batch_size = 128
            for i in range(0, n, batch_size):
                j = min(n, i + batch_size)
                batch = graphs[i:j] if rank == 0 else [None] * (j - i)
                dist.broadcast_object_list(batch, src=0)
                if rank != 0:
                    graphs[i:j] = batch

        if rank == 0 and existing_dataset is not None:
            return existing_dataset

        dataset = GraphDatasetLMDB(lmdb_paths=[], in_memory=True, cache_size=cache_size)
        dataset.data_list = graphs
        dataset.length = len(graphs)
        return dataset

    # === LMDB MODE ===
    def _collect_lmdb_paths():
        paths = []
        for directory in shard_dirs:
            paths.extend(sorted(directory.glob(f"{stage}_shard*.lmdb")))
        return paths

    # if shards already exist, load them
    lmdb_paths = _collect_lmdb_paths()
    if lmdb_paths:
        rank_zero_info(
            f"Found existing LMDB files for {stage}: {len(lmdb_paths)} shards"
        )
        return GraphDatasetLMDB(
            lmdb_paths,
            in_memory=False,
            cache_size=cache_size,
        )

    # Only rank 0 will create LMDB shards (if atoms_list provided)
    if rank == 0 and atoms_list:
        rank_zero_info(
            f"Rank 0 building and saving {len(atoms_list)} graphs "
            f"for stage={stage} (stream LMDB mode)"
        )
        buffer = []
        shard_idx = 0
        n_atoms = len(atoms_list)
        # choose directory as round-robin across provided shard_dirs for distribution
        n_dirs = max(1, len(shard_dirs))
        for d in shard_dirs:
            d.mkdir(parents=True, exist_ok=True)

        for idx, atoms in enumerate(atoms_list):
            graph = from_atoms(element, atoms, **for_dataset)
            buffer.append(pickle.dumps(graph, protocol=pickle.HIGHEST_PROTOCOL))

            if len(buffer) >= shard_size or idx == n_atoms - 1:
                dir_path = shard_dirs[shard_idx % n_dirs]
                lmdb_path = dir_path / f"{stage}_shard{shard_idx:04d}.lmdb"
                temp_path = Path(f"{lmdb_path}.tmp-{os.getpid()}")
                if temp_path.exists():
                    shutil.rmtree(temp_path)
                if lmdb_path.exists():
                    shutil.rmtree(lmdb_path)

                serialized_graphs = buffer
                avg_graph_size = avg_graph_size_in_KB * 1024
                estimated_size = int(len(buffer) * avg_graph_size * 1.1)
                serialized_size = sum(len(value) for value in serialized_graphs)
                map_size = max(
                    estimated_size,
                    int(serialized_size * 1.25) + 16 * 1024 * 1024,
                )

                max_retries = 3
                for attempt in range(max_retries):
                    env = None
                    try:
                        env = lmdb.open(
                            str(temp_path),
                            map_size=map_size,
                            subdir=True,
                            writemap=True,
                            metasync=True,
                            sync=True,
                            map_async=False,
                        )
                        with env.begin(write=True) as txn:
                            # local keys start from 0 for each shard
                            for i_local, value in enumerate(serialized_graphs):
                                key = f"{i_local:09d}".encode("ascii")
                                txn.put(key, value)
                        env.close()
                        os.replace(temp_path, lmdb_path)
                        break
                    except lmdb.MapFullError:
                        if env is not None:
                            env.close()
                        if temp_path.exists():
                            shutil.rmtree(temp_path)
                        map_size *= 2
                        if attempt == max_retries - 1:
                            raise
                    except lmdb.Error:
                        if env is not None:
                            env.close()
                        if temp_path.exists():
                            shutil.rmtree(temp_path)
                        if attempt == max_retries - 1:
                            raise
                        time.sleep(1)

                rank_zero_info(f"Saved shard {lmdb_path} with {len(buffer)} graphs")
                buffer.clear()
                shard_idx += 1
                gc.collect()

        rank_zero_info(f"All {n_atoms} graphs saved successfully for stage={stage}")
        gc.collect()

        #  other ranks wait for files to appear
    if world_size > 1 and dist.is_initialized():
        if rank != 0:
            # poll for lmdb files with timeout to avoid indefinite hang
            waited = 0
            poll_interval = 1.0
            while waited < lmdb_wait_timeout:
                lmdb_paths = _collect_lmdb_paths()
                if lmdb_paths:
                    break
                time.sleep(poll_interval)
                waited += poll_interval
            if not lmdb_paths:
                # after timeout, still no files -> error to avoid silent hang
                raise RuntimeError(
                    f"Timeout: no LMDB files for stage={stage} found after {lmdb_wait_timeout}s"
                )
        # ensure all processes reach here after files exist or after rank0 created them
        dist.barrier()

    lmdb_paths = _collect_lmdb_paths()
    if not lmdb_paths:
        logging.error(f"No LMDB files found for {stage} after creation")
        raise RuntimeError(f"No LMDB files found for {stage} after creation")

    rank_zero_info(f"Loaded {len(lmdb_paths)} LMDB shards for {stage}")
    dataset = GraphDatasetLMDB(
        lmdb_paths,
        in_memory=False,
        cache_size=cache_size,
    )
    return dataset


# === DataModule ===
class GraphDataModule(LightningDataModule):
    def __init__(
        self,
        cfg: dict,
        element: TorchElement,
        target_property: List[str],
        keyspec: KeySpecification,
        embedding_property: List[str],
        threeAtomsList=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.keyspec = keyspec
        self.element = element
        self.target_property = target_property
        self.embedding_property = embedding_property

        self.train_dataset = None
        self.val_dataset = None
        self.test_datasets = None
        self.threeAtomsList = threeAtomsList

        self.storage_mode = cfg.get("dataset", {}).get("storage_mode", "memory")
        self.shard_dirs = [
            Path(p) for p in cfg.get("dataset", {}).get("shard_dirs", ["graphCache"])
        ]
        test_pattern = re.compile(r"^test(\d+)_shard\d+\.lmdb$")
        test_ids = set()
        for shard_dir in self.shard_dirs:
            if not shard_dir.exists():
                continue
            for f in shard_dir.iterdir():
                m = test_pattern.match(f.name)
                if m:
                    test_ids.add(int(m.group(1)))
        self.presave_test_sets = bool(test_ids)
        self.num_test_sets = len(test_ids)
        self.shard_size = cfg.get("dataset", {}).get("shard_size", 1_000_000)
        self.cache_size = cfg.get("dataset", {}).get("cache_size", 1024)
        self.avg_graph_size_in_KB = cfg.get("dataset", {}).get(
            "avg_graph_size_in_KB", 200
        )
        self.lmdb_wait_timeout = cfg.get("dataset", {}).get("lmdb_wait_timeout", 86400)
        self.no_valid_set = cfg.get("dataset", {}).get("no_valid_set", False)
        self.neighborlist_backend = self.cfg.get("dataset", {}).get(
            "neighborlist_backend", "matscipy"
        )
        self._for_dataset_config = {
            "cutoff": float(cfg["model"]["config"].get("cutoff", 6.0)),
            "max_neighbors": cfg["model"]["config"].get("max_neighbors", None),
            "keyspec": self.keyspec,
            "target_property": self.target_property,
            "embedding_property": self.embedding_property,
            "universal_embedding": self.cfg.get("model", {})
            .get("config", {})
            .get("universal_embedding", None),
            "neighborlist_backend": self.neighborlist_backend,
        }
        logging.info(f"Neighborlist backend is {self.neighborlist_backend}")

    def _stage_cache_exists(self, stage: str) -> bool:
        return any(
            any(directory.glob(f"{stage}_shard*.lmdb")) for directory in self.shard_dirs
        )

    def _all_required_caches_exist(self) -> bool:
        required_stages = ["train"]
        if not self.no_valid_set:
            required_stages.append("valid")
        test_files = self.cfg.get("dataset", {}).get("test_files")
        if test_files:
            num_test_sets = len(test_files) if isinstance(test_files, list) else 1
            required_stages.extend(f"test{idx}" for idx in range(num_test_sets))
        return all(self._stage_cache_exists(stage) for stage in required_stages)

    @rank_zero_only
    def prepare_data(self):
        if self.storage_mode == "lmdb":
            for d in self.shard_dirs:
                d.mkdir(parents=True, exist_ok=True)
            if self._all_required_caches_exist():
                logging.info(
                    "[prepare_data] Existing LMDB cache found; "
                    "skipping raw data reading."
                )
                self.threeAtomsList = None
                return

        logging.info("[prepare_data] Reading raw atoms from source files...")
        if self.threeAtomsList is None:
            self.threeAtomsList = tace_read_all_files(
                self.cfg,
                self.target_property,
                self.embedding_property,
                self.keyspec,
                in_datamodule=True,
            )

    def setup(self, stage: Optional[str] = None):
        """
        Key logic:
         - In LMDB mode: do NOT broadcast threeAtomsList. Only rank0 will have it.
           All ranks will call create_graphs(...) (with atoms_list possibly None), which
           will either find existing LMDB shards (and load) or rank0 will create them.
         - In memory mode: broadcast threeAtomsList to other ranks.
        """
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        sync_memory_datasets = (
            self.storage_mode == "memory" and world_size > 1 and dist.is_initialized()
        )

        # === memory mode ===
        if self.storage_mode != "lmdb" and world_size > 1 and dist.is_initialized():
            if rank == 0:
                objects = [self.threeAtomsList]
            else:
                objects = [None]
            logging.info("Start broadcast of threeAtomsList (memory mode)")
            dist.broadcast_object_list(objects, src=0)
            logging.info("End broadcast of threeAtomsList (memory mode)")
            self.threeAtomsList = objects[0]
            dist.barrier()

        if stage in (None, "fit"):
            # TRAIN
            atoms_for_train = (
                self.threeAtomsList[0]
                if (self.threeAtomsList and self.threeAtomsList[0] is not None)
                else None
            )
            if self.train_dataset is None or sync_memory_datasets:
                self.train_dataset = create_graphs(
                    atoms_for_train,
                    self.element,
                    self._for_dataset_config,
                    "train",
                    self.shard_dirs,
                    storage_mode=self.storage_mode,
                    shard_size=self.shard_size,
                    lmdb_wait_timeout=self.lmdb_wait_timeout,
                    avg_graph_size_in_KB=self.avg_graph_size_in_KB,
                    cache_size=self.cache_size,
                    existing_dataset=self.train_dataset,
                )

            logging.info(
                f"Rank {rank}: Number of configs in train: {len(self.train_dataset)}"
            )
            if rank == 0 and self.threeAtomsList and self.threeAtomsList[0] is not None:
                self.threeAtomsList[0] = None
                gc.collect()

            # VALID
            if not self.no_valid_set:
                atoms_for_valid = (
                    self.threeAtomsList[1]
                    if (self.threeAtomsList and self.threeAtomsList[1] is not None)
                    else None
                )
                if self.val_dataset is None or sync_memory_datasets:
                    self.val_dataset = create_graphs(
                        atoms_for_valid,
                        self.element,
                        self._for_dataset_config,
                        "valid",
                        self.shard_dirs,
                        storage_mode=self.storage_mode,
                        shard_size=self.shard_size,
                        lmdb_wait_timeout=self.lmdb_wait_timeout,
                        avg_graph_size_in_KB=self.avg_graph_size_in_KB,
                        cache_size=self.cache_size,
                        existing_dataset=self.val_dataset,
                    )
                logging.info(
                    f"Rank {rank}: Number of configs in valid: {len(self.val_dataset)}"
                )
                if (
                    rank == 0
                    and self.threeAtomsList
                    and self.threeAtomsList[1] is not None
                ):
                    self.threeAtomsList[1] = None
                    gc.collect()
            else:
                self.val_dataset = None

            # TEST
            atoms_for_test_container = (
                self.threeAtomsList[2]
                if (self.threeAtomsList and self.threeAtomsList[2] is not None)
                else None
            )
            if sync_memory_datasets:
                if rank == 0:
                    existing_test_datasets = self.test_datasets
                    num_test_sets = len(existing_test_datasets or [])
                    if (
                        existing_test_datasets is None
                        and atoms_for_test_container is not None
                    ):
                        num_test_sets = len(atoms_for_test_container)
                    num_test_sets_obj = [num_test_sets]
                else:
                    existing_test_datasets = None
                    num_test_sets_obj = [None]
                dist.broadcast_object_list(num_test_sets_obj, src=0)

                self.test_datasets = []
                for idx in range(int(num_test_sets_obj[0])):
                    test_atoms_list = (
                        atoms_for_test_container[idx]
                        if atoms_for_test_container is not None
                        else None
                    )
                    existing_test_dataset = (
                        existing_test_datasets[idx]
                        if existing_test_datasets is not None
                        else None
                    )
                    self.test_datasets.append(
                        create_graphs(
                            test_atoms_list,
                            self.element,
                            self._for_dataset_config,
                            f"test{idx}",
                            self.shard_dirs,
                            storage_mode=self.storage_mode,
                            shard_size=self.shard_size,
                            lmdb_wait_timeout=self.lmdb_wait_timeout,
                            avg_graph_size_in_KB=self.avg_graph_size_in_KB,
                            cache_size=self.cache_size,
                            existing_dataset=existing_test_dataset,
                        )
                    )
            elif self.test_datasets is None:
                self.test_datasets = []
                if not self.presave_test_sets:
                    if atoms_for_test_container is not None:
                        for idx, test_atoms_list in enumerate(atoms_for_test_container):
                            test_dataset = create_graphs(
                                test_atoms_list,
                                self.element,
                                self._for_dataset_config,
                                f"test{idx}",
                                self.shard_dirs,
                                storage_mode=self.storage_mode,
                                shard_size=self.shard_size,
                                lmdb_wait_timeout=self.lmdb_wait_timeout,
                                avg_graph_size_in_KB=self.avg_graph_size_in_KB,
                                cache_size=self.cache_size,
                            )
                            self.test_datasets.append(test_dataset)
                        self.presave_test_sets = True
                else:
                    for idx in range(self.num_test_sets):
                        test_dataset = create_graphs(
                            None,
                            self.element,
                            self._for_dataset_config,
                            f"test{idx}",
                            self.shard_dirs,
                            storage_mode=self.storage_mode,
                            shard_size=self.shard_size,
                            lmdb_wait_timeout=self.lmdb_wait_timeout,
                            avg_graph_size_in_KB=self.avg_graph_size_in_KB,
                            cache_size=self.cache_size,
                        )
                        self.test_datasets.append(test_dataset)
            if rank == 0 and self.threeAtomsList and self.threeAtomsList[2] is not None:
                self.threeAtomsList[2] = None
                gc.collect()

    def train_dataloader(self):
        return instantiate(
            self.cfg["dataset"]["train_dataloader"], dataset=self.train_dataset
        )

    def statistics_dataloader(self):
        return build_statistics_dataloader(
            self.cfg, self.train_dataset, self.target_property
        )

    def val_dataloader(self):
        if self.no_valid_set or self.val_dataset is None:
            # if None, will warning dataloader's length is zero, just ignore it
            return instantiate(self.cfg["dataset"]["valid_dataloader"], dataset=[])

        return instantiate(
            self.cfg["dataset"]["valid_dataloader"], dataset=self.val_dataset
        )

    def test_dataloader(self):
        if self.test_datasets is None:
            return None
        return [
            instantiate(self.cfg["dataset"]["test_dataloader"], dataset=td)
            for td in self.test_datasets
        ]


# === Datamodule Builder ===
def build_datamodule(
    cfg: Dict,
    atomic_numbers: List[int],
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeySpecification,
    threeAtomsList,
):
    element = build_element_lookup(atomic_numbers)
    datamodule = GraphDataModule(
        cfg,
        element,
        target_property,
        keyspec,
        embedding_property,
        threeAtomsList,
    )
    return datamodule

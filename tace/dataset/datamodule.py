"""Graph datasets with bounded preprocessing and optional online construction."""

import logging
import multiprocessing
import time
from collections import deque
from collections.abc import Sequence
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from contextlib import ExitStack
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
from lightning.pytorch import LightningDataModule
from torch.utils.data import Dataset

from .augmentation import AugmentedDataset, validate_augmentations
from .cache import graph_cache_directories, read_cache_metadata
from .dataloader import build_graph_dataloader, build_statistics_dataloader
from .element import build_element_lookup
from .graph import batch_geometry, from_records
from .neighbour_list import get_neighborhood_batch
from .quantity import KeySpecification
from .read import tace_read_all_files
from .records import record_from_atoms
from .split import IndexedSource
from .store import GraphDatasetLMDB, build_graph_store, write_graphs


class GraphProgress:
    """Log graph counts, throughput, and remaining time every ten seconds.

    Parameters
    ----------
    total : int
        Number of structures to process.
    stage : str
        Dataset split name.
    """

    def __init__(self, total, stage):
        self.total, self.stage = total, stage
        self.count = 0
        self.started = self.updated = time.monotonic()

    def update(self, count, *, finished=False):
        self.count += count
        now = time.monotonic()
        if not finished and (now - self.updated < 10 or self.count >= self.total):
            return
        if not self.count or (
            dist.is_available() and dist.is_initialized() and dist.get_rank() != 0
        ):
            return
        elapsed = max(now - self.started, 1e-9)
        rate = self.count / elapsed
        remaining = max(0, self.total - self.count) / rate
        logging.info(
            "%s graphs: %d/%d (%.1f%%), %.1f graphs/s, elapsed %s, ETA %s%s",
            self.stage,
            self.count,
            self.total,
            100 * self.count / max(1, self.total),
            rate,
            timedelta(seconds=int(elapsed)),
            timedelta(seconds=int(remaining)),
            "; complete" if finished else "",
        )
        self.updated = now


def read_records(source, indices):
    """Read a batch without constructing ASE calculators."""
    if hasattr(source, "get_records"):
        return source.get_records(indices)
    return [record_from_atoms(source[index]) for index in indices]


def graph_devices(device, backend):
    """Resolve a scalar device, CUDA device sequence, or ``'all'``."""
    if device is None and backend == "alchemiops":
        device = "all" if torch.cuda.is_available() else "cpu"
    multiple = isinstance(device, Sequence) and not isinstance(device, (str, bytes))
    if not multiple and device != "all":
        return [device]
    if backend != "alchemiops":
        raise ValueError(
            "Multiple graph devices require neighborlist_backend='alchemiops'."
        )
    count = torch.cuda.device_count()
    devices = list(device) if multiple else [f"cuda:{index}" for index in range(count)]
    if not devices:
        raise ValueError(
            "Multi-GPU graph construction requires at least one CUDA device."
        )
    resolved = []
    for value in devices:
        item = torch.device(value)
        index = (
            torch.cuda.current_device()
            if item.type == "cuda" and item.index is None and count
            else item.index
        )
        if item.type != "cuda" or index is None or not 0 <= index < count:
            raise ValueError(f"Graph device {value!r} is not a visible CUDA device.")
        name = f"cuda:{index}"
        if name in resolved:
            raise ValueError(f"Duplicate graph device: {name}")
        resolved.append(name)
    return resolved


def initialize_graph_worker(device):
    """Give each spawned neighbor worker its own CUDA context."""
    torch.set_num_threads(1)
    torch.cuda.set_device(device)


def write_gpu_graphs(
    source,
    element,
    options,
    device,
    partitions,
    directories,
    stage,
    batch_size,
    writer_options,
    progress_connection,
):
    """Read, construct, and write assigned graph partitions on one GPU."""
    count = 0
    updated = time.monotonic()

    def progress(written):
        nonlocal count, updated
        count += written
        now = time.monotonic()
        if now - updated >= 1:
            progress_connection.send(count)
            updated = now

    completed = []
    for index, start, stop in partitions:
        offset = index % len(directories)
        part_directories = directories[offset:] + directories[:offset]
        graphs = iter_graphs(
            IndexedSource(source, range(start, stop)),
            element,
            options,
            batch_size=batch_size,
            device=device,
        )
        dataset = write_graphs(
            graphs,
            part_directories,
            f"{stage}_shard{index:06d}_part",
            progress=progress,
            **writer_options,
        )
        completed.append((index, dataset.metadata))
        dataset.close()
    progress_connection.send(count)
    progress_connection.close()
    return completed


def write_multi_gpu_graphs(
    source, element, options, devices, shard_dirs, stage, batch_size, **writer_options
):
    """Build independent LMDB partitions and publish their ordered index."""
    if batch_size < 1:
        raise ValueError("Graph batch_size must be positive.")
    progress = None

    def build(directories, stage, writer_options):
        nonlocal progress
        size = len(source)
        per_device = max(batch_size, (size + len(devices) - 1) // len(devices))
        # Balance several partitions per GPU without creating many tiny LMDBs.
        target_size = (size + 4 * len(devices) - 1) // (4 * len(devices))
        part_size = min(
            writer_options["shard_size"],
            max(128 * batch_size, target_size),
            per_device,
        )
        if part_size < 1:
            raise ValueError("Graph shard_size must be positive.")
        num_parts = (size + part_size - 1) // part_size
        num_parts = min(
            size, ((num_parts + len(devices) - 1) // len(devices)) * len(devices)
        )
        partitions = [
            (index, size * index // num_parts, size * (index + 1) // num_parts)
            for index in range(num_parts)
        ]
        logging.info(
            "Reading, building, and writing %s graphs across %s", stage, devices
        )
        progress = GraphProgress(size, stage)
        context = multiprocessing.get_context("spawn")
        with ExitStack() as stack:
            futures, receivers, counts = [], [], []
            for slot, device in enumerate(devices):
                assigned = partitions[slot :: len(devices)]
                if not assigned:
                    continue
                executor = stack.enter_context(
                    ProcessPoolExecutor(
                        max_workers=1,
                        mp_context=context,
                        initializer=initialize_graph_worker,
                        initargs=(device,),
                    )
                )
                receiver, sender = context.Pipe(duplex=False)
                stack.enter_context(receiver)
                stack.enter_context(sender)
                receivers.append(receiver)
                counts.append(0)
                futures.append(
                    executor.submit(
                        write_gpu_graphs,
                        source,
                        element,
                        options,
                        device,
                        assigned,
                        directories,
                        stage,
                        batch_size,
                        writer_options,
                        sender,
                    )
                )
            completed, pending = [], set(futures)
            while pending:
                done, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
                for slot, receiver in enumerate(receivers):
                    count = counts[slot]
                    while receiver.poll():
                        count = receiver.recv()
                    progress.update(count - counts[slot])
                    counts[slot] = count
                for future in done:
                    completed.extend(future.result())
        return [metadata for _, metadata in sorted(completed)]

    dataset = build_graph_store(build, shard_dirs, stage, **writer_options)
    if progress is not None:
        progress.update(0, finished=True)
    return dataset


def iter_gpu_graphs(source, element, options, chunks, devices):
    """Build independent CUDA batches with bounded, ordered results.

    Only packed geometry and NumPy neighbor arrays cross process boundaries.
    Source reads, labels, graph assembly, and cache writes stay in the parent.
    """
    with ExitStack() as stack:
        executors = [
            stack.enter_context(
                ProcessPoolExecutor(
                    max_workers=1,
                    mp_context=multiprocessing.get_context("spawn"),
                    initializer=initialize_graph_worker,
                    initargs=(device,),
                )
            )
            for device in devices
        ]
        pending = deque()
        try:
            for batch, indices in enumerate(chunks):
                records = read_records(source, indices)
                slot = batch % len(devices)
                future = executors[slot].submit(
                    get_neighborhood_batch,
                    **batch_geometry(records),
                    cutoff=options["cutoff"],
                    max_neighbors=options.get("max_neighbors"),
                    backend="alchemiops",
                    device=devices[slot],
                )
                pending.append((records, future))
                if len(pending) >= 2 * len(devices):
                    records, future = pending.popleft()
                    yield from from_records(
                        element, records, neighborhood=future.result(), **options
                    )
            while pending:
                records, future = pending.popleft()
                yield from from_records(
                    element, records, neighborhood=future.result(), **options
                )
        finally:
            for _, future in pending:
                future.cancel()


def iter_graphs(source, element, options, batch_size=256, workers=0, device=None):
    """Build bounded batches; source database access stays in the calling thread."""
    if batch_size < 1 or workers < 0:
        raise ValueError("Graph batch_size must be positive and workers non-negative.")
    chunks = (
        range(start, min(start + batch_size, len(source)))
        for start in range(0, len(source), batch_size)
    )
    devices = graph_devices(device, options.get("neighborlist_backend", "matscipy"))
    if len(devices) > 1:
        logging.info("Building graph batches across %s", ", ".join(devices))
        yield from iter_gpu_graphs(source, element, options, chunks, devices)
        return
    device = devices[0]
    if not workers or (device is not None and torch.device(device).type != "cpu"):
        for indices in chunks:
            yield from from_records(
                element, read_records(source, indices), **options, device=device
            )
        return
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = []
        for indices in chunks:
            records = read_records(source, indices)
            pending.append(
                executor.submit(
                    from_records, element, records, **options, device=device
                )
            )
            if len(pending) >= 2 * workers:
                yield from pending.pop(0).result()
        for future in pending:
            yield from future.result()


class OnTheFlyGraphDataset(Dataset):
    """Construct graph batches from an indexed structure source.

    Parameters
    ----------
    source : Sequence
        Structures or an indexed record source.
    options : dict
        Arguments forwarded to graph construction.
    device : str, optional
        Neighbor-search device. CUDA requires a loader without worker processes.
    """

    def __init__(self, source, options, device=None, fields=None):
        if device == "all" or (
            isinstance(device, Sequence) and not isinstance(device, (str, bytes))
        ):
            raise ValueError(
                "on_the_fly graph construction requires one explicit device; "
                "use storage_mode='lmdb' or 'memory' for multi-GPU preprocessing."
            )
        self.source = source
        self.options = options
        if device is None and options.get("neighborlist_backend") == "alchemiops":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.fields = None if fields is None else frozenset(fields)

    def __len__(self):
        return len(self.source)

    def __getitems__(self, indices):
        graphs = from_records(
            None, read_records(self.source, indices), **self.options, device=self.device
        )
        if self.fields is not None:
            for graph in graphs:
                for key in set(graph.keys()) - self.fields - {"num_nodes"}:
                    del graph[key]
        return graphs

    def __getitem__(self, index):
        return self.__getitems__([index])[0]

    def project(self, fields):
        return type(self)(self.source, self.options, self.device, fields)


def create_graphs(
    atoms_list,
    element,
    for_dataset,
    stage,
    shard_dirs,
    storage_mode="lmdb",
    shard_size=100000,
    cache_size=0,
    existing_dataset=None,
    batch_size=256,
    workers=0,
    device=None,
    transaction_bytes=16 * 1024**2,
    shard_bytes=2 * 1024**3,
    max_buffer_bytes=64 * 1024**2,
    max_open_shards=32,
):
    """Load complete graph shards or build them with bounded memory."""
    if existing_dataset is not None:
        return existing_dataset
    if storage_mode not in {"lmdb", "memory", "on_the_fly"}:
        raise ValueError(f"Unknown storage_mode: {storage_mode!r}")
    if storage_mode == "lmdb" and read_cache_metadata(shard_dirs, stage) is not None:
        return GraphDatasetLMDB.from_directories(
            shard_dirs, stage, cache_size=cache_size, max_open_shards=max_open_shards
        )
    if atoms_list is None:
        raise ValueError(f"No source or complete graph cache available for {stage}.")
    if storage_mode == "on_the_fly":
        return OnTheFlyGraphDataset(atoms_list, for_dataset, device)
    devices = graph_devices(device, for_dataset.get("neighborlist_backend", "matscipy"))
    if (
        storage_mode == "memory"
        and dist.is_available()
        and dist.is_initialized()
        and dist.get_world_size() > 1
        and len(devices) > 1
    ):
        raise ValueError(
            "DDP memory-mode preprocessing cannot use multiple graph GPUs: "
            "every rank would build the full dataset on all selected devices. "
            "Use storage_mode='lmdb' for rank-zero multi-GPU preprocessing, "
            "or select one rank-local graph_device."
        )
    writer_options = dict(
        shard_size=shard_size,
        transaction_bytes=transaction_bytes,
        shard_bytes=shard_bytes,
        cache_size=cache_size,
        max_buffer_bytes=max_buffer_bytes,
        max_open_shards=max_open_shards,
    )
    if storage_mode == "lmdb" and len(devices) > 1:
        if workers < 0:
            raise ValueError("Graph workers must be non-negative.")
        return write_multi_gpu_graphs(
            atoms_list,
            element,
            for_dataset,
            devices,
            shard_dirs,
            stage,
            batch_size,
            **writer_options,
        )
    graphs = iter_graphs(atoms_list, element, for_dataset, batch_size, workers, device)
    progress = GraphProgress(len(atoms_list), stage)
    if storage_mode == "memory":
        dataset = GraphDatasetLMDB([], in_memory=True)
        dataset.data_list = []
        for graph in graphs:
            dataset.data_list.append(graph)
            if len(dataset.data_list) % batch_size == 0:
                progress.update(batch_size)
        dataset.length = len(dataset.data_list)
        progress.update(dataset.length % batch_size, finished=True)
        return dataset
    logging.info(
        "Building %d %s graphs in batches of %d", len(atoms_list), stage, batch_size
    )
    dataset = write_graphs(
        graphs,
        shard_dirs,
        stage,
        progress=progress.update,
        **writer_options,
    )
    progress.update(0, finished=True)
    return dataset


class GraphDataModule(LightningDataModule):
    """Read indexed structures and prepare reusable FP64 graph datasets.

    Parameters
    ----------
    cfg : dict
        Dataset, model, and loader configuration.
    element : TorchElement or None
        Configured chemical elements; inferred when absent.
    target_property, embedding_property : list of str
        Labels and input properties.
    keyspec : KeySpecification
        Source property names.
    threeAtomsList : ThreeDataset, optional
        Preselected indexed training, validation, and test sources.
    """

    def __init__(
        self,
        cfg,
        element,
        target_property,
        keyspec,
        embedding_property,
        threeAtomsList=None,
    ):
        super().__init__()
        self.cfg = cfg
        self.element = element
        self._elements_ready = False
        self.keyspec = keyspec
        self.target_property = target_property
        self.embedding_property = embedding_property
        self.threeAtomsList = threeAtomsList
        self._test_sources = (
            threeAtomsList[2]
            if threeAtomsList is not None and threeAtomsList[2]
            else None
        )
        self.train_dataset = self.val_dataset = self.test_datasets = None
        data = cfg["dataset"]
        self.storage_mode = data.get("storage_mode", "memory")
        self.no_valid_set = data.get("no_valid_set", False)
        self.augmentation = validate_augmentations(data.get("augmentation", []))
        self.cache_size = data.get("cache_size", 0)
        self.build_options = dict(
            batch_size=data.get("graph_batch_size", 256),
            workers=data.get("graph_workers", 0),
            device=data.get("graph_device"),
            shard_size=data.get("shard_size", 100000),
            transaction_bytes=data.get("transaction_bytes", 16 * 1024**2),
            shard_bytes=data.get("shard_bytes", 2 * 1024**3),
            max_buffer_bytes=data.get("max_buffer_bytes", 64 * 1024**2),
            max_open_shards=data.get("max_open_shards", 32),
        )
        model = cfg["model"]["config"]
        self.graph_options = dict(
            cutoff=float(model.get("cutoff", 6.0)),
            max_neighbors=model.get("max_neighbors"),
            keyspec=keyspec,
            target_property=target_property,
            embedding_property=embedding_property,
            universal_embedding=model.get("universal_embedding"),
            neighborlist_backend=data.get("neighborlist_backend", "matscipy"),
        )
        if self.storage_mode not in {"lmdb", "memory", "on_the_fly"}:
            raise ValueError(f"Unknown storage_mode: {self.storage_mode!r}")
        if self.storage_mode == "on_the_fly":
            logging.info(
                "Online graph construction enabled; benchmark loader throughput before large-scale training."
            )

    def cache_directories(self, stage):
        return graph_cache_directories(
            self.cfg,
            self.target_property,
            self.embedding_property,
            self.keyspec,
            stage=stage,
        )

    @property
    def shard_dirs(self):
        """Keep the training cache path available without indexing it at construction."""
        return self.cache_directories("train")

    def prepare_data(self):
        # Lightning may call this only on rank zero. setup opens independent,
        # lightweight source indexes on each rank; no Atoms lists are broadcast.
        return None

    def sources(self):
        if self.threeAtomsList is None:
            self.threeAtomsList = tace_read_all_files(
                self.cfg,
                self.target_property,
                self.embedding_property,
                self.keyspec,
                in_datamodule=True,
                stages=("train", "valid"),
            )
        return self.threeAtomsList

    def source(self, stage):
        if stage == "train":
            return self.sources()[0]
        if stage == "valid":
            return self.sources()[1]
        if self._test_sources is None:
            self._test_sources = tace_read_all_files(
                self.cfg,
                self.target_property,
                self.embedding_property,
                self.keyspec,
                in_datamodule=True,
                stages=("test",),
            ).test
            if self.threeAtomsList is not None:
                self.threeAtomsList[2] = self._test_sources
        return self._test_sources[int(stage[4:])]

    def dataset(self, stage):
        directories = self.cache_directories(stage)
        distributed = dist.is_available() and dist.is_initialized()
        # Only rank zero writes shared caches. The writer uses a filesystem lock
        # as well, so independent jobs cannot publish over one another.
        writer = not distributed or dist.get_rank() == 0
        result = None
        error = None
        if self.storage_mode != "lmdb" or writer:
            try:
                complete = (
                    self.storage_mode == "lmdb"
                    and read_cache_metadata(directories, stage) is not None
                )
                result = create_graphs(
                    None if complete else self.source(stage),
                    self.element,
                    self.graph_options,
                    stage,
                    directories,
                    storage_mode=self.storage_mode,
                    cache_size=self.cache_size,
                    **self.build_options,
                )
            except Exception as exc:
                if not distributed or self.storage_mode != "lmdb":
                    raise
                error = f"{type(exc).__name__}: {exc}"
        if distributed and self.storage_mode == "lmdb":
            status = [error]
            dist.broadcast_object_list(status, src=0)
            if status[0] is not None:
                raise RuntimeError(f"Graph preparation failed for {stage}: {status[0]}")
            if not writer:
                result = GraphDatasetLMDB.from_directories(
                    directories,
                    stage,
                    cache_size=self.cache_size,
                    max_open_shards=self.build_options["max_open_shards"],
                )
        logging.info(
            "%s dataset: %d structures (%s)", stage, len(result), self.storage_mode
        )
        return result

    def setup(self, stage=None):
        if stage in (None, "fit", "validate"):
            if self.train_dataset is None:
                self.train_dataset = self.dataset("train")
            if not self.no_valid_set and self.val_dataset is None:
                self.val_dataset = self.dataset("valid")
            if not self._elements_ready:
                self.update_elements()
                self._elements_ready = True
        if stage in (None, "test"):
            if self.test_datasets is None:
                tests = self.cfg["dataset"].get("test_files")
                count = (
                    len(tests) if isinstance(tests, (list, tuple)) else int(bool(tests))
                )
                if self._test_sources is not None:
                    count = len(self._test_sources)
                self.test_datasets = [
                    self.dataset(f"test{index}") for index in range(count)
                ]

    def update_elements(self):
        present = set()
        for dataset in (self.train_dataset, self.val_dataset):
            if dataset is None:
                continue
            metadata = getattr(dataset, "metadata", None)
            if metadata is not None and "atomic_numbers" in metadata:
                present.update(metadata["atomic_numbers"])
            elif isinstance(dataset, OnTheFlyGraphDataset):
                source = dataset.source
                for start in range(0, len(source), self.build_options["batch_size"]):
                    for record in read_records(
                        source,
                        range(
                            start,
                            min(start + self.build_options["batch_size"], len(source)),
                        ),
                    ):
                        present.update(record.numbers.tolist())
            else:
                for graph in dataset:
                    present.update(graph.atomic_numbers.tolist())
        configured = self.cfg["model"]["config"].get("atomic_numbers")
        if configured is not None and not present.issubset(configured):
            raise ValueError(
                f"Configured elements omit dataset atomic numbers: {sorted(present - set(configured))}"
            )
        present.update(configured or [])
        for fidelity in self.cfg["model"]["config"].get("fidelity", []):
            present.update(int(z) for z in (fidelity.get("atomic_energy") or {}))
        if not present:
            raise ValueError("The training dataset contains no atoms.")
        self.element = build_element_lookup(sorted(present))

    def train_dataloader(self):
        dataset = self.train_dataset
        if self.augmentation:
            dataset = AugmentedDataset(dataset, self.augmentation)
        return build_graph_dataloader(self.cfg["dataset"]["train_dataloader"], dataset)

    def statistics_dataloader(self):
        return build_statistics_dataloader(
            self.cfg, self.train_dataset, self.target_property
        )

    def val_dataloader(self):
        return build_graph_dataloader(
            self.cfg["dataset"]["valid_dataloader"],
            self.val_dataset if self.val_dataset is not None else [],
        )

    def test_dataloader(self):
        if self.test_datasets is None:
            self.setup("test")
        return [
            build_graph_dataloader(self.cfg["dataset"]["test_dataloader"], dataset)
            for dataset in self.test_datasets
        ]


def build_datamodule(
    cfg,
    atomic_numbers,
    target_property,
    embedding_property,
    keyspec: KeySpecification,
    threeAtomsList=None,
):
    element = (
        build_element_lookup(atomic_numbers) if atomic_numbers is not None else None
    )
    return GraphDataModule(
        cfg, element, target_property, keyspec, embedding_property, threeAtomsList
    )


def load_cached_datamodule(cfg, target_property, embedding_property, keyspec):
    """Attach training caches without opening structure sources."""
    if cfg["dataset"].get("storage_mode", "memory") != "lmdb":
        return None
    datamodule = build_datamodule(
        cfg, None, target_property, embedding_property, keyspec
    )
    stages = ["train"] + ([] if datamodule.no_valid_set else ["valid"])
    if any(
        read_cache_metadata(datamodule.cache_directories(stage), stage) is None
        for stage in stages
    ):
        return None
    datamodule.setup("fit")
    return datamodule

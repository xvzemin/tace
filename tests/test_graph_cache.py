"""Graph construction, cache storage and restart behavior."""

import json
import multiprocessing
import os
import pickle
import shutil
import struct
import zlib
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.data import covalent_radii
from filelock import FileLock, Timeout
from omegaconf import OmegaConf
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import tace.dataset.cache as cache_module
import tace.dataset.statistics as statistics_module
from tace.dataset.cache import (
    ARRAY_CODEC,
    COMPRESSED_ARRAY_CODEC,
    LMDB_MODES,
    graph_cache_directories,
    read_cache_metadata,
    statistics_cache_key,
)
from tace.dataset.datamodule import GraphDatasetLMDB, create_graphs
from tace.dataset.element import build_element_lookup
from tace.dataset.filter import filter_graphs, filter_options
from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KeySpecification
from tace.dataset.store import (
    build_graph_store,
    compress_array_record,
    decode_graph,
    encode_graph,
    write_graphs,
)

from .testing import file_snapshot, reader_keys, write_reader_database


@pytest.mark.parametrize("visible_devices", [0, 1, 3])
def test_graph_device_auto_uses_visible_count(monkeypatch, visible_devices):
    from tace.dataset.datamodule import OnTheFlyGraphDataset, graph_devices

    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible_devices)

    def no_runtime_probe():
        raise AssertionError("Device discovery must not initialize the CUDA runtime")

    monkeypatch.setattr(torch.cuda, "is_available", no_runtime_probe)
    monkeypatch.setattr(torch.cuda, "current_device", no_runtime_probe)
    assert graph_devices(None, "alchemiops") == (
        [f"cuda:{index}" for index in range(visible_devices)]
        if visible_devices
        else ["cpu"]
    )
    dataset = OnTheFlyGraphDataset([], {"neighborlist_backend": "alchemiops"})
    assert dataset.device == ("cuda" if visible_devices else "cpu")
    assert graph_devices(None, "matscipy") == [None]


def test_graph_device_selection(monkeypatch):
    from tace.dataset.datamodule import OnTheFlyGraphDataset, graph_devices

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("Unexpected CUDA initialization")
    )
    assert graph_devices(None, "ase") == [None]
    assert graph_devices("cpu", "alchemiops") == ["cpu"]
    assert graph_devices("all", "alchemiops") == ["cuda:0", "cuda:1"]
    for devices in (
        [1, 0],
        (1, 0),
        OmegaConf.create([1, 0]),
        OmegaConf.create(["cuda:1", "cuda:0"]),
    ):
        assert graph_devices(devices, "alchemiops") == ["cuda:1", "cuda:0"]
    assert graph_devices([0, "cuda:1"], "alchemiops") == ["cuda:0", "cuda:1"]
    for device in (0, 1):
        expected = f"cuda:{device}"
        assert graph_devices(device, "alchemiops") == [expected]
        assert (
            OnTheFlyGraphDataset(
                [], {"neighborlist_backend": "alchemiops"}, device
            ).device
            == expected
        )
    for invalid in ([], ["cuda:0", "cuda:0"], ["cuda", "cuda:1"], ["cpu"]):
        with pytest.raises(ValueError):
            graph_devices(invalid, "alchemiops")
    with pytest.raises(ValueError, match="Duplicate"):
        graph_devices([0, "cuda:0"], "alchemiops")
    for invalid in ([2], ["cuda:2"]):
        with pytest.raises(ValueError, match="not a visible CUDA device"):
            graph_devices(invalid, "alchemiops")
    with pytest.raises(ValueError, match="alchemiops"):
        graph_devices("all", "matscipy")
    for devices in ("all", ["cuda:0", "cuda:1"], [0, 1]):
        with pytest.raises(ValueError, match="one explicit device"):
            OnTheFlyGraphDataset([], {"neighborlist_backend": "alchemiops"}, devices)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="at least one CUDA"):
        graph_devices("all", "alchemiops")


@pytest.mark.parametrize(
    "cuda_available,device,expected_device,uses_thread_pool",
    [
        (True, None, "cuda:0", False),
        (False, None, "cpu", True),
        (True, "cpu", "cpu", True),
        (True, "cuda:1", "cuda:1", False),
    ],
)
def test_alchemiops_iterator_resolves_device_before_threading(
    monkeypatch, cuda_available, device, expected_device, uses_thread_pool
):
    import tace.dataset.datamodule as module

    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: int(cuda_available))
    pool_calls, observed_devices = [], []
    real_thread_pool = module.ThreadPoolExecutor

    def thread_pool(*args, **kwargs):
        pool_calls.append(True)
        return real_thread_pool(*args, **kwargs)

    def from_records(element, records, *, device, **options):
        observed_devices.append(device)
        return records

    def no_gpu_processes(*args, **kwargs):
        raise AssertionError("Single-device construction must not spawn GPU workers")

    monkeypatch.setattr(module, "ThreadPoolExecutor", thread_pool)
    monkeypatch.setattr(module, "from_records", from_records)
    monkeypatch.setattr(module, "iter_gpu_graphs", no_gpu_processes)
    monkeypatch.setattr(
        module, "read_records", lambda source, indices: [source[i] for i in indices]
    )
    assert list(
        module.iter_graphs(
            [0, 1, 2],
            None,
            {"neighborlist_backend": "alchemiops"},
            batch_size=2,
            workers=2,
            device=device,
        )
    ) == [0, 1, 2]
    assert bool(pool_calls) == uses_thread_pool
    assert observed_devices == [expected_device, expected_device]


@pytest.mark.parametrize("device", [None, "all", ["cuda:0", "cuda:1"]])
def test_distributed_memory_rejects_multiple_graph_gpus(monkeypatch, device):
    import tace.dataset.datamodule as module

    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def no_graph_work(*args, **kwargs):
        raise AssertionError("Reject this configuration before starting graph work")

    monkeypatch.setattr(module, "iter_graphs", no_graph_work)
    with pytest.raises(ValueError, match="DDP memory-mode.*storage_mode='lmdb'"):
        module.create_graphs(
            [object()],
            None,
            {"neighborlist_backend": "alchemiops"},
            "train",
            [],
            storage_mode="memory",
            device=device,
        )


@pytest.mark.parametrize(
    "initialized,world_size,device,visible_devices",
    [
        (True, 2, "cuda:0", 2),
        (True, 2, ["cuda:0"], 2),
        (True, 2, None, 1),
        (True, 2, "cpu", 2),
        (True, 2, "all", 1),
        (True, 1, ["cuda:0", "cuda:1"], 2),
        (False, 2, ["cuda:0", "cuda:1"], 2),
    ],
)
def test_memory_graph_devices_preserve_supported_modes(
    monkeypatch, initialized, world_size, device, visible_devices
):
    import tace.dataset.datamodule as module

    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: initialized)
    monkeypatch.setattr(module.dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: visible_devices)
    graph = Data(num_nodes=1)
    monkeypatch.setattr(module, "iter_graphs", lambda *args, **kwargs: iter([graph]))
    dataset = module.create_graphs(
        [object()],
        None,
        {"neighborlist_backend": "alchemiops"},
        "train",
        [],
        storage_mode="memory",
        device=device,
    )
    assert len(dataset) == 1
    assert dataset[0] is graph


@pytest.fixture
def graph_progress(monkeypatch):
    import tace.dataset.datamodule as module

    reporters = []
    original = module.GraphProgress

    def record(*args, **kwargs):
        reporter = original(*args, **kwargs)
        reporters.append(reporter)
        return reporter

    monkeypatch.setattr(module, "GraphProgress", record)
    return reporters


@pytest.mark.parametrize("rank", [None, 0, 1])
def test_graph_progress_counts_and_reports_only_on_rank_zero(
    monkeypatch, capsys, caplog, rank
):
    import tace.dataset.datamodule as module

    monkeypatch.setattr(module.dist, "is_available", lambda: True)
    monkeypatch.setattr(module.dist, "is_initialized", lambda: rank is not None)
    monkeypatch.setattr(module.dist, "get_rank", lambda: rank)
    caplog.set_level("INFO")
    progress = module.GraphProgress(10, "train")
    progress.update(1)
    progress.update(3)
    if rank != 1:
        assert progress.bar.n == 4
        progress.bar.refresh()
        assert "4/10" in capsys.readouterr().err
    progress.update(1)
    progress.update(5)
    if rank != 1:
        assert progress.bar.n == 10
        assert not progress.bar.disable
    progress.update(0, finished=True)
    assert progress.count == 10
    output = capsys.readouterr().err
    if rank == 1:
        assert progress.bar is None
        assert not output
    else:
        assert progress.bar.disable
        assert "train graphs:" in output and "10/10" in output
    assert not caplog.messages


@pytest.mark.parametrize("count", [0, 3])
def test_graph_progress_finish_preserves_actual_count(capsys, count):
    from tace.dataset.datamodule import GraphProgress

    progress = GraphProgress(10, "train")
    progress.update(count, finished=True)
    progress.close()
    assert progress.count == count
    output = capsys.readouterr().err
    if count:
        assert progress.bar.n == count
        assert progress.bar.disable
        assert "3/10" in output and "10/10" not in output
    else:
        assert progress.bar is None
        assert not output


@pytest.mark.parametrize("mode", ["memory", *LMDB_MODES])
def test_create_graphs_reports_final_partial_batch_and_skips_warm_cache(
    tmp_path, monkeypatch, capsys, graph_progress, mode
):
    import tace.dataset.datamodule as module

    graph = Data(
        num_nodes=1,
        atomic_numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
    )
    monkeypatch.setattr(module, "iter_graphs", lambda *args: iter([graph] * 5))
    options = {"neighborlist_backend": "ase"}
    dataset = module.create_graphs(
        [object()] * 5,
        None,
        options,
        "train",
        [tmp_path],
        storage_mode=mode,
        batch_size=2,
        transaction_bytes=2 * len(encode_graph(graph)),
    )
    assert len(dataset) == 5
    assert len(graph_progress) == 1
    progress = graph_progress[0]
    assert progress.count == progress.bar.n == 5
    assert progress.bar.disable
    output = capsys.readouterr().err
    assert "train graphs:" in output and "5/5" in output
    dataset.close()

    if mode in LMDB_MODES:
        assert dataset.metadata["codec"] == (
            COMPRESSED_ARRAY_CODEC if mode == "clmdb" else ARRAY_CODEC
        )

        def no_graph_work(*args, **kwargs):
            pytest.fail("Warm cache must not construct graphs or progress reporters")

        monkeypatch.setattr(module, "iter_graphs", no_graph_work)
        monkeypatch.setattr(module, "GraphProgress", no_graph_work)
        reused = module.create_graphs(None, None, options, "train", [tmp_path])
        assert len(reused) == 5
        assert not capsys.readouterr().err
        reused.close()


@pytest.mark.parametrize("mode", ["memory", *LMDB_MODES])
@pytest.mark.parametrize("completed", [1, 3])
def test_interrupted_graph_build_does_not_report_completion(
    tmp_path, monkeypatch, capsys, graph_progress, mode, completed
):
    import tace.dataset.datamodule as module

    graph = Data(
        num_nodes=1,
        atomic_numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
    )

    def interrupted(*args):
        yield from [graph] * completed
        raise RuntimeError("injected source failure")

    monkeypatch.setattr(module, "iter_graphs", interrupted)
    with pytest.raises(RuntimeError, match="injected source failure"):
        module.create_graphs(
            [object()] * 5,
            None,
            {"neighborlist_backend": "ase"},
            "train",
            [tmp_path],
            storage_mode=mode,
            batch_size=2,
            transaction_bytes=1,
        )
    assert len(graph_progress) == 1
    progress = graph_progress[0]
    assert progress.count == progress.bar.n == completed
    assert progress.bar.disable
    output = capsys.readouterr().err
    assert f"{completed}/5" in output
    assert "5/5" not in output and "complete" not in output
    assert read_cache_metadata([tmp_path], "train") is None


def test_graph_cache_publication_failure_does_not_report_completion(
    tmp_path, monkeypatch, capsys, graph_progress
):
    import tace.dataset.datamodule as module
    import tace.dataset.store as store_module

    graph = Data(
        num_nodes=1,
        atomic_numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
    )
    monkeypatch.setattr(module, "iter_graphs", lambda *args: iter([graph]))

    def no_publication(*args, **kwargs):
        raise RuntimeError("injected publication failure")

    monkeypatch.setattr(store_module.os, "rename", no_publication)
    with pytest.raises(RuntimeError, match="injected publication failure"):
        module.create_graphs(
            [object()], None, {"neighborlist_backend": "ase"}, "train", [tmp_path]
        )
    progress = graph_progress[0]
    # The record was committed, but no cache completion is claimed.
    assert progress.count == progress.bar.n == 1
    assert progress.bar.disable
    output = capsys.readouterr().err
    assert "1/1" in output and "complete" not in output
    assert read_cache_metadata([tmp_path], "train") is None


@pytest.mark.parametrize("default_dtype", [torch.float32, torch.float64])
def test_graph_data_precision_and_indices(default_dtype):
    torch.set_default_dtype(default_dtype)
    atoms = Atoms("Si2", positions=[[0.1234567890123, 0, 0], [2, 0, 0]])
    atoms.info["energy"] = -100000.000123
    graph = from_atoms(
        build_element_lookup([14]),
        atoms,
        cutoff=3.0,
        max_neighbors=None,
        keyspec=KeySpecification.from_defaults(),
        target_property=["energy"],
        embedding_property=[],
        neighborlist_backend="ase",
    )
    assert graph.energy.item() == atoms.info["energy"]
    assert graph.positions[0, 0].item() == atoms.positions[0, 0]
    assert "node_attrs" not in graph
    assert "node_type" not in graph
    assert graph.num_nodes == 2
    for _, value in graph.items():
        if torch.is_tensor(value) and value.is_floating_point():
            assert value.dtype == torch.float64
    for name in ("atomic_numbers", "edge_index", "edge_shifts", "fidelity_idx"):
        assert graph[name].dtype == torch.int64
    assert graph.pbc.dtype == torch.bool


@pytest.mark.parametrize("context", multiprocessing.get_all_start_methods())
def test_graph_cache_precision_and_process_reuse(tmp_path, context):
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2, 0, 0]])
    atoms.info["energy"] = -100000.000123
    options = {
        "cutoff": 3.0,
        "max_neighbors": None,
        "keyspec": KeySpecification.from_defaults(),
        "target_property": ["energy"],
        "embedding_property": [],
        "neighborlist_backend": "ase",
    }
    element = build_element_lookup([14])
    dataset = create_graphs([atoms], element, options, "train", [tmp_path])
    expected = dataset[0].energy.clone()
    torch.set_default_dtype(torch.float32)
    copied = pickle.loads(pickle.dumps(dataset))
    assert torch.equal(copied[0].energy, expected)
    copied.close()
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        multiprocessing_context=context,
        timeout=30,
    )
    assert torch.equal(next(iter(loader)).energy, expected)
    reused = create_graphs(None, element, options, "train", [tmp_path])
    assert torch.equal(reused[0].energy, expected)
    reused.close()
    dataset.close()


def test_graph_cache_directories_are_configured_paths_without_namespaces(tmp_path):
    directory = tmp_path / "graphs"
    cfg = {
        "dataset": {"train_file": "original.xyz", "shard_dirs": [str(directory)]},
        "model": {"config": {"cutoff": 3.0}},
    }
    assert graph_cache_directories(cfg) == [directory]
    cfg["dataset"].update(train_file="different.xyz", valid_ratio=0.5, split_seed=9)
    cfg["model"]["config"].update(cutoff=4.0, max_neighbors=12)
    cfg["dtype"] = {"model": "float32"}
    assert graph_cache_directories(cfg) == [directory]
    cfg["dataset"]["shard_dirs"] = str(directory)
    assert graph_cache_directories(cfg) == [directory]


def _prebuilt_graph_store(
    directory,
    energies,
    targets=("energy",),
    embeddings=(),
    atomic_number=1,
    stage="train",
):
    keyspec = KeySpecification.from_defaults()
    loader = {
        "_target_": "torch_geometric.loader.DataLoader",
        "batch_size": 2,
        "shuffle": False,
        "num_workers": 0,
    }
    cfg = {
        "misc": {"device": "cpu"},
        "model": {
            "config": {
                "cutoff": 2.0,
                "fidelity": [{"name": "DFT", "atomic_energy": {1: 0.0}}],
            }
        },
        "dataset": {
            "storage_mode": "lmdb",
            "train_file": str(directory.parent / "original.xyz"),
            "no_valid_set": True,
            "neighborlist_backend": "ase",
            "shard_dirs": [str(directory)],
            "train_dataloader": loader,
            "valid_dataloader": dict(loader),
            "test_dataloader": dict(loader),
            "statistics_dataloader": {"batch_size": 2},
        },
    }
    graphs = []
    for index, energy in enumerate(energies):
        atoms = Atoms(
            numbers=[atomic_number] * (index + 1),
            positions=np.arange((index + 1) * 3).reshape(-1, 3) * 0.2,
        )
        atoms.info["energy"] = energy
        atoms.arrays["forces"] = np.ones((len(atoms), 3))
        atoms.arrays["charges"] = np.zeros(len(atoms))
        graphs.append(
            from_atoms(
                None,
                atoms,
                cutoff=2.0,
                max_neighbors=None,
                keyspec=keyspec,
                target_property=list(targets),
                embedding_property=list(embeddings),
                neighborlist_backend="ase",
            )
        )
    dataset = write_graphs(
        graphs,
        [directory],
        stage,
        shard_size=1,
        graph_config={"cutoff": 2.0, "max_neighbors": None},
    )
    dataset.close()
    return cfg, keyspec, dataset


def _forbid_graph_rebuild(monkeypatch):
    def fail(*args, **kwargs):
        pytest.fail("Completed graph stores must not reopen raw sources or rebuild")

    monkeypatch.setattr("tace.dataset.datamodule.tace_read_all_files", fail)
    monkeypatch.setattr("tace.dataset.datamodule.from_records", fail)


@pytest.mark.parametrize("change", ["removed", "modified"])
def test_completed_graph_cache_survives_raw_source_changes(
    tmp_path, monkeypatch, change
):
    from tace.dataset.datamodule import build_datamodule, load_cached_datamodule
    from tace.dataset.statistics import compute_statistics, prepare_atomic_energies

    monkeypatch.chdir(tmp_path)
    filename = tmp_path / "structures.db"
    write_reader_database(filename)
    keyspec = reader_keys()
    cfg = {
        "misc": {"device": "cpu"},
        "model": {
            "config": {
                "cutoff": 2.0,
                "fidelity": [{"name": "DFT", "atomic_energy": None}],
            }
        },
        "dataset": {
            "reader": "ase_db",
            "train_file": str(filename),
            "storage_mode": "lmdb",
            "shard_dirs": [str(tmp_path / "cache")],
            "neighborlist_backend": "ase",
            "no_valid_set": True,
            "train_dataloader": {
                "_target_": "torch_geometric.loader.DataLoader",
                "batch_size": 2,
                "num_workers": 0,
            },
            "statistics_dataloader": {"batch_size": 2},
        },
    }
    cold = build_datamodule(cfg, None, ["energy", "forces"], [], keyspec)
    cold.setup("fit")
    paths = list(cold.train_dataset.lmdb_paths)
    cold.train_dataset.close()
    fingerprint = statistics_cache_key(cfg, ["energy", "forces"], [], keyspec)
    if change == "removed":
        filename.unlink()
    else:
        before = filename.stat()
        filename.write_bytes(
            b"Replaced raw file, deliberately no longer an ASE database"
        )
        os.utime(filename, ns=(before.st_atime_ns, before.st_mtime_ns + 10**9))
    _forbid_graph_rebuild(monkeypatch)
    assert statistics_cache_key(cfg, ["energy", "forces"], [], keyspec) == fingerprint
    warm = load_cached_datamodule(cfg, ["energy", "forces"], [], keyspec)
    assert warm is not None and warm.threeAtomsList is None
    assert warm.train_dataset.lmdb_paths == paths
    element, references = prepare_atomic_energies(cfg, warm)
    statistics = compute_statistics(cfg, warm, element, references)
    assert statistics[0]["num_graphs"] == 4
    assert statistics[0]["data_fingerprint"] == fingerprint
    warm.train_dataset.close()


@pytest.mark.parametrize("change", ["cutoff", "max_neighbors", "properties", "mask"])
def test_incompatible_completed_cache_fails_without_rebuilding(
    tmp_path, monkeypatch, change
):
    from tace.dataset.datamodule import load_cached_datamodule

    directory = tmp_path / "graphs"
    cfg, keyspec, dataset = _prebuilt_graph_store(directory, [1.0, 2.0])
    targets = ["energy"]
    if change in {"cutoff", "max_neighbors"}:
        cfg["model"]["config"][change] = 3
    elif change == "properties":
        targets.append("forces")
    else:
        manifest = dataset.lmdb_paths[0] / "metadata.json"
        metadata = json.loads(manifest.read_text())
        metadata["fields"].remove("energy_weight")
        manifest.write_text(json.dumps(metadata))
    original = {
        path: path.read_bytes() for path in directory.rglob("*") if path.is_file()
    }
    _forbid_graph_rebuild(monkeypatch)
    message = (
        "cutoff or max_neighbors differs"
        if change in {"cutoff", "max_neighbors"}
        else "lack required fields"
    )
    with pytest.raises(ValueError, match=message):
        load_cached_datamodule(cfg, targets, [], keyspec)
    assert all(path.read_bytes() == value for path, value in original.items())


def test_completed_graph_reuse_ignores_raw_configuration(tmp_path, monkeypatch):
    from tace.dataset.datamodule import load_cached_datamodule

    cfg, keyspec, dataset = _prebuilt_graph_store(tmp_path / "graphs", [1.0, 2.0])
    cfg["dtype"] = {"model": "float32", "autocast": "bfloat16", "matmul": "high"}
    cfg["model"]["config"]["atomic_numbers"] = [1, 8]
    cfg["dataset"].update(
        train_file="unavailable-different-source.xyz",
        valid_ratio=0.75,
        split_seed=19,
        neighborlist_backend="matscipy",
        reader="ase",
    )
    keyspec.info_keys["energy"] = "irrelevant_raw_energy"
    keyspec.info_keys["stress"] = "unused_stress"
    keyspec.arrays_keys["charges"] = "unused_charges"
    _forbid_graph_rebuild(monkeypatch)
    warm = load_cached_datamodule(cfg, ["energy"], [], keyspec)
    assert warm.train_dataset.lmdb_paths == dataset.lmdb_paths
    assert warm.train_dataset[0].energy.dtype == torch.float64
    assert warm.element.zs == [1, 8]
    warm.train_dataset.close()
    for key, value in (("type", "ase"), ("format", "extxyz"), ("options", {})):
        cfg["dataset"][key] = value
        with pytest.raises(ValueError, match="Unsupported dataset settings"):
            load_cached_datamodule(cfg, ["energy"], [], keyspec)
        del cfg["dataset"][key]
    cfg["dataset"]["reader"] = []
    with pytest.raises(ValueError, match="dataset.reader has 0 values"):
        load_cached_datamodule(cfg, ["energy"], [], keyspec)


@pytest.mark.parametrize("move_shards", [False, True])
def test_shard_directories_combine_independent_stores_without_raw_sources(
    tmp_path, monkeypatch, move_shards
):
    from tace.dataset.datamodule import load_cached_datamodule

    first_dir, second_dir = tmp_path / "run-a", tmp_path / "run-b"
    cfg, keyspec, first = _prebuilt_graph_store(
        first_dir,
        [1.0, 2.0],
        targets=["energy", "forces"],
        embeddings=["initial_noncollinear_magmoms"],
    )
    _, _, second = _prebuilt_graph_store(
        second_dir, [3.0], targets=["energy", "charges"], atomic_number=8
    )
    expected = {
        path.name: (
            dataset[index].energy.item(),
            dataset.get_sizes([index])[0].tolist(),
        )
        for dataset in (first, second)
        for index, path in enumerate(dataset.lmdb_paths)
    }
    first.close()
    second.close()
    paths = first.lmdb_paths + second.lmdb_paths
    assert len({path.name for path in paths}) == len(paths)
    for path in paths:
        metadata = json.loads((path / "metadata.json").read_text())
        assert metadata["graph_config"] == {"cutoff": 2.0, "max_neighbors": None}
        assert "graph_selection" not in metadata and "store_id" not in metadata
        assert "paths" not in metadata
    assert not list(tmp_path.rglob("*_metadata.json"))
    directories = [first_dir, second_dir]
    if move_shards:
        combined = tmp_path / "combined"
        combined.mkdir()
        paths = [path.rename(combined / path.name) for path in paths]
        directories = [combined]
        assert not list(first_dir.glob("*.lmdb"))
        assert not list(second_dir.glob("*.lmdb"))
    cfg["dataset"]["shard_dirs"] = list(map(str, directories))
    cfg["dataset"].pop("train_file")
    keyspec.info_keys["energy"] = "irrelevant_raw_energy_key"
    _forbid_graph_rebuild(monkeypatch)
    warm = load_cached_datamodule(cfg, ["energy"], [], keyspec)
    dataset = warm.train_dataset
    assert warm.threeAtomsList is None
    assert not dataset._opened
    assert dataset.lmdb_paths == sorted(paths, key=lambda path: path.name)
    expected_values = [expected[path.name][0] for path in dataset.lmdb_paths]
    assert dataset.get_sizes().tolist() == [
        expected[path.name][1] for path in dataset.lmdb_paths
    ]
    indices = [2, 0, 1]
    assert dataset.get_sizes(indices).tolist() == dataset.get_sizes()[indices].tolist()
    graphs = dataset.__getitems__(indices)
    assert [graph.energy.item() for graph in graphs] == [
        expected_values[index] for index in indices
    ]
    assert all("forces" not in graph and "charges" not in graph for graph in graphs)
    assert all("initial_noncollinear_magmoms" not in graph for graph in graphs)
    assert dataset.metadata["atomic_numbers"] == [1, 8]
    assert "energy" in dataset.metadata["fields"]
    assert not {"forces", "charges", "initial_noncollinear_magmoms"} & set(
        dataset.metadata["fields"]
    )
    assert torch.cat([batch.energy for batch in warm.train_dataloader()]).tolist() == (
        expected_values
    )
    assert sum(batch.num_graphs for batch in warm.statistics_dataloader()) == 3
    dataset.close()


@pytest.mark.parametrize("missing", ["metadata.json", "data.mdb", "counts.bin"])
def test_incomplete_graph_shards_fail_without_rebuilding(
    tmp_path, monkeypatch, missing
):
    from tace.dataset.datamodule import load_cached_datamodule

    cfg, keyspec, dataset = _prebuilt_graph_store(tmp_path / "graphs", [1.0])
    (dataset.lmdb_paths[0] / missing).unlink()
    _forbid_graph_rebuild(monkeypatch)
    with pytest.raises(ValueError, match="Cannot read graph shard"):
        load_cached_datamodule(cfg, ["energy"], [], keyspec)


def test_cached_shards_supply_all_stages_without_raw_configuration(
    tmp_path, monkeypatch
):
    from tace.dataset.datamodule import build_datamodule

    directory = tmp_path / "graphs"
    cfg, keyspec, _ = _prebuilt_graph_store(directory, [1.0, 2.0])
    _prebuilt_graph_store(directory, [3.0], stage="valid")
    _prebuilt_graph_store(directory, [4.0], stage="test0")
    _prebuilt_graph_store(directory, [5.0, 6.0], stage="test1")
    cfg["dataset"].pop("train_file")
    cfg["dataset"]["no_valid_set"] = False
    _forbid_graph_rebuild(monkeypatch)
    module = build_datamodule(cfg, None, ["energy"], [], keyspec)
    module.setup()
    assert module.threeAtomsList is None
    assert len(module.train_dataset) == 2 and len(module.val_dataset) == 1
    assert [len(dataset) for dataset in module.test_datasets] == [1, 2]
    for dataset in (module.train_dataset, module.val_dataset, *module.test_datasets):
        dataset.close()


def test_combined_shards_reject_different_graph_configuration(tmp_path):
    first, second = tmp_path / "first", tmp_path / "second"
    _prebuilt_graph_store(first, [1.0])
    _, _, dataset = _prebuilt_graph_store(second, [2.0])
    manifest = dataset.lmdb_paths[0] / "metadata.json"
    metadata = json.loads(manifest.read_text())
    metadata["graph_config"]["cutoff"] = 3.0
    manifest.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="Different cutoffs or neighbor limits"):
        read_cache_metadata([first, second], "train")


def _convert_test_shard(path, compressed):
    with lmdb.open(str(path)) as env:
        with env.begin(write=True) as txn:
            for key, value in list(txn.cursor()):
                if value[:8] == b"TACEZLB\0":
                    value = zlib.decompress(value[16:])
                txn.put(key, compress_array_record(value) if compressed else value)
    manifest = path / "metadata.json"
    metadata = json.loads(manifest.read_text())
    metadata["codec"] = COMPRESSED_ARRAY_CODEC if compressed else ARRAY_CODEC
    manifest.write_text(json.dumps(metadata))


@pytest.mark.parametrize("compressed", [False, True])
def test_statistics_fingerprint_tracks_added_shards_but_not_moves(tmp_path, compressed):
    original_dir, moved_dir = tmp_path / "original", tmp_path / "moved"
    cfg, keyspec, first = _prebuilt_graph_store(original_dir, [1.0])
    key = statistics_cache_key(cfg, ["energy"], [], keyspec)
    moved_dir.mkdir()
    for path in first.lmdb_paths:
        path.rename(moved_dir / path.name)
        _convert_test_shard(moved_dir / path.name, compressed)
    cfg["dataset"].update(
        shard_dirs=[str(moved_dir)],
        train_file="different-unavailable-source.xyz",
        valid_ratio=0.4,
        split_seed=9,
    )
    keyspec.info_keys["energy"] = "different_raw_label"
    for mode in LMDB_MODES:
        cfg["dataset"]["storage_mode"] = mode
        assert statistics_cache_key(cfg, ["energy"], [], keyspec) == key
    _, _, second = _prebuilt_graph_store(tmp_path / "second", [2.0])
    for path in second.lmdb_paths:
        path.rename(moved_dir / path.name)
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) != key


def test_statistics_fingerprint_tracks_validation_elements_not_labels(tmp_path):
    directory = tmp_path / "graphs"
    cfg, keyspec, _ = _prebuilt_graph_store(directory, [1.0])
    _prebuilt_graph_store(directory, [2.0], stage="valid")
    cfg["dataset"]["no_valid_set"] = False
    key = statistics_cache_key(cfg, ["energy"], [], keyspec)
    _, _, extra = _prebuilt_graph_store(tmp_path / "more-valid", [3.0], stage="valid")
    for path in extra.lmdb_paths:
        path.rename(directory / path.name)
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == key
    _, _, oxygen = _prebuilt_graph_store(
        tmp_path / "oxygen", [4.0], atomic_number=8, stage="valid"
    )
    for path in oxygen.lmdb_paths:
        path.rename(directory / path.name)
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) != key


def test_graph_cache_rejects_lossy_data(tmp_path):
    graph = Data(num_nodes=1, positions=torch.zeros(1, 3, dtype=torch.float32))
    with pytest.raises(ValueError, match="not float64"):
        encode_graph(graph)
    with pytest.raises(ValueError, match="not float64"):
        write_graphs([graph], [tmp_path], "train")
    assert read_cache_metadata([tmp_path], "train") is None


@pytest.mark.parametrize("compressed", [False, True])
def test_typed_graph_store_batched_projection_and_size_index(tmp_path, compressed):
    graphs = [
        Data(
            num_nodes=2,
            atomic_numbers=torch.tensor([1, 8]),
            positions=torch.full((2, 3), float(index), dtype=torch.float64),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_shifts=torch.zeros(2, 3, dtype=torch.int64),
            energy=torch.tensor([float(index)], dtype=torch.float64),
            forces=torch.arange(6, dtype=torch.float64).reshape(2, 3) / 7,
            stress=torch.arange(9, dtype=torch.float64).reshape(1, 3, 3) / 11,
            fidelity_idx=torch.tensor(0),
        )
        for index in range(7)
    ]
    for graph in graphs:
        encoded = encode_graph(graph)
        decoded = decode_graph(
            compress_array_record(encoded) if compressed else encoded
        )
        for name, value in graph.items():
            if torch.is_tensor(value):
                assert torch.equal(decoded[name], value)
                assert decoded[name].dtype == value.dtype
            else:
                assert decoded[name] == value
    dataset = write_graphs(
        iter(graphs),
        [tmp_path / "a", tmp_path / "b"],
        "train",
        shard_size=2,
        transaction_bytes=2000,
        max_buffer_bytes=10000,
        compressed=compressed,
    )
    assert len(dataset.lmdb_paths) == 4
    assert dataset.metadata["codec"] == (
        COMPRESSED_ARRAY_CODEC if compressed else ARRAY_CODEC
    )
    if compressed:
        dataset.close()
        for path in dataset.lmdb_paths[::2]:
            _convert_test_shard(path, False)
        metadata = read_cache_metadata([tmp_path / "a", tmp_path / "b"], "train")
        assert sum(metadata["shards"].values()) == 7
    indices = [6, 0, -2, 6, 1]
    assert [graph.energy.item() for graph in dataset.__getitems__(indices)] == [
        6,
        0,
        5,
        6,
        1,
    ]
    assert dataset.get_sizes().tolist() == [[2, 2]] * 7
    assert dataset.get_sizes(indices).tolist() == [[2, 2]] * 5
    projected = dataset.project(["energy"])
    saved = projected[0]
    assert set(saved.keys()) == {"num_nodes", "energy"}
    dataset.close()
    assert projected[1].energy.item() == 1
    projected.close()
    assert saved.energy.item() == 0

    # Eviction and shared ownership must leave decoded tensors valid. With
    # enough handles, a second random traversal reuses every transaction.
    dataset.max_open_shards = 2
    dataset.__getitems__([0, 2, 4, 6])
    assert len(dataset._opened) == 2
    dataset.close()
    dataset.max_open_shards = 4
    dataset.__getitems__([0, 2, 4, 6])
    transactions = [dataset._transaction(i) for i in range(4)]
    dataset.__getitems__([6, 4, 2, 0])
    assert all(dataset._transaction(i) is txn for i, txn in enumerate(transactions))
    dataset.close()
    assert saved.energy.item() == 0


def test_compressed_graph_records_reject_corruption():
    raw = encode_graph(
        Data(num_nodes=1, positions=torch.ones(1, 3, dtype=torch.float64))
    )
    value = compress_array_record(raw)
    for malformed in (
        value[:15],
        value[:-1],
        value + b"trailing data",
        value[:16] + bytes([value[16] ^ 255]) + value[17:],
        value[:8] + struct.pack("<Q", len(raw) - 1) + value[16:],
        value[:8] + struct.pack("<Q", len(raw) + 1) + value[16:],
        value[:8] + struct.pack("<Q", (2 << 30) + 1) + value[16:],
    ):
        with pytest.raises(ValueError):
            decode_graph(malformed)
    with pytest.raises(ValueError, match="uncompressed"):
        compress_array_record(value)


@pytest.mark.parametrize("in_memory", [False, True])
@pytest.mark.parametrize(
    "options,expected",
    [
        (None, list(range(7))),
        ({"enabled": False, "max_edges": 1}, list(range(7))),
        ({"enabled": True}, list(range(7))),
        ({"enabled": True, "max_edges": 24000}, [0, 2, 3, 4, 5, 6]),
        ({"enabled": True, "max_edges": 24001}, list(range(7))),
        ({"enabled": True, "max_edges": None}, list(range(7))),
        ({"enabled": True, "max_edges": None, "drop_isolated": False}, list(range(7))),
        ({"enabled": True, "max_forces": 5}, [0, 2, 6]),
        ({"enabled": True, "max_energy_per_atom": -1}, [0, 1, 3, 6]),
        ({"enabled": True, "max_stress": 5}, [0, 1, 2, 6]),
        ({"enabled": True, "max_atoms": 1}, [6]),
        ({"enabled": True, "max_edges": 1}, [4, 6]),
        (
            {
                "enabled": True,
                "max_edges": None,
                "max_atoms": 2,
                "max_forces": 5,
                "max_energy_per_atom": -1,
                "max_stress": 5,
            },
            [0, 6],
        ),
    ],
)
def test_cached_graph_filters_are_inclusive_and_readonly(
    tmp_path, monkeypatch, options, expected, in_memory
):
    graphs = []
    for index, (atoms, edges, energy, force) in enumerate(
        [
            (2, 2, -4, [3, 4, 0]),
            (4, 24001, -4, [6, 0, 0]),
            (2, 24000, 6, [0, 0, 0]),
            (2, 2, -20, [4, 4, 0]),
            (2, 0, -4, [3, 4, 0]),
            (2, 2, float("nan"), [float("nan"), 0, 0]),
            (1, 1, -1, [0, 0, 0]),
        ]
    ):
        stress = torch.zeros(1, 3, 3, dtype=torch.float64)
        if index in (0, 4):
            stress[0, 0, 0], stress[0, 1, 1] = 3, 4
        elif index == 3:
            stress[0, 0, 1] = stress[0, 1, 0] = 4
        elif index == 5:
            stress.fill_(float("nan"))
        graphs.append(
            Data(
                num_nodes=atoms,
                atomic_numbers=torch.ones(atoms, dtype=torch.int64),
                edge_index=torch.zeros(2, edges, dtype=torch.int64),
                energy=torch.tensor([energy], dtype=torch.float64),
                forces=torch.tensor([force] * atoms, dtype=torch.float64),
                stress=stress,
                **{
                    f"{name}_weight": torch.tensor(
                        [float(index != 4)], dtype=torch.float64
                    )
                    for name in ("energy", "forces", "stress")
                },
            )
        )
    if in_memory:
        dataset = GraphDatasetLMDB([], in_memory=True)
        dataset.data_list, dataset.length = graphs, len(graphs)
    else:
        dataset = write_graphs(graphs, [tmp_path], "train", shard_size=2)
    before = file_snapshot(tmp_path)
    bounds = filter_options(options)
    if not set(bounds) & {"max_forces", "max_energy_per_atom", "max_stress"}:
        with monkeypatch.context() as patch:
            patch.setattr(
                GraphDatasetLMDB,
                "_decode",
                lambda *args: pytest.fail("Count filters must not decode graphs"),
            )
            selected = filter_graphs(dataset, bounds)
    else:
        selected = filter_graphs(dataset, bounds)
    actual = (
        list(range(len(dataset)))
        if selected.indices is None
        else selected.indices.tolist()
    )
    assert actual == expected
    if expected == list(range(len(graphs))):
        assert selected is dataset
    np.testing.assert_array_equal(selected.get_sizes(), dataset.get_sizes(expected))
    projected = selected.project(["energy"])
    assert len(projected) == len(expected)
    for view_index in [-1, 0, -1]:
        torch.testing.assert_close(
            projected[view_index].energy,
            graphs[expected[view_index]].energy,
            equal_nan=True,
        )
    projected.close()
    selected.close()
    dataset.close()
    assert file_snapshot(tmp_path) == before


@pytest.mark.parametrize("storage", ["memory", "lmdb", "clmdb"])
def test_min_covalent_ratio_filters_periodic_graphs(tmp_path, storage):
    hh_limit = 0.5 * (covalent_radii[1] + covalent_radii[1])
    frames = [
        Atoms("H2", positions=[[0, 0, 0], [hh_limit, 0, 0]]),
        Atoms("H2", positions=[[0, 0, 0], [np.nextafter(hh_limit, 0.0), 0, 0]]),
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0.95, 1.85, 0]],
            cell=[[3, 0, 0], [1, 2, 0], [0, 0, 3]],
            pbc=True,
        ),
        Atoms("H"),
        Atoms("Si", cell=[0.8, 2, 2], pbc=[True, False, False]),
        Atoms("H2", positions=[[0, 0, 0], [0, 0, 0]]),
        Atoms("H2", positions=[[0, 0, 0], [5, 0, 0]]),
        # H-H is the shortest distance, but only the longer Sb-Sb contact fails.
        Atoms("H2Sb2", positions=[[0, 0, 0], [0.7, 0, 0], [8, 0, 0], [9.25, 0, 0]]),
        Atoms("HO", positions=[[0, 0, 0], [0.49, 0, 0]]),
        Atoms("HO", positions=[[0, 0, 0], [0.48, 0, 0]]),
    ]
    for i, atoms in enumerate(frames):
        atoms.info["energy"] = 1.0 if i == 0 else -1.0
    config = {"cutoff": 1.5, "max_neighbors": None}
    dataset = create_graphs(
        frames,
        build_element_lookup([1, 8, 14, 51]),
        dict(
            config,
            keyspec=KeySpecification.from_defaults(),
            target_property=["energy"],
            embedding_property=[],
            neighborlist_backend="ase",
        ),
        "train",
        [tmp_path],
        storage_mode=storage,
        graph_config=config,
    )
    before = file_snapshot(tmp_path)
    view = dataset.project(["energy"])
    options = filter_options(
        {"enabled": True, "max_edges": None, "min_covalent_ratio": 0.5}
    )
    cutoff = config["cutoff"] if storage == "memory" else None
    selected = filter_graphs(view, options, graph_cutoff=cutoff)
    assert selected.indices.tolist() == [0, 3, 6, 8]
    assert set(selected[0].keys()) == {"energy", "num_nodes"}
    selected.close()
    selected = filter_graphs(
        view, {**options, "max_energy_per_atom": 0}, graph_cutoff=cutoff
    )
    assert selected.indices.tolist() == [3, 6, 8]
    selected.close()
    with pytest.raises(ValueError, match="min_covalent_ratio.*cutoff"):
        filter_graphs(view, {"min_covalent_ratio": 0.55}, graph_cutoff=cutoff)
    view.close()
    dataset.close()
    assert file_snapshot(tmp_path) == before


@pytest.mark.parametrize("storage", ["memory", "lmdb", "clmdb"])
@pytest.mark.parametrize(
    "cutoff,expected",
    [(1.0, [0, 1, 4, 5, 8]), (6.0, [0, 1, 4, 5, 6, 7, 8])],
)
def test_drop_isolated_keeps_single_atoms_and_uses_the_graph_cutoff(
    tmp_path, storage, cutoff, expected
):
    def chain(*x):
        return Atoms("H" * len(x), positions=[[r, 0, 0] for r in x])

    frames = [
        chain(0),
        chain(0, 0.74),
        chain(0, 7),
        chain(0, 0.7, 1.4, 10),  # Six directed edges, but the last atom has none.
        chain(0, 0.7, 10, 10.7),  # Two disconnected dimers; no isolated atom.
        Atoms("H2", positions=[[0.2, 0, 0], [9.8, 0, 0]], cell=[10] * 3, pbc=True),
        Atoms(
            "H2",
            positions=[[0, 0, 0], [0, 10, 0]],
            cell=[1.5, 25, 25],
            pbc=[True, False, False],
        ),
        chain(0, 2),
        chain(0, 0.1),
    ]
    config = {"cutoff": cutoff, "max_neighbors": None}
    dataset = create_graphs(
        frames,
        build_element_lookup([1]),
        dict(
            config,
            keyspec=KeySpecification.from_defaults(),
            target_property=[],
            embedding_property=[],
            neighborlist_backend="ase",
        ),
        "train",
        [tmp_path],
        storage_mode=storage,
        graph_config=config,
    )
    before = file_snapshot(tmp_path)
    view = dataset.project(["atomic_numbers"])
    options = filter_options(
        {"enabled": True, "max_edges": None, "drop_isolated": True}
    )
    selected = filter_graphs(view, options)
    assert selected.indices.tolist() == expected
    assert set(selected[0].keys()) == {"num_nodes", "atomic_numbers"}
    selected.close()
    selected = filter_graphs(
        view, {**options, "min_covalent_ratio": 0.5}, graph_cutoff=cutoff
    )
    assert selected.indices.tolist() == [i for i in expected if i != 8]
    selected.close()
    if storage == "memory":
        for graph in dataset.data_list:
            del graph["num_neighbors"]
            del graph["positions"]
        selected = filter_graphs(view, options)
        assert selected.indices.tolist() == expected
        selected.close()
    view.close()
    dataset.close()
    assert file_snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "options,error",
    [
        ({"enabled": "true"}, TypeError),
        ({"enabled": True, "max_atoms": 2.5}, TypeError),
        ({"enabled": True, "max_edges": True}, TypeError),
        ({"enabled": True, "max_forces": -1}, ValueError),
        ({"enabled": True, "min_covalent_ratio": -0.5}, ValueError),
        ({"enabled": True, "min_covalent_ratio": True}, TypeError),
        ({"enabled": True, "drop_isolated": 6.0}, TypeError),
        ({"enabled": True, "drop_isolated": None}, TypeError),
        ({"enabled": True, "min_distance": 0.5}, ValueError),
        ({"enabled": True, "max_energy_per_atom": float("inf")}, ValueError),
        ({"enabled": True, "max_stress": float("nan")}, ValueError),
        ({"enabled": True, "max_edge": 3}, ValueError),
    ],
)
def test_graph_filter_rejects_invalid_configuration(options, error):
    with pytest.raises(error, match="dataset.filter"):
        filter_options(options)


@pytest.mark.parametrize("epoch_size", [None, 0.5])
def test_cached_filter_only_changes_training_loader(tmp_path, monkeypatch, epoch_size):
    from tace.dataset.datamodule import load_cached_datamodule
    from tace.dataset.sampler import EpochSampler
    from tace.dataset.statistics import prepare_atomic_energies

    cfg, keyspec, _ = _prebuilt_graph_store(
        tmp_path, [-1, 2, -9, 8], targets=("energy", "forces")
    )
    for stage in ("valid", "test0"):
        _prebuilt_graph_store(
            tmp_path, [10, 20, 30, 40], targets=("energy",), stage=stage
        )
    cfg["dataset"]["no_valid_set"] = False
    cfg["dataset"]["epoch_sampler"] = {
        "epoch_size": epoch_size,
        "balance_batches": "edges" if epoch_size is not None else None,
    }
    cfg["model"]["config"]["fidelity"][0]["atomic_energy"] = None
    old_key = statistics_cache_key(cfg, ["energy"], [], keyspec)
    cfg["dataset"]["filter"] = {
        "enabled": True,
        "max_energy_per_atom": -1,
        "max_forces": 2,
    }
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == old_key
    before = file_snapshot(tmp_path)
    _forbid_graph_rebuild(monkeypatch)
    module = load_cached_datamodule(cfg, ["energy"], [], keyspec)
    module.setup("test")
    datasets = [module.train_dataset, module.val_dataset, *module.test_datasets]
    assert [len(dataset) for dataset in datasets] == [4, 4, 4]
    assert module._filtered_train_dataset is None
    assert "forces" not in module.train_dataset[0]
    loader = module.train_dataloader()
    assert len(loader.dataset) == 2
    assert "forces" not in loader.dataset[0]
    assert isinstance(loader.sampler, EpochSampler) == (epoch_size is not None)
    if epoch_size is not None:
        assert loader.sampler.epoch_size == 1
        assert loader.sampler.balance == "edges"
    expected = [-1] if epoch_size is not None else [-1, -9]
    assert torch.cat([batch.energy for batch in loader]).tolist() == expected
    with monkeypatch.context() as patch:
        patch.setattr(
            "tace.dataset.datamodule.filter_graphs",
            lambda *args: pytest.fail("Loader reload must reuse the filter index"),
        )
        assert module.train_dataloader().dataset is loader.dataset

    statistics_loader = module.statistics_dataloader()
    assert not isinstance(statistics_loader.sampler, EpochSampler)
    assert torch.cat([batch.energy for batch in statistics_loader]).tolist() == [
        -1,
        2,
        -9,
        8,
    ]
    element, references = prepare_atomic_energies(cfg, module)
    assert references[0][1] == pytest.approx(8 / 30)
    written = []
    monkeypatch.setattr(statistics_module, "log_statistics_to_yaml", written.append)
    statistics = statistics_module.statistics_from_loader(
        statistics_loader,
        element.zs,
        references,
        target_property=["energy"],
        device="cpu",
        num_fidelities=1,
        data_fingerprint=old_key,
    )
    assert statistics[0]["num_graphs"] == 4
    assert statistics[0]["num_atoms"] == 10
    assert statistics[0]["data_fingerprint"] == old_key
    assert written == [statistics]
    statistics_loader.dataset.close()
    for evaluation_loader in [module.val_dataloader(), *module.test_dataloader()]:
        assert not isinstance(evaluation_loader.sampler, EpochSampler)
        assert torch.cat([batch.energy for batch in evaluation_loader]).tolist() == [
            10,
            20,
            30,
            40,
        ]
        evaluation_loader.dataset.close()
    with pytest.raises(ValueError, match="removed all"):
        filter_graphs(datasets[0], {"max_atoms": 0})
    with pytest.raises(ValueError, match="lack fields"):
        filter_graphs(datasets[0], {"max_stress": 1})
    # Even an empty training selection must not block statistics or evaluation.
    cfg["dataset"]["filter"]["max_atoms"] = 0
    rejected = load_cached_datamodule(cfg, ["energy"], [], keyspec)
    assert len(rejected.train_dataset) == len(rejected.val_dataset) == 4
    with pytest.raises(ValueError, match="removed all"):
        rejected.train_dataloader()
    rejected.train_dataset.close()
    rejected.val_dataset.close()
    loader.dataset.close()
    for dataset in datasets:
        dataset.close()
    assert file_snapshot(tmp_path) == before


def test_store_progress_counts_committed_batches_and_final_partial(tmp_path):
    graph = Data(
        num_nodes=1,
        atomic_numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
    )
    written = []
    dataset = write_graphs(
        [graph] * 5,
        [tmp_path],
        "train",
        transaction_bytes=2 * len(compress_array_record(encode_graph(graph))),
        progress=written.append,
    )
    assert written == [2, 2, 1]
    assert dataset.metadata["codec"] == COMPRESSED_ARRAY_CODEC
    assert sum(written) == len(dataset)
    dataset.close()
    written.clear()
    reused = write_graphs(iter(()), [tmp_path], "train", progress=written.append)
    assert len(reused) == 5 and not written
    reused.close()


@pytest.mark.parametrize("compressed", [False, True])
def test_store_progress_does_not_double_count_map_full_retries(
    tmp_path, monkeypatch, compressed
):
    open_lmdb = lmdb.open
    resized = []

    class TrackingEnvironment:
        def __init__(self, environment):
            self.environment = environment

        def __getattr__(self, name):
            return getattr(self.environment, name)

        def set_mapsize(self, size):
            resized.append(size)
            self.environment.set_mapsize(size)

    def track_map_size(*args, **kwargs):
        environment = open_lmdb(*args, **kwargs)
        if kwargs.get("readonly"):
            return environment
        return TrackingEnvironment(environment)

    monkeypatch.setattr(lmdb, "open", track_map_size)
    graph = Data(
        num_nodes=1,
        atomic_numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
        payload=torch.randn(
            300_000, dtype=torch.float64, generator=torch.Generator().manual_seed(7)
        ),
    )
    written = []
    dataset = write_graphs(
        [graph],
        [tmp_path],
        "train",
        map_size=1 << 20,
        progress=written.append,
        compressed=compressed,
    )
    assert resized == [2 << 20, 4 << 20]
    assert written == [1]
    assert torch.equal(dataset[0].payload, graph.payload)
    dataset.close()


def test_store_failure_does_not_publish_completion(tmp_path):
    graph = Data(
        num_nodes=1,
        atomic_numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
    )

    def interrupted():
        yield graph
        yield graph
        raise RuntimeError("injected source failure")

    with pytest.raises(RuntimeError, match="injected source failure"):
        write_graphs(interrupted(), [tmp_path], "train", shard_size=1)
    assert read_cache_metadata([tmp_path], "train") is None
    assert (tmp_path / "train_failed.json").is_file()
    assert not (tmp_path / "train.build.lock").exists()


def test_independent_graph_stores_publish_in_input_order(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    graphs = [
        Data(
            num_nodes=index + 1,
            atomic_numbers=torch.full((index + 1,), index + 1, dtype=torch.int64),
            positions=torch.full((index + 1, 3), float(index), dtype=torch.float64),
            edge_index=torch.empty((2, 0), dtype=torch.int64),
            energy=torch.tensor([float(index)], dtype=torch.float64),
        )
        for index in range(7)
    ]
    directories = [tmp_path / "a", tmp_path / "b"]

    def builder(directories, stage, options):
        assert options["graph_config"] == {"cutoff": 2.0, "max_neighbors": None}

        def partition(part):
            dataset = write_graphs(
                graphs[part * 3 : (part + 1) * 3],
                directories,
                f"{stage}_shard{part:06d}_part",
                **options,
            )
            dataset.close()
            return dataset.metadata

        with ThreadPoolExecutor(max_workers=3) as workers:
            return list(workers.map(partition, range(3)))

    dataset = build_graph_store(
        builder,
        directories,
        "train",
        shard_size=2,
        graph_config={"cutoff": 2.0, "max_neighbors": None},
    )
    assert len(dataset) == len(graphs)
    assert len(dataset.lmdb_paths) == 5
    assert [graph.energy.item() for graph in dataset] == list(range(7))
    assert dataset.get_sizes().tolist() == [[index + 1, 0] for index in range(7)]
    assert dataset.metadata["num_nodes"] == 28
    assert dataset.metadata["num_edges"] == 0
    assert dataset.metadata["graph_config"] == {"cutoff": 2.0, "max_neighbors": None}
    assert not list(tmp_path.rglob("*_metadata.json"))
    assert all(
        "sha256" not in stats for stats in dataset.metadata["shard_stats"].values()
    )
    assert dataset.metadata["atomic_numbers"] == list(range(1, 8))
    assert set(dataset.metadata["fields"]) == set(graphs[0].keys())
    dataset.close()

    def unused(*args):
        pytest.fail("A completed graph store must not invoke its builder")

    reused = build_graph_store(unused, directories, "train")
    assert reused[-1].energy.item() == 6
    reused.close()


@pytest.mark.parametrize("failure", ["interrupted", "missing"])
def test_independent_graph_store_failure_is_not_published(tmp_path, failure):
    graph = Data(
        num_nodes=1,
        atomic_numbers=torch.tensor([1]),
        positions=torch.zeros(1, 3, dtype=torch.float64),
        edge_index=torch.empty((2, 0), dtype=torch.int64),
    )

    def builder(directories, stage, options):
        dataset = write_graphs(
            [graph], directories, f"{stage}_shard000000_part", **options
        )
        dataset.close()
        if failure == "interrupted":
            raise RuntimeError("injected partition failure")
        if failure == "missing":
            (dataset.lmdb_paths[0] / "counts.bin").unlink()
        return [dataset.metadata]

    with pytest.raises((ValueError, RuntimeError)):
        build_graph_store(builder, [tmp_path], "train")
    assert read_cache_metadata([tmp_path], "train") is None
    assert not (tmp_path / "train_metadata.json").exists()
    assert (tmp_path / "train_failed.json").is_file()
    assert not (tmp_path / "train.build.lock").exists()
    with pytest.raises(ValueError, match="No completed"):
        GraphDatasetLMDB.from_directories([tmp_path], "train")
    with pytest.raises(RuntimeError, match="previous train cache writer failed"):
        build_graph_store(
            lambda *_: pytest.fail("Retried partial cache"), [tmp_path], "train"
        )


def test_independent_graph_store_uses_shared_writer_lock(tmp_path):
    from filelock import FileLock

    with FileLock(str(tmp_path / ".train.writer.lock")):
        with pytest.raises(TimeoutError, match="waiting for train cache writer"):
            build_graph_store(
                lambda *_: pytest.fail("Builder ran without owning the cache lock"),
                [tmp_path],
                "train",
                lock_timeout=0,
            )
    assert not (tmp_path / "train_metadata.json").exists()
    assert not (tmp_path / "train_failed.json").exists()


def sample_graphs():
    """Return tiny CPU graphs with nontrivial labels and integer topology."""
    return [
        Data(
            num_nodes=len(numbers),
            atomic_numbers=torch.tensor(numbers),
            positions=torch.arange(len(numbers) * 3, dtype=torch.float64).reshape(
                -1, 3
            ),
            edge_index=torch.stack(
                (torch.arange(len(numbers)), torch.arange(len(numbers)))
            ),
            energy=torch.tensor([energy], dtype=torch.float64),
            energy_weight=torch.ones(1, dtype=torch.float64),
            forces=torch.ones(len(numbers), 3, dtype=torch.float64),
            forces_weight=torch.ones(1, dtype=torch.float64),
            fidelity_idx=torch.tensor(0),
        )
        for numbers, energy in (([26], 2.0), ([8], 3.0), ([26, 8], 5.0))
    ]


def make_cache(directory, *, stage="train", compressed=False, shard_size=1):
    stored = write_graphs(
        sample_graphs(),
        [directory],
        stage,
        compressed=compressed,
        shard_size=shard_size,
        graph_config={"cutoff": 3.0, "max_neighbors": None},
    )
    stored.close()
    return sorted(directory.glob("*_shard*.lmdb"))


def records(path):
    with lmdb.open(str(path), readonly=True, create=False, lock=False) as env:
        with env.begin() as transaction:
            return dict(transaction.cursor())


def copy(sources, destination, **kwargs):
    return cache_module.copy_graph_cache(
        sources, destination, reserve_bytes=0, transaction_bytes=128, **kwargs
    )


@pytest.mark.parametrize(
    "workers,directory_descriptors", [(1, True), (2, True), (1, False)]
)
def test_copy_mixed_codecs_preserves_records_and_resumes(
    tmp_path, monkeypatch, workers, directory_descriptors
):
    if not directory_descriptors:
        monkeypatch.delattr(cache_module.os, "O_DIRECTORY", raising=False)
        fsync = os.fsync

        def sync_writable_file(descriptor):
            # Windows requires write access even when fsync has no new data.
            os.write(descriptor, b"")
            fsync(descriptor)

        monkeypatch.setattr(cache_module.os, "fsync", sync_writable_file)
    raw, compressed, destination = (
        tmp_path / name for name in ("raw", "compressed", "copy")
    )
    make_cache(raw)
    make_cache(compressed, stage="valid", compressed=True)
    before = {root: file_snapshot(root) for root in (raw, compressed)}
    manifest = copy([raw, compressed], destination, workers=workers)
    assert manifest["status"] == "complete"
    assert len(manifest["completed"]) == 6
    for spec in manifest["plan"]["shards"]:
        source, target = Path(spec["source"]), destination / spec["name"]
        original, copied = records(source), records(target)
        assert original.keys() == copied.keys()
        assert (source / "counts.bin").read_bytes() == (
            target / "counts.bin"
        ).read_bytes()
        assert json.loads((target / "metadata.json").read_text()) == dict(
            spec["metadata"], codec=COMPRESSED_ARRAY_CODEC
        )
        for key, value in original.items():
            assert copied[key].startswith(b"TACEZLB\0")
            assert bytes(cache_module.raw_record(value)) == bytes(
                cache_module.raw_record(copied[key])
            )
            left, right = decode_graph(value), decode_graph(copied[key])
            assert left.keys() == right.keys()
            for name in left.keys():
                if isinstance(left[name], torch.Tensor):
                    torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
                else:
                    assert left[name] == right[name]
    shards_before = {p.name: file_snapshot(p) for p in destination.glob("*.lmdb")}
    resumed = copy(
        [raw, compressed], destination, workers=workers, verify_existing=True
    )
    assert resumed["completed"] == manifest["completed"]
    assert shards_before == {
        p.name: file_snapshot(p) for p in destination.glob("*.lmdb")
    }
    assert all(file_snapshot(root) == snapshot for root, snapshot in before.items())
    assert not list(destination.glob("*.build.lock"))


def test_interruption_hides_partial_cache_and_preserves_completed_shards(
    tmp_path, monkeypatch
):
    source, destination = tmp_path / "source", tmp_path / "copy"
    make_cache(source, stage="train_shard_outer_part")
    before = file_snapshot(source)
    original = cache_module.copy_shard
    calls = 0

    def interrupt_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise InterruptedError("test interruption")
        return original(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(cache_module, "copy_shard", interrupt_second)
        with pytest.raises(InterruptedError, match="test interruption"):
            copy([source], destination)
    assert (destination / "train.build.lock").is_file()
    assert not (destination / "train_shard_outer_part.build.lock").exists()
    assert read_cache_metadata([destination], "train") is None
    partial = json.loads((destination / cache_module.MANIFEST).read_text())
    assert partial["status"] == "interrupted"
    assert len(partial["completed"]) == 1
    published = next(destination.glob("*.lmdb"))
    published_before = file_snapshot(published)
    assert copy([source], destination)["status"] == "complete"
    assert file_snapshot(published) == published_before
    assert sum(read_cache_metadata([destination], "train")["shards"].values()) == 3
    assert file_snapshot(source) == before


def test_raw_statistics_reused_after_compression_and_online_u_shift(
    tmp_path, monkeypatch
):
    import tace.dataset.datamodule as datamodule_module
    import tace.scripts.train as train_module
    from tace.foundations.u_shift import shift_energy_labels

    cuda_initialized = torch.cuda.is_initialized()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TACE_USE_U_SHIFT", "0")
    source, destination = tmp_path / "raw", tmp_path / "compressed"
    make_cache(source, shard_size=3)
    stored = write_graphs(
        [sample_graphs()[-1]],
        [source],
        "valid",
        compressed=False,
        graph_config={"cutoff": 3.0, "max_neighbors": None},
    )
    stored.close()
    source_before = file_snapshot(source)

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "Raw source access or statistics recomputation is forbidden"
        )

    monkeypatch.setattr(torch.cuda, "_lazy_init", forbidden)
    monkeypatch.setattr(datamodule_module, "tace_read_all_files", forbidden)
    monkeypatch.setattr(datamodule_module, "from_records", forbidden)
    monkeypatch.setattr(train_module, "initialize", lambda cfg: cfg)
    monkeypatch.setattr(train_module, "create_model", lambda *args: None)
    monkeypatch.setattr(train_module, "configure_model_precision", lambda *args: None)
    cfg = OmegaConf.create(
        {
            "misc": {"device": "cpu"},
            "dtype": {"model": "float32", "autocast": None, "matmul": "ieee"},
            "loss": {"loss_property": ["energy", "forces"]},
            "model": {"config": {"cutoff": 3.0, "fidelity": [{"name": "raw"}]}},
            "dataset": {
                "storage_mode": "lmdb",
                "shard_dirs": [str(source)],
                "train_file": "must-not-read.db",
                "valid_file": "must-not-read.db",
                "train_dataloader": {
                    "_target_": "torch_geometric.loader.DataLoader",
                    "batch_size": 2,
                    "num_workers": 0,
                },
                "statistics_dataloader": {"batch_size": 3},
            },
        }
    )
    keyspec = KeySpecification.from_defaults()
    original_key = statistics_cache_key(cfg, ["energy", "forces"], [], keyspec)
    _, original_stats, _, _, _, module = train_module.build(cfg)
    assert original_stats[0]["atomic_energy"] == pytest.approx({8: 3.0, 26: 2.0})
    module.train_dataset.close()
    module.val_dataset.close()
    yaml_before = (tmp_path / "statistics_0.yaml").read_bytes()

    copy([source], destination)
    cfg.dataset.storage_mode = "clmdb"
    cfg.dataset.shard_dirs = [str(destination)]
    monkeypatch.setenv("TACE_USE_U_SHIFT", "1")
    assert statistics_cache_key(cfg, ["energy", "forces"], [], keyspec) == original_key
    monkeypatch.setattr(train_module, "compute_statistics", forbidden)
    monkeypatch.setattr(train_module, "load_cached_datamodule", forbidden)
    monkeypatch.setattr(train_module, "prepare_atomic_energies", forbidden)
    _, reused, _, _, _, module = train_module.build(cfg)
    assert reused == original_stats
    module.prepare_data()
    module.setup("fit")
    graph = module.train_dataset[2]
    assert graph.energy.item() == 5.0
    assert shift_energy_labels(graph).energy.item() == pytest.approx(5.0 - 1.925)
    assert graph.energy.item() == 5.0
    module.train_dataset.close()
    module.val_dataset.close()
    assert (tmp_path / "statistics_0.yaml").read_bytes() == yaml_before
    assert file_snapshot(source) == source_before
    assert torch.cuda.is_initialized() == cuda_initialized


@pytest.mark.parametrize("overlap", ["same", "child", "parent"])
def test_copy_rejects_overlapping_paths(tmp_path, overlap):
    source = tmp_path / "source"
    make_cache(source)
    destination = {"same": source, "child": source / "copy", "parent": tmp_path}[
        overlap
    ]
    with pytest.raises(ValueError, match="overlap"):
        copy([source], destination)


def test_copy_rejects_duplicate_names_and_foreign_destination(tmp_path):
    source, duplicate, destination = (
        tmp_path / name for name in ("source", "duplicate", "copy")
    )
    make_cache(source)
    shutil.copytree(source, duplicate)
    with pytest.raises(ValueError, match="Duplicate source shard"):
        copy([source, duplicate], destination)
    destination.mkdir()
    (destination / "foreign.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="nonempty destination"):
        copy([source], destination)
    assert (destination / "foreign.txt").read_text() == "keep me"


def test_resume_rejects_changed_source(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    shards = make_cache(source)
    copy([source], destination)
    metadata = shards[0] / "metadata.json"
    metadata.write_text(metadata.read_text() + "\n")
    with pytest.raises(ValueError, match="different source plan"):
        copy([source], destination)


@pytest.mark.parametrize("failure", ["space", "record"])
def test_copy_failure_never_publishes_partial_shards(tmp_path, monkeypatch, failure):
    source, destination = tmp_path / "source", tmp_path / "copy"
    shards = make_cache(source)
    if failure == "space":
        usage = shutil.disk_usage(tmp_path)
        monkeypatch.setattr(
            cache_module.shutil,
            "disk_usage",
            lambda _: type(usage)(usage.total, usage.used, 0),
        )
        error, match = OSError, "Insufficient free space"
    else:
        with lmdb.open(str(shards[0])) as env:
            with env.begin(write=True) as transaction:
                transaction.put(b"000000000", b"broken")
        error, match = ValueError, "Truncated typed-array record"
    before = file_snapshot(source)
    with pytest.raises(error, match=match):
        copy([source], destination)
    assert not list(destination.glob("*.lmdb"))
    assert read_cache_metadata([destination], "train") is None
    assert file_snapshot(source) == before


@pytest.mark.parametrize("failure", ["bootstrap", "transaction"])
def test_unfinished_shard_can_restart_after_write_failure(
    tmp_path, monkeypatch, failure
):
    source, destination = tmp_path / "source", tmp_path / "copy"
    make_cache(source, shard_size=3)
    source_before = file_snapshot(source)
    calls = 0
    with monkeypatch.context() as patch:
        if failure == "bootstrap":
            original = cache_module.write_json

            def fail_bootstrap(path, value):
                if Path(path).name == "source.json":
                    raise OSError("test bootstrap failure")
                return original(path, value)

            patch.setattr(cache_module, "write_json", fail_bootstrap)
        else:
            original = cache_module.require_space

            def fail_transaction(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("test transaction failure")
                return original(*args, **kwargs)

            patch.setattr(cache_module, "require_space", fail_transaction)
        with pytest.raises(OSError, match=f"test {failure} failure"):
            copy([source], destination)
    assert not list(destination.glob("*.lmdb"))
    assert read_cache_metadata([destination], "train") is None
    staged = list((destination / cache_module.STAGING).iterdir())
    if failure == "bootstrap":
        assert not staged
    else:
        assert len(staged) == 1
        assert (staged[0] / "source.json").is_file()
        assert len(records(staged[0])) == 1
    assert copy([source], destination)["status"] == "complete"
    assert sum(read_cache_metadata([destination], "train")["shards"].values()) == 3
    assert file_snapshot(source) == source_before


def test_copy_respects_native_cache_writer_lock(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    make_cache(source)
    destination.mkdir()
    with FileLock(str(destination / ".train.writer.lock"), timeout=0):
        with pytest.raises(Timeout):
            copy([source], destination)
    assert not list(destination.glob("*.lmdb"))
    assert copy([source], destination)["status"] == "complete"


def test_resume_rejects_foreign_files_and_changed_destination(tmp_path):
    source, destination = tmp_path / "source", tmp_path / "copy"
    make_cache(source)
    copy([source], destination)
    foreign = destination / "foreign.txt"
    foreign.write_text("unrelated content")
    with pytest.raises(ValueError, match="Unrecognized destination contents"):
        copy([source], destination)
    assert foreign.read_text() == "unrelated content"
    foreign.unlink()
    counts = next(destination.glob("*.lmdb")) / "counts.bin"
    counts.write_bytes(counts.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="Destination file changed"):
        copy([source], destination, verify_existing=True)

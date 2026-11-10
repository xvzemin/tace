import hashlib
import json
import math
import multiprocessing
import os
import pickle
import shutil
import sqlite3
import struct
import time
import zlib
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import lmdb
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write
from filelock import FileLock, Timeout
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

import tace.dataset.cache as cache_module
import tace.dataset.statistics as statistics_module
from tace.dataset.cache import (
    ARRAY_CODEC,
    COMPRESSED_ARRAY_CODEC,
    GRAPH_SCHEMA,
    LMDB_MODES,
    graph_cache_directories,
    read_cache_metadata,
    source_cache_key,
    statistics_cache_key,
)
from tace.dataset.dataloader import (
    StatisticsDataLoader,
    build_statistics_dataloader,
    collate_statistics,
)
from tace.dataset.datamodule import GraphDatasetLMDB, build_datamodule, create_graphs
from tace.dataset.element import build_element_lookup
from tace.dataset.filter import filter_graphs, filter_options
from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KeySpecification
from tace.dataset.read import (
    ASEDatabaseSource,
    ASEFileSource,
    BoundSource,
    ConcatSource,
    RecordSource,
    read_all_files,
)
from tace.dataset.records import record_from_atoms
from tace.dataset.split import IndexedSource
from tace.dataset.statistics import statistics_devices
from tace.dataset.store import (
    build_graph_store,
    compress_array_record,
    decode_graph,
    encode_graph,
    write_graphs,
    write_records,
)
from tace.models.blocks import ScaleShift


def test_graph_device_selection(monkeypatch):
    from tace.dataset.datamodule import OnTheFlyGraphDataset, graph_devices

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert graph_devices(None, "ase") == [None]
    assert graph_devices(None, "alchemiops") == ["cuda:0", "cuda:1"]
    assert (
        OnTheFlyGraphDataset([], {"neighborlist_backend": "alchemiops"}).device
        == "cuda"
    )
    assert graph_devices("cpu", "alchemiops") == ["cpu"]
    assert graph_devices("all", "alchemiops") == ["cuda:0", "cuda:1"]
    assert graph_devices(OmegaConf.create(["cuda:1", "cuda:0"]), "alchemiops") == [
        "cuda:1",
        "cuda:0",
    ]
    for invalid in ([], ["cuda:0", "cuda:0"], ["cuda", "cuda:1"], ["cuda:2"], ["cpu"]):
        with pytest.raises(ValueError):
            graph_devices(invalid, "alchemiops")
    with pytest.raises(ValueError, match="alchemiops"):
        graph_devices("all", "matscipy")
    for devices in ("all", ["cuda:0", "cuda:1"]):
        with pytest.raises(ValueError, match="one explicit device"):
            OnTheFlyGraphDataset([], {}, devices)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    with pytest.raises(ValueError, match="at least one CUDA"):
        graph_devices("all", "alchemiops")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert graph_devices(None, "alchemiops") == ["cpu"]
    assert (
        OnTheFlyGraphDataset([], {"neighborlist_backend": "alchemiops"}).device == "cpu"
    )


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


@pytest.fixture
def statistics_progress(monkeypatch):
    bars = []
    original = statistics_module.tqdm

    def record(*args, **kwargs):
        bar = original(*args, **kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setattr(statistics_module, "tqdm", record)
    return bars


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


def test_reader_aliases_preserve_explicit_fields_and_validate_scope():
    from ase.calculators.singlepoint import SinglePointCalculator

    from tace.dataset.graph import read_properties
    from tace.dataset.read import check_keys

    atoms = Atoms("H2", cell=[4, 4, 4], pbc=True)
    atoms.set_initial_magnetic_moments([1.0, -1.0])
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=3.0,
        forces=np.zeros((2, 3)),
        magmom=2.0,
        stresses=np.arange(12.0).reshape(2, 6),
    )
    atoms.info["reference_energy"] = 7.0
    # Real database producers put per-atom fields in info or row.data too.
    atoms.info["reference_forces"] = np.ones((2, 3))
    keys = KeySpecification.from_defaults()
    keys.info_keys["energy"] = "reference_energy"
    keys.arrays_keys["forces"] = "reference_forces"
    names = [
        "energy",
        "forces",
        "initial_collinear_magmoms",
        "total_collinear_magmom",
        "atomic_stresses",
    ]
    check_keys([atoms], names, keys)
    assert atoms.info["reference_energy"] == 7.0
    values = read_properties(atoms, keys, names, periodic_cell=True)
    assert values["energy"].item() == 7.0
    assert values["forces"].tolist() == [[1.0] * 3] * 2
    assert values["initial_collinear_magmoms"].tolist() == [1.0, -1.0]
    assert values["total_collinear_magmom"].item() == 2.0
    assert values["atomic_stresses"].shape == (2, 3, 3)
    assert values["atomic_stresses"][0].tolist() == [
        [0.0, 5.0, 4.0],
        [5.0, 1.0, 3.0],
        [4.0, 3.0, 2.0],
    ]
    atoms.arrays["reference_forces"] = np.zeros((1, 3))
    with pytest.raises(RuntimeError, match="per-atom property requires 2 entries"):
        read_properties(atoms, keys, ["forces"], periodic_cell=True)


@pytest.mark.parametrize("value", [-1, np.nan, [1.0, 2.0]])
def test_reader_rejects_invalid_weights(value):
    from tace.dataset.graph import read_properties

    atoms = Atoms("H")
    atoms.info.update(energy=1.0, energy_weight=value)
    with pytest.raises(RuntimeError, match="energy_weight"):
        read_properties(atoms, KeySpecification.from_defaults(), ["energy"], False)


@pytest.mark.parametrize("value", [-1, True, 1.5, np.nan])
def test_graph_rejects_invalid_fidelity(value):
    atoms = Atoms("H", info={"fidelity_idx": value})
    with pytest.raises(ValueError, match="fidelity_idx"):
        from_atoms(
            None,
            atoms,
            2.0,
            None,
            KeySpecification.from_defaults(),
            [],
            [],
            neighborlist_backend="ase",
        )


def test_property_dependencies_are_transitive(monkeypatch):
    from tace.dataset.quantity import PROPERTIES, get_need_property

    monkeypatch.setitem(PROPERTIES["energy"], "must_be_with", ["total_charge"])
    assert get_need_property(["hessian"], [], True) == [
        "total_charge",
        "energy",
        "forces",
        "hessian",
    ]


def _write_reader_database(path):
    from ase.db import connect

    open_lmdb = lmdb.open

    def open_small_lmdb(*args, **kwargs):
        kwargs["map_size"] = 1 << 20
        return open_lmdb(*args, **kwargs)

    # ASE-LMDB defaults to a 2 TiB map, which expands the file on Windows.
    # Restrict only fixture creation; readers still use the real lmdb.open.
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lmdb, "open", open_small_lmdb)
        database = connect(path)

    atoms_list = []
    with database:
        for index, symbols in enumerate(("H2", "O", "HO", "H2O")):
            atoms = Atoms(symbols)
            atoms.positions = np.arange(len(atoms) * 3).reshape(-1, 3) * 0.2
            atoms.info["reference_energy"] = -float(
                np.where(atoms.numbers == 1, 1, 2).sum()
            )
            atoms.arrays["reference_forces"] = np.full((len(atoms), 3), index * 0.1)
            database.write(
                atoms,
                data={
                    "reference_energy": atoms.info["reference_energy"],
                    "reference_forces": atoms.arrays["reference_forces"],
                },
            )
            atoms_list.append(atoms)
    return atoms_list


def _file_digest(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.digest()


def _reader_keys():
    keys = KeySpecification.from_defaults()
    keys.info_keys["energy"] = "reference_energy"
    keys.arrays_keys["forces"] = "reference_forces"
    return keys


def test_no_validation_takes_precedence_and_index_splits_do_not_overlap(
    tmp_path, monkeypatch
):
    from tace.dataset.read import tace_read_all_files

    monkeypatch.chdir(tmp_path)
    filename = tmp_path / "structures.db"
    _write_reader_database(filename)
    cfg = {
        "model": {"config": {"cutoff": 2.0}},
        "dataset": {
            "train_file": str(filename),
            "reader": "ase_db",
            "no_valid_set": True,
        },
    }
    keys = _reader_keys()
    cfg["dataset"].update(valid_file="missing-validation.db", valid_from_index=True)
    (tmp_path / "train.index").write_text("0\n1\n")
    (tmp_path / "valid.index").write_text("1\n2\n")
    source = tace_read_all_files(cfg, ["energy"], [], keys)
    assert len(source.train) == 4 and source.valid is None
    cfg["dataset"].pop("valid_file")
    cfg["dataset"]["no_valid_set"] = False
    with pytest.raises(ValueError, match="train.index and valid.index overlap"):
        tace_read_all_files(cfg, ["energy"], [], keys)
    (tmp_path / "valid.index").write_text("3\n2\n")
    split = tace_read_all_files(cfg, ["energy"], [], keys)
    assert [len(record) for record in split.valid.get_records([0, 1])] == [3, 2]


@pytest.mark.parametrize("mode", ["memory", *LMDB_MODES, "on_the_fly"])
def test_fit_defers_test_sources_until_test_stage(tmp_path, mode):
    from tace.dataset.datamodule import build_datamodule

    filename = tmp_path / "structures.db"
    _write_reader_database(filename)
    missing = tmp_path / "missing-test.db"
    cfg = {
        "model": {"config": {"cutoff": 2.0}},
        "dataset": {
            "train_file": str(filename),
            "test_files": str(missing),
            "reader": "ase_db",
            "no_valid_set": True,
            "storage_mode": mode,
            "neighborlist_backend": "ase",
            "shard_dirs": [str(tmp_path / "cache")],
        },
    }
    module = build_datamodule(cfg, None, ["energy"], [], _reader_keys())
    module.setup("fit")
    assert len(module.train_dataset) == 4
    assert module._test_sources is None and not module.threeAtomsList.test
    with pytest.raises(FileNotFoundError, match="missing-test.db"):
        module.setup("test")
    if hasattr(module.train_dataset, "close"):
        module.train_dataset.close()

    # A test-only setup also must not open an irrelevant training source.
    empty_training_directory = tmp_path / "unused-training-directory"
    empty_training_directory.mkdir()
    cfg["dataset"].update(
        train_file=str(empty_training_directory), test_files=str(filename)
    )
    test_only = build_datamodule(cfg, None, ["energy"], [], _reader_keys())
    test_only.setup("test")
    assert test_only.train_dataset is None and test_only.threeAtomsList is None
    assert len(test_only.test_datasets) == 1 and len(test_only.test_datasets[0]) == 4
    if hasattr(test_only.test_datasets[0], "close"):
        test_only.test_datasets[0].close()


@pytest.mark.parametrize(
    "suffix,reader", [("db", "ase_db"), ("aselmdb", "fair_aselmdb")]
)
def test_database_readers_are_lazy_readonly_and_batched(
    tmp_path, monkeypatch, suffix, reader
):
    from ase.db.row import AtomsRow

    from tace.dataset.read import read_all_files

    if suffix == "aselmdb":
        pytest.importorskip("ase_db_backends.aselmdb")
    path = tmp_path / f"structures.{suffix}"
    _write_reader_database(path)
    before = _file_digest(path)

    def no_atoms(*args, **kwargs):
        raise AssertionError("Record reads must not materialize ASE Atoms")

    monkeypatch.setattr(AtomsRow, "toatoms", no_atoms)
    source = read_all_files(
        {
            "path": path,
            "fidelity_idx": 2,
        },
        ["energy", "forces"],
        _reader_keys(),
        [],
        reader=reader,
    )
    assert not isinstance(source, list)
    assert len(source) == 4
    records = source.get_records([3, 0, 3])
    assert [len(record) for record in records] == [3, 2, 3]
    assert [record.info["fidelity_idx"] for record in records] == [2, 2, 2]
    assert records[0].data["reference_energy"] == -4.0
    assert _file_digest(path) == before
    restored = pickle.loads(pickle.dumps(source))
    assert len(restored.get_record(-1)) == 3


def test_generic_ase_raw_cache_and_reader_fingerprints(tmp_path, monkeypatch):
    from ase.io import write

    from tace.dataset.read import read_all_files

    source_dir = tmp_path / "source"
    source_dir.mkdir()
    filename = source_dir / "structures.cif"
    write(filename, [Atoms("Si", cell=[4, 4, 4], pbc=True)] * 2)
    cfg = {"dataset": {"reader": "ase", "train_file": str(source_dir)}}
    key = source_cache_key(cfg)
    source = read_all_files(
        source_dir,
        [],
        _reader_keys(),
        [],
        record_cache_dir=tmp_path / "cache",
    )
    assert len(source) == 2
    assert source.get_record(0).info["spacegroup"].no == 1

    def no_decode(*args, **kwargs):
        raise AssertionError("Warm raw cache must not decode the original file")

    with monkeypatch.context() as patch:
        patch.setattr("ase.io.iread", no_decode)
        warm = read_all_files(
            source_dir,
            [],
            _reader_keys(),
            [],
            record_cache_dir=tmp_path / "cache",
        )
        assert len(warm.get_records([1, 0, 1])) == 3
    cfg["dataset"]["train_file"] = {"path": str(filename)}
    assert source_cache_key(cfg) == key
    cfg["dataset"]["reader"] = "ase_db"
    assert source_cache_key(cfg) != key
    cfg["dataset"]["reader"] = "ase"
    cfg["dataset"]["train_file"]["fidelity_idx"] = 1
    assert source_cache_key(cfg) != key
    del cfg["dataset"]["train_file"]["fidelity_idx"]
    filename.write_text(filename.read_text() + "\n# changed\n")
    assert source_cache_key(cfg) != key
    write(source_dir / "z-atoms.xyz", Atoms("He"))
    mixed = read_all_files(
        source_dir, [], _reader_keys(), record_cache_dir=tmp_path / "cache"
    )
    assert len(mixed) == 3
    assert mixed.get_record(2).numbers.tolist() == [2]
    source.close()
    warm.close()
    mixed.close()


@pytest.mark.parametrize("selection", ["list", "omega", "fidelity"])
def test_source_readers_follow_entries_before_sorting(tmp_path, selection):
    from tace.dataset.cache import source_signature
    from tace.dataset.read import discover_sources

    database = tmp_path / "z-structures.db"
    xyz = tmp_path / "a-atoms.xyz"
    _write_reader_database(database)
    write(xyz, Atoms("He"))
    sources = [str(database), str(xyz)]
    reader = ["ase_db", "ase"]
    if selection == "omega":
        reader = OmegaConf.create(reader)
        sources = OmegaConf.create(sources)
    elif selection == "fidelity":
        sources = [
            {"path": str(database), "fidelity_idx": 1},
            {"path": str(xyz), "fidelity_idx": 2},
        ]
    specs = discover_sources(sources, reader)
    assert [(Path(spec.path).name, spec.reader) for spec in specs] == [
        (xyz.name, "ase"),
        (database.name, "ase_db"),
    ]
    source = read_all_files(sources, [], _reader_keys(), reader=reader)
    assert len(source) == 5
    assert source.get_record(0).numbers.tolist() == [2]
    assert len(source.get_record(1)) == 2
    if selection == "fidelity":
        assert source.get_record(0).info["fidelity_idx"] == 2
        assert source.get_record(1).info["fidelity_idx"] == 1
    source.close()
    signature = source_signature(sources, reader)
    assert [(Path(item["path"]).name, item["reader"]) for item in signature] == [
        (xyz.name, "ase"),
        (database.name, "ase_db"),
    ]
    if selection != "fidelity":
        assert source_cache_key(
            {"dataset": {"reader": reader, "train_file": sources}}
        ) == source_cache_key(
            {
                "dataset": {
                    "reader": ["ase_db", "ase"],
                    "train_file": [
                        {"path": str(database)},
                        {"path": str(xyz)},
                    ],
                }
            }
        )
    missing = tmp_path / "missing.xyz"
    signature = source_signature([database, missing], ["ase_db", "ase"])
    missing_item = next(item for item in signature if item["path"] == str(missing))
    assert missing_item["reader"] == "ase"
    assert missing_item["size"] is None


def test_source_readers_match_seven_entries_with_a_directory(tmp_path):
    from tace.dataset.cache import source_signature
    from tace.dataset.read import discover_sources, stage_readers, tace_read_all_files

    sources = []
    for index in range(7):
        if index == 5:
            directory = tmp_path / "sixth-source"
            (directory / "nested").mkdir(parents=True)
            _write_reader_database(directory / "first.db")
            _write_reader_database(directory / "nested" / "second.db")
            sources.append(str(directory))
        else:
            path = tmp_path / f"source-{index}.xyz"
            write(path, Atoms("He"))
            sources.append(str(path))
    readers = ["ase"] * 5 + ["ase_db", "ase"]
    specs = discover_sources(sources, readers)
    assert len(specs) == 8
    database_specs = [
        spec for spec in specs if Path(sources[5]) in Path(spec.path).parents
    ]
    assert len(database_specs) == 2
    assert all(spec.reader == "ase_db" for spec in database_specs)
    source = read_all_files(sources, [], _reader_keys(), reader=readers)
    assert len(source) == 14
    assert (
        sum(
            len(source.sources[index])
            for index, spec in enumerate(specs)
            if spec.reader == "ase_db"
        )
        == 8
    )
    source.close()
    cfg = {
        "dataset": {
            "reader": readers,
            "train_file": sources[:3],
            "valid_file": sources[3:5],
            "test_files": sources[5:],
            "shard_dirs": [str(tmp_path / "graphs")],
        }
    }
    assigned = stage_readers(cfg["dataset"])
    assert assigned == {
        "train": ["ase"] * 3,
        "valid": ["ase"] * 2,
        "test0": ["ase_db"],
        "test1": ["ase"],
    }
    split = tace_read_all_files(cfg, [], [], _reader_keys())
    assert len(split.train) == 3
    assert len(split.valid) == 2
    assert [len(part) for part in split.test] == [8, 1]
    split.train.close()
    split.valid.close()
    for part in split.test:
        part.close()
    valid_only = tace_read_all_files(cfg, [], [], _reader_keys(), stages=("valid",))
    assert valid_only.train is None and len(valid_only.valid) == 2
    valid_only.valid.close()
    test_only = tace_read_all_files(cfg, [], [], _reader_keys(), stages=("test",))
    assert [len(part) for part in test_only.test] == [8, 1]
    for part in test_only.test:
        part.close()
    signature = source_signature(sources[5], assigned["test0"])
    assert len(signature) == 2 and all(item["reader"] == "ase_db" for item in signature)
    assert source_cache_key(cfg, "test0") != source_cache_key(cfg, "test1")
    for bad_readers in (readers[:-1], readers + ["ase"]):
        cfg["dataset"]["reader"] = bad_readers
        with pytest.raises(ValueError, match="expected 7"):
            tace_read_all_files(cfg, [], [], _reader_keys(), stages=("train",))
        with pytest.raises(ValueError, match="expected 7"):
            source_cache_key(cfg, "test0")
    cfg["dataset"]["reader"] = "ase_db"
    assert set(stage_readers(cfg["dataset"]).values()) == {"ase_db"}
    cfg["dataset"].update(reader=["ase"] * 3, valid_file=None, test_files=None)
    assert stage_readers(cfg["dataset"])["valid"] == ["ase"] * 3


def test_source_reader_selection_rejects_invalid_configuration(tmp_path):
    from tace.dataset.read import discover_sources

    filename = tmp_path / "atoms.xyz"
    write(filename, Atoms("He"))
    with pytest.raises(ValueError, match="one value per source entry"):
        discover_sources([filename, filename], ["ase"])
    for reader in (None, {}, 1):
        with pytest.raises(ValueError, match="string or a list"):
            discover_sources(filename, reader)
    with pytest.raises(ValueError, match="Unknown dataset reader"):
        discover_sources(filename, ["unknown"])
    with pytest.raises(ValueError, match="conflicting source readers"):
        discover_sources([filename, filename], ["ase", "ase_db"])
    for key, value in (
        ("format", "extxyz"),
        ("options", {}),
        ("type", "ase"),
        ("reader", "ase"),
    ):
        entry = {"path": str(filename), key: value}
        with pytest.raises(ValueError, match="path and optionally fidelity_idx"):
            discover_sources(entry)
        with pytest.raises(ValueError, match="path and optionally fidelity_idx"):
            source_cache_key({"dataset": {"train_file": entry}})


def test_storage_modes_cold_statistics_and_warm_training(tmp_path, monkeypatch):
    from tace.dataset.datamodule import build_datamodule, load_cached_datamodule
    from tace.dataset.statistics import compute_statistics, prepare_atomic_energies

    pytest.importorskip("ase_db_backends.aselmdb")
    monkeypatch.chdir(tmp_path)
    filename = tmp_path / "structures.aselmdb"
    atoms_list = _write_reader_database(filename)
    keys = _reader_keys()
    cfg = {
        "misc": {"device": "cpu"},
        "model": {
            "config": {
                "cutoff": 2.0,
                "fidelity": [{"name": "DFT", "atomic_energy": None}],
            }
        },
        "dataset": {
            "reader": "fair_aselmdb",
            "train_file": str(filename),
            "no_valid_set": True,
            "shard_dirs": [str(tmp_path / "cache")],
            "neighborlist_backend": "ase",
            "graph_batch_size": 2,
            "graph_workers": 1,
            "train_dataloader": {
                "_target_": "torch_geometric.loader.DataLoader",
                "batch_size": 2,
                "shuffle": False,
                "num_workers": 0,
            },
            "statistics_dataloader": {"batch_size": 3},
        },
    }
    options = dict(
        cutoff=2.0,
        max_neighbors=None,
        keyspec=keys,
        target_property=["energy", "forces"],
        embedding_property=[],
        neighborlist_backend="ase",
    )
    expected_graphs = [from_atoms(None, atoms, **options) for atoms in atoms_list]
    statistics_by_mode = []
    for mode in ("memory", *LMDB_MODES, "on_the_fly"):
        cfg["dataset"]["storage_mode"] = mode
        cfg["dataset"]["shard_dirs"] = [str(tmp_path / f"cache-{mode}")]
        module = build_datamodule(cfg, None, ["energy", "forces"], [], keys)
        module.prepare_data()
        module.setup("fit")
        if mode in LMDB_MODES:
            assert module.train_dataset.metadata["codec"] == (
                COMPRESSED_ARRAY_CODEC if mode == "clmdb" else ARRAY_CODEC
            )
        element, references = prepare_atomic_energies(cfg, module)
        assert element.zs == [1, 8]
        assert references[0] == pytest.approx({1: -1.0, 8: -2.0}, abs=1e-12)
        statistics = compute_statistics(cfg, module, element, references)
        assert statistics[0]["num_graphs"] == 4
        assert statistics[0]["rms_forces"][1] == pytest.approx(
            math.sqrt(0.045), abs=1e-12
        )
        statistics_by_mode.append(statistics[0])
        graphs = [
            graph
            for batch in module.train_dataloader()
            for graph in batch.to_data_list()
        ]
        for actual, expected in zip(graphs, expected_graphs):
            for name, value in expected.items():
                if torch.is_tensor(value):
                    assert actual[name].dtype == value.dtype
                    torch.testing.assert_close(
                        actual[name].reshape(value.shape),
                        value,
                        rtol=0,
                        atol=0,
                        check_device=False,
                    )
        if hasattr(module.train_dataset, "close"):
            module.train_dataset.close()
    for key in ("rms_forces", "mean_energy", "avg_num_neighbors"):
        assert all(
            item[key] == statistics_by_mode[0][key] for item in statistics_by_mode
        )

    def no_source(*args, **kwargs):
        raise AssertionError("Warm training must not open sources or construct graphs")

    monkeypatch.setattr("tace.dataset.datamodule.tace_read_all_files", no_source)
    monkeypatch.setattr("tace.dataset.datamodule.from_records", no_source)
    for stored_mode in LMDB_MODES:
        cfg["dataset"]["shard_dirs"] = [str(tmp_path / f"cache-{stored_mode}")]
        for read_mode in LMDB_MODES:
            cfg["dataset"]["storage_mode"] = read_mode
            warm = load_cached_datamodule(cfg, ["energy", "forces"], [], keys)
            assert warm is not None and warm.threeAtomsList is None
            assert sum(batch.num_graphs for batch in warm.train_dataloader()) == 4
            warm.train_dataset.close()


@pytest.mark.parametrize("context", multiprocessing.get_all_start_methods())
def test_aselmdb_online_loader_process_reuse(tmp_path, context):
    from tace.dataset.datamodule import OnTheFlyGraphDataset
    from tace.dataset.read import read_all_files

    pytest.importorskip("ase_db_backends.aselmdb")
    filename = tmp_path / "structures.aselmdb"
    _write_reader_database(filename)
    keys = _reader_keys()
    source = read_all_files(
        filename, ["energy", "forces"], keys, [], reader="fair_aselmdb"
    )
    options = dict(
        cutoff=2.0,
        max_neighbors=None,
        keyspec=keys,
        target_property=["energy", "forces"],
        embedding_property=[],
        neighborlist_backend="ase",
    )
    dataset = OnTheFlyGraphDataset(source, options)
    expected = dataset[0].energy.clone()  # Open the environment before forking.
    loader = DataLoader(
        dataset,
        batch_size=2,
        num_workers=1,
        multiprocessing_context=context,
        timeout=30,
    )
    actual = next(iter(loader))
    torch.testing.assert_close(actual.energy[:1], expected)
    assert actual.energy.dtype == torch.float64


@pytest.fixture
def structures():
    frames = []
    for index, symbols in enumerate(("H2", "O", "HO", "H2O")):
        atoms = Atoms(symbols)
        atoms.positions = np.arange(len(atoms) * 3).reshape(-1, 3) * 0.2
        atoms.calc = SinglePointCalculator(
            atoms, energy=-float(index + 1), forces=np.zeros((len(atoms), 3))
        )
        frames.append(atoms)
    return frames


@pytest.fixture
def sqlite_source(tmp_path, structures):
    path = tmp_path / "structures.db"
    with connect(path) as database:
        for atoms in structures:
            database.write(atoms)
    return path


def _track_connections(monkeypatch):
    connections = []
    real_connect = sqlite3.connect

    def track(*args, **kwargs):
        connection = real_connect(*args, **kwargs)
        connections.append(connection)
        return connection

    monkeypatch.setattr(sqlite3, "connect", track)
    return connections


def _assert_closed(connections):
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")


def _options():
    return dict(
        cutoff=2.0,
        max_neighbors=None,
        keyspec=KeySpecification.from_defaults(),
        target_property=["energy", "forces"],
        embedding_property=[],
        neighborlist_backend="ase",
    )


def test_wrapped_sqlite_source_closes_and_reopens_readonly(sqlite_source, monkeypatch):
    original = sqlite_source.read_bytes()
    connections = _track_connections(monkeypatch)
    source = read_all_files(
        sqlite_source, ["energy", "forces"], _options()["keyspec"], reader="ase_db"
    )
    subset = IndexedSource(source, [3, 0, 3])
    assert [len(record) for record in subset.get_records([0, 1, 2])] == [3, 2, 3]
    subset.close()
    subset.close()
    _assert_closed(connections)

    assert len(subset.get_record(1)) == 2
    assert len(connections) == 2
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        connections[-1].execute("DELETE FROM systems")
    subset.close()
    _assert_closed(connections)
    assert sqlite_source.read_bytes() == original
    sqlite_source.unlink()


@pytest.mark.parametrize("mode", ["memory", "lmdb"])
def test_offline_graphs_release_raw_source_and_split_readers(
    tmp_path, sqlite_source, monkeypatch, mode
):
    monkeypatch.chdir(tmp_path)
    connections = _track_connections(monkeypatch)
    loader = {
        "_target_": "torch_geometric.loader.DataLoader",
        "batch_size": 2,
        "num_workers": 0,
    }
    cfg = {
        "misc": {"device": "cpu", "seed": 42},
        "model": {"config": {"cutoff": 2.0}},
        "dataset": {
            "reader": "ase_db",
            "train_file": str(sqlite_source),
            "storage_mode": mode,
            "shard_dirs": [str(tmp_path / "graphs")],
            "neighborlist_backend": "ase",
            "valid_ratio": 0.5,
            "train_dataloader": loader,
            "valid_dataloader": dict(loader),
        },
    }
    module = build_datamodule(
        cfg, None, ["energy", "forces"], [], _options()["keyspec"]
    )
    module.setup("fit")
    assert len(connections) == 2  # Validation reopens the shared raw source.
    _assert_closed(connections)
    sqlite_source.unlink()
    for dataset in (module.train_dataset, module.val_dataset):
        assert len(dataset) == 2
        assert len(dataset.__getitems__([0, 1])) == 2
        dataset.close()

    if mode == "lmdb":

        def no_raw_source(*args, **kwargs):
            pytest.fail("Completed graph caches must not reopen raw sources")

        monkeypatch.setattr(
            "tace.dataset.datamodule.tace_read_all_files", no_raw_source
        )
        warm = build_datamodule(
            cfg, None, ["energy", "forces"], [], _options()["keyspec"]
        )
        warm.setup("fit")
        assert warm.threeAtomsList is None
        assert len(warm.train_dataset) == len(warm.val_dataset) == 2
        warm.train_dataset.close()
        warm.val_dataset.close()
    _assert_closed(connections)


@pytest.mark.parametrize("mode", ["memory", "lmdb", "multi_gpu"])
def test_failed_offline_build_releases_source(
    tmp_path, sqlite_source, monkeypatch, mode
):
    connections = _track_connections(monkeypatch)
    source = read_all_files(
        sqlite_source, ["energy", "forces"], _options()["keyspec"], reader="ase_db"
    )

    def fail(*args, **kwargs):
        raise RuntimeError("injected graph failure")

    if mode == "multi_gpu":
        monkeypatch.setattr(
            "tace.dataset.datamodule.graph_devices", lambda *args: ["cuda:0", "cuda:1"]
        )
        monkeypatch.setattr("tace.dataset.datamodule.write_multi_gpu_graphs", fail)
    else:
        monkeypatch.setattr("tace.dataset.datamodule.from_records", fail)
    with pytest.raises(RuntimeError, match="injected graph failure"):
        create_graphs(
            source,
            None,
            _options(),
            "train",
            [tmp_path / "graphs"],
            storage_mode="lmdb" if mode == "multi_gpu" else mode,
        )
    _assert_closed(connections)
    assert len(source.get_record(0)) == 2
    source.close()
    _assert_closed(connections)


def test_online_graphs_keep_source_available_and_reopen_after_close(
    tmp_path, sqlite_source, monkeypatch
):
    connections = _track_connections(monkeypatch)
    source = ASEDatabaseSource(sqlite_source)
    dataset = create_graphs(
        source, None, _options(), "train", [tmp_path], storage_mode="on_the_fly"
    )
    assert connections == []
    assert dataset[0].num_nodes == 2
    assert len(connections) == 1
    assert connections[0].execute("SELECT COUNT(*) FROM systems").fetchone() == (4,)
    dataset.close()
    _assert_closed(connections)
    assert dataset[3].num_nodes == 3
    assert len(connections) == 2
    dataset.close()
    _assert_closed(connections)


def test_cache_hit_releases_an_already_open_raw_source(
    tmp_path, sqlite_source, monkeypatch
):
    connections = _track_connections(monkeypatch)
    source = ASEDatabaseSource(sqlite_source)
    arguments = (source, None, _options(), "train", [tmp_path / "graphs"])
    cold = create_graphs(*arguments)
    cold.close()
    _assert_closed(connections)
    source.get_record(0)

    def no_records(*args, **kwargs):
        pytest.fail("A cache published after source discovery must not read raw rows")

    monkeypatch.setattr(source, "get_records", no_records)
    warm = create_graphs(*arguments)
    assert len(warm) == 4
    _assert_closed(connections)
    warm.close()


def test_trajectory_source_releases_real_file_and_reopens(tmp_path, structures):
    path = tmp_path / "structures.traj"
    write(path, structures)
    source = ASEFileSource(path)
    assert len(source.get_record(0)) == 2
    reader = source._trajectory
    source.close()
    assert reader.backend._fd.closed
    assert len(source.get_record(3)) == 3
    source.close()
    path.unlink()


def test_record_cache_wrappers_release_store_and_reopen(tmp_path, structures):
    dataset = write_records(
        (record_from_atoms(atoms) for atoms in structures), [tmp_path / "records"]
    )
    assert dataset.metadata["codec"] == COMPRESSED_ARRAY_CODEC
    source = IndexedSource(
        ConcatSource(
            [
                BoundSource(
                    RecordSource(dataset, "records"), _options()["keyspec"], [], []
                )
            ]
        ),
        [3, 0],
    )
    assert len(source.get_record(0)) == 3
    assert dataset._opened
    source.close()
    assert not dataset._opened
    assert len(source.get_record(1)) == 2
    assert dataset._opened
    source.close()


def test_aselmdb_shared_readers_can_release_and_reopen(
    tmp_path, structures, monkeypatch
):
    pytest.importorskip("ase_db_backends.aselmdb")
    path = tmp_path / "structures.aselmdb"
    real_open = lmdb.open

    def small_open(*args, **kwargs):
        kwargs["map_size"] = 1 << 20
        return real_open(*args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(lmdb, "open", small_open)
        with connect(path) as database:
            for atoms in structures:
                database.write(atoms)
    first = ASEDatabaseSource(path, lmdb=True)
    second = ASEDatabaseSource(path, lmdb=True)
    assert len(first.get_record(0)) == len(second.get_record(0)) == 2
    shared = first._lmdb()
    assert shared is second._lmdb()
    first.close()
    with pytest.raises(lmdb.Error):
        shared.stat()
    assert len(second.get_record(3)) == 3
    assert len(first.get_record(1)) == 1
    first.close()
    second.close()
    path.unlink()


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
    _write_reader_database(filename)
    keyspec = _reader_keys()
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
        ({"enabled": True}, [0, 2, 3, 4, 5, 6]),
        ({"enabled": True, "max_edges": 24001}, list(range(7))),
        ({"enabled": True, "max_edges": None}, list(range(7))),
        ({"enabled": True, "max_forces": 5}, [0, 2, 6]),
        ({"enabled": True, "max_energy_per_atom": -1}, [0, 3, 6]),
        ({"enabled": True, "max_stress": 5}, [0, 2, 6]),
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
    before = _snapshot(tmp_path)
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
    assert _snapshot(tmp_path) == before


@pytest.mark.parametrize(
    "options,error",
    [
        ({"enabled": "true"}, TypeError),
        ({"enabled": True, "max_atoms": 2.5}, TypeError),
        ({"enabled": True, "max_edges": True}, TypeError),
        ({"enabled": True, "max_forces": -1}, ValueError),
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
    before = _snapshot(tmp_path)
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
    assert _snapshot(tmp_path) == before


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


def test_raw_and_statistics_cache_dependencies_are_separate(tmp_path, monkeypatch):
    monkeypatch.delenv("TACE_USE_U_SHIFT", raising=False)
    source = tmp_path / "train.xyz"
    source.write_text("source")
    cfg = {
        "dataset": {
            "train_file": str(source),
            "valid_ratio": 0.1,
            "test_files": [str(tmp_path / "test-a.xyz")],
        },
        "model": {
            "config": {
                "cutoff": 3.0,
                "fidelity": [{"name": "PBE", "atomic_energy": {1: 1.0}}],
            }
        },
    }
    keyspec = KeySpecification.from_defaults()
    raw_key = source_cache_key(cfg, keyspec=keyspec)
    stats_key = statistics_cache_key(cfg, ["energy"], [], keyspec)
    monkeypatch.setenv("TACE_USE_U_SHIFT", "1")
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == stats_key
    assert source_cache_key(cfg, keyspec=keyspec) == raw_key
    monkeypatch.setenv("TACE_USE_U_SHIFT", "0")
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == stats_key
    cfg["dataset"]["filter"] = {"enabled": False, "max_forces": 1}
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == stats_key
    cfg["dataset"]["filter"]["enabled"] = True
    filtered_key = statistics_cache_key(cfg, ["energy"], [], keyspec)
    assert filtered_key == stats_key
    assert source_cache_key(cfg, keyspec=keyspec) == raw_key
    cfg["dataset"]["filter"]["max_forces"] = 2
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == filtered_key
    del cfg["dataset"]["filter"]
    cfg["dataset"]["test_files"] = [str(tmp_path / "test-b.xyz")]
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == stats_key
    cfg["model"]["config"]["cutoff"] = 4.0
    assert source_cache_key(cfg, keyspec=keyspec) == raw_key
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) != stats_key
    cfg["model"]["config"]["cutoff"] = 3.0
    cfg["model"]["config"]["fidelity"][0]["name"] = "renamed"
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) == stats_key
    cfg["model"]["config"]["fidelity"][0]["atomic_energy"][1] = 2.0
    assert statistics_cache_key(cfg, ["energy"], [], keyspec) != stats_key
    assert source_cache_key(cfg, keyspec=keyspec) == raw_key


def test_missing_model_statistics_only_checks_requested_inputs():
    model_config = {
        "fidelity": [{"name": "PBE", "magnetic_scale": None}],
        "scale_shift": {
            "enable": True,
            "scale_type": "rms_forces",
            "shift_type": None,
        },
    }
    cached_statistics = [{"rms_forces": {26: 1.0}, "atomic_energy": {26: 0.0}}]

    assert not statistics_module.missing_model_statistics(
        cached_statistics,
        model_config,
        target_property=["energy"],
        embedding_property=[],
    )
    assert statistics_module.missing_model_statistics(
        cached_statistics,
        model_config,
        target_property=["energy"],
        embedding_property=["initial_noncollinear_magmoms"],
    ) == {"max_noncollinear_magmoms_norm_by_element"}

    model_config["fidelity"][0]["magnetic_scale"] = {26: 2.0}
    assert statistics_module.missing_model_statistics(
        cached_statistics,
        model_config,
        target_property=["energy"],
        embedding_property=["initial_noncollinear_magmoms"],
    ) == {"max_noncollinear_magmoms_norm_by_element"}

    cached_statistics[0]["max_noncollinear_magmoms_norm_by_element"] = {26: 3.0}
    assert not statistics_module.missing_model_statistics(
        cached_statistics,
        model_config,
        target_property=["energy"],
        embedding_property=["initial_noncollinear_magmoms"],
    )

    del cached_statistics[0]["atomic_energy"]
    assert statistics_module.missing_model_statistics(
        cached_statistics, model_config, ["energy"], []
    ) == {"atomic_energy"}


@pytest.mark.parametrize("targets", [["energy", "forces"], ["direct_hessian"]])
def test_training_statistics_reuse_graph_cache(tmp_path, monkeypatch, targets):
    import tace.dataset.datamodule as datamodule_module
    import tace.scripts.train as train_module

    monkeypatch.chdir(tmp_path)
    records = [([1, 1], 2.0, 0), ([8], 3.0, 0), ([1, 8], 4.0, 0), ([8], 5.0, 1)]
    graphs = []
    for numbers, energy, fidelity in records:
        numbers = torch.tensor(numbers)
        graphs.append(
            Data(
                atomic_numbers=numbers,
                num_nodes=len(numbers),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                fidelity_idx=torch.tensor(fidelity),
                energy=torch.tensor([energy], dtype=torch.float64),
                energy_weight=torch.ones(1, dtype=torch.float64),
                forces=torch.ones(len(numbers), 3, dtype=torch.float64),
                forces_weight=torch.ones(1, dtype=torch.float64),
                direct_hessian=torch.ones(len(numbers) ** 2, 3, 3, dtype=torch.float64),
                direct_hessian_weight=torch.ones(1, dtype=torch.float64),
            )
        )
    cache = tmp_path / "graphs"
    cache.mkdir()

    def no_source_read(*args, **kwargs):
        raise AssertionError("Existing LMDB graphs must not read source files")

    monkeypatch.setattr(datamodule_module, "tace_read_all_files", no_source_read)
    monkeypatch.setattr(datamodule_module, "from_records", no_source_read)
    monkeypatch.setattr(train_module, "initialize", lambda cfg: cfg)
    monkeypatch.setattr(train_module, "create_model", lambda *args: None)
    monkeypatch.setattr(train_module, "configure_model_precision", lambda *args: None)
    cfg = OmegaConf.create(
        {
            "misc": {"device": "cuda" if torch.cuda.is_available() else "cpu"},
            "dtype": {"model": "float32", "autocast": None, "matmul": "ieee"},
            "loss": {"loss_property": targets},
            "model": {
                "config": {
                    "cutoff": 3.0,
                    "fidelity": [
                        {"name": "fitted"},
                        {"name": "fixed", "atomic_energy": {1: 4.0, 8: 5.0}},
                    ],
                }
            },
            "dataset": {
                "storage_mode": "lmdb",
                "shard_dirs": [str(cache)],
                "train_file": "does-not-exist.db",
                "valid_file": "does-not-exist.db",
                "train_dataloader": {
                    "_target_": "torch_geometric.loader.DataLoader",
                    "batch_size": 2,
                    "num_workers": 0,
                },
                "statistics_dataloader": {"batch_size": 3},
            },
        }
    )
    for stage, data in (("train", graphs), ("valid", [graphs[-1]])):
        plain_cfg = OmegaConf.to_container(cfg, resolve=True)
        directories = graph_cache_directories(plain_cfg)
        stored = write_graphs(
            data,
            directories,
            stage,
            graph_config={"cutoff": 3.0, "max_neighbors": None},
        )
        stored.close()
    contents = {p: p.read_bytes() for p in cache.rglob("*") if p.is_file()}
    _, statistics, _, _, _, datamodule = train_module.build(cfg)
    assert datamodule.threeAtomsList is None
    assert statistics[0]["atomic_numbers"] == [1, 8]
    assert statistics[0]["data_schema"] == GRAPH_SCHEMA
    assert statistics[0]["data_dtype"] == "float64"
    assert statistics[0]["num_graphs"] == 3
    assert statistics[1]["num_graphs"] == 1
    if "energy" in targets:
        assert statistics[0]["atomic_energy"] == pytest.approx({1: 1.0, 8: 3.0})
        assert statistics[1]["atomic_energy"] == {1: 4.0, 8: 5.0}
        with monkeypatch.context() as patch:
            patch.setattr(statistics_module, "log_statistics_to_yaml", lambda _: None)
            reference = statistics_module.statistics_from_loader(
                DataLoader(graphs, batch_size=3, exclude_keys=["direct_hessian"]),
                [1, 8],
                [{1: 1.0, 8: 3.0}, {1: 4.0, 8: 5.0}],
                targets,
                device=cfg.misc.device,
                num_fidelities=2,
            )
        for actual, expected in zip(statistics, reference):
            assert actual["mean_energy"] == pytest.approx(expected["mean_energy"])
            assert actual["rms_forces"] == pytest.approx(expected["rms_forces"])
    datamodule.train_dataset.close()
    datamodule.val_dataset.close()

    monkeypatch.setattr(train_module, "load_cached_datamodule", no_source_read)
    monkeypatch.setattr(train_module, "prepare_atomic_energies", no_source_read)
    monkeypatch.setattr(train_module, "compute_statistics", no_source_read)
    _, reused, _, _, _, datamodule = train_module.build(cfg)
    assert reused == statistics
    datamodule.prepare_data()
    assert datamodule.threeAtomsList is None
    assert all(p.read_bytes() == value for p, value in contents.items())


def _statistics_loader_graphs():
    return [
        Data(
            num_nodes=n,
            atomic_numbers=torch.ones(n, dtype=torch.int64),
            edge_index=torch.stack((torch.arange(n), torch.arange(n))),
            num_neighbors=torch.ones(n, dtype=torch.int64),
            positions=torch.zeros(n, 3, dtype=torch.float64),
            energy=torch.tensor([n * 0.1], dtype=torch.float64),
            energy_weight=torch.ones(1, dtype=torch.float64),
            forces=torch.arange(n * 3, dtype=torch.float64).reshape(n, 3),
            forces_weight=torch.ones(1, dtype=torch.float64),
            fidelity_idx=torch.tensor(n % 2),
        )
        for n in (2, 1, 4, 3, 1)
    ]


@pytest.mark.parametrize("as_dict", [False, True])
def test_statistics_collation_matches_pyg_and_keeps_integer_indices(as_dict):
    samples = _statistics_loader_graphs()
    expected = Batch.from_data_list(samples)
    actual = collate_statistics(
        [sample.to_dict() for sample in samples] if as_dict else samples
    )
    assert actual.num_graphs == expected.num_graphs
    assert actual.num_nodes == expected.num_nodes
    assert set(actual.keys()) == set(expected.keys())
    for key in expected.keys():
        if key != "num_nodes":
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    assert actual.batch.dtype == actual.ptr.dtype == torch.int64


def test_lmdb_statistics_projection_is_owned_and_readonly(tmp_path):
    samples = _statistics_loader_graphs()
    dataset = write_graphs(samples, [tmp_path], "train")
    before = {p: _file_digest(p) for p in tmp_path.rglob("*") if p.is_file()}
    cfg = {
        "dataset": {
            "train_dataloader": {
                "_target_": "torch_geometric.loader.DataLoader",
                "batch_size": 1,
                "num_workers": 4,
                "prefetch_factor": 2,
                "persistent_workers": True,
            },
            "statistics_dataloader": {"batch_size": 3},
        },
    }
    loader = build_statistics_dataloader(cfg, dataset, ["energy", "forces"])
    assert isinstance(loader, StatisticsDataLoader)
    assert loader.num_workers == 0
    assert loader.prefetch_factor is None
    assert not loader.persistent_workers
    assert loader.batch_size == 3
    batches = list(loader)
    assert sum(b.num_graphs for b in batches) == len(samples)
    assert "positions" not in batches[0]
    assert "edge_index" not in batches[0]
    assert batches[0].energy.dtype == torch.float64
    # Returning dictionaries avoids one PyG object per graph. Mutating a decoded
    # tensor must never change the mapped source bytes or subsequent reads.
    decoded = loader.dataset[0]
    assert isinstance(decoded, dict)
    decoded["energy"].fill_(12345)
    torch.testing.assert_close(loader.dataset[0]["energy"], samples[0].energy)
    loader.dataset.close()
    dataset.close()
    assert all(_file_digest(p) == v for p, v in before.items())


def test_explicit_statistics_workers_have_independent_prefetch_default():
    cfg = {
        "dataset": {
            "train_dataloader": {
                "_target_": "torch_geometric.loader.DataLoader",
                "batch_size": 1,
                "num_workers": 8,
                "prefetch_factor": 4,
            },
            "statistics_dataloader": {"num_workers": 2},
        }
    }
    loader = build_statistics_dataloader(cfg, _statistics_loader_graphs(), ["energy"])
    assert loader.num_workers == 2
    assert loader.prefetch_factor == 1
    cfg["dataset"]["statistics_dataloader"]["prefetch_factor"] = 3
    assert (
        build_statistics_dataloader(cfg, _statistics_loader_graphs()).prefetch_factor
        == 3
    )


def test_statistics_devices_respect_visibility_and_dataset_mode(tmp_path, monkeypatch):
    dataset = write_graphs(_statistics_loader_graphs(), [tmp_path], "train")
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    cfg = OmegaConf.create({"dataset": {}, "misc": {"device": "cuda"}})
    assert statistics_devices(cfg, dataset) == ["cuda:0", "cuda:1", "cuda:2"]
    assert statistics_devices(cfg, _statistics_loader_graphs()) == ["cuda"]
    cfg.misc.device = "cuda:2"
    assert statistics_devices(cfg, dataset) == ["cuda:2"]
    cfg.misc.device = "cpu"
    assert statistics_devices(cfg, dataset) == ["cpu"]
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    cfg.misc.device = "cuda"
    assert statistics_devices(cfg, dataset) == ["cuda:0"]
    dataset.close()


@pytest.mark.parametrize(
    ("environment_size", "group_size", "local_rank", "expected"),
    [
        (4, None, 2, ["cuda:2"]),
        (4, None, None, ["cuda"]),
        (1, 4, None, ["cuda"]),
        (1, 4, 1, ["cuda:1"]),
    ],
)
def test_statistics_devices_do_not_expand_all_gpus_in_each_distributed_rank(
    tmp_path, monkeypatch, environment_size, group_size, local_rank, expected
):
    dataset = write_graphs(_statistics_loader_graphs(), [tmp_path], "train")
    monkeypatch.setenv("WORLD_SIZE", str(environment_size))
    if local_rank is None:
        monkeypatch.delenv("LOCAL_RANK", raising=False)
    else:
        monkeypatch.setenv("LOCAL_RANK", str(local_rank))
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "is_initialized", lambda: group_size is not None
    )
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: group_size)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    try:
        cfg = {"misc": {"device": "cuda"}}
        assert statistics_devices(cfg, dataset) == expected
        cfg["misc"]["device"] = "cuda:3"
        assert statistics_devices(cfg, dataset) == ["cuda:3"]
        cfg["misc"]["device"] = "cpu"
        assert statistics_devices(cfg, dataset) == ["cpu"]
    finally:
        dataset.close()


def test_statistics_dataloader_overrides_training_batch_size():
    dataset = [
        Data(
            atomic_numbers=torch.tensor([1]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            positions=torch.zeros(1, 3),
            energy=torch.ones(1),
            energy_weight=torch.ones(1),
        )
        for _ in range(5)
    ]
    loader = build_statistics_dataloader(
        {
            "dataset": {
                "epoch_sampler": {"epoch_size": 1},
                "train_dataloader": {
                    "_target_": "torch_geometric.loader.DataLoader",
                    "batch_size": 2,
                    "shuffle": True,
                    "drop_last": True,
                    "num_workers": 0,
                },
                "statistics_dataloader": {"batch_size": 4},
            }
        },
        dataset,
        target_property=["energy"],
    )

    assert loader.batch_size == 4
    assert loader.drop_last is False
    assert isinstance(loader.sampler, SequentialSampler)
    assert set(loader.collate_fn.exclude_keys) == {"positions"}
    batch = next(iter(loader))
    assert "energy" in batch
    assert "energy_weight" in batch
    assert "positions" not in batch
    assert sum(batch.num_graphs for batch in loader) == len(dataset)


def test_statistics_does_not_move_the_complete_batch(monkeypatch):
    monkeypatch.setattr(statistics_module, "log_statistics_to_yaml", lambda _: None)

    class Batch(dict):
        ptr = torch.tensor([0, 2])
        edge_index = torch.tensor([[0, 1], [1, 0]])

        def to(self, *args, **kwargs):
            raise AssertionError("the complete batch must not be moved")

    batch = Batch(
        batch=torch.tensor([0, 0]),
        atomic_numbers=torch.tensor([1, 1]),
        fidelity_idx=torch.tensor([0]),
        unused=torch.ones(1_000),
    )

    stats = statistics_module.statistics_from_loader(
        [batch],
        atomic_numbers=[1],
        atomic_energies=None,
        target_property=[],
    )[0]

    assert stats["num_atoms"] == 2
    assert stats["avg_num_neighbors"] == 1.0


def test_compute_atomic_energies_handles_all_fidelities_in_one_call():
    atoms_list = [
        Atoms(numbers=[1, 1], info={"energy": 2.0, "fidelity_idx": 0}),
        Atoms(numbers=[2], info={"energy": 3.0, "fidelity_idx": 0}),
        Atoms(numbers=[1], info={"energy": 4.0, "fidelity_idx": 1}),
        Atoms(numbers=[2, 2], info={"energy": 10.0, "fidelity_idx": 1}),
    ]

    energies = statistics_module.compute_atomic_energies(
        atoms_list,
        build_element_lookup([1, 2]),
        KeySpecification.from_defaults(),
        [0, 1, 2],
    )

    assert energies[0] == pytest.approx({1: 1.0, 2: 3.0})
    assert energies[1] == pytest.approx({1: 4.0, 2: 5.0})
    assert energies[2] == {1: 0.0, 2: 0.0}


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize("loader_type", [DataLoader, StatisticsDataLoader])
def test_u_shift_preserves_raw_statistics_fit_and_filter(
    tmp_path, monkeypatch, compressed, loader_type
):
    numbers = [[26], [8], [26, 8], [26, 26, 8], [28, 9, 9], [26, 8]]
    raw = [-2.0, -1.0, -2.0, -3.0, -3.0, 999.0]
    atoms_list, graphs = [], []
    for index, (zs, energy) in enumerate(zip(numbers, raw)):
        atoms = Atoms(numbers=zs, positions=np.arange(len(zs) * 3).reshape(-1, 3))
        atoms.info.update(energy=energy, energy_weight=float(index != 5))
        atoms.arrays["forces"] = np.arange(len(zs) * 3).reshape(-1, 3).astype(float)
        atoms_list.append(atoms)
        graphs.append(
            from_atoms(
                None,
                atoms,
                cutoff=2.0,
                max_neighbors=None,
                keyspec=KeySpecification.from_defaults(),
                target_property=["energy", "forces"],
                embedding_property=[],
                neighborlist_backend="ase",
            )
        )
    dataset = write_graphs(
        graphs, [tmp_path], "train", compressed=compressed, shard_size=2
    )
    before = _snapshot(tmp_path)
    element = build_element_lookup([8, 9, 26, 28])
    counts = np.array([[zs.count(z) for z in element.zs] for zs in numbers])
    expected_fit = dict(
        zip(element.zs, np.linalg.lstsq(counts[:5], raw[:5], rcond=None)[0])
    )
    monkeypatch.setenv("TACE_USE_U_SHIFT", "1")
    loader = loader_type(dataset, batch_size=2, shuffle=False)
    fits = [
        statistics_module.compute_atomic_energies(
            atoms_list, element, KeySpecification.from_defaults(), [0]
        ),
        statistics_module.compute_atomic_energies_from_graphs(loader, element, [0]),
    ]
    incremental = statistics_module.AtomicEnergyFit(element.zs, [0])
    for index in range(5):
        incremental.update(0, counts[index : index + 1], raw[index : index + 1])
    fits.append(incremental.result())
    for fit in fits:
        assert fit[0] == pytest.approx(expected_fit)
    options = dict(
        atomic_numbers=element.zs,
        atomic_energies=[dict.fromkeys(element.zs, 0.0)],
        target_property=["energy", "forces"],
    )
    accumulator = statistics_module.StatisticsAccumulator(**options)
    for batch in loader:
        accumulator.update(batch)
    actual = accumulator.result()[0]
    selected = filter_graphs(dataset, {"max_energy_per_atom": -1.5})
    assert selected.indices.tolist() == [0]
    assert float(selected[0].energy) == raw[0]
    selected.close()
    monkeypatch.setenv("TACE_USE_U_SHIFT", "0")
    selected = filter_graphs(dataset, {"max_energy_per_atom": -1.5})
    assert selected.indices.tolist() == [0]
    selected.close()
    expected = _accumulator(graphs, options, batch_size=2).result()[0]
    _assert_statistics_equal(actual, expected)
    assert [float(graph.energy) for graph in dataset] == raw
    assert [atoms.info["energy"] for atoms in atoms_list] == raw
    dataset.close()
    assert _snapshot(tmp_path) == before


def test_energy_statistics_use_canonical_names(monkeypatch):
    monkeypatch.setattr(statistics_module, "log_statistics_to_yaml", lambda _: None)

    graphs = [
        Data(
            atomic_numbers=torch.tensor([1, 1]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=2,
            energy=torch.tensor([4.0]),
            energy_weight=torch.tensor([1.0]),
            fidelity_idx=torch.tensor([0]),
        ),
        Data(
            atomic_numbers=torch.tensor([2]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            energy=torch.tensor([8.0]),
            energy_weight=torch.tensor([1.0]),
            fidelity_idx=torch.tensor([0]),
        ),
    ]

    stats = statistics_module.statistics_from_loader(
        DataLoader(graphs, batch_size=2),
        atomic_numbers=[1, 2],
        atomic_energies=[{1: 1.0, 2: 2.0}],
        target_property=["energy"],
    )[0]

    assert stats["mean_energy"] == 6.0
    assert stats["std_energy"] == pytest.approx(math.sqrt(8.0), rel=1e-14, abs=0.0)
    assert stats["atomic_energy"] == {1: 1.0, 2: 2.0}
    assert stats["mean_energy_per_atom"] == {1: 5.0, 2: 5.0}
    assert stats["mean_delta_energy_per_atom"] == {1: 3.5, 2: 3.5}

    scale_shift = ScaleShift.build_from_config(
        [stats],
        {
            "scale_type": None,
            "shift_type": "mean_delta_energy_per_atom",
            "scale_trainable": False,
            "shift_trainable": False,
            "all_atoms": True,
        },
        atomic_numbers=[1, 2],
    )
    torch.testing.assert_close(
        scale_shift.shift, torch.tensor([[3.5, 3.5]], dtype=torch.float64)
    )


def test_forces_statistics_use_canonical_names(monkeypatch):
    monkeypatch.setattr(statistics_module, "log_statistics_to_yaml", lambda _: None)

    graphs = [
        Data(
            atomic_numbers=torch.tensor([1, 1]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=2,
            forces=torch.tensor([[1.0, 2.0, 2.0], [-1.0, 0.0, 0.0]]),
            forces_weight=torch.tensor([1.0]),
            fidelity_idx=torch.tensor([0]),
        ),
        Data(
            atomic_numbers=torch.tensor([2]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            forces=torch.tensor([[0.0, 3.0, 4.0]]),
            forces_weight=torch.tensor([1.0]),
            fidelity_idx=torch.tensor([0]),
        ),
    ]

    stats = statistics_module.statistics_from_loader(
        DataLoader(graphs, batch_size=2),
        atomic_numbers=[1, 2],
        atomic_energies=None,
        target_property=["forces"],
    )[0]

    torch.testing.assert_close(
        torch.tensor(stats["mean_forces_xyz"]),
        torch.tensor([0.0, 5.0 / 3.0, 2.0]),
    )
    torch.testing.assert_close(
        torch.tensor(stats["std_forces_xyz"]),
        torch.tensor(
            [math.sqrt(2.0 / 3.0), math.sqrt(14.0) / 3.0, math.sqrt(8.0 / 3.0)]
        ),
    )
    torch.testing.assert_close(
        torch.tensor(stats["rms_forces_xyz"]),
        torch.tensor(
            [math.sqrt(2.0 / 3.0), math.sqrt(13.0 / 3.0), math.sqrt(20.0 / 3.0)]
        ),
    )
    assert stats["mean_forces_norm"] == 3.0
    assert stats["std_forces_norm"] == pytest.approx(
        math.sqrt(8.0 / 3.0), rel=1e-14, abs=0.0
    )
    assert stats["mean_forces_xyz_by_element"] == {
        1: [0.0, 1.0, 1.0],
        2: [0.0, 3.0, 4.0],
    }
    assert stats["std_forces_xyz_by_element"] == {
        1: [1.0, 1.0, 1.0],
        2: [0.0, 0.0, 0.0],
    }
    assert stats["mean_forces_norm_by_element"] == {1: 2.0, 2: 5.0}
    assert stats["std_forces_norm_by_element"] == {1: 1.0, 2: 0.0}
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_forces"].values())),
        torch.full((2,), math.sqrt(35.0 / 9.0)),
    )
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_forces_by_element"].values())),
        torch.tensor([math.sqrt(5.0 / 3.0), math.sqrt(25.0 / 3.0)]),
    )
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_forces_norm"].values())),
        torch.full((2,), math.sqrt(35.0 / 3.0)),
    )
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_forces_norm_by_element"].values())),
        torch.tensor([math.sqrt(5.0), 5.0]),
    )
    scale_shift = ScaleShift.build_from_config(
        [stats],
        {
            "scale_type": "rms_forces",
            "shift_type": None,
            "scale_trainable": False,
            "shift_trainable": False,
            "all_atoms": True,
        },
        atomic_numbers=[1, 2],
    )
    torch.testing.assert_close(
        scale_shift.scale,
        torch.full((1, 2), math.sqrt(35.0 / 9.0), dtype=torch.float64),
    )
    assert stats["num_forces_by_element"] == {1: 2, 2: 1}
    torch.testing.assert_close(
        torch.tensor(list(stats["mean_squared_forces_xyz_by_element"].values())),
        torch.tensor([5.0 / 3.0, 25.0 / 3.0]),
    )


def test_noncollinear_magmoms_statistics_use_canonical_names(monkeypatch):
    monkeypatch.setattr(statistics_module, "log_statistics_to_yaml", lambda _: None)

    graphs = [
        Data(
            atomic_numbers=torch.tensor([1, 1]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=2,
            initial_noncollinear_magmoms=torch.tensor(
                [[1.0, 2.0, 2.0], [-1.0, 0.0, 0.0]]
            ),
            fidelity_idx=torch.tensor([0]),
        ),
        Data(
            atomic_numbers=torch.tensor([2]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            initial_noncollinear_magmoms=torch.tensor([[0.0, 3.0, 4.0]]),
            fidelity_idx=torch.tensor([0]),
        ),
    ]

    stats = statistics_module.statistics_from_loader(
        DataLoader(graphs, batch_size=2),
        atomic_numbers=[1, 2],
        atomic_energies=None,
        target_property=[],
    )[0]

    torch.testing.assert_close(
        torch.tensor(stats["mean_noncollinear_magmoms_xyz"]),
        torch.tensor([0.0, 5.0 / 3.0, 2.0]),
    )
    torch.testing.assert_close(
        torch.tensor(stats["std_noncollinear_magmoms_xyz"]),
        torch.tensor(
            [math.sqrt(2.0 / 3.0), math.sqrt(14.0) / 3.0, math.sqrt(8.0 / 3.0)]
        ),
    )
    torch.testing.assert_close(
        torch.tensor(stats["rms_noncollinear_magmoms_xyz"]),
        torch.tensor(
            [math.sqrt(2.0 / 3.0), math.sqrt(13.0 / 3.0), math.sqrt(20.0 / 3.0)]
        ),
    )
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_noncollinear_magmoms"].values())),
        torch.full((2,), math.sqrt(35.0) / 3.0),
    )
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_noncollinear_magmoms_by_element"].values())),
        torch.tensor([math.sqrt(5.0 / 3.0), 5.0 / math.sqrt(3.0)]),
    )
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_noncollinear_magmoms_norm"].values())),
        torch.full((2,), math.sqrt(35.0 / 3.0)),
    )
    assert stats["mean_noncollinear_magmoms_norm"] == 3.0
    assert stats["std_noncollinear_magmoms_norm"] == pytest.approx(
        math.sqrt(8.0 / 3.0), rel=1e-14, abs=0.0
    )
    assert stats["max_noncollinear_magmoms_norm"] == 5.0
    assert stats["mean_noncollinear_magmoms_xyz_by_element"] == {
        1: [0.0, 1.0, 1.0],
        2: [0.0, 3.0, 4.0],
    }
    assert stats["std_noncollinear_magmoms_xyz_by_element"] == {
        1: [1.0, 1.0, 1.0],
        2: [0.0, 0.0, 0.0],
    }
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_noncollinear_magmoms_norm_by_element"].values())),
        torch.tensor([math.sqrt(5.0), 5.0]),
    )
    component_rms = torch.tensor(
        list(stats["rms_noncollinear_magmoms_by_element"].values())
    )
    norm_rms = torch.tensor(
        list(stats["rms_noncollinear_magmoms_norm_by_element"].values())
    )
    torch.testing.assert_close(norm_rms, math.sqrt(3.0) * component_rms)
    global_component_rms = torch.tensor(
        list(stats["rms_noncollinear_magmoms"].values())
    )
    global_norm_rms = torch.tensor(
        list(stats["rms_noncollinear_magmoms_norm"].values())
    )
    torch.testing.assert_close(
        global_norm_rms,
        math.sqrt(3.0) * global_component_rms,
    )
    assert stats["max_noncollinear_magmoms_norm_by_element"] == {
        1: 3.0,
        2: 5.0,
    }
    assert stats["num_noncollinear_magmoms_by_element"] == {1: 2, 2: 1}


def test_noncollinear_magmoms_statistics_are_per_fidelity(monkeypatch):
    monkeypatch.setattr(statistics_module, "log_statistics_to_yaml", lambda _: None)

    graphs = [
        Data(
            atomic_numbers=torch.tensor([26]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            initial_noncollinear_magmoms=torch.tensor([[1.0, 0.0, 0.0]]),
            fidelity_idx=torch.tensor([0]),
        ),
        Data(
            atomic_numbers=torch.tensor([26]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            initial_noncollinear_magmoms=torch.tensor([[5.0, 0.0, 0.0]]),
            fidelity_idx=torch.tensor([1]),
        ),
    ]

    stats = statistics_module.statistics_from_loader(
        DataLoader(graphs, batch_size=2),
        atomic_numbers=[26],
        atomic_energies=None,
        target_property=[],
        num_fidelities=2,
    )

    assert stats[0]["max_noncollinear_magmoms_norm_by_element"] == {26: 1.0}
    assert stats[1]["max_noncollinear_magmoms_norm_by_element"] == {26: 5.0}


def _assert_statistics_equal(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_statistics_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for value, reference in zip(actual, expected):
            _assert_statistics_equal(value, reference)
    elif isinstance(expected, float):
        assert np.isfinite(actual)
        assert actual == pytest.approx(expected, rel=2e-12, abs=2e-12)
    else:
        assert actual == expected


@pytest.fixture
def statistics_graphs():
    generator = torch.Generator().manual_seed(71)
    graphs = []
    for index in range(11):
        count = 1 + index % 4
        energy = torch.randn(1, generator=generator, dtype=torch.float64) + 12
        forces = torch.randn(count, 3, generator=generator, dtype=torch.float64)
        valid_energy = index % 5 != 0
        valid_forces = index % 4 != 0
        if not valid_energy:
            energy.fill_(float("nan"))
        if not valid_forces:
            forces.fill_(float("nan"))
        magmoms = torch.randn(count, 3, generator=generator, dtype=torch.float64)
        if index == 9:
            magmoms[0] = torch.tensor([30.0, 40.0, 0.0], dtype=torch.float64)
        graphs.append(
            Data(
                num_nodes=count,
                atomic_numbers=torch.tensor(([1, 8] * count)[:count]),
                num_neighbors=torch.arange(count) + index % 3,
                edge_index=torch.empty((2, 0), dtype=torch.int64),
                positions=torch.zeros(count, 3, dtype=torch.float64),
                energy=energy,
                energy_weight=torch.tensor([float(valid_energy)], dtype=torch.float64),
                forces=forces,
                forces_weight=torch.tensor([float(valid_forces)], dtype=torch.float64),
                fidelity_idx=torch.tensor([index % 2]),
                initial_noncollinear_magmoms=magmoms,
            )
        )
    return graphs


@pytest.fixture
def statistics_options():
    return {
        "atomic_numbers": [1, 6, 8],
        "atomic_energies": [{1: 1.5, 8: 2.5}, {1: 3.0, 8: 5.0}, {}],
        "target_property": ["energy", "forces"],
        "num_fidelities": 3,
    }


def _accumulator(graphs, options, batch_size=3):
    accumulator = statistics_module.StatisticsAccumulator(**options)
    for batch in DataLoader(graphs, batch_size=batch_size):
        accumulator.update(batch)
    return accumulator


@pytest.mark.parametrize("mode", ["loader", "iterable", "callback"])
@pytest.mark.parametrize("failure", [False, True])
def test_statistics_progress_counts_only_successful_batches(
    monkeypatch,
    capsys,
    statistics_progress,
    statistics_graphs,
    statistics_options,
    mode,
    failure,
):
    loader = DataLoader(statistics_graphs, batch_size=4)
    if mode == "iterable":
        loader = list(loader)
    accumulator = statistics_module.StatisticsAccumulator(**statistics_options)
    update = accumulator.update
    now = [0.0]
    batches = 0

    def consume(data):
        nonlocal batches
        if failure and batches:
            raise RuntimeError("injected statistics failure")
        update(data)
        batches += 1
        now[0] += 0.6

    monkeypatch.setattr(accumulator, "update", consume)
    monkeypatch.setattr(
        statistics_module, "time", SimpleNamespace(monotonic=lambda: now[0])
    )
    reported = []
    callback = reported.append if mode == "callback" else None
    if failure:
        with pytest.raises(RuntimeError, match="injected statistics failure"):
            statistics_module.accumulate_statistics(loader, accumulator, callback)
    else:
        assert statistics_module.accumulate_statistics(
            loader, accumulator, callback
        ) == len(statistics_graphs)
    output = capsys.readouterr().err
    if mode == "callback":
        # Workers report at most once a second, then flush the actual tail even
        # after a failure. Only the coordinator owns a terminal progress bar.
        assert reported == ([4] if failure else [8, 11])
        assert not statistics_progress
        assert "statistics:" not in output
    else:
        assert len(statistics_progress) == 1
        progress = statistics_progress[0]
        assert progress.total == (None if mode == "iterable" else 11)
        assert progress.n == (4 if failure else 11)
        assert progress.disable
        assert "statistics:" in output


@pytest.mark.parametrize("force_key", ["forces", "direct_forces"])
def test_statistics_merge_preserves_masks_fidelities_and_element_table(
    statistics_graphs, statistics_options, force_key
):
    graphs = statistics_graphs
    options = statistics_options
    if force_key == "direct_forces":
        options["target_property"] = ["energy", force_key]
        for graph in graphs:
            graph[force_key] = graph.pop("forces")
            graph[f"{force_key}_weight"] = graph.pop("forces_weight")

    merged = statistics_module.StatisticsAccumulator(**options)
    # Unequal partitions, including a neutral empty state, require weighted
    # moment merging; averaging finalized standard deviations is incorrect.
    for partition in ([], graphs[:1], graphs[1:7], graphs[7:]):
        merged.merge(_accumulator(partition, options, batch_size=2).state())
    result = merged.result("test-fingerprint")
    _assert_statistics_equal(
        result, _accumulator(graphs, options).result("test-fingerprint")
    )

    neighbors = torch.cat([graph.num_neighbors for graph in graphs]).double()
    for fidelity in (0, 1):
        local = [graph for graph in graphs if graph.fidelity_idx.item() == fidelity]
        valid_energy = [graph for graph in local if graph.energy_weight.item() > 0]
        energies = torch.cat([graph.energy for graph in valid_energy])
        per_atom = torch.tensor(
            [graph.energy.item() / graph.num_nodes for graph in valid_energy],
            dtype=torch.float64,
        )
        valid_forces = [
            graph for graph in local if graph[f"{force_key}_weight"].item() > 0
        ]
        forces = torch.cat([graph[force_key] for graph in valid_forces])
        entry = result[fidelity]
        assert entry["num_graphs"] == len(local)
        assert entry["num_atoms"] == sum(graph.num_nodes for graph in local)
        assert entry["avg_num_neighbors"] == pytest.approx(neighbors.mean().item())
        assert entry["mean_energy"] == pytest.approx(energies.mean().item())
        assert entry["std_energy"] == pytest.approx(energies.std().item())
        assert entry["mean_energy_per_atom"][6] == pytest.approx(per_atom.mean().item())
        assert entry["std_forces_xyz"] == pytest.approx(
            forces.std(dim=0, correction=0).tolist()
        )
        assert entry["rms_forces_xyz"] == pytest.approx(
            forces.square().mean(dim=0).sqrt().tolist()
        )
        for number in options["atomic_numbers"]:
            assert entry["num_forces_by_element"][number] == sum(
                int((graph.atomic_numbers == number).sum()) for graph in valid_forces
            )
            norms = [
                graph.initial_noncollinear_magmoms[graph.atomic_numbers == number].norm(
                    dim=-1
                )
                for graph in local
            ]
            norms = torch.cat(norms)
            expected_max = norms.max().item() if norms.numel() else 0.0
            assert entry["max_noncollinear_magmoms_norm_by_element"][number] == (
                pytest.approx(expected_max)
            )
        assert entry["atomic_numbers"] == [1, 6, 8]
        assert entry["data_dtype"] == "float64"
        assert entry["data_fingerprint"] == "test-fingerprint"

    assert result[1]["max_noncollinear_magmoms_norm_by_element"][1] == 50.0
    assert result[2]["available"] is False
    assert result[2]["num_graphs"] == result[2]["num_atoms"] == 0
    assert result[2]["atomic_energy"] == {1: 0.0, 6: 0.0, 8: 0.0}
    assert result[2]["rms_forces_by_element"] == {1: 1.0, 6: 1.0, 8: 1.0}


def test_statistics_merge_retains_small_variance_on_large_energy_offset():
    offset = float(2**40)
    options = {
        "atomic_numbers": [1, 8],
        "atomic_energies": [{1: offset}],
        "target_property": ["energy"],
    }
    graphs = [
        Data(
            num_nodes=1,
            atomic_numbers=torch.tensor([1]),
            num_neighbors=torch.tensor([0]),
            energy=torch.tensor([offset + value], dtype=torch.float64),
        )
        for value in (-6.0, 0.0, 2.0, 4.0)
    ]
    merged = statistics_module.StatisticsAccumulator(**options)
    for partition in (graphs[:1], graphs[1:]):
        merged.merge(_accumulator(partition, options).state())
    entry = merged.result()[0]
    assert entry["mean_energy"] == offset
    assert entry["std_energy"] == pytest.approx(np.sqrt(56.0 / 3.0), rel=1e-14)
    assert entry["mean_delta_energy_per_atom"] == {1: 0.0, 8: 0.0}


def _snapshot(directory):
    return {
        str(path.relative_to(directory)): (
            path.stat().st_size,
            path.stat().st_mtime_ns,
            _file_digest(path),
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


@pytest.fixture
def statistics_store(tmp_path, statistics_graphs):
    dataset = write_graphs(
        iter(statistics_graphs), [tmp_path / "cache"], "train", shard_size=3
    )
    projected = dataset.project(
        [
            "atomic_numbers",
            "num_neighbors",
            "energy",
            "energy_weight",
            "forces",
            "forces_weight",
            "fidelity_idx",
            "initial_noncollinear_magmoms",
        ]
    )
    yield projected
    projected.close()
    dataset.close()


@pytest.mark.parametrize("num_workers", [0, 1])
@pytest.mark.parametrize("filtered", [False, True])
def test_spawned_statistics_read_all_shards_once_without_changing_cache(
    tmp_path,
    statistics_progress,
    statistics_store,
    statistics_graphs,
    statistics_options,
    num_workers,
    filtered,
):
    if filtered:
        statistics_store = statistics_store.select([0, 1, 3, 4, 9, 10])
        statistics_graphs = [statistics_graphs[i] for i in [0, 1, 3, 4, 9, 10]]
    loader_options = {"batch_size": 2, "num_workers": num_workers}
    if num_workers:
        loader_options["prefetch_factor"] = 1
    loader = StatisticsDataLoader(statistics_store, **loader_options)
    before = _snapshot(tmp_path)
    children_before = {process.pid for process in multiprocessing.active_children()}
    try:
        tasks = statistics_module.statistics_tasks(
            loader, **statistics_options, devices=[torch.device("cpu")] * 2
        )
        assigned_paths = [path for task in tasks for path in task["paths"]]
        expected_paths = [
            path
            for i, path in enumerate(statistics_store.lmdb_paths)
            if not filtered or i != 2
        ]
        assert Counter(assigned_paths) == Counter(map(str, expected_paths))
        assert sum(
            len(task["indices"])
            if task["indices"] is not None
            else sum(task["lengths"])
            for task in tasks
        ) == len(statistics_graphs)
        assert all(task["paths"] for task in tasks)
        assert all(task["loader"]["num_workers"] == num_workers for task in tasks)
        accumulator = statistics_module.StatisticsAccumulator(**statistics_options)
        statistics_module.parallel_statistics(tasks, accumulator)
        _assert_statistics_equal(
            accumulator.result(),
            _accumulator(statistics_graphs, statistics_options).result(),
        )
    finally:
        loader.dataset.close()
    statistics_store.close()
    assert _snapshot(tmp_path) == before
    assert {process.pid for process in multiprocessing.active_children()} <= (
        children_before
    )
    assert len(statistics_progress) == 1
    progress = statistics_progress[0]
    assert progress.total == len(statistics_graphs)
    assert progress.n == len(statistics_graphs)
    assert progress.disable


def test_two_gpu_statistics_match_cpu_and_only_parent_writes_yaml(
    tmp_path,
    monkeypatch,
    statistics_progress,
    statistics_store,
    statistics_graphs,
    statistics_options,
):
    if torch.cuda.device_count() < 2:
        pytest.skip("Two visible CUDA devices are required")
    monkeypatch.chdir(tmp_path)
    writes = []
    monkeypatch.setattr(statistics_module, "log_statistics_to_yaml", writes.append)
    before = _snapshot(tmp_path)
    loader = StatisticsDataLoader(statistics_store, batch_size=2, num_workers=0)
    try:
        result = statistics_module.statistics_from_loader(
            loader, **statistics_options, devices=["cuda:0", "cuda:1"]
        )
    finally:
        loader.dataset.close()
    _assert_statistics_equal(
        result, _accumulator(statistics_graphs, statistics_options).result()
    )
    assert writes == [result]
    assert _snapshot(tmp_path) == before
    assert len(statistics_progress) == 1
    progress = statistics_progress[0]
    assert progress.total == len(statistics_graphs)
    assert progress.n == len(statistics_graphs)
    assert progress.disable


def _fail_or_wait(index, task, messages):
    if index == 0:
        messages.put(("progress", index, 1))
        if task["failure"] == "exit":
            messages.close()
            messages.join_thread()
            os._exit(7)
        messages.put(("error", index, "deliberate statistics worker failure"))
    # The coordinator must terminate a remaining worker after its peer fails.
    time.sleep(30)


@pytest.mark.parametrize("failure", ["error", "exit"])
def test_parallel_statistics_reports_failure_and_reaps_its_workers(
    monkeypatch, statistics_progress, failure, statistics_options
):
    context = multiprocessing.get_context("spawn")
    processes = []

    def record_process(**kwargs):
        process = context.Process(**kwargs)
        processes.append(process)
        return process

    recorded_context = SimpleNamespace(Queue=context.Queue, Process=record_process)
    monkeypatch.setattr(
        statistics_module,
        "multiprocessing",
        SimpleNamespace(get_context=lambda _: recorded_context),
    )
    monkeypatch.setattr(statistics_module, "statistics_process", _fail_or_wait)
    tasks = [
        {
            "device": "cpu",
            "lengths": [2],
            "loader": {"batch_size": 1},
            "failure": failure,
        }
        for _ in range(2)
    ]
    expected = (
        "deliberate statistics worker failure" if failure == "error" else "code 7"
    )
    with pytest.raises(RuntimeError, match=expected):
        statistics_module.parallel_statistics(
            tasks, statistics_module.StatisticsAccumulator(**statistics_options)
        )
    assert len(processes) == 2
    assert all(not process.is_alive() for process in processes)
    assert all(process.exitcode is not None for process in processes)
    assert processes[1].exitcode != 0
    assert len(statistics_progress) == 1
    progress = statistics_progress[0]
    assert progress.total == 4
    assert progress.n == 1
    assert progress.disable


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


def file_snapshot(directory):
    return {
        str(path.relative_to(directory)): (
            path.stat().st_ino,
            path.stat().st_mtime_ns,
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
        for path in directory.rglob("*")
        if path.is_file()
    }


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
    assert not torch.cuda.is_initialized()


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

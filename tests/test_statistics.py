import hashlib
import json
import math
import multiprocessing
import os
import pickle

import lmdb
import numpy as np
import pytest
import torch
from ase import Atoms
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import tace.dataset.statistics as statistics_module
from tace.dataset.cache import (
    GRAPH_SCHEMA,
    graph_cache_directories,
    read_cache_metadata,
    source_cache_key,
    statistics_cache_key,
)
from tace.dataset.dataloader import build_statistics_dataloader
from tace.dataset.datamodule import GraphDatasetLMDB, create_graphs
from tace.dataset.element import build_element_lookup
from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KeySpecification
from tace.dataset.store import (
    build_graph_store,
    decode_graph,
    encode_graph,
    write_graphs,
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


@pytest.mark.parametrize("mode", ["memory", "lmdb"])
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

    if mode == "lmdb":

        def no_graph_work(*args, **kwargs):
            pytest.fail("Warm cache must not construct graphs or progress reporters")

        monkeypatch.setattr(module, "iter_graphs", no_graph_work)
        monkeypatch.setattr(module, "GraphProgress", no_graph_work)
        reused = module.create_graphs(None, None, options, "train", [tmp_path])
        assert len(reused) == 5
        assert not capsys.readouterr().err
        reused.close()


@pytest.mark.parametrize("mode", ["memory", "lmdb"])
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
            "type": "ase_db",
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


@pytest.mark.parametrize("mode", ["memory", "lmdb", "on_the_fly"])
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
            "type": "ase_db",
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
            "options": {"serial": True, "use_lock_file": False},
        },
        ["energy", "forces"],
        _reader_keys(),
        [],
        backend=reader,
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
    cfg = {
        "dataset": {
            "train_file": str(source_dir),
            "format": "cif",
            "options": {"store_tags": True},
        }
    }
    key = source_cache_key(cfg)
    source = read_all_files(
        source_dir,
        [],
        _reader_keys(),
        [],
        format="cif",
        options={"store_tags": True},
        record_cache_dir=tmp_path / "cache",
    )
    assert len(source) == 2
    assert source.get_record(0).info["spacegroup"].no == 1

    def no_decode(*args, **kwargs):
        raise AssertionError("Warm raw cache must not decode the original file")

    monkeypatch.setattr("ase.io.iread", no_decode)
    warm = read_all_files(
        source_dir,
        [],
        _reader_keys(),
        [],
        format="cif",
        options={"store_tags": True},
        record_cache_dir=tmp_path / "cache",
    )
    assert len(warm.get_records([1, 0, 1])) == 3
    cfg["dataset"]["options"]["store_tags"] = False
    assert source_cache_key(cfg) != key
    cfg["dataset"]["options"]["store_tags"] = True
    filename.write_text(filename.read_text() + "\n# changed\n")
    assert source_cache_key(cfg) != key
    cfg["dataset"]["train_file"] = {
        "path": str(filename),
        "reader": "ase",
        "format": "cif",
        "options": {"store_tags": False},
    }
    mapped_key = source_cache_key(cfg)
    cfg["dataset"]["train_file"]["options"]["store_tags"] = True
    assert source_cache_key(cfg) != mapped_key


def test_storage_modes_cold_statistics_and_warm_training(tmp_path, monkeypatch):
    from tace.dataset.dataloader import build_atomsList, compute_statistics
    from tace.dataset.datamodule import build_datamodule, load_cached_datamodule

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
            "type": "fair_aselmdb",
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
    for mode in ("memory", "lmdb", "on_the_fly"):
        cfg["dataset"]["storage_mode"] = mode
        module = build_datamodule(cfg, None, ["energy", "forces"], [], keys)
        module.prepare_data()
        module.setup("fit")
        element, raw, references = build_atomsList(
            cfg, ["energy", "forces"], [], keys, datamodule=module
        )
        assert element.zs == [1, 8]
        assert references[0] == pytest.approx({1: -1.0, 8: -2.0}, abs=1e-12)
        statistics = compute_statistics(
            cfg,
            ["energy", "forces"],
            [],
            keys,
            element,
            raw,
            cfg["model"]["config"]["fidelity"],
            references,
            dataloader_train=module.statistics_dataloader(),
        )
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

    cfg["dataset"]["storage_mode"] = "lmdb"

    def no_source(*args, **kwargs):
        raise AssertionError("Warm training must not open sources or construct graphs")

    monkeypatch.setattr("tace.dataset.datamodule.tace_read_all_files", no_source)
    monkeypatch.setattr("tace.dataset.datamodule.from_records", no_source)
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
        filename, ["energy", "forces"], keys, [], backend="fair_aselmdb"
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
    from tace.dataset.dataloader import build_atomsList, compute_statistics
    from tace.dataset.datamodule import build_datamodule, load_cached_datamodule

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
            "type": "ase_db",
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
    element, raw, references = build_atomsList(
        cfg, ["energy", "forces"], [], keyspec, datamodule=warm
    )
    statistics = compute_statistics(
        cfg,
        ["energy", "forces"],
        [],
        keyspec,
        element,
        raw,
        cfg["model"]["config"]["fidelity"],
        references,
        dataloader_train=warm.statistics_dataloader(),
    )
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
        format="extxyz",
        options={"index": ":"},
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


def test_statistics_fingerprint_tracks_added_shards_but_not_moves(tmp_path):
    original_dir, moved_dir = tmp_path / "original", tmp_path / "moved"
    cfg, keyspec, first = _prebuilt_graph_store(original_dir, [1.0])
    key = statistics_cache_key(cfg, ["energy"], [], keyspec)
    moved_dir.mkdir()
    for path in first.lmdb_paths:
        path.rename(moved_dir / path.name)
    cfg["dataset"].update(
        shard_dirs=[str(moved_dir)],
        train_file="different-unavailable-source.xyz",
        valid_ratio=0.4,
        split_seed=9,
    )
    keyspec.info_keys["energy"] = "different_raw_label"
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


def test_typed_graph_store_batched_projection_and_size_index(tmp_path):
    graphs = [
        Data(
            num_nodes=2,
            atomic_numbers=torch.tensor([1, 8]),
            positions=torch.full((2, 3), float(index), dtype=torch.float64),
            edge_index=torch.tensor([[0, 1], [1, 0]]),
            edge_shifts=torch.zeros(2, 3, dtype=torch.int64),
            energy=torch.tensor([float(index)], dtype=torch.float64),
            fidelity_idx=torch.tensor(0),
        )
        for index in range(7)
    ]
    for graph in graphs:
        decoded = decode_graph(encode_graph(graph))
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
    )
    assert len(dataset.lmdb_paths) == 4
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
        transaction_bytes=2 * len(encode_graph(graph)),
        progress=written.append,
    )
    assert written == [2, 2, 1]
    assert sum(written) == len(dataset)
    dataset.close()
    written.clear()
    reused = write_graphs(iter(()), [tmp_path], "train", progress=written.append)
    assert len(reused) == 5 and not written
    reused.close()


def test_store_progress_does_not_double_count_map_full_retries(tmp_path, monkeypatch):
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
        payload=torch.zeros(300_000, dtype=torch.float64),
    )
    written = []
    dataset = write_graphs(
        [graph],
        [tmp_path],
        "train",
        map_size=1 << 20,
        progress=written.append,
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


def test_raw_and_statistics_cache_dependencies_are_separate(tmp_path):
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
            reference = statistics_module._compute_statistics(
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
    monkeypatch.setattr(train_module, "build_atomsList", no_source_read)
    monkeypatch.setattr(train_module, "compute_statistics", no_source_read)
    _, reused, _, _, _, datamodule = train_module.build(cfg)
    assert reused == statistics
    datamodule.prepare_data()
    assert datamodule.threeAtomsList is None
    assert all(p.read_bytes() == value for p, value in contents.items())


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
                "epoch_size": 1,
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

    stats = statistics_module._compute_statistics(
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

    stats = statistics_module._compute_statistics(
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

    stats = statistics_module._compute_statistics(
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

    stats = statistics_module._compute_statistics(
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

    stats = statistics_module._compute_statistics(
        DataLoader(graphs, batch_size=2),
        atomic_numbers=[26],
        atomic_energies=None,
        target_property=[],
        num_fidelities=2,
    )

    assert stats[0]["max_noncollinear_magmoms_norm_by_element"] == {26: 1.0}
    assert stats[1]["max_noncollinear_magmoms_norm_by_element"] == {26: 5.0}

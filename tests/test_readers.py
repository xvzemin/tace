"""Atomic structure readers, data splits and source lifetimes."""

import math
import multiprocessing
import pickle
import sqlite3
from pathlib import Path

import lmdb
import numpy as np
import pytest
import torch
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write
from omegaconf import OmegaConf
from torch_geometric.loader import DataLoader

from tace.dataset.cache import (
    ARRAY_CODEC,
    COMPRESSED_ARRAY_CODEC,
    LMDB_MODES,
    source_cache_key,
)
from tace.dataset.datamodule import build_datamodule, create_graphs
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
from tace.dataset.store import write_records

from .testing import file_digest, reader_keys, write_reader_database


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


def test_no_validation_takes_precedence_and_index_splits_do_not_overlap(
    tmp_path, monkeypatch
):
    from tace.dataset.read import tace_read_all_files

    monkeypatch.chdir(tmp_path)
    filename = tmp_path / "structures.db"
    write_reader_database(filename)
    cfg = {
        "model": {"config": {"cutoff": 2.0}},
        "dataset": {
            "train_file": str(filename),
            "reader": "ase_db",
            "no_valid_set": True,
        },
    }
    keys = reader_keys()
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
    write_reader_database(filename)
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
    module = build_datamodule(cfg, None, ["energy"], [], reader_keys())
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
    test_only = build_datamodule(cfg, None, ["energy"], [], reader_keys())
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
    write_reader_database(path)
    before = file_digest(path)

    def no_atoms(*args, **kwargs):
        raise AssertionError("Record reads must not materialize ASE Atoms")

    monkeypatch.setattr(AtomsRow, "toatoms", no_atoms)
    source = read_all_files(
        {
            "path": path,
            "fidelity_idx": 2,
        },
        ["energy", "forces"],
        reader_keys(),
        [],
        reader=reader,
    )
    assert not isinstance(source, list)
    assert len(source) == 4
    records = source.get_records([3, 0, 3])
    assert [len(record) for record in records] == [3, 2, 3]
    assert [record.info["fidelity_idx"] for record in records] == [2, 2, 2]
    assert records[0].data["reference_energy"] == -4.0
    assert file_digest(path) == before
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
        reader_keys(),
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
            reader_keys(),
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
        source_dir, [], reader_keys(), record_cache_dir=tmp_path / "cache"
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
    write_reader_database(database)
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
    source = read_all_files(sources, [], reader_keys(), reader=reader)
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
            write_reader_database(directory / "first.db")
            write_reader_database(directory / "nested" / "second.db")
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
    source = read_all_files(sources, [], reader_keys(), reader=readers)
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
    split = tace_read_all_files(cfg, [], [], reader_keys())
    assert len(split.train) == 3
    assert len(split.valid) == 2
    assert [len(part) for part in split.test] == [8, 1]
    split.train.close()
    split.valid.close()
    for part in split.test:
        part.close()
    valid_only = tace_read_all_files(cfg, [], [], reader_keys(), stages=("valid",))
    assert valid_only.train is None and len(valid_only.valid) == 2
    valid_only.valid.close()
    test_only = tace_read_all_files(cfg, [], [], reader_keys(), stages=("test",))
    assert [len(part) for part in test_only.test] == [8, 1]
    for part in test_only.test:
        part.close()
    signature = source_signature(sources[5], assigned["test0"])
    assert len(signature) == 2 and all(item["reader"] == "ase_db" for item in signature)
    assert source_cache_key(cfg, "test0") != source_cache_key(cfg, "test1")
    for bad_readers in (readers[:-1], readers + ["ase"]):
        cfg["dataset"]["reader"] = bad_readers
        with pytest.raises(ValueError, match="expected 7"):
            tace_read_all_files(cfg, [], [], reader_keys(), stages=("train",))
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
    atoms_list = write_reader_database(filename)
    keys = reader_keys()
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
    write_reader_database(filename)
    keys = reader_keys()
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

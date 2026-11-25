"""Training statistics, precision and distributed accumulation."""

import math
import multiprocessing
import os
import time
from collections import Counter
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase import Atoms
from omegaconf import OmegaConf
from torch.utils.data import SequentialSampler
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

import tace.dataset.statistics as statistics_module
from tace.dataset.cache import (
    GRAPH_SCHEMA,
    graph_cache_directories,
    source_cache_key,
    statistics_cache_key,
)
from tace.dataset.dataloader import (
    StatisticsDataLoader,
    build_statistics_dataloader,
    collate_statistics,
)
from tace.dataset.element import build_element_lookup
from tace.dataset.filter import filter_graphs
from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KeySpecification
from tace.dataset.statistics import statistics_devices
from tace.dataset.store import write_graphs
from tace.models.blocks import ScaleShift
from tace.utils.cfg import save_full_cfg

from .testing import file_digest, file_snapshot


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


def test_training_yaml_only_written_by_global_rank_zero(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for name in (
        "RANK",
        "NODE_RANK",
        "GROUP_RANK",
        "LOCAL_RANK",
        "SLURM_PROCID",
        "JSM_NAMESPACE_RANK",
    ):
        monkeypatch.delenv(name, raising=False)

    def write():
        save_full_cfg({"seed": 42})
        statistics_module.log_statistics_to_yaml([{"num_atoms": 2}])

    monkeypatch.setenv("LOCAL_RANK", "0")
    for name in ("RANK", "NODE_RANK", "SLURM_PROCID"):
        monkeypatch.setenv(name, "1")
        write()
        assert not list(tmp_path.iterdir())
        monkeypatch.delenv(name)
    write()
    assert {p.name for p in tmp_path.iterdir()} == {
        "_full_config.yaml",
        "statistics_0.yaml",
    }


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
    before = {p: file_digest(p) for p in tmp_path.rglob("*") if p.is_file()}
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
    assert all(file_digest(p) == v for p, v in before.items())


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
    raw = [-2.0, -1.0, -2.0, -4.5, -3.0, 999.0]
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
    before = file_snapshot(tmp_path)
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
    # Both isolated references pass. Fe2O meets the raw threshold exactly;
    # FeO would pass only if the training-time U shift leaked into filtering.
    expected_indices = [0, 1, 3]
    for use_u_shift in ("1", "0"):
        monkeypatch.setenv("TACE_USE_U_SHIFT", use_u_shift)
        selected = filter_graphs(dataset, {"max_energy_per_atom": -1.5})
        assert selected.indices.tolist() == expected_indices
        assert [float(graph.energy) for graph in selected] == [
            raw[i] for i in expected_indices
        ]
        selected.close()
    expected = _accumulator(graphs, options, batch_size=2).result()[0]
    _assert_statistics_equal(actual, expected)
    assert [float(graph.energy) for graph in dataset] == raw
    assert [atoms.info["energy"] for atoms in atoms_list] == raw
    dataset.close()
    assert file_snapshot(tmp_path) == before


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
    before = file_snapshot(tmp_path)
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
    assert file_snapshot(tmp_path) == before
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
    before = file_snapshot(tmp_path)
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
    assert file_snapshot(tmp_path) == before
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

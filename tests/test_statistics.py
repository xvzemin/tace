import math

import pytest
import torch
from ase import Atoms
from torch.utils.data import SequentialSampler
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

import tace.dataset.statistics as statistics_module
from tace.dataset.dataloader import build_statistics_dataloader
from tace.dataset.element import build_element_lookup
from tace.dataset.quantity import KeySpecification
from tace.models.blocks import ScaleShift


def test_statistics_dataloader_overrides_training_batch_size():
    dataset = [
        Data(
            atomic_numbers=torch.tensor([1]),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            num_nodes=1,
            positions=torch.zeros(1, 3),
            node_attrs=torch.ones(1, 1),
            energy=torch.ones(1),
            energy_weight=torch.ones(1),
        )
        for _ in range(5)
    ]
    loader = build_statistics_dataloader(
        {
            "dataset": {
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
    assert set(loader.collate_fn.exclude_keys) == {"node_attrs", "positions"}
    batch = next(iter(loader))
    assert "energy" in batch
    assert "energy_weight" in batch
    assert "node_attrs" not in batch
    assert "positions" not in batch


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
    assert stats["std_energy"] == math.sqrt(8.0)
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
    torch.testing.assert_close(scale_shift.shift, torch.tensor([[3.5, 3.5]]))

    removed_names = {
        "__mean_energy",
        "__std_energy",
        "__mean_energy_per_atom",
        "__mean_delta_energy_per_atom",
        "scalar_mean_energy_per_atom",
        "mean_energy_by_element",
        "std_energy_by_element",
        "mean_energy_per_atom_by_element",
        "mean_delta_energy_per_atom_by_element",
    }
    assert removed_names.isdisjoint(stats)


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
    assert stats["std_forces_norm"] == math.sqrt(8.0 / 3.0)
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
        torch.full((2,), math.sqrt(35.0 / 3.0)),
    )
    torch.testing.assert_close(
        torch.tensor(list(stats["rms_forces_by_element"].values())),
        torch.tensor([math.sqrt(5.0), 5.0]),
    )
    assert stats["num_forces_by_element"] == {1: 2, 2: 1}
    torch.testing.assert_close(
        torch.tensor(list(stats["mean_squared_forces_xyz_by_element"].values())),
        torch.tensor([5.0 / 3.0, 25.0 / 3.0]),
    )

    removed_names = {
        "__mean_forces_3d",
        "__std_forces_3d",
        "__rms_forces_3d",
        "__mean_forces_1d",
        "__std_forces_1d",
        "__rms_forces_1d",
        "__mean_forces_3d_by_element",
        "__std_forces_3d_by_element",
        "__rms_forces_3d_by_element",
        "__mean_forces_1d_by_element",
        "__std_forces_1d_by_element",
        "__rms_forces_1d_by_element",
        "mean_forces_for_normalize",
        "std_forces_for_normalize",
        "std_forces",
        "std_forces_by_element",
        "force_atom_counts_by_element",
        "force_mse_by_element",
        "recommended_force_element_weights",
    }
    assert removed_names.isdisjoint(stats)


def test_initial_noncollinear_magmoms_statistics_use_canonical_names(monkeypatch):
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
        torch.tensor(stats["mean_initial_noncollinear_magmoms_xyz"]),
        torch.tensor([0.0, 5.0 / 3.0, 2.0]),
    )
    torch.testing.assert_close(
        torch.tensor(stats["std_initial_noncollinear_magmoms_xyz"]),
        torch.tensor(
            [math.sqrt(2.0 / 3.0), math.sqrt(14.0) / 3.0, math.sqrt(8.0 / 3.0)]
        ),
    )
    torch.testing.assert_close(
        torch.tensor(stats["rms_initial_noncollinear_magmoms_xyz"]),
        torch.tensor(
            [math.sqrt(2.0 / 3.0), math.sqrt(13.0 / 3.0), math.sqrt(20.0 / 3.0)]
        ),
    )
    assert stats["mean_initial_noncollinear_magmoms_norm"] == 3.0
    assert stats["std_initial_noncollinear_magmoms_norm"] == math.sqrt(8.0 / 3.0)
    assert stats["rms_initial_noncollinear_magmoms_norm"] == math.sqrt(35.0 / 3.0)
    assert stats["max_initial_noncollinear_magmoms_norm"] == 5.0
    assert stats["mean_initial_noncollinear_magmoms_xyz_by_element"] == {
        1: [0.0, 1.0, 1.0],
        2: [0.0, 3.0, 4.0],
    }
    assert stats["std_initial_noncollinear_magmoms_xyz_by_element"] == {
        1: [1.0, 1.0, 1.0],
        2: [0.0, 0.0, 0.0],
    }
    torch.testing.assert_close(
        torch.tensor(
            list(stats["rms_initial_noncollinear_magmoms_norm_by_element"].values())
        ),
        torch.tensor([math.sqrt(5.0), 5.0]),
    )
    assert stats["max_initial_noncollinear_magmoms_norm_by_element"] == {
        1: 3.0,
        2: 5.0,
    }
    assert stats["num_initial_noncollinear_magmoms_by_element"] == {1: 2, 2: 1}
    assert "magmoms_norm_by_element" not in stats


def test_initial_noncollinear_magmoms_statistics_are_per_fidelity(monkeypatch):
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

    assert stats[0]["max_initial_noncollinear_magmoms_norm_by_element"] == {26: 1.0}
    assert stats[1]["max_initial_noncollinear_magmoms_norm_by_element"] == {26: 5.0}

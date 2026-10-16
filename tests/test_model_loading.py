"""Precision conversion and isolation of model state during loading."""

from copy import deepcopy

import numpy as np
import pytest
import torch
from ase import Atoms
from torch_geometric.data import Batch

from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KeySpecification
from tace.interface.ase import TACEAseCalc
from tace.lightning import create_model, export_tace, load_tace
from tace.lightning.lit_model import LightningWrapperModel, _load_model_state_dict
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.utils.utils import torch_default_dtype


@pytest.fixture(params=[torch.float32, torch.float64], ids=["fp32", "fp64"])
def precision_model(request, monkeypatch):
    monkeypatch.setenv("TACE_USE_TF32", "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=2,
        lmax=2,
        mmax=2,
        cutoff=4.0,
        max_neighbors=None,
        fidelity=[{"name": "first"}, {"name": "second"}],
        target_property=["energy", "forces", "stress"],
    )
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"].update(scale_type=None, all_atoms=True)
    config["short_range"]["zbl"]["enable"] = True
    statistics = [
        dict(
            atomic_numbers=[1, 14],
            atomic_energy={1: -0.1, 14: -0.8},
            avg_num_neighbors=2.0,
        )
        for _ in range(2)
    ]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch_default_dtype(request.param):
        model = create_model(config, statistics, ["energy", "forces", "stress"], [])
    model.to(device=device, dtype=request.param).eval()
    # Saved physical state deliberately differs from the construction metadata.
    with torch.no_grad():
        model.readout_fn.atomic_energy_layer.atomic_energy.copy_(
            torch.tensor([[-0.3, -0.86888875], [-0.5, -1.23456789]])
        )
        model.readout_fn.scale_shift.scale.fill_(1.25)
        model.readout_fn.scale_shift.shift.fill_(-0.125)
        model.readout_fn.representation.radial_basis.radial_fn.bessel_weights.mul_(1.1)
    return model


def model_batch(model):
    structures = [
        Atoms("Si2", positions=[[0, 0, 0], [2.35, 0.13, -0.17]]),
        Atoms("HSi", positions=[[0, 0, 0], [1.7, 0.2, 0.3]], cell=[6] * 3, pbc=True),
    ]
    for index, atoms in enumerate(structures):
        atoms.info["fidelity_idx"] = index
    graphs = [
        from_atoms(
            model.get_torch_element(),
            atoms,
            model.get_cutoff(),
            max_neighbors=None,
            keyspec=KeySpecification(info_keys={"fidelity_idx": "fidelity_idx"}),
            target_property=model.get_target_property(),
            embedding_property=[],
            training=False,
            neighborlist_backend="ase",
            dtype=model.get_model_dtype(),
        )
        for atoms in structures
    ]
    return Batch.from_data_list(graphs).to(next(model.parameters()).device).to_dict()


def save_checkpoint(model, path):
    torch.save(
        {
            "state_dict": {
                "model." + key: value for key, value in model.state_dict().items()
            },
            "hyper_parameters": {
                "cfg": {
                    "model": {"config": model.readout_fn.model_config},
                    "loss": {"loss_property": model.get_target_property()},
                },
                "statistics": model.readout_fn.statistics,
            },
        },
        path,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("format", ["pt", "pth", "ckpt", "module"])
def test_model_precision_roundtrip(precision_model, tmp_path, dtype, format):
    source = precision_model
    device = next(source.parameters()).device
    if format == "module":
        path = deepcopy(source)
    elif format == "ckpt":
        path = tmp_path / "model.ckpt"
        save_checkpoint(source, path)
    else:
        path = tmp_path / "model.pt"
        export_tace(source, str(path))
        path = path.rename(tmp_path / ("model." + format))

    default_dtype = torch.float64 if dtype == torch.float32 else torch.float32
    torch.set_default_dtype(default_dtype)
    loaded = load_tace(path, device=device, dtype=dtype).eval()
    assert torch.get_default_dtype() == default_dtype
    assert loaded.get_model_dtype() == dtype
    assert all(p.dtype == dtype for p in loaded.parameters())
    assert all(b.dtype == dtype for b in loaded.buffers() if b.is_floating_point())
    for name, value in loaded.state_dict().items():
        if name.rsplit(".", 1)[-1].startswith(("_w3j_", "U_matrix_")):
            continue
        torch.testing.assert_close(
            value, source.state_dict()[name].to(value), atol=0, rtol=0
        )

    reference = deepcopy(source).to(dtype=dtype)
    expected = reference(model_batch(reference))
    actual = loaded(model_batch(loaded))
    for key in ("energy", "forces", "stress"):
        assert actual[key].dtype == dtype
        assert torch.isfinite(actual[key]).all()
        torch.testing.assert_close(actual[key], expected[key], atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("dtype", [None, torch.float32, torch.float64])
def test_checkpoint_precision(precision_model, tmp_path, dtype):
    path = tmp_path / "model.ckpt"
    save_checkpoint(precision_model, path)
    device = next(precision_model.parameters()).device
    default_dtype = torch.get_default_dtype()
    loaded = LightningWrapperModel.load_from_checkpoint(
        path, map_location=device, dtype=dtype
    )
    expected_dtype = dtype or precision_model.get_model_dtype()
    assert loaded.get_model_dtype() == expected_dtype
    assert torch.get_default_dtype() == default_dtype
    assert all(
        b.dtype == expected_dtype for b in loaded.buffers() if b.is_floating_point()
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_loaded_force_training(precision_model, tmp_path, dtype):
    path = tmp_path / "model.pt"
    export_tace(precision_model, str(path))
    loaded = load_tace(
        path, device=next(precision_model.parameters()).device, dtype=dtype
    ).train()
    output = loaded(model_batch(loaded))
    loss = sum(output[key].square().mean() for key in ("energy", "forces", "stress"))
    loss.backward()
    gradients = [p.grad for p in loaded.parameters() if p.grad is not None]
    assert gradients
    assert all(g.dtype == dtype and torch.isfinite(g).all() for g in gradients)
    assert any(g.abs().sum() > 0 for g in gradients)


def test_precision_conversion_rebuilds_only_angular_constants():
    from e3nn import o3

    def module(dtype):
        with torch_default_dtype(dtype):
            model = o3.FullyConnectedTensorProduct("1o", "1o", "2e")
            model.register_buffer("atomic_energy", torch.tensor([-0.8]))
            model.register_buffer("U_matrix_2", torch.tensor([2.0**-0.5]))
        return model

    original = module(torch.float32)
    original.atomic_energy.fill_(-0.86888875)
    expected = module(torch.float64)
    model = module(torch.float64)
    state = original.state_dict()
    _load_model_state_dict(model, state, strict=True, rebuild_constants=True)
    torch.testing.assert_close(
        model.atomic_energy, original.atomic_energy.double(), atol=0, rtol=0
    )
    torch.testing.assert_close(model.weight, original.weight.double(), atol=0, rtol=0)
    constants = [
        name
        for name in state
        if name.rsplit(".", 1)[-1].startswith(("_w3j_", "U_matrix_"))
    ]
    assert len(constants) >= 2
    for name in constants:
        actual = model.state_dict()[name]
        torch.testing.assert_close(actual, expected.state_dict()[name], atol=0, rtol=0)
        assert not torch.equal(actual, state[name].double())
        assert state[name].dtype == torch.float32

    del state["atomic_energy"]
    with pytest.raises(RuntimeError, match="Missing key"):
        _load_model_state_dict(model, state, strict=True, rebuild_constants=True)


def test_ase_mixed_precision_calculators(precision_model, tmp_path):
    path = tmp_path / "model.pt"
    export_tace(precision_model, str(path))
    device = next(precision_model.parameters()).device
    calculators = [
        TACEAseCalc(path, device=device, dtype=dtype, neighborlist_backend="ase")
        for dtype in ("float32", "float64")
    ]
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2.3500001234567, 0.13, -0.17]])
    results = []
    for index, calculator in enumerate(calculators + calculators[::-1]):
        default_dtype = torch.float32 if index < 2 else torch.float64
        torch.set_default_dtype(default_dtype)
        calculator.reset()
        atoms.calc = calculator
        results.append((atoms.get_potential_energy(), atoms.get_forces()))
        assert torch.get_default_dtype() == default_dtype
    for actual, expected in zip(results, results[::-1]):
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a, b)
    for a, b in zip(results[0], results[1]):
        np.testing.assert_allclose(a, b, atol=5e-6, rtol=5e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graph_explicit_precision(dtype):
    from tace.dataset.element import TorchElement

    torch.set_default_dtype(torch.float64 if dtype == torch.float32 else torch.float32)
    atoms = Atoms("Si2", positions=[[0, 0, 0], [2.3500001234567, 0.1, -0.2]])
    graph = from_atoms(
        TorchElement([14]),
        atoms,
        4.0,
        max_neighbors=None,
        keyspec=KeySpecification(info_keys={"fidelity_idx": "fidelity_idx"}),
        target_property=["energy", "forces", "stress"],
        embedding_property=[],
        training=True,
        neighborlist_backend="ase",
        dtype=dtype,
    )
    assert all(value.dtype == dtype for _, value in graph if value.is_floating_point())
    assert graph.edge_index.dtype == graph.atomic_numbers.dtype == torch.int64
    assert graph.pbc.dtype == graph.stress_mask.dtype == torch.bool
    torch.testing.assert_close(
        graph.positions, torch.tensor(atoms.positions, dtype=dtype), atol=0, rtol=0
    )

"""Model loading, precision conversion, and element and representation transfer."""

import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms
from torch_geometric.data import Batch

from tace.dataset.graph import from_atoms
from tace.dataset.quantity import KeySpecification
from tace.foundations import resolve_model_path, tace_foundations
from tace.interface.ase import TACEAseCalc
from tace.lightning import convert_cgtp, create_model, export_tace, load_tace
from tace.lightning.lit_model import LightningWrapperModel, _load_model_state_dict
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.tace import e3nnTACE
from tace.models.adapter import TensorModel
from tace.models.blocks import ScaleShift
from tace.models.linear import switch_e3nn_weight_layout
from tace.scripts.export_elements import select_elements
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


@pytest.mark.parametrize("all_atoms", [None, False, True])
def test_scale_shift_additivity(caplog, all_atoms):
    assert DEFAULT_MODEL_CONFIG["scale_shift"]["all_atoms"] is False
    options = {} if all_atoms is None else {"all_atoms": all_atoms}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layer = ScaleShift([1], [{1: 2.0}], [{1: 0.5}], **options).to(device)
    energy = torch.full((2,), 3.0, device=device, dtype=torch.float64)
    attrs = torch.ones(2, 1, device=device, dtype=torch.float64)
    edges = torch.empty(2, 0, dtype=torch.long, device=device)
    fidelity = torch.zeros(2, dtype=torch.long, device=device)
    together = layer(
        energy,
        attrs,
        torch.tensor([0, 2], device=device),
        edges,
        torch.tensor([0, 0], device=device),
        fidelity,
    )
    separate = layer(
        energy,
        attrs,
        torch.tensor([0, 1, 2], device=device),
        edges,
        torch.tensor([0, 1], device=device),
        fidelity,
    )
    torch.testing.assert_close(together, torch.full_like(energy, 6.5))
    torch.testing.assert_close(
        separate, together if all_atoms is True else torch.zeros_like(energy)
    )
    messages = [
        record.message
        for record in caplog.records
        if "scale_shift.all_atoms=False is a legacy option" in record.message
    ]
    assert not messages


@pytest.mark.parametrize("format", ["pt", "ckpt"])
@pytest.mark.parametrize("all_atoms", [None, False, True])
def test_saved_scale_shift_default(
    precision_model, tmp_path, caplog, format, all_atoms
):
    config = precision_model.readout_fn.model_config["scale_shift"]
    if all_atoms is None:
        config.pop("all_atoms")
    else:
        config["all_atoms"] = all_atoms
    path = tmp_path / f"model.{format}"
    if format == "ckpt":
        save_checkpoint(precision_model, path)
    else:
        export_tace(precision_model, str(path))
    caplog.clear()
    model = load_tace(path, device=next(precision_model.parameters()).device)
    assert model.readout_fn.scale_shift.all_atoms is (all_atoms is True)
    messages = [
        record.message
        for record in caplog.records
        if "scale_shift.all_atoms=False is a legacy option" in record.message
    ]
    assert len(messages) == int(all_atoms is not True)


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
    assert graph.pbc.dtype == torch.bool
    torch.testing.assert_close(
        graph.positions, torch.tensor(atoms.positions, dtype=dtype), atol=0, rtol=0
    )


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(interaction="cgtp", matrix=False, v2=False):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=4,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
        target_property=["energy", "forces", "stress"],
    )
    config["atomic_basis"].update(type=["cgtp", interaction], num_head=1)
    config["edge_update"]["type"] = "element2"
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["apply_cutoff"] = False
    if v2:
        config["edge_update"]["type"] = "element2_v2"
        config["mlp_type"] = "mlp_v2"
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"].update(
        scale_type="rms_forces",
        shift_type="mean_energy_per_atom",
        scale_trainable=True,
        shift_trainable=True,
    )
    config["fidelity"] = [{"name": "a"}, {"name": "b"}]
    if interaction == "so2":
        config["product_basis"].update(num_expert=2, num_channel_per_expert=2)
    elif interaction == "o2_mag":
        config["parity"] = True
        config["node_update"]["magnetic_type"] = "element2"
        config["angular_basis"]["magnetic_Lmax"] = 1
        config["fidelity"][0]["magnetic_scale"] = {1: 2.0, 6: 3.0, 8: 4.0}
        config["fidelity"][1]["magnetic_scale"] = {1: 3.0, 6: 4.0, 8: 5.0}
    statistics = [
        {
            "atomic_numbers": [1, 6, 8],
            "atomic_energy": {1: -1.0, 6: -2.0, 8: -3.0},
            "avg_num_neighbors": 2.0,
            "rms_forces": {1: 1.0 + fidelity, 6: 1.5, 8: 2.0 + fidelity},
            "mean_energy_per_atom": {1: 0.3, 6: 0.4, 8: 0.5 + fidelity},
        }
        for fidelity in range(2)
    ]
    model = create_model(config, statistics, ["energy", "forces", "stress"], [])
    if matrix:
        switch_e3nn_weight_layout(model, "matrix")
    return model.to(device=DEVICE, dtype=torch.float64).eval()


def evaluate(model, fidelity=0, numbers=(1, 1, 8)):
    return model(
        {
            "positions": torch.tensor(
                [[0.0, 0.1, 0.0], [1.0, 0.0, 0.2], [0.1, 1.0, 0.0]],
                device=DEVICE,
            ),
            "node_attrs": model.get_torch_element()
            .z2onehot(torch.tensor(numbers, device=DEVICE))
            .to(torch.float64),
            "edge_index": torch.tensor(
                [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], device=DEVICE
            ),
            "edge_shifts": torch.zeros(6, 3, device=DEVICE),
            "lattice": torch.eye(3, device=DEVICE).unsqueeze(0) * 10.0,
            "batch": torch.zeros(3, dtype=torch.long, device=DEVICE),
            "ptr": torch.tensor([0, 3], device=DEVICE),
            "fidelity_idx": torch.tensor([fidelity], device=DEVICE),
            "initial_noncollinear_magmoms": torch.tensor(
                [[0.1, 0.2, 0.3], [0.3, -0.1, 0.2], [-0.1, 0.1, 0.2]],
                device=DEVICE,
            ),
        }
    )


@pytest.mark.parametrize("interaction", ["cgtp", "so2", "o2", "o2_mag"])
@pytest.mark.parametrize("elements", [["O", "H"], [14, 8, 6, 1], ["H", "O", "Si"]])
def test_select_elements(interaction, elements, tmp_path, double_precision):
    model = make_model(interaction)
    original = {name: value.clone() for name, value in model.state_dict().items()}
    selected = select_elements(model, elements)
    path = tmp_path / "selected.pt"
    export_tace(selected, str(path))
    restored = load_tace(path, device=DEVICE).eval()
    assert restored.get_atomic_numbers() == selected.get_atomic_numbers()
    assert restored.readout_fn.num_fidelities == 2
    for fidelity in range(2):
        expected = evaluate(model, fidelity)
        actual = evaluate(restored, fidelity)
        for key in ("energy", "forces", "stress"):
            torch.testing.assert_close(
                actual[key], expected[key], atol=1e-10, rtol=1e-10
            )
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, original[name], atol=0, rtol=0)


@pytest.mark.parametrize("v2", [False, True])
def test_new_element_initialization(double_precision, v2):
    model = make_model(v2=v2)
    torch.manual_seed(12)
    selected = select_elements(model, [8, 14])
    torch.manual_seed(12)
    initialized = create_model(
        deepcopy(selected.readout_fn.model_config),
        selected.readout_fn.statistics,
        selected.get_target_property(),
        selected.get_embedding_property(),
    ).to(device=DEVICE, dtype=torch.float64)
    for name, module in selected.named_modules():
        original = dict(initialized.named_modules())[name]
        if (
            name.endswith("node_embedding.embedding")
            or name.endswith("source_embedding")
            or name.endswith("target_embedding")
        ):
            torch.testing.assert_close(
                module.weight.reshape(2, -1)[1],
                original.weight.reshape(2, -1)[1],
                atol=0,
                rtol=0,
            )
        elif hasattr(module, "num_elements") and hasattr(module, "weight"):
            torch.testing.assert_close(
                module.weight[1], original.weight[1], atol=0, rtol=0
            )
    assert torch.equal(
        selected.readout_fn.atomic_energy_layer.atomic_energy[:, 1],
        torch.zeros(2, device=DEVICE),
    )
    assert torch.equal(
        selected.readout_fn.scale_shift.scale[:, 1], torch.ones(2, device=DEVICE)
    )
    assert torch.equal(
        selected.readout_fn.scale_shift.shift[:, 1], torch.zeros(2, device=DEVICE)
    )
    for value in evaluate(selected, numbers=(8, 14, 8)).values():
        if isinstance(value, torch.Tensor):
            assert torch.isfinite(value).all()


@pytest.mark.parametrize("elements", [[1, 8], [1, 6, 8, 14]])
def test_v2_element_export_roundtrip(elements, tmp_path, double_precision):
    model = make_model(v2=True)
    selected = select_elements(model, elements)
    path = tmp_path / "selected.pt"
    export_tace(selected, str(path))
    restored = load_tace(path, device=DEVICE, dtype=torch.float64).eval()
    for fidelity in range(2):
        expected = evaluate(model, fidelity)
        actual = evaluate(restored, fidelity)
        for key in ("energy", "forces", "stress"):
            torch.testing.assert_close(
                actual[key], expected[key], atol=1e-10, rtol=1e-10
            )


def test_legacy_mlp_defaults(tmp_path, double_precision):
    model = make_model()
    path = tmp_path / "legacy.pt"
    export_tace(model, str(path))
    checkpoint = torch.load(path, weights_only=False)
    del checkpoint["cfg"]["mlp_type"]
    del checkpoint["cfg"]["radial_basis"]["mlp_norm"]
    torch.save(checkpoint, path)
    restored = load_tace(path, device=DEVICE, dtype=torch.float64).eval()
    for name, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], value, atol=0, rtol=0)
    for fidelity in range(2):
        actual, expected = evaluate(restored, fidelity), evaluate(model, fidelity)
        for key in ("energy", "forces", "stress"):
            torch.testing.assert_close(actual[key], expected[key], atol=0, rtol=0)


def test_matrix_weights(double_precision):
    model = make_model(matrix=True)
    selected = select_elements(model, [1, 8])
    actual = evaluate(selected)
    for key, value in evaluate(model).items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(actual[key], value, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("elements", [[], ["Unknown"], [0], [119], [1, "H"]])
def test_invalid_elements(elements, double_precision):
    with pytest.raises(ValueError):
        select_elements(make_model(), elements)


def test_export_elements_cli(tmp_path, monkeypatch, double_precision, capsys):
    from tace.scripts.export_elements import main

    path, output = tmp_path / "original.pt", tmp_path / "selected.pt"
    export_tace(make_model(), str(path))
    monkeypatch.setattr(
        "sys.argv",
        [
            "tace-export-elements",
            "-m",
            str(path),
            "-e",
            "8",
            "H",
            "Si",
            "-o",
            str(output),
        ],
    )
    main()
    assert load_tace(output, device="cpu").get_atomic_numbers() == [1, 8, 14]
    assert "New elements initialized (training required): Si" in capsys.readouterr().out
    original = output.read_bytes()
    with pytest.raises(SystemExit, match="2"):
        main()
    assert output.read_bytes() == original


@pytest.mark.parametrize(
    "parity,node_embedding",
    [(False, "linear"), (True, "linear"), (True, "spherical_tensor")],
)
@pytest.mark.parametrize("fused", [False, True])
def test_cartesian_conversion(
    double_precision, monkeypatch, tmp_path, parity, node_embedding, fused
):
    for name in ("TACE_USE_EQT", "TACE_USE_OEQ", "TACE_USE_CUE", "TACE_USE_EQX"):
        monkeypatch.setenv(name, "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=3,
        Lmax=2,
        lmax=3,
        parity=parity,
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["node_embedding"]["type"] = node_embedding
    config["atomic_basis"]["type"] = "cgtp"
    config["readout_emlp"].update(use_one_body_magmoms=False, hidden=[2])
    config["radial_basis"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    reference = TensorModel(e3nnTACE(**config)).to(DEVICE).train()
    if node_embedding == "spherical_tensor":
        assert (
            reference.readout_fn.representation.node_embedding.edge_info.dims[1:-1]
            == config["radial_basis"]["hidden"]
        )
    module = convert_cgtp(reference, implementation="co3")
    parameters = dict(reference.named_parameters())
    assert parameters.keys() == dict(module.named_parameters()).keys()
    for name, p in module.named_parameters():
        torch.testing.assert_close(p, parameters[name], rtol=0, atol=0)
    representation = module.readout_fn.representation
    assert representation.co3_angular_basis is not None
    assert representation.use_o3_angular_basis == (node_embedding == "spherical_tensor")
    for original, converted in zip(
        reference.readout_fn.representation.products, representation.products
    ):
        assert type(original) is type(converted)
    if node_embedding == "linear":

        def no_spherical_harmonics(*args):
            raise AssertionError(
                "Cartesian convolution must evaluate Cartesian harmonics."
            )

        monkeypatch.setattr(
            representation.o3_angular_basis, "forward", no_spherical_harmonics
        )
    data = dict(
        positions=torch.tensor(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.3, 0.2],
                [0.4, 1.1, -0.2],
                [0.2, 0.1, 0.3],
                [0.7, -0.3, 1.1],
            ],
            device=DEVICE,
        ),
        node_attrs=torch.ones(5, 1, device=DEVICE),
        edge_index=torch.tensor(
            [[0, 1, 0, 2, 1, 2, 3, 4], [1, 0, 2, 0, 2, 1, 4, 3]], device=DEVICE
        ),
        edge_shifts=torch.zeros(8, 3, device=DEVICE),
        lattice=torch.eye(3, device=DEVICE).repeat(2, 1, 1) * 8,
        batch=torch.tensor([0, 0, 0, 1, 1], device=DEVICE),
        ptr=torch.tensor([0, 3, 5], device=DEVICE),
        fidelity_idx=torch.zeros(2, dtype=torch.long, device=DEVICE),
    )
    expected = reference({k: v.clone() for k, v in data.items()})
    if fused:
        monkeypatch.setenv("TACE_USE_EQX", "1")

        def no_edge_harmonics(*args):
            raise AssertionError(
                "The fused convolution evaluates harmonics in registers."
            )

        monkeypatch.setattr(
            representation.co3_angular_basis, "forward", no_edge_harmonics
        )
    actual = module({k: v.clone() for k, v in data.items()})
    for key in ("energy", "forces", "stress", "virials"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-10, rtol=2e-10)
    for model, output in ((reference, expected), (module, actual)):
        sum(
            output[key].square().sum() for key in ("energy", "forces", "stress")
        ).backward()
    for name, p in module.named_parameters():
        if parameters[name].grad is not None:
            torch.testing.assert_close(
                p.grad, parameters[name].grad, atol=2e-9, rtol=2e-9
            )
    module.eval()
    export_tace(module, str(tmp_path / "co3.pt"))
    restored = load_tace(tmp_path / "co3.pt", device=DEVICE, dtype=torch.float64).eval()
    spherical = convert_cgtp(module, "o3").eval()
    for model in (restored, spherical):
        output = model({k: v.clone() for k, v in data.items()})
        for key in ("energy", "forces", "stress", "virials"):
            torch.testing.assert_close(
                output[key], expected[key], atol=2e-10, rtol=2e-10
            )


@pytest.fixture
def foundation_model(tmp_path, monkeypatch, double_precision):
    import tace.foundations.download_link as downloads
    from tace.lightning import create_model, export_tace

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=1, num_channel=2, Lmax=1, lmax=1, cutoff=4.0, max_neighbors=None
    )
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        {
            "atomic_numbers": [1],
            "atomic_energy": {1: 0.0},
            "avg_num_neighbors": 2.0,
        }
    ]
    model = create_model(config, statistics, ["energy", "forces"], [])
    path = tmp_path / "download.pt"
    export_tace(model, str(path))
    cache = tmp_path / "cache"
    cache.mkdir()
    download = Mock(return_value=str(path))
    monkeypatch.setattr(downloads, "CACHE_DIR", cache)
    monkeypatch.setattr(downloads, "hf_hub_download", download)
    return path, download


@pytest.mark.parametrize("name", ["TACE-OAM-7M", "TECE-OAM-RRA-1.0"])
def test_foundation_name_resolution(foundation_model, name):
    source, download = foundation_model
    assert name in tace_foundations
    assert "unknown" not in tace_foundations
    download.assert_not_called()
    path = resolve_model_path(name)
    assert path.name == name + ".pt"
    assert path.read_bytes() == source.read_bytes()
    assert resolve_model_path(Path(name)) == path
    assert resolve_model_path(path) == path
    download.assert_called_once_with(
        repo_id="xvzemin/tace-foundations",
        filename=name + ".pt",
        revision="main",
        cache_dir=path.parent,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_load_foundation_matches_file(foundation_model, dtype):
    from tace.lightning import load_tace

    source, download = foundation_model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference = load_tace(source, device=device, dtype=dtype).eval()
    module = load_tace("TACE-OAM-7M", device=device, dtype=dtype).eval()
    assert module.get_model_dtype() == dtype
    for name, value in reference.state_dict().items():
        torch.testing.assert_close(module.state_dict()[name], value, atol=0, rtol=0)
    atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0.2, 0]], cell=[6, 6, 6])
    graph = from_atoms(
        reference.get_torch_element(),
        atoms,
        reference.get_cutoff(),
        max_neighbors=None,
        keyspec=KeySpecification(info_keys={"fidelity_idx": "fidelity_idx"}),
        target_property=["energy", "forces"],
        embedding_property=[],
        training=False,
        neighborlist_backend="ase",
        dtype=dtype,
    )
    data = Batch.from_data_list([graph]).to(device).to_dict()
    results = [
        model({key: value.clone() for key, value in data.items()})
        for model in (reference, module)
    ]
    for key in ("energy", "forces"):
        torch.testing.assert_close(results[0][key], results[1][key])
    assert load_tace(module, device=device, target_property=["energy"]) is module
    assert module.get_target_property() == ["energy"]
    download.assert_called_once()


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".PT", ".ckpt", ".pt2"])
def test_local_model_dispatch(foundation_model, tmp_path, monkeypatch, suffix):
    import tace.models.compile as compile_model
    from tace.lightning import load_tace
    from tace.lightning.lit_model import LightningWrapperModel

    source, download = foundation_model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    reference = load_tace(source, device=device)
    path = tmp_path / ("TACE-OAM-7M" + suffix)
    checkpoint = Mock(return_value=reference)
    aoti = Mock(return_value=reference)
    monkeypatch.setattr(LightningWrapperModel, "load_from_checkpoint", checkpoint)
    monkeypatch.setattr(compile_model, "load_aotinductor", aoti)
    if suffix.lower() in (".pt", ".pth"):
        path.write_bytes(source.read_bytes())
    module = load_tace(path, device=device, use_ema=False)
    assert module.get_model_dtype() == reference.get_model_dtype()
    assert checkpoint.call_count == (suffix == ".ckpt")
    assert aoti.call_count == (suffix == ".pt2")
    if suffix == ".ckpt":
        checkpoint.assert_called_once_with(
            path, map_location=device, strict=True, use_ema=False, dtype=None
        )
    elif suffix == ".pt2":
        aoti.assert_called_once_with(path, device)
    download.assert_not_called()


def test_unknown_model_does_not_download(foundation_model, tmp_path):
    from tace.lightning import load_tace

    _, download = foundation_model
    for name in ("unknown", "model.bin", str(tmp_path / "TACE-OAM-7M")):
        with pytest.raises(ValueError, match="registered foundation model"):
            load_tace(name)
    with pytest.raises(FileNotFoundError):
        load_tace(tmp_path / "TACE-OAM-7M.pt")
    download.assert_not_called()


@pytest.mark.parametrize(
    "script,output",
    [
        ("export_train", "-state.pt"),
        ("export_eval", "-state_dict.pt"),
        ("convert_cgtp", "-converted.pt"),
    ],
)
def test_foundation_cli_exports(
    foundation_model, monkeypatch, tmp_path, script, output
):
    import importlib

    name = "TECE-OAM-RRA-1.0"
    module = importlib.import_module("tace.scripts." + script)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [script, "-m", name, "--device", device])
    module.main()
    assert (tmp_path / (name + output)).is_file()
    foundation_model[1].assert_called_once()


def test_foundation_ase_and_finetune(foundation_model):
    from ase import Atoms

    from tace.interface.ase import TACEAseCalc
    from tace.lightning.lit_model import finetune

    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = []
    for model in (foundation_model[0], "TACE-OAM-7M"):
        atoms = Atoms("H2", positions=[[0, 0, 0], [1, 0.2, 0]], cell=[6, 6, 6])
        atoms.calc = TACEAseCalc(
            model, device=device, dtype="float64", neighborlist_backend="ase"
        )
        results.append((atoms.get_potential_energy(), atoms.get_forces()))
    for actual, expected in zip(*results):
        torch.testing.assert_close(
            torch.as_tensor(actual), torch.as_tensor(expected), atol=1e-12, rtol=1e-12
        )
    model = finetune(
        {"finetune_from_model": "TACE-OAM-7M", "trainer": {"precision": 64}}
    )
    assert model.training
    assert model.get_model_dtype() == torch.float64
    foundation_model[1].assert_called_once()

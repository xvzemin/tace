"""Element vocabulary transfer and checkpoint round trips."""

from copy import deepcopy

import pytest
import torch

from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models.linear import switch_e3nn_weight_layout
from tace.scripts.export_elements import select_elements

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_model(interaction="cgtp", matrix=False):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=4,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["atomic_basis"].update(type=["cgtp", interaction], num_head=1)
    config["edge_update"]["type"] = "element2"
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["apply_cutoff"] = False
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
        config["node_embedding"]["type"] = "linear_spin"
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


def test_new_element_initialization(double_precision):
    model = make_model()
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
            name.endswith("elem_emb1")
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

"""Cartesian and spherical model equivalence."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3

from eqx import co3
from tace.models import TensorModel, cartTACE, e3nnTACE
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG


@pytest.mark.parametrize("fused", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "parity,correlation,variant",
    [
        (False, 2, "default"),
        (True, 3, "default"),
        (False, 2, "tensor"),
        (False, 2, "bilinear"),
        (True, 2, "multihead"),
    ],
)
def test_cartesian_model(monkeypatch, fused, dtype, parity, correlation, variant):
    torch.set_default_dtype(dtype)
    monkeypatch.setenv("TACE_USE_EQX", str(int(fused)))
    monkeypatch.setenv("TACE_USE_EQT", "0")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=3,
        Lmax=2,
        lmax=3,
        parity=parity,
        cutoff=4.0,
        max_neighbors=None,
        target_property=["energy", "forces", "stress", "virials"],
        statistics=[
            dict(
                atomic_numbers=[1, 6],
                atomic_energy={1: 0.0, 6: 0.0},
                avg_num_neighbors=2.0,
            )
        ],
    )
    config["radial_basis"]["hidden"] = [5]
    config["product_basis"]["correlation"] = correlation
    config["scale_shift"]["enable"] = False
    config["readout_emlp"]["bias"] = True
    if variant == "tensor":
        config["node_embedding"]["type"] = "spherical_tensor_element2"
        config["layer_norm"]["pre_norm_type"] = "merge_layer_norm"
        config["target_property"] += [
            "direct_forces",
            "direct_stress",
            "direct_virials",
        ]
        config["atomic_basis"]["gate_m0"] = False
    elif variant == "bilinear":
        config["num_channel"] = 4
        config["product_basis"]["nonlinear"] = "bilinear"
        config["product_basis"]["num_expert"] = 2
    elif variant == "multihead":
        config["fidelity"] = [{"name": "A"}, {"name": "B"}]
        config["statistics"] *= 2
        config["target_property"] += [
            "direct_forces",
            "direct_stress",
            "direct_virials",
        ]
        config["node_embedding"]["type"] = "wigner_tensor_element2"
    torch.manual_seed(21)
    spherical = TensorModel(e3nnTACE(**deepcopy(config))).to(device).train()
    torch.manual_seed(21)
    cartesian = TensorModel(cartTACE(**deepcopy(config))).to(device).train()
    assert not any(
        isinstance(module, (co3.ChangeOfBasis, o3.TensorProduct, o3.SphericalHarmonics))
        for module in cartesian.modules()
    )
    left, right = dict(spherical.named_parameters()), dict(cartesian.named_parameters())
    assert left.keys() == right.keys()
    for name in left:
        torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        if name.endswith("bias"):
            with torch.no_grad():
                left[name].uniform_(-0.1, 0.1)
                right[name].copy_(left[name])
    data = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]]),
        lattice=4 * torch.eye(3).unsqueeze(0),
        node_attrs=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        edge_shifts=torch.zeros(6, 3),
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
        fidelity_idx=torch.zeros(1, dtype=torch.long),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    data = {key: value.to(device) for key, value in data.items()}
    if variant == "multihead":
        data["fidelity_idx"].fill_(1)
    outputs = [
        model({key: value.clone() for key, value in data.items()})
        for model in (spherical, cartesian)
    ]
    for key in config["target_property"]:
        torch.testing.assert_close(
            outputs[0][key],
            outputs[1][key],
            rtol=2e-9 if dtype == torch.float64 else 3e-4,
            atol=2e-10 if dtype == torch.float64 else 3e-5,
        )
    for output in outputs:
        (output["energy"].square().sum() + output["forces"].square().sum()).backward()
    for name in left:
        if left[name].grad is not None:
            torch.testing.assert_close(
                left[name].grad,
                right[name].grad,
                rtol=2e-8 if dtype == torch.float64 else 1e-3,
                atol=2e-9 if dtype == torch.float64 else 3e-5,
            )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_cartesian_loading(double_precision, monkeypatch, tmp_path, dtype):
    from tace.lightning import export_tace, load_tace

    monkeypatch.setenv("TACE_USE_EQX", "1")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_channel=2,
        num_layers=1,
        Lmax=2,
        lmax=2,
        cutoff=4.0,
        max_neighbors=None,
        statistics=[
            dict(atomic_numbers=[1], atomic_energy={1: -1.0}, avg_num_neighbors=1.0)
        ],
    )
    config["radial_basis"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    model = TensorModel(cartTACE(**config)).to(device).eval()
    filename = str(tmp_path / "cartesian.pt")
    export_tace(model, filename)
    loaded = load_tace(filename, device=device, dtype=dtype)
    assert isinstance(loaded.readout_fn, cartTACE)
    reference = model.to(dtype=dtype)
    data = dict(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.2, 0.1]], device=device, dtype=dtype
        ),
        node_attrs=torch.ones(2, 1, device=device, dtype=dtype),
        edge_index=torch.tensor([[0, 1], [1, 0]], device=device),
        edge_shifts=torch.zeros(2, 3, device=device, dtype=dtype),
        lattice=5 * torch.eye(3, device=device, dtype=dtype)[None],
        ptr=torch.tensor([0, 2], device=device),
        batch=torch.zeros(2, device=device, dtype=torch.long),
        fidelity_idx=torch.zeros(1, device=device, dtype=torch.long),
        pbc=torch.zeros(1, 3, device=device, dtype=torch.bool),
    )
    for empty in (False, True):
        if empty:
            data["edge_index"] = data["edge_index"][:, :0]
            data["edge_shifts"] = data["edge_shifts"][:0]
        actual, expected = loaded(dict(data)), reference(dict(data))
        for name in ("energy", "forces"):
            torch.testing.assert_close(actual[name], expected[name])
            assert torch.isfinite(actual[name]).all()
        if empty:
            torch.testing.assert_close(
                actual["forces"], torch.zeros_like(actual["forces"])
            )

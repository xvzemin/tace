"""Cartesian interaction conversion with a spherical product basis."""

from copy import deepcopy

import pytest
import torch

from tace.lightning import convert_cgtp, export_tace, load_tace
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.tace import e3nnTACE
from tace.models.adapter import TensorModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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

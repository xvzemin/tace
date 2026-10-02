"""Residual ECE models and fused force training."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3

from tace.models import TECE, TensorModel
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG


@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("parity", [False, True])
def test_tece_algorithms_and_force_training(asymmetric, parity, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=2,
        mmax=1,
        parity=parity,
        target_property=["energy", "forces", "stress", "virials"],
        statistics=[
            {
                "atomic_numbers": [1, 8],
                "avg_num_neighbors": 2.0,
                "atomic_energy": {1: 0.0, 8: 0.0},
            }
        ],
    )
    cfg["atomic_basis"].update(
        type="ece_o2",
        correlation=2,
        algorithm="recursive",
        use_asymmetric_contraction=asymmetric,
    )
    cfg["radial_basis"].update(num_radial_basis=3, hidden=[4], bias=True)
    cfg["readout_emlp"].update(hidden=[4], use_alllayer=True)
    cfg["scale_shift"]["enable"] = False
    cfg["node_embedding"]["type"] = "tensor"
    model = TensorModel(TECE(**cfg)).cuda().train()
    representation = model.readout_fn.representation
    assert not hasattr(representation, "products")
    for layer in representation.interactions:
        assert layer.source_weight.shape == (2, layer.contraction.weight_numel)
        if not parity:
            assert all(ir.m > 0 or ir.p == 1 for ir, _ in layer.contraction.irreps_in)
    data = dict(
        positions=torch.randn(3, 3, device="cuda"),
        node_attrs=torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device="cuda"),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]], device="cuda"
        ),
        edge_shifts=torch.zeros(6, 3, device="cuda"),
        lattice=torch.eye(3, device="cuda")[None] * 10,
        batch=torch.zeros(3, dtype=torch.long, device="cuda"),
        ptr=torch.tensor([0, 3], device="cuda"),
    )
    baseline = None
    state = {key: value.clone() for key, value in model.state_dict().items()}
    for algorithm, fused in [
        ("recursive", False),
        ("dense", False),
        ("recursive", True),
        ("dense", True),
    ]:
        representation.set_algorithm(algorithm)
        for layer in representation.interactions:
            layer.use_eqx = fused
        model.load_state_dict(state, strict=True)
        result = model({key: value.clone() for key, value in data.items()})
        loss = result["energy"].square().sum() + result["forces"].square().sum()
        loss = loss + result["stress"].square().sum()
        gradients = torch.autograd.grad(
            loss, tuple(model.parameters()), allow_unused=True
        )
        output = [
            result[key].detach() for key in ("energy", "forces", "stress", "virials")
        ]
        output.extend(
            torch.zeros_like(p) if grad is None else grad.detach()
            for p, grad in zip(model.parameters(), gradients)
        )
        if baseline is None:
            baseline = output
        else:
            for value, expected in zip(output, baseline):
                torch.testing.assert_close(value, expected, atol=1e-8, rtol=1e-8)
        assert all(torch.isfinite(value).all() for value in output)
    # Joint proper and improper transformations of positions.
    for determinant in (-1, 1):
        rotation = determinant * o3.rand_matrix(device="cuda")
        transformed = {key: value.clone() for key, value in data.items()}
        transformed["positions"] = data["positions"] @ rotation.T
        transformed["lattice"] = data["lattice"] @ rotation.T
        result = model(transformed)
        torch.testing.assert_close(result["energy"], baseline[0], atol=1e-8, rtol=1e-8)
        torch.testing.assert_close(
            result["forces"], baseline[1] @ rotation.T, atol=1e-8, rtol=1e-8
        )
    empty = {key: value.clone() for key, value in data.items()}
    empty["edge_index"] = data["edge_index"][:, :0]
    empty["edge_shifts"] = data["edge_shifts"][:0]
    result = model(empty)
    assert torch.isfinite(result["energy"]).all()
    torch.testing.assert_close(result["forces"], torch.zeros_like(data["positions"]))

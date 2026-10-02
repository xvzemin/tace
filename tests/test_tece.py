"""Residual ECE models and fused force training."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3

from eqx import o2
from tace.models import TECE, TensorModel
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.prod import CgtpACE


@pytest.mark.parametrize("asymmetric", [False, True])
@pytest.mark.parametrize("parity", [False, True])
@pytest.mark.parametrize("element_dependent", [False, True])
def test_tece_algorithms_and_force_training(
    asymmetric, parity, element_dependent, double_precision
):
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
        element_dependent=element_dependent,
        gate_m0=parity,
    )
    cfg["node_embedding"]["type"] = "wigner_tensor"
    cfg["radial_basis"].update(num_radial_basis=3, hidden=[4], bias=True)
    cfg["readout_emlp"].update(hidden=[4], use_alllayer=True)
    cfg["scale_shift"]["enable"] = False
    model = TensorModel(TECE(**cfg)).cuda().train()
    representation = model.readout_fn.representation
    assert not hasattr(representation, "products")
    assert len(representation.interactions) == cfg["num_layers"] - 1
    assert isinstance(representation.product, CgtpACE)
    assert len(representation.irreps_outs) == cfg["num_layers"]
    for layer in representation.interactions:
        assert layer.source_weight.shape == (2, layer.contraction.weight_numel)
        assert layer.target_weight.shape == layer.source_weight.shape
        assert layer.edge_info.mlp[-1].out_dim == layer.radial_linear.weight_numel
        if element_dependent:
            assert layer.radial_source_weight.shape == (2, layer.weight_numel)
        else:
            assert layer.radial_source_weight is None
            assert layer.radial_target_weight is None
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
    # A zero message leaves the only residual branch unchanged, including Gate.
    source, target = data["edge_index"]
    vectors = data["positions"][target] - data["positions"][source]
    radial, cutoff = representation.radial_basis(
        vectors.norm(dim=-1, keepdim=True) + 1e-9,
        data["node_attrs"],
        data["edge_index"],
        representation.atomic_numbers,
    )
    wigner, inverse = representation.wigner(vectors)
    features = torch.randn(
        3, representation.interactions[0].irreps_in.dim, device="cuda"
    )
    for layer in representation.interactions:
        result = layer(
            features,
            data["node_attrs"].argmax(-1),
            radial,
            data["edge_index"],
            wigner,
            inverse,
            torch.zeros_like(cutoff),
        )
        torch.testing.assert_close(result, features, atol=0, rtol=0)


@pytest.mark.parametrize("mmax", [0, 1, 3])
def test_tece_scalar_lift(mmax, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    vectors = torch.randn(7, 3, device="cuda", requires_grad=True)
    _, inverse = o2.WignerD(mmax, 3).cuda()(vectors)
    frame = o2.LocalFrame("2x0e+2x1o+2x2e+2x3o", mmax=0).cuda()
    actual = frame.to_global(torch.ones(7, 8, device="cuda"), inverse)
    expected = (
        o3.spherical_harmonics(
            list(range(4)), vectors, normalize=True, normalization="component"
        )
        .unsqueeze(-1)
        .expand(-1, -1, 2)
        .flatten(1)
    )
    for order in range(3):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        if order < 2:
            seed = torch.randn_like(actual)
            actual, expected = (
                torch.autograd.grad((y * seed).sum(), vectors, create_graph=True)[0]
                for y in (actual, expected)
            )


@pytest.mark.parametrize(
    "num_layers,embedding,nonlinear,gate_m0",
    [
        (1, "spherical_tensor", None, False),
        (1, "wigner_tensor", "gate", True),
        (3, "spherical_tensor_element2", ["gate", None, "gate"], False),
        (3, "wigner_tensor_element2", [None, "gate", None], True),
    ],
)
def test_tece_layer_layout(
    num_layers,
    embedding,
    nonlinear,
    gate_m0,
    monkeypatch,
    double_precision,
):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setenv("TACE_USE_EQX", "1")
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=num_layers,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        target_property=["energy", "forces"],
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=1.0, atomic_energy={1: 0.0})
        ],
    )
    cfg["node_embedding"]["type"] = embedding
    cfg["atomic_basis"].update(
        correlation=2,
        nonlinear=nonlinear,
        gate_m0=gate_m0,
        scalar_act="tanh",
        tensor_act="silu",
        use_asymmetric_contraction=False,
    )
    cfg["radial_basis"].update(num_radial_basis=3, hidden=[4])
    cfg["product_basis"]["correlation"] = 2
    cfg["readout_emlp"].update(hidden=[4], use_alllayer=True)
    cfg["scale_shift"]["enable"] = False
    model = TensorModel(TECE(**cfg)).cuda().train()
    rep = model.readout_fn.representation
    assert rep.node_embedding.edge_info.dims[1:-1] == cfg["radial_basis"]["hidden"]
    assert len(rep.interactions) + 1 == num_layers
    assert hasattr(rep.product, "eqx_ace")
    gates = [rep.embedding_gate, *(layer.gate for layer in rep.interactions)]
    names = nonlinear if isinstance(nonlinear, list) else [nonlinear] * num_layers
    for gate, name in zip(gates, names):
        assert isinstance(gate, torch.nn.Identity) == (name is None)
    data = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [0.5, 0.8, 0.3]], device="cuda"),
        node_attrs=torch.ones(2, 1, device="cuda"),
        edge_index=torch.tensor([[0, 1], [1, 0]], device="cuda"),
        edge_shifts=torch.zeros(2, 3, device="cuda"),
        lattice=torch.eye(3, device="cuda")[None] * 10,
        batch=torch.zeros(2, dtype=torch.long, device="cuda"),
        ptr=torch.tensor([0, 2], device="cuda"),
    )
    result = model(data)
    loss = result["energy"].square().sum() + result["forces"].square().sum()
    gradients = torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True)
    assert all(torch.isfinite(g).all() for g in gradients if g is not None)
    # The last ACE has exactly one projected skip, even with EQX enabled.
    features = torch.randn(2, rep.product.irreps_in.dim, device="cuda")
    skip = rep.product_skip(features)
    base = rep.product(features, data["node_attrs"], None, data["batch"])
    actual = rep.product(features, data["node_attrs"], skip, data["batch"])
    torch.testing.assert_close(actual, base + skip, atol=1e-12, rtol=1e-12)
    cfg["node_embedding"]["type"] = "linear"
    with pytest.raises(ValueError, match="tensor node embedding"):
        TECE(**cfg)

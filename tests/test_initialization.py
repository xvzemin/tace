"""Second-moment initialization and indexed feature projections."""

from copy import deepcopy
from functools import partial

import pytest
import torch

from tace.models._e3nn import readout
from tace.models._e3nn.base import EdgeUpdate, NodeEmbedding
from tace.models._e3nn.edge import Element2EdgeUpdateV2
from tace.models._e3nn.node import LinearNodeEmbeddingV2
from tace.models.linear import mlpLinear
from tace.models.mlp import MLP, MLPV2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.usefixtures("double_precision")


@pytest.mark.parametrize("mlp_type", ["mlp", "mlp_v2"])
@pytest.mark.parametrize("norm", [None, "layer_norm", "rms_norm"])
@pytest.mark.parametrize(
    "embedding", ["linear_v2", "spherical_tensor_element2", "wigner_tensor_element2"]
)
def test_radial_mlp_configuration(mlp_type, norm, embedding):
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
    from tace.models._e3nn.tace import e3nnTACE

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=1,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
        statistics=[
            dict(
                atomic_numbers=[1, 8],
                avg_num_neighbors=2.0,
                atomic_energy={1: 0.0, 8: 0.0},
            )
        ],
    )
    config["node_embedding"]["type"] = embedding
    config["edge_update"]["type"] = "element2_v2"
    config["radial_basis"].update(
        mlp_type=mlp_type,
        mlp_norm=norm,
        hidden=[4],
        apply_cutoff=False,
    )
    config["atomic_basis"]["scatter_norm"] = "density"
    config["scale_shift"]["enable"] = False
    representation = e3nnTACE(**config).to(DEVICE).representation
    networks = [representation.interactions[0].edge_info]
    if hasattr(representation.node_embedding, "edge_info"):
        networks.append(representation.node_embedding.edge_info)
    expected_norm = norm or ("layer_norm" if mlp_type == "mlp" else None)
    for network in networks:
        assert type(network) is (MLP if mlp_type == "mlp" else MLPV2)
        assert any(isinstance(m, torch.nn.LayerNorm) for m in network.mlp) == (
            expected_norm == "layer_norm"
        )
        assert any(isinstance(m, torch.nn.RMSNorm) for m in network.mlp) == (
            expected_norm == "rms_norm"
        )
    density = representation.interactions[0].scatter_norm.edge_density
    assert type(density) is MLP
    assert any(isinstance(m, torch.nn.LayerNorm) for m in density.mlp)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"apply_cutoff": True}, "apply_cutoff=false"),
        ({"mlp_type": "unknown"}, "mlp_type"),
        ({"mlp_norm": "unknown"}, "mlp_norm"),
        ({"mlp_norm": "none"}, "mlp_norm"),
    ],
)
def test_v2_configuration_errors(options, match):
    from tace.models._e3nn.default import check_model_config

    with pytest.raises(ValueError, match=match):
        check_model_config(
            {
                "edge_update": {"type": "element2_v2"},
                "radial_basis": {"apply_cutoff": False, **options},
            }
        )


@pytest.mark.parametrize(
    "normalization", [{}, {"layer_norm": True}, {"rms_norm": True}]
)
def test_mlp_v2_second_moment(normalization):
    model = MLPV2([256] * 5, **normalization).to(DEVICE)
    x = torch.randn(4096, 256, device=DEVICE)
    with torch.no_grad():
        for layer in model.mlp:
            x = layer(x)
            if isinstance(layer, mlpLinear):
                assert layer.alpha == 256**-0.5
                assert 0.7 < x.square().mean().item() < 1.3


def test_mlp_v2_activation_scaled_once():
    plain = MLPV2([8, 16, 4], act="silu").to(DEVICE)
    scaled = MLPV2([8, 16, 4], act="scaled_silu").to(DEVICE)
    scaled.load_state_dict(plain.state_dict())
    x = torch.randn(12, 8, device=DEVICE, requires_grad=True)
    torch.testing.assert_close(plain(x), scaled(x), atol=0, rtol=0)
    with pytest.raises(ValueError, match="mutually exclusive"):
        MLPV2([8, 4], layer_norm=True, rms_norm=True)


@pytest.mark.parametrize("num_elements", [4, 80])
def test_embedding_v2_second_moment(num_elements):
    model = LinearNodeEmbeddingV2(
        num_elements=num_elements,
        num_radial_basis=4,
        num_channel=1024,
        Lmax=0,
        lmax=0,
        avg_num_neighbors=1.0,
    ).to(DEVICE)
    assert isinstance(model, NodeEmbedding)
    indices = torch.arange(num_elements, device=DEVICE)
    attrs = torch.nn.functional.one_hot(indices, num_elements).to(
        model.embedding.weight
    )
    radial = torch.empty(0, 4, device=DEVICE)
    edges = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    angular = torch.empty(0, 1, device=DEVICE)
    args = (attrs, radial, edges, angular, None, None, None)
    output = model(*args)
    torch.testing.assert_close(output, model(*args, node_type=indices), atol=0, rtol=0)
    assert 0.9 < output.square().mean().item() < 1.1
    torch.testing.assert_close(output, model.embedding.weight, atol=0, rtol=0)
    output.sum().backward()
    torch.testing.assert_close(
        model.embedding.weight.grad, torch.ones_like(model.embedding.weight)
    )


@pytest.mark.parametrize("zero_elements", [False, True])
def test_element2_v2_projection_and_derivatives(zero_elements):
    kwargs = dict(
        layer=0,
        num_layers=2,
        num_elements=3,
        num_radial_basis=4,
        num_channel=6,
        edge_embedding_channel=4,
    )
    with pytest.raises(TypeError, match="element_std"):
        Element2EdgeUpdateV2(**kwargs, element_std=0.0)
    update = Element2EdgeUpdateV2(**kwargs).to(DEVICE)
    if zero_elements:
        torch.nn.init.zeros_(update.source_embedding.weight)
        torch.nn.init.zeros_(update.target_embedding.weight)
    assert isinstance(update, EdgeUpdate)
    model = MLPV2([update.out_dim, 12, 3], bias=True).to(DEVICE)
    types = torch.tensor([0, 1, 2, 0], device=DEVICE)
    edges = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], device=DEVICE)
    radial = torch.randn(5, 4, device=DEVICE, requires_grad=True)
    attrs = torch.nn.functional.one_hot(types, 3).to(radial)
    node_feats = torch.randn(4, 6, device=DEVICE)
    parts = update(node_feats, attrs, radial, edges, None)
    explicit = update(node_feats, attrs, radial, edges, None, node_type=types)
    torch.testing.assert_close(parts, explicit, atol=0, rtol=0)
    dense = torch.cat([x if index is None else x[index] for x, index in parts], dim=-1)
    actual, expected = model(parts), model(dense)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)

    inputs = (radial, *update.parameters(), *model.parameters())
    values = [actual.square().sum(), expected.square().sum()]
    for _ in range(2):
        gradients = [
            torch.autograd.grad(value, inputs, create_graph=True, retain_graph=True)
            for value in values
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        values = [sum(g.square().sum() for g in grads) for grads in gradients]

    if zero_elements:
        radial_model = MLPV2([4, 12, 3], bias=True).to(DEVICE)
        with torch.no_grad():
            radial_model.mlp[0].weight.copy_(model.mlp[0].weight[:4])
            radial_model.mlp[0].bias.copy_(model.mlp[0].bias)
        radial_model.mlp[2].load_state_dict(model.mlp[2].state_dict())
        torch.testing.assert_close(actual, radial_model(radial), atol=1e-12, rtol=1e-12)
        for grad in gradients[0][1:3]:
            assert grad.abs().sum() > 0


@pytest.mark.parametrize("num_elements,num_channel", [(4, 16), (80, 16), (80, 128)])
def test_element2_v2_second_moment_without_mlp(num_elements, num_channel):
    update = Element2EdgeUpdateV2(
        layer=0,
        num_layers=1,
        num_elements=num_elements,
        num_radial_basis=8,
        num_channel=num_channel,
        edge_embedding_channel=8,
    ).to(DEVICE)
    types = torch.arange(num_elements, device=DEVICE)
    attrs = torch.nn.functional.one_hot(types, num_elements).to(
        torch.get_default_dtype()
    )
    edges = torch.randint(num_elements, (2, 4096), device=DEVICE)
    radial = torch.randn(4096, 8, device=DEVICE)
    nodes = torch.empty(num_elements, num_channel, device=DEVICE)
    linear = torch.nn.Linear(update.out_dim, 256, bias=False).to(DEVICE)
    torch.nn.init.normal_(linear.weight, std=update.out_dim**-0.5)

    with torch.no_grad():
        parts = update(nodes, attrs, radial, edges, None)
        dense = torch.cat(
            [x if index is None else x[index] for x, index in parts], dim=-1
        )
        # The edge module sets the input scale; no MLP-specific initialization.
        radial_moment = radial.square().mean()
        expected = radial_moment + sum(
            x[index].square().sum(dim=-1).mean() / update.out_dim
            for x, index in parts[1:]
        )
        torch.testing.assert_close(dense.square().mean(), expected)
        assert abs(dense.square().mean().item() / radial_moment.item() - 1) < 1e-4
        assert abs(linear(dense).square().mean().item() / expected.item() - 1) < 0.15


@pytest.mark.parametrize("matrix", ["0", "1"])
@pytest.mark.parametrize(
    "cls,irrep",
    [
        (readout.ScalarReadOut, "0e"),
        (readout.ElementScalarReadOut, "0e"),
        (readout.TensorReadOut, "1o"),
        (readout.ElementTensorReadOut, "1o"),
    ],
)
def test_readout_head_initialization_and_loading(monkeypatch, cls, irrep, matrix):
    for name in ("e3nnLinear", "e3nnElementLinear"):
        monkeypatch.setattr(
            readout, name, partial(getattr(readout, name), use_matrix_weight=matrix)
        )
    kwargs = dict(
        layer=0,
        num_layers=1,
        hidden_channel=[128, 128],
        bias=True,
        num_elements=3,
        num_fidelities=4,
        parity=True,
        irreps_in="128x0e+128x1o",
        irreps_out=f"1x{irrep}",
    )
    model = cls(**kwargs).to(DEVICE)
    for index, linear in enumerate(model.linear2):
        weights = linear.weight
        if isinstance(weights, torch.nn.ParameterList):
            weights = torch.cat([w.flatten() for w in weights])
        expected = 1 if index == 0 else 4
        assert abs(weights.square().mean().item() / expected - 1) < 0.15

    # Arbitrary saved parameters must not be rescaled at load time.
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.3)
    restored = cls(**kwargs).to(DEVICE)
    restored.load_state_dict(model.state_dict(), strict=True)
    x = torch.randn(4, 512, device=DEVICE, requires_grad=True)
    fidelity = torch.arange(4, device=DEVICE)
    types = fidelity % 3
    attrs = torch.nn.functional.one_hot(types, 3).to(x)
    outputs = [module(x, fidelity, attrs, types) for module in (model, restored)]
    torch.testing.assert_close(*outputs, atol=0, rtol=0)
    gradients = [torch.autograd.grad(y.square().sum(), x)[0] for y in outputs]
    torch.testing.assert_close(*gradients, atol=0, rtol=0)

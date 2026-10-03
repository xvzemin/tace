"""Scatter normalization values, derivatives, and checkpoint migration."""

from copy import deepcopy

import pytest
import torch

from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.scatter_norm import (
    AvgNumNeighborsScatterNorm,
    DensityScatterNorm,
    IdentityScatterNorm,
    NoCutoffDensityScatterNorm,
    get_scatter_norm_layer,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.mark.parametrize(
    "key",
    [
        None,
        "identity",
        "avg_num_neighbors",
        "sqrt_avg_num_neighbors",
        "density",
        "no_cutoff_density",
    ],
)
@pytest.mark.parametrize("shape", [(4, 4, 7), (2, 4, 7), (4, 4, 0), (0, 0, 0)])
@pytest.mark.parametrize("use_cutoff", [False, True])
def test_scatter_norm_values_and_derivatives(double_precision, key, shape, use_cutoff):
    nlocal, num_nodes, num_edges = shape
    module = get_scatter_norm_layer(
        key, 9.0, 4, radial_bias=True, radial_layer_norm=True
    ).to(DEVICE)
    node_feats = torch.randn(nlocal, 3, device=DEVICE, requires_grad=True)
    edge_feats = torch.randn(num_edges, 4, device=DEVICE, requires_grad=True)
    edge_index = (
        torch.randint(num_nodes, (2, num_edges), device=DEVICE)
        if num_nodes
        else torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    )
    cutoff = torch.rand(num_edges, 1, device=DEVICE, requires_grad=True)
    edge_cutoff = cutoff if use_cutoff else None
    if isinstance(module, DensityScatterNorm):
        with torch.no_grad():
            module.alpha.fill_(1.7)
            module.beta.fill_(0.3)
        density = torch.tanh(module.edge_density(edge_feats) ** 2)
        if use_cutoff and key == "density":
            density = density * cutoff
        density = density.new_zeros(num_nodes, 1).index_add(0, edge_index[1], density)
        expected = node_feats / (density[:nlocal] * module.beta + module.alpha)
    elif key == "avg_num_neighbors":
        expected = node_feats / 9.0
    elif key == "sqrt_avg_num_neighbors":
        expected = node_feats / 3.0
    else:
        expected = node_feats
    actual = module(node_feats, edge_feats, edge_index, edge_cutoff, num_nodes)
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
    inputs = (node_feats, edge_feats, cutoff, *module.parameters())
    values = [actual.square().sum(), expected.square().sum()]
    for _ in range(2):
        gradients = [
            torch.autograd.grad(
                value, inputs, create_graph=True, retain_graph=True, allow_unused=True
            )
            for value in values
        ]
        for first, second in zip(*gradients):
            assert (first is None) == (second is None)
            if first is not None:
                torch.testing.assert_close(first, second, atol=1e-11, rtol=1e-11)
        values = [
            sum(g.square().sum() for g in grads if g is not None) for grads in gradients
        ]


@pytest.mark.parametrize("key", ["density", "no_cutoff_density"])
def test_density_accepts_indexed_features(double_precision, key):
    module = get_scatter_norm_layer(key, 4.0, 7, radial_layer_norm=True).to(DEVICE)
    with torch.no_grad():
        module.beta.fill_(0.3)
    nodes = torch.randn(4, 2, device=DEVICE)
    edges = torch.tensor([[0, 1, 2, 3], [1, 0, 3, 2]], device=DEVICE)
    radial = torch.randn(4, 3, device=DEVICE)
    indexed = ((radial, None), (nodes, edges[0]), (nodes, edges[1]))
    dense = torch.cat((radial, nodes[edges[0]], nodes[edges[1]]), dim=-1)
    cutoff = torch.rand(4, 1, device=DEVICE)
    torch.testing.assert_close(
        module(nodes, indexed, edges, cutoff, 4),
        module(nodes, dense, edges, cutoff, 4),
    )
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(
        compiled(nodes, indexed, edges, cutoff, 4),
        module(nodes, indexed, edges, cutoff, 4),
    )


@pytest.fixture
def small_model():
    def build(key, attention=False):
        config = deepcopy(DEFAULT_MODEL_CONFIG)
        config.update(
            num_layers=2,
            num_channel=2,
            Lmax=1,
            lmax=1,
            mmax=1,
            cutoff=4.0,
            max_neighbors=None,
        )
        config["atomic_basis"].update(
            type="o2" if attention else "cgtp",
            scatter_norm=key,
            use_radial_rotary_attention=attention,
            num_head=1,
        )
        config["edge_update"]["type"] = "element2"
        config["radial_basis"].update(hidden=[4], bias=True, apply_cutoff=False)
        config["readout_emlp"]["hidden"] = [4]
        config["scale_shift"]["enable"] = False
        statistics = [
            {"atomic_numbers": [1], "atomic_energy": {1: 0.0}, "avg_num_neighbors": 2.0}
        ]
        return create_model(config, statistics, ["energy", "forces"], []).to(DEVICE)

    return build


@pytest.mark.parametrize(
    "key, norm_class",
    [
        (None, IdentityScatterNorm),
        ("avg_num_neighbors", AvgNumNeighborsScatterNorm),
        ("density", DensityScatterNorm),
        ("no_cutoff_density", NoCutoffDensityScatterNorm),
    ],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_legacy_scatter_norm_loading(tmp_path, small_model, key, norm_class, dtype):
    torch.set_default_dtype(dtype)
    reference = small_model(key).eval()
    for layer in reference.readout_fn.representation.interactions:
        if isinstance(layer.scatter_norm, DensityScatterNorm):
            with torch.no_grad():
                layer.scatter_norm.beta.fill_(0.3)
    path = tmp_path / "model.pt"
    export_tace(reference, str(path))
    checkpoint = torch.load(path, weights_only=False)
    # Density parameters previously belonged directly to the interaction.
    checkpoint["state_dict"] = {
        name.replace(".scatter_norm.", "."): value
        for name, value in checkpoint["state_dict"].items()
    }
    torch.save(checkpoint, path)

    loaded = load_tace(path, device=DEVICE, dtype=dtype, strict=True).eval()
    expected = reference.state_dict()
    assert loaded.state_dict().keys() == expected.keys()
    for name, value in loaded.state_dict().items():
        torch.testing.assert_close(value, expected[name], atol=0, rtol=0)
    for layer in loaded.readout_fn.representation.interactions:
        assert type(layer.scatter_norm) is norm_class
        assert not hasattr(layer, "alpha") and not hasattr(layer, "edge_density")

    data = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.3]], device=DEVICE),
        node_attrs=torch.ones(2, 1, device=DEVICE),
        edge_index=torch.tensor([[0, 1], [1, 0]], device=DEVICE),
        edge_shifts=torch.zeros(2, 3, device=DEVICE),
        lattice=torch.eye(3, device=DEVICE).unsqueeze(0) * 8,
        batch=torch.zeros(2, dtype=torch.long, device=DEVICE),
        ptr=torch.tensor([0, 2], device=DEVICE),
        fidelity_idx=torch.zeros(1, dtype=torch.long, device=DEVICE),
    )
    outputs = [
        model({k: v.clone() for k, v in data.items()}) for model in (reference, loaded)
    ]
    for name in ("energy", "forces"):
        torch.testing.assert_close(outputs[0][name], outputs[1][name])


def test_attention_owns_normalization(small_model):
    model = small_model("density", attention=True)
    uses_attention = False
    for layer in model.readout_fn.representation.interactions:
        if layer.rejector.attention is not None:
            uses_attention = True
            assert isinstance(layer.scatter_norm, IdentityScatterNorm)
            assert not any("edge_density" in name for name in layer.state_dict())
        else:
            assert isinstance(layer.scatter_norm, DensityScatterNorm)
    assert uses_attention


def test_unknown_scatter_norm(small_model):
    with pytest.raises(RuntimeError) as exc:
        small_model("unknown")
    assert "Unknown scatter normalization" in str(exc.value.__cause__)

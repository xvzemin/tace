"""Feature embeddings, readouts, initialization, and scatter normalization."""

from copy import deepcopy
from functools import partial
from unittest.mock import patch

import pytest
import torch
from e3nn import o3

from eqx import o2
from tace.lightning import create_model, export_tace, load_tace
from tace.models._e3nn import readout
from tace.models._e3nn.base import (
    EdgeInteraction,
    EdgeUpdate,
    NodeEmbedding,
)
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG, check_model_config
from tace.models._e3nn.edge import Element2EdgeUpdateV2
from tace.models._e3nn.fused import UvO2TensorProduct
from tace.models._e3nn.node import NODE_EMBEDDING, NODE_UPDATE, LinearNodeEmbedding
from tace.models._e3nn.readout import EdgeReadOut
from tace.models._e3nn.representation import PairRepresentation
from tace.models._e3nn.scatter_norm import (
    AvgNumNeighborsScatterNorm,
    DensityScatterNorm,
    IdentityScatterNorm,
    NoCutoffDensityScatterNorm,
    get_scatter_norm_layer,
)
from tace.models.layout import LayoutTransform
from tace.models.linear import (
    e3nnElementLinear,
    e3nnLinear,
    e3nnMoEElementLinear,
    enable_lora,
    mlpLinear,
)
from tace.models.mlp import MLP, MLPV2
from tace.utils.env import EQX_KERNELS
from tace.utils.utils import torch_default_dtype

DTYPE = torch.float64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pytestmark = pytest.mark.usefixtures("double_precision")


@pytest.mark.parametrize("mlp_type", ["mlp_v1", "mlp_v2"])
@pytest.mark.parametrize("norm", [None, "layer_norm", "rms_norm"])
@pytest.mark.parametrize(
    "embedding", ["linear", "spherical_tensor_element2", "wigner_tensor_element2"]
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
    config["mlp_type"] = mlp_type
    config["radial_basis"].update(
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
    expected_norm = norm or ("layer_norm" if mlp_type == "mlp_v1" else None)
    for network in networks:
        assert type(network) is MLP[mlp_type]
        assert any(isinstance(m, torch.nn.LayerNorm) for m in network.mlp) == (
            expected_norm == "layer_norm"
        )
        assert any(isinstance(m, torch.nn.RMSNorm) for m in network.mlp) == (
            expected_norm == "rms_norm"
        )
    density = representation.interactions[0].scatter_norm.edge_density
    assert type(density) is MLP[mlp_type]
    assert any(isinstance(m, torch.nn.LayerNorm) for m in density.mlp)


@pytest.mark.parametrize("mlp_type", ["mlp_v1", "mlp_v2"])
def test_magnetic_mlp_configuration(mlp_type):
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
                atomic_numbers=[26],
                avg_num_neighbors=2.0,
                atomic_energy={26: 0.0},
            )
        ],
    )
    config["fidelity"]["magnetic_scale"] = 2.0
    config["atomic_basis"].update(type="o2_mag", use_radial_rotary_attention=False)
    config["angular_basis"].update(magnetic_Lmax=1, use_spin_orbit_coupling=False)
    config["mlp_type"] = mlp_type
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["scale_shift"]["enable"] = False
    interaction = e3nnTACE(**config).to(DEVICE).representation.interactions[0]
    networks = [interaction.source_magnetic_info, interaction.target_magnetic_info]
    for network in networks:
        assert type(network) is MLP[mlp_type]
        assert not any(
            isinstance(layer, (torch.nn.LayerNorm, torch.nn.RMSNorm))
            for layer in network.mlp
        )
    inputs = tuple(
        torch.randn(3, network.dims[0], device=DEVICE, requires_grad=True)
        for network in networks
    )
    edges = torch.tensor([[0, 1, 2], [1, 2, 0]], device=DEVICE)
    weights = interaction._magnetic_weights(inputs, edges)
    assert weights.shape == (3, interaction.magnetic_linear.weight_numel)
    parameters = tuple(
        parameter for network in networks for parameter in network.parameters()
    )
    gradients = torch.autograd.grad(weights.square().sum(), (*inputs, *parameters))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize("mlp_type", ["mlp_v1", "mlp_v2"])
def test_universal_invariant_mlp_configuration(mlp_type):
    from tace.models._e3nn.ue import UniversalInvariantEmbedding

    module = UniversalInvariantEmbedding(
        4,
        {"energy": {}, "fidelity_idx": {"num_embeddings": 2}},
        bias=True,
        mlp_type=mlp_type,
    ).to(DEVICE)
    assert type(module.uie["energy"]) is MLP[mlp_type]
    assert type(module.project) is MLP[mlp_type]
    assert isinstance(module.uie["fidelity_idx"], torch.nn.Embedding)
    energy = torch.randn(2, device=DEVICE, requires_grad=True)
    batch = torch.tensor([0, 0, 1], device=DEVICE)
    fidelity = torch.tensor([0, 1], device=DEVICE)
    actual = module({"batch": batch, "energy": energy, "fidelity_idx": fidelity})
    expected = module.act(
        module.project(
            torch.cat(
                (
                    module.uie["energy"](energy[batch].unsqueeze(-1)),
                    module.uie["fidelity_idx"](fidelity[batch]),
                ),
                dim=-1,
            )
        )
    ).unsqueeze(1)
    assert actual.shape == (3, 1, 4)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(
        actual.square().sum(), (energy, *module.parameters())
    )
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


@pytest.mark.parametrize(
    "options,match",
    [
        ({"apply_cutoff": True}, "apply_cutoff=false"),
        ({"mlp_type": "mlp_v2"}, "model.config.mlp_type"),
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


@pytest.mark.parametrize("mlp_type", ["unknown", "mlp", "v1", "v2", None])
def test_unknown_model_mlp_type(mlp_type):
    from tace.models._e3nn.default import check_model_config

    with pytest.raises(ValueError, match="mlp_type"):
        check_model_config({"mlp_type": mlp_type})


def test_model_mlp_type_defaults():
    from tace.models._e3nn.default import check_model_config

    assert set(MLP) == {"mlp_v1", "mlp_v2"}
    assert DEFAULT_MODEL_CONFIG["mlp_type"] == "mlp_v1"
    assert "mlp_type" not in DEFAULT_MODEL_CONFIG["radial_basis"]
    config = check_model_config(
        {
            "statistics": [
                dict(
                    atomic_numbers=[1],
                    avg_num_neighbors=2.0,
                    atomic_energy={1: 0.0},
                )
            ]
        }
    )
    assert config["mlp_type"] == "mlp_v1"


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
def test_embedding_second_moment(num_elements):
    model = LinearNodeEmbedding(
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


@pytest.mark.parametrize("matrix", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_embedding_legacy_state_dict(matrix, dtype):
    model = LinearNodeEmbedding(
        num_elements=7,
        num_radial_basis=4,
        num_channel=5,
        Lmax=0,
        lmax=0,
        avg_num_neighbors=1.0,
    ).to(device=DEVICE, dtype=dtype)
    legacy = e3nnLinear(
        "7x0e", "5x0e", bias=False, use_matrix_weight="1" if matrix else "0"
    ).to(device=DEVICE, dtype=torch.float32)
    state = {"elem_emb1." + k: v.clone() for k, v in legacy.state_dict().items()}
    original = {k: v.clone() for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    indices = torch.tensor([6, 0, 2, 2, 4], device=DEVICE)
    attrs = torch.nn.functional.one_hot(indices, 7).to(dtype=dtype)
    expected = legacy.to(dtype=dtype)(attrs)
    actual = model(attrs, None, None, None, None, None, None, node_type=indices)
    torch.testing.assert_close(actual, expected)
    assert state.keys() == original.keys()
    for key in state:
        torch.testing.assert_close(state[key], original[key], atol=0, rtol=0)
    model.load_state_dict(model.state_dict(), strict=True)
    torch.testing.assert_close(model.embedding(indices), actual, atol=0, rtol=0)


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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interaction_type", ["cgtp", "o2"])
@pytest.mark.parametrize("mlp_type", ["mlp_v1", "mlp_v2"])
@pytest.mark.parametrize("norm", [None, "layer_norm", "rms_norm"])
def test_edge_representation_readout(dtype, interaction_type, mlp_type, norm):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    irreps = o3.Irreps("2x0e+2x0o+2x1o+2x1e+2x2e+2x3o")
    radial = deepcopy(DEFAULT_MODEL_CONFIG["radial_basis"])
    radial.update(hidden=[4], mlp_norm=norm)
    with torch_default_dtype(dtype):
        descriptor = PairRepresentation(
            irreps,
            o3.Irreps.spherical_harmonics(3),
            irreps,
            2,
            [1, 8],
            4.0,
            radial,
            interaction_type=interaction_type,
            mlp_type=mlp_type,
        ).to(device=device, dtype=dtype)
        readouts = torch.nn.ModuleList(
            EdgeReadOut(descriptor.irreps_out, irreps, output)
            for output in ("0e+0o", "2x1o", "0e+1e+2e")
        ).to(device=device, dtype=dtype)

    assert descriptor.num_layers == len(descriptor.interactions) == 1
    for head in readouts:
        assert head.nonlinearity.irreps_scalars == irreps.filter(lmax=0)
        assert all(ir.l > 0 for _, ir in head.nonlinearity.irreps_gated)
    interaction = descriptor.interactions[0]
    assert isinstance(interaction, EdgeInteraction)
    assert type(interaction.edge_info) is MLP[mlp_type]
    assert not hasattr(interaction, "angular_basis")
    assert not hasattr(interaction, "wigner")
    layers = interaction.edge_info.mlp
    assert any(isinstance(layer, torch.nn.LayerNorm) for layer in layers) == (
        norm in (None, "layer_norm")
    )
    assert any(isinstance(layer, torch.nn.RMSNorm) for layer in layers) == (
        norm == "rms_norm"
    )
    x = torch.randn(3, irreps.dim, dtype=dtype, device=device, requires_grad=True)
    positions = torch.randn(3, 3, dtype=dtype, device=device, requires_grad=True)
    attrs = x.new_tensor([[1, 0], [0, 1], [1, 0]])
    edge_index = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], device=device)
    shifts = x.new_zeros((4, 3))
    lattice = x.new_zeros((1, 3, 3))
    batch = torch.zeros(3, dtype=torch.long, device=device)
    features = descriptor(x, attrs, positions, edge_index, shifts, lattice, batch)
    assert features.shape == (4, descriptor.irreps_out.dim)
    if interaction_type == "o2":
        assert isinstance(interaction.linear_up, e3nnLinear)
        assert interaction.linear_up.irreps_in == irreps
        assert isinstance(interaction.rejector, UvO2TensorProduct)
        linear_up = interaction.rejector.linear_up
        gate = interaction.rejector.gate
        linear_down = interaction.rejector.linear_down
        assert isinstance(linear_up, o2.Linear)
        assert isinstance(gate, o2.Gate)
        assert isinstance(linear_down, o2.Linear)
        assert linear_up.irreps_out == gate.irreps_in
        assert linear_down.irreps_in == gate.irreps_out
        assert not torch.allclose(
            descriptor(2 * x, attrs, positions, edge_index, shifts, lattice, batch),
            2 * features,
        )
    if interaction_type == "cgtp":
        source, target = edge_index
        vectors = positions[target] - positions[source]
        edge_feats, cutoff = descriptor.radial_basis(
            vectors.norm(dim=-1, keepdim=True),
            attrs,
            edge_index,
            torch.tensor(descriptor.atomic_numbers, dtype=torch.int64, device=device),
        )
        weights = interaction.edge_info(
            torch.cat((edge_feats, attrs[source], attrs[target]), dim=-1)
        )
        expected = interaction.rejector(
            interaction.linear_up(torch.cat((x[source], x[target]), dim=-1)),
            o3.spherical_harmonics(
                interaction.irreps_sh,
                vectors,
                normalize=True,
                normalization="component",
            ),
            weights * cutoff,
        )
        torch.testing.assert_close(features, expected, atol=0, rtol=0)
    torch.testing.assert_close(
        descriptor(x, attrs, positions + 1, edge_index, shifts, lattice, batch),
        features,
    )
    outputs = [readout(features) for readout in readouts]

    atol = 2e-5 if dtype == torch.float32 else 1e-8
    for sign in (1, -1):
        rotation = sign * o3.rand_matrix(dtype=torch.float64)
        with torch_default_dtype(torch.float64):
            d_in = irreps.D_from_matrix(rotation).to(device=device, dtype=dtype)
            d_out = [
                readout.irreps_out.D_from_matrix(rotation).to(
                    device=device, dtype=dtype
                )
                for readout in readouts
            ]
        transformed = descriptor(
            x @ d_in.T,
            attrs,
            positions @ rotation.to(device=device, dtype=dtype).T,
            edge_index,
            shifts,
            lattice,
            batch,
        )
        for head, output, d in zip(readouts, outputs, d_out):
            torch.testing.assert_close(
                head(transformed), output @ d.T, atol=atol, rtol=atol
            )

    first = torch.autograd.grad(
        sum(output.square().sum() for output in outputs),
        (x, positions),
        create_graph=True,
    )
    second = torch.autograd.grad(
        sum(grad.square().sum() for grad in first),
        (x, positions, *descriptor.parameters(), *readouts.parameters()),
    )
    assert all(torch.isfinite(grad).all() for grad in first + second)

    empty = descriptor(
        x, attrs, positions, edge_index[:, :0], shifts[:0], lattice, batch
    )
    for head in readouts:
        assert head(empty).shape == (0, head.irreps_out.dim)
        torch.testing.assert_close(
            head(torch.zeros_like(features)), torch.zeros_like(head(features))
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_uv_o2_tensor_product(dtype):
    with torch_default_dtype(dtype):
        product = UvO2TensorProduct(
            "2x0e+3x0o+2x1o+1x1e+2x2e",
            "2x0e+1x0o+3x1o+2x1e+1x3o",
        ).to(device=DEVICE, dtype=dtype)
        wigner = o2.WignerD(3, 3, method="recursive").to(device=DEVICE, dtype=dtype)
    x = torch.randn(4, product.irreps_in.dim, device=DEVICE, dtype=dtype)
    vectors = torch.randn(4, 3, device=DEVICE, dtype=dtype)
    weights = torch.randn(4, product.weight_numel, device=DEVICE, dtype=dtype)
    packed = wigner.forward_packed(vectors)
    dense, _ = wigner(vectors)
    expected = product(x, packed, weights)
    torch.testing.assert_close(product(x, dense, weights), expected)
    torch.testing.assert_close(
        product(x, packed, torch.zeros_like(weights)), torch.zeros_like(expected)
    )
    assert product(x[:0], packed[:0], weights[:0]).shape == (0, product.irreps_out.dim)
    with pytest.raises(ValueError, match="radial weights"):
        product(x, packed, weights[:, :-1])


@pytest.mark.parametrize("interaction_type", ["cgtp", "o2"])
def test_pair_representation_bases(interaction_type):
    irreps = o3.Irreps("2x0e+2x1o+2x2e")
    radial = deepcopy(DEFAULT_MODEL_CONFIG["radial_basis"])
    radial["hidden"] = [4]
    descriptor = PairRepresentation(
        irreps,
        o3.Irreps.spherical_harmonics(2),
        irreps,
        2,
        [1, 8],
        4.0,
        radial,
        interaction_type=interaction_type,
        mlp_type="mlp_v2",
    ).to(DEVICE)
    assert descriptor.num_layers == len(descriptor.interactions) == 1
    assert descriptor.atomic_numbers == [1, 8]
    assert "atomic_numbers" not in dict(descriptor.named_buffers())
    assert "atomic_numbers" not in descriptor.state_dict()
    x = torch.randn(3, irreps.dim, device=DEVICE, requires_grad=True)
    positions = torch.randn(3, 3, device=DEVICE, requires_grad=True)
    attrs = x.new_tensor([[1, 0], [0, 1], [1, 0]])
    edges = torch.tensor([[0, 1, 0, 2], [1, 0, 2, 0]], device=DEVICE)
    shifts = x.new_tensor([[0, 0, 0], [0, 0, 0], [1, 0, 0], [-1, 0, 0]])
    lattice = torch.eye(3, device=DEVICE).unsqueeze(0).requires_grad_()
    batch = torch.zeros(3, dtype=torch.long, device=DEVICE)
    calls = []
    linear_inputs = []
    hooks = [
        layer.register_forward_hook(
            lambda module, inputs, output: calls.append((inputs, output))
        )
        for layer in descriptor.interactions
    ]
    hooks.append(
        descriptor.interactions[0].linear_up.register_forward_pre_hook(
            lambda module, inputs: linear_inputs.append(inputs[0])
        )
    )
    angular_basis = (
        descriptor.o3_angular_basis
        if interaction_type == "cgtp"
        else descriptor.o2_angular_basis
    )
    method = "forward" if interaction_type == "cgtp" else "forward_packed"
    with (
        patch.object(
            angular_basis, method, wraps=getattr(angular_basis, method)
        ) as angular,
        patch.object(
            descriptor.radial_basis, "forward", wraps=descriptor.radial_basis.forward
        ) as radial_call,
    ):
        output = descriptor(x, attrs, positions, edges, shifts, lattice, batch)
    for hook in hooks:
        hook.remove()
    assert angular.call_count == radial_call.call_count == 1
    vectors = positions[edges[1]] - positions[edges[0]] + shifts @ lattice[0]
    torch.testing.assert_close(angular.call_args.args[0], vectors)
    torch.testing.assert_close(
        radial_call.call_args.args[0], vectors.norm(dim=-1, keepdim=True)
    )
    atomic_numbers = radial_call.call_args.args[3]
    assert atomic_numbers.dtype == torch.int64
    assert atomic_numbers.device == vectors.device
    assert len(calls) == 1
    torch.testing.assert_close(calls[0][0][0], x)
    torch.testing.assert_close(calls[0][0][1], edges)
    assert len(linear_inputs) == 1
    torch.testing.assert_close(
        linear_inputs[0],
        x if interaction_type == "o2" else torch.cat((x[edges[0]], x[edges[1]]), -1),
    )
    assert output.shape == (4, descriptor.irreps_out.dim)
    first = torch.autograd.grad(
        output.square().sum(), (x, positions, lattice), create_graph=True
    )
    second = torch.autograd.grad(
        sum(g.square().sum() for g in first),
        (x, positions, lattice, *descriptor.parameters()),
    )
    assert all(torch.isfinite(g).all() for g in first + second)
    assert descriptor(
        x, attrs, positions, edges[:, :0], shifts[:0], lattice, batch
    ).shape == (
        0,
        descriptor.irreps_out.dim,
    )


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
def test_scatter_norm_values_and_derivatives(
    double_precision, caplog, key, shape, use_cutoff
):
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
    messages = [
        record.message
        for record in caplog.records
        if "scatter_norm='no_cutoff_density' is a legacy option" in record.message
    ]
    assert not messages


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
def test_legacy_scatter_norm_loading(
    tmp_path, caplog, small_model, key, norm_class, dtype
):
    torch.set_default_dtype(dtype)
    reference = small_model(key).eval()
    messages = [
        record.message
        for record in caplog.records
        if "is a legacy option" in record.message
    ]
    assert len(messages) == int(key == "no_cutoff_density")
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

    caplog.clear()
    loaded = load_tace(path, device=DEVICE, dtype=dtype, strict=True).eval()
    messages = [
        record.message
        for record in caplog.records
        if "is a legacy option" in record.message
    ]
    assert len(messages) == int(key == "no_cutoff_density")
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


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("hidden", [[], [7]])
@pytest.mark.parametrize("num_edges", [0, 8])
@pytest.mark.parametrize("use_eqx", [False, True])
def test_element2_projects_nodes_before_gather(
    bias, lora, hidden, num_edges, use_eqx, monkeypatch
):
    from tace.models._e3nn.edge import Element2EdgeUpdate
    from tace.models.linear import enable_lora
    from tace.models.mlp import MLP

    if use_eqx and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    monkeypatch.setenv("TACE_USE_EQX", str(int(use_eqx)))
    device = DEVICE if use_eqx else torch.device("cpu")
    module = Element2EdgeUpdate(
        layer=0,
        num_layers=2,
        num_elements=3,
        num_radial_basis=4,
        num_channel=4,
        edge_embedding_channel=6,
        bias=bias,
    ).to(device=device, dtype=DTYPE)
    mlp = MLP["mlp_v1"]([14, *hidden, 3], bias=bias).to(device=device, dtype=DTYPE)
    if lora:
        enable_lora(mlp.mlp[0], r=2, freeze_base=False)
        with torch.no_grad():
            mlp.mlp[0].lora_B.normal_()
    with torch.no_grad():
        for name, parameter in (*module.named_parameters(), *mlp.named_parameters()):
            if name.endswith("bias"):
                parameter.normal_()
    attrs = torch.randn(5, 3, dtype=DTYPE, device=device, requires_grad=True)
    edges = torch.randint(5, (2, num_edges), device=device)
    feats = torch.randn(num_edges, 6, dtype=DTYPE, device=device, requires_grad=True)
    source = module.source_embedding(attrs[edges[0]])
    target = module.target_embedding(attrs[edges[1]])
    expected = mlp(torch.cat((feats, target, source), -1))
    sizes = []
    hook = module.source_embedding.register_forward_pre_hook(
        lambda module, inputs: sizes.append(inputs[0].size(0))
    )
    actual = mlp(module(None, attrs, feats, edges, None))
    hook.remove()
    assert sizes == [attrs.size(0)]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (attrs, feats, *module.parameters(), *mlp.parameters())
    for _ in range(2):
        seed = torch.randn_like(actual)
        gradients = [
            torch.autograd.grad(
                (y * seed).sum(),
                inputs,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            for y in (actual, expected)
        ]
        actual, expected = [
            torch.cat(
                [
                    (torch.zeros_like(x) if g is None else g).reshape(-1)
                    for x, g in zip(inputs, values)
                ]
            )
            for values in gradients
        ]
        torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize(
    "name",
    [
        "spherical_tensor",
        "spherical_tensor_element2",
        "wigner_tensor",
        "wigner_tensor_element2",
    ],
)
@pytest.mark.parametrize("determinant", [-1, 1])
def test_tensor_node_embedding_is_equivariant(name, determinant, double_precision):
    embedding = NODE_EMBEDDING[name](
        num_elements=2,
        num_radial_basis=4,
        num_channel=3,
        Lmax=2,
        lmax=2,
        avg_num_neighbors=2.0,
        bias=False,
    ).to(DEVICE, DTYPE)
    node_attrs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 0], [1, 2, 0, 2]],
        device=DEVICE,
    )
    edge_vectors = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    edge_feats = torch.randn(4, 4, dtype=DTYPE, device=DEVICE)
    edge_cutoff = torch.rand(4, 1, dtype=DTYPE, device=DEVICE)
    wigner_module = o2.WignerD(2, 2).to(DEVICE, DTYPE)
    wigner, wigner_inv = wigner_module(edge_vectors)
    output = embedding(
        node_attrs,
        edge_feats,
        edge_index,
        o3.spherical_harmonics(
            list(range(3)), edge_vectors, True, normalization="component"
        ),
        edge_cutoff,
        wigner,
        wigner_inv,
    )

    rotation = determinant * o3.rand_matrix(dtype=DTYPE, device=DEVICE)
    rotated_wigner, rotated_wigner_inv = wigner_module(edge_vectors @ rotation.T)
    rotated_output = embedding(
        node_attrs,
        edge_feats,
        edge_index,
        o3.spherical_harmonics(
            list(range(3)), edge_vectors @ rotation.T, True, normalization="component"
        ),
        edge_cutoff,
        rotated_wigner,
        rotated_wigner_inv,
    )

    assert embedding.irreps_out == o3.Irreps("3x0e+3x1o+3x2e")
    assert torch.isfinite(
        embedding(
            node_attrs,
            edge_feats,
            edge_index,
            o3.spherical_harmonics(
                list(range(3)), edge_vectors, True, normalization="component"
            ),
            None,
            wigner,
            wigner_inv,
        )
    ).all()
    torch.testing.assert_close(
        rotated_output,
        output @ embedding.irreps_out.D_from_matrix(rotation.cpu()).to(output).T,
        atol=1.0e-6,
        rtol=1.0e-5,
    )


@pytest.mark.parametrize("element_dependent", [False, True])
@pytest.mark.parametrize("mmax", [0, 2])
@pytest.mark.parametrize("edges", [0, 5])
@pytest.mark.parametrize("hidden", [None, [], [7, 5]])
@pytest.mark.parametrize("mlp_type", ["mlp_v1", "mlp_v2"])
def test_tensor_embedding_constructions(
    element_dependent, mmax, edges, hidden, mlp_type, double_precision
):
    suffix = "_element2" if element_dependent else ""
    spherical = NODE_EMBEDDING[f"spherical_tensor{suffix}"](
        2, 4, 3, 2, 2, 2.0, radial_mlp=hidden, mlp_type=mlp_type
    ).to(DEVICE)
    wigner = NODE_EMBEDDING[f"wigner_tensor{suffix}"](
        2, 4, 3, 2, 2, 2.0, radial_mlp=hidden, mlp_type=mlp_type
    ).to(DEVICE)
    wigner.load_state_dict(spherical.state_dict(), strict=True)
    positions = torch.randn(edges, 3, device=DEVICE, requires_grad=True)
    radial = torch.randn(edges, 4, device=DEVICE, requires_grad=True)
    index = torch.randint(3, (2, edges), device=DEVICE)
    attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]], device=DEVICE)
    rotation, inverse = o2.WignerD(mmax, 2).to(DEVICE)(positions)
    harmonics = o3.spherical_harmonics(
        list(range(3)), positions, True, normalization="component"
    )
    actual = wigner(attrs, radial, index, None, None, rotation, inverse)
    expected = spherical(attrs, radial, index, harmonics, None, None, None)
    widths = [3, 3] if hidden is None else hidden
    for embedding in (spherical, wigner):
        assert embedding.edge_info.dims == [4 + 6 * element_dependent, *widths, 9]
        norms = [
            layer.normalized_shape
            for layer in embedding.edge_info.mlp
            if isinstance(layer, torch.nn.LayerNorm)
        ]
        assert norms == (
            [(width,) for width in widths]
            if element_dependent and mlp_type == "mlp_v1"
            else []
        )
    for order in range(3):
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
        if order < 2:
            seed = torch.randn_like(actual)
            actual, expected = (
                torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(
                            (value * seed).sum(),
                            (positions, radial),
                            create_graph=True,
                        )
                    ]
                )
                for value in (actual, expected)
            )
    assert "tensor" not in NODE_EMBEDDING and "o2_tensor" not in NODE_EMBEDDING


def test_universal_embedding_is_filtered_by_default_config():
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["statistics"] = [
        {
            "atomic_numbers": [26],
            "avg_num_neighbors": 2.0,
            "atomic_energy": {26: 0.0},
        }
    ]
    config["universal_embedding"]["initial_noncollinear_magmoms"] = {
        "enable": True,
        "normalizer": 1.0,
    }

    filtered = check_model_config(config)["universal_embedding"]

    assert filtered == DEFAULT_MODEL_CONFIG["universal_embedding"]


@pytest.mark.parametrize(
    ("shape", "dim", "use_ptr"),
    [
        ((5, 3), 0, False),
        ((3, 5), 1, False),
        ((3, 5), -1, False),
        ((5, 3), 0, True),
        ((0, 3), 0, False),
        ((0, 3), 0, True),
    ],
)
def test_graph_softmax_matches_groupwise_sums_and_gradients(shape, dim, use_ptr):
    from tace.models.softmax import GraphSoftmax

    module = GraphSoftmax()
    src = torch.randn(shape, dtype=DTYPE, device=DEVICE, requires_grad=True)
    scale = torch.rand_like(src, requires_grad=True)
    counts = [2, 0, 3, 0] if src.size(dim) else [0, 0, 0, 0]
    count = torch.tensor(counts, device=DEVICE)
    index = torch.arange(len(counts), device=DEVICE).repeat_interleave(count)
    ptr = torch.cat((count.new_zeros(1), count.cumsum(0)))
    actual = module(
        src,
        dim=dim,
        exp_rescale=scale,
        **({"ptr": ptr} if use_ptr else {"index": index, "num_nodes": len(counts)}),
    )
    expected = []
    for value, weight in zip(src.split(counts, dim), scale.split(counts, dim)):
        if value.size(dim):
            value = (value - value.detach().amax(dim, keepdim=True)).exp()
        value = value * weight
        expected.append(value / (value.sum(dim, keepdim=True) + module.eps))
    expected = torch.cat(expected, dim=dim)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.square().sum(), (src, scale)),
        torch.autograd.grad(expected.square().sum(), (src, scale)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("magnetic_type", ["identity", "element", "element2"])
def test_node_update_returns_source_and_target_info(magnetic_type):
    update = NODE_UPDATE[magnetic_type](
        num_elements=2,
        num_radial_basis=3,
        num_channel=5,
    ).to(DEVICE, DTYPE)
    magnetic_radial_basis = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    node_attrs = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    output = update(magnetic_radial_basis, node_attrs)
    if magnetic_type == "identity":
        source_info = target_info = magnetic_radial_basis
    elif magnetic_type == "element":
        source_info = target_info = update.embedding(
            magnetic_radial_basis,
            node_attrs,
        )
    else:
        source_info = update.source_embedding(
            magnetic_radial_basis,
            node_attrs,
        )
        target_info = update.target_embedding(
            magnetic_radial_basis,
            node_attrs,
        )

    assert output[0].shape == (magnetic_radial_basis.size(0), update.out_dim)
    assert output[1].shape == (magnetic_radial_basis.size(0), update.out_dim)
    torch.testing.assert_close(output[0], source_info)
    torch.testing.assert_close(output[1], target_info)


@pytest.mark.parametrize(
    "layout_in",
    ("ir_mul", "mul_ir", "flatten_ir_mul", "flatten_mul_ir"),
)
@pytest.mark.parametrize(
    "layout_out",
    ("ir_mul", "mul_ir", "flatten_ir_mul", "flatten_mul_ir"),
)
def test_layout_transform_supports_every_layout_pair(layout_in, layout_out) -> None:
    irreps = o3.Irreps("3x0e+3x1o+3x1o+3x2e")
    flatten_mul_ir = torch.randn(
        5, irreps.dim, dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    blocks = []
    offset = 0
    for mul, ir in irreps:
        width = mul * ir.dim
        blocks.append(
            flatten_mul_ir[..., offset : offset + width].reshape(5, mul, ir.dim)
        )
        offset += width
    mul_ir = torch.cat(blocks, dim=-1)
    ir_mul = mul_ir.transpose(-1, -2)
    layouts = {
        "ir_mul": ir_mul,
        "mul_ir": mul_ir,
        "flatten_ir_mul": ir_mul.flatten(-2),
        "flatten_mul_ir": flatten_mul_ir,
    }
    transform = LayoutTransform(
        irreps,
        layout_in=layout_in,
        layout_out=layout_out,
    ).to(DEVICE)
    observed = transform(layouts[layout_in])
    torch.testing.assert_close(observed, layouts[layout_out])
    torch.testing.assert_close(transform.inverse(observed), layouts[layout_in])
    seed = torch.randn_like(observed)
    gradients = [
        torch.autograd.grad((output * seed).sum(), flatten_mul_ir, retain_graph=True)[0]
        for output in (observed, layouts[layout_out])
    ]
    torch.testing.assert_close(*gradients)
    assert not transform.state_dict()


def test_layout_transform_flattened_layouts_allow_different_multiplicities() -> None:
    irreps = o3.Irreps("2x0e+3x1o")
    features = torch.randn(4, irreps.dim, dtype=DTYPE, device=DEVICE)
    transform = LayoutTransform(
        irreps,
        layout_in="flatten_mul_ir",
        layout_out="flatten_ir_mul",
    ).to(DEVICE)
    torch.testing.assert_close(transform.inverse(transform(features)), features)


@pytest.mark.parametrize("wigner_lmax", [2, 4])
@pytest.mark.parametrize("basis_change", [True, False])
def test_local_frame_roundtrip_flattened_ir_mul(wigner_lmax, basis_change):
    irreps = o3.Irreps("2x0e+1x0o+3x1e+2x1o+1x2e+3x2o")
    frame = o2.LocalFrame(irreps, basis_change=basis_change).to(DEVICE, DTYPE)
    layout = LayoutTransform(
        irreps,
        layout_in="flatten_mul_ir",
        layout_out="flatten_ir_mul",
    ).to(DEVICE)
    vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(wigner_lmax, wigner_lmax).to(DEVICE, DTYPE)(vectors)
    features = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)

    local = frame(layout(features), wigner)
    assert frame.irreps_out == o2.Irreps("5x0e+7x0o+9x1m+4x2m")
    options = "mmax=2" + ("" if basis_change else ", basis_change=False")
    assert repr(frame) == (
        f"LocalFrame({frame.global_irreps} -> {frame.local_irreps})({options})"
    )
    reverse_frame = o2.LocalFrame(irreps, reverse=True, basis_change=basis_change)
    assert repr(reverse_frame) == (
        f"LocalFrame({frame.local_irreps} -> {frame.global_irreps})({options})"
    )
    if basis_change:
        default = o2.LocalFrame(irreps).to(DEVICE, DTYPE)
        torch.testing.assert_close(
            default(layout(features), wigner), local, atol=0, rtol=0
        )
    assert local.shape == (7, frame.irreps_out.dim)
    torch.testing.assert_close(
        layout.inverse(frame.to_global(local, wigner_inv)),
        features,
    )


@pytest.mark.parametrize("matrix", ["0", "1"])
@pytest.mark.parametrize("internal", [False, True])
@pytest.mark.parametrize("num_nodes", [0, 7])
def test_shared_linear(matrix, internal, num_nodes, double_precision):
    import copy

    actual = e3nnLinear(
        "2x0e+0e+3x1o+2x5o",
        "3x0e+0e+2x1o+5o",
        bias=True,
        internal_weights=internal,
        use_matrix_weight=matrix,
    )
    reference = copy.deepcopy(actual)
    reference.linear = o3.Linear(
        actual.irreps_in,
        actual.irreps_out,
        internal_weights=False,
        shared_weights=False,
    )
    reference.load_state_dict(actual.state_dict(), strict=True)
    x = torch.randn(num_nodes, actual.irreps_in.dim, requires_grad=True)
    weight = (
        None
        if internal
        else torch.randn(num_nodes, actual.weight_numel, requires_grad=True)
    )
    values = [module(x, weight) for module in (actual, reference)]
    torch.testing.assert_close(*values, atol=1e-12, rtol=1e-12)
    inputs = [
        (x, *module.parameters()) if internal else (x, weight, *module.parameters())
        for module in (actual, reference)
    ]
    for _ in range(3):
        derivatives = [
            torch.autograd.grad(y.sin().sum(), args, create_graph=True)
            for y, args in zip(values, inputs)
        ]
        for a, b in zip(*derivatives):
            torch.testing.assert_close(a, b, atol=1e-10, rtol=1e-10)
        values = [torch.cat([g.flatten() for g in grads]) for grads in derivatives]


@pytest.mark.parametrize("experts", [1, 2])
@pytest.mark.parametrize("matrix", ["0", "1"])
@pytest.mark.parametrize("lora", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_tace_indexed_linear(monkeypatch, experts, matrix, lora, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.set_default_dtype(dtype)
    cls = e3nnElementLinear if experts == 1 else e3nnMoEElementLinear
    module = cls(
        "4x0e+2x0e+4x1o+2x1e",
        "6x0e+4x1o+2x1o+2x2o",
        num_elements=3,
        bias=True,
        use_matrix_weight=matrix,
        **({} if experts == 1 else {"num_experts": experts}),
    ).cuda()
    if lora:
        enable_lora(module, r=2, freeze_base=False)
        with torch.no_grad():
            for parameter in module.lora_B:
                parameter.normal_()
    with torch.no_grad():
        module.bias.normal_()
    attrs = torch.rand(5, 3, device="cuda", requires_grad=True)
    x = torch.randn(5, module.irreps_in.dim, device="cuda", requires_grad=True)
    # A supplied element index takes precedence over attrs.argmax for weights.
    types = torch.tensor([2, 0, 1, 2, 1], device="cuda")
    monkeypatch.setenv("TACE_USE_EQX", "1")
    outputs, derivatives = [], []
    inputs = (x, attrs, *module.parameters())
    for enabled in (False, True):
        monkeypatch.setitem(EQX_KERNELS, "linear", enabled)
        output = module(x, attrs, types)
        first = torch.autograd.grad(output.square().mean(), x, create_graph=True)[0]
        derivatives.append(
            torch.autograd.grad(
                output.square().mean() + first.square().mean(),
                inputs,
                allow_unused=True,
            )
        )
        outputs.append(output)
    tol = 3e-4 if dtype == torch.float32 else 1e-10
    torch.testing.assert_close(*outputs, atol=tol, rtol=tol)
    for actual, expected in zip(*derivatives):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
    if dtype == torch.float64 and matrix == "1" and lora:
        compiled = torch.compile(
            module, backend="aot_eager", fullgraph=True, dynamic=True
        )
        output = compiled(x, attrs, types)
        torch.testing.assert_close(output, outputs[1], atol=tol, rtol=tol)
        torch.testing.assert_close(
            torch.autograd.grad(output.square().mean(), x)[0],
            torch.autograd.grad(module(x, attrs, types).square().mean(), x)[0],
            atol=tol,
            rtol=tol,
        )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 13])
def test_graph_attention_derivatives(device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.conv.attention import graph_softmax
    from tace.models.softmax import GraphSoftmax

    target = torch.arange(edges, device=device) % 4
    scores = torch.randn(
        edges, 3, dtype=torch.float64, device=device, requires_grad=True
    )
    weight = torch.rand(edges, 1, dtype=torch.float64, device=device)
    weight[::3] = 0
    weight.requires_grad_()
    actual = graph_softmax(scores, target, 5, weight, eps=1e-3)
    expected = GraphSoftmax(eps=1e-3)(scores, target, num_nodes=5, exp_rescale=weight)
    torch.testing.assert_close(actual, expected)
    losses = [x.square().sum() for x in (actual, expected)]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, (scores, weight), create_graph=True)
            for loss in losses
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-8)
        losses = [sum(g.square().sum() for g in values) for values in grads]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("nodes", [0, 5])
def test_streaming_graph_attention_weights(device, nodes):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from eqx.conv.attention import StreamingGraphAttention
    from tace.models.softmax import GraphSoftmax

    edges, heads = (9 if nodes else 0), 2
    target = torch.arange(edges, device=device) % 3
    inputs = [
        torch.zeros(nodes, 1, device=device, dtype=torch.float64),
        torch.randn(edges, heads, device=device, dtype=torch.float64).requires_grad_(),
        torch.randn(edges, 3, 4, device=device, dtype=torch.float64).requires_grad_(),
        torch.rand(edges, 1, device=device, dtype=torch.float64).requires_grad_(),
        torch.rand(edges, heads, device=device, dtype=torch.float64).requires_grad_(),
        target,
        torch.ones(edges, device=device, dtype=torch.bool),
    ]
    with torch.no_grad():
        inputs[3][::3] = 0

    def fake(v):
        return (
            v[0].new_empty(nodes, 3, 4),
            v[0].new_empty(nodes, heads),
            v[0].new_empty(nodes, heads),
        )

    attention = StreamingGraphAttention(
        lambda v: (v[1], v[2]),
        fake,
        {i: 0 for i in range(1, 7)},
        target_slot=5,
        normalizer_slot=3,
        value_weight_slot=4,
        valid_slot=6,
        tile_size=4,
        eps=1e-3,
    )
    actual = attention(*inputs)[0]
    weight = (
        GraphSoftmax(eps=1e-3)(
            inputs[1], target, num_nodes=nodes, exp_rescale=inputs[3]
        )
        * inputs[4]
    )
    value = (inputs[2].view(edges, 3, heads, 2) * weight[:, None, :, None]).reshape(
        edges, 3, 4
    )
    expected = value.new_zeros(nodes, 3, 4).index_add(0, target, value)
    torch.testing.assert_close(actual, expected)
    losses = [x.square().sum() for x in (actual, expected)]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, inputs[1:5], create_graph=True) for loss in losses
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-8)
        losses = [sum(g.square().sum() for g in values) for values in grads]

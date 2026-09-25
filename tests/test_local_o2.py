################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import sys
from copy import deepcopy
from unittest.mock import Mock

import pytest
import torch
from e3nn import o3

from eqx import o2
from tace.models._e3nn.base import NodeUpdate
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG, check_model_config
from tace.models._e3nn.inter import O2MagneticInteraction
from tace.models._e3nn.magnetic import MagneticBasis
from tace.models._e3nn.node import (
    NODE_EMBEDDING,
    NODE_UPDATE,
    LinearSpinNodeEmbedding,
    NonLinearSpinNodeEmbedding,
    O2TensorNodeEmbedding,
)
from tace.models._e3nn.o2 import (
    O2ScatterMagneticTensorProduct,
    O2ScatterTensorProduct,
)
from tace.models._e3nn.representation import Representation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("num_edges", [0, 8])
def test_element_edge_update_embeds_nodes_before_gather(reverse, num_edges):
    from tace.models._e3nn.edge import Element2EdgeUpdate, ElementEdgeUpdate

    cls = Element2EdgeUpdate if reverse else ElementEdgeUpdate
    module = cls(
        layer=0,
        num_layers=2,
        num_elements=3,
        num_radial_basis=4,
        num_channel=4,
        edge_embedding_channel=4,
        bias=True,
    ).double()
    attrs = torch.randn(5, 3, dtype=DTYPE, requires_grad=True)
    edges = torch.randint(5, (2, num_edges))
    feats = torch.randn(num_edges, 4, dtype=DTYPE, requires_grad=True)
    source = module.source_embedding(attrs[edges[0]])
    target = module.target_embedding(attrs[edges[1]])
    expected = torch.cat(
        (feats, target, source) if reverse else (feats, source, target), -1
    )
    sizes = []
    hook = module.source_embedding.register_forward_pre_hook(
        lambda module, inputs: sizes.append(inputs[0].size(0))
    )
    actual = module(None, attrs, feats, edges, None)
    hook.remove()
    assert sizes == [attrs.size(0)]
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    inputs = (attrs, feats, *module.parameters())
    gradients = [
        torch.autograd.grad(y.square().sum(), inputs, retain_graph=True)
        for y in (actual, expected)
    ]
    for actual_grad, expected_grad in zip(*gradients):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("num_graphs", [1, 2])
def test_periodic_graph_derivatives(num_graphs):
    from tace.models.adapter import TensorModel

    torch.manual_seed(71)
    positions = torch.randn(3 * num_graphs, 3, dtype=DTYPE, requires_grad=True)
    lattice = torch.randn(num_graphs, 3, 3, dtype=DTYPE, requires_grad=True)
    edges = torch.cat(
        [torch.tensor([[0, 1, 2, 1], [2, 0, 1, 2]]) + 3 * i for i in range(num_graphs)],
        dim=1,
    )
    data = dict(
        positions=positions,
        lattice=lattice,
        node_attrs=torch.ones(3 * num_graphs, 1, dtype=DTYPE),
        edge_index=edges,
        edge_shifts=torch.randint(-1, 2, (edges.size(1), 3)).to(DTYPE),
        batch=torch.arange(num_graphs).repeat_interleave(3),
        ptr=torch.arange(num_graphs + 1) * 3,
        fidelity_idx=torch.zeros(num_graphs, dtype=torch.long),
    )
    model = Mock(
        lmp=False,
        readout_fn=torch.nn.Module(),
        flags=Mock(compute_virials=True, compute_stress=True),
    )
    model.get_target_property.return_value = ["energy", "forces", "stress"]
    graph = TensorModel.prepare_graph(model, data)
    torch.testing.assert_close(graph.node_type, data["node_attrs"].argmax(-1))
    source, target = edges
    reference = (
        data["positions"][target]
        - data["positions"][source]
        + torch.einsum(
            "ni,nij->nj", data["edge_shifts"], data["lattice"][data["batch"][source]]
        )
    )
    torch.testing.assert_close(graph.edge_vector, reference)
    inputs = positions, lattice, graph.displacement
    actual, expected = graph.edge_vector.sin(), reference.sin()
    for _ in range(3):
        seed = torch.randn_like(actual)
        derivatives = [
            torch.autograd.grad(
                (value * seed).sum(), inputs, create_graph=True, retain_graph=True
            )
            for value in (actual, expected)
        ]
        actual, expected = [
            torch.cat([g.flatten() for g in grads]) for grads in derivatives
        ]
        torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)


def _time_reverse(features: torch.Tensor, irreps) -> torch.Tensor:
    output = features.clone()
    for ir_mul, ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir_mul.ir.t
    return output


@pytest.mark.parametrize(
    "arguments,dtype,device",
    [
        ([], None, "cpu"),
        (["--dtype", "float32"], "float32", "cpu"),
        (["--dtype", "float64", "--device", "cuda:0"], "float64", "cuda:0"),
    ],
)
def test_convert_cgtp_script_arguments(monkeypatch, tmp_path, arguments, dtype, device):
    from tace.scripts import convert_cgtp as script

    path = tmp_path / "model.pth"
    load, convert, export = Mock(), Mock(), Mock()
    monkeypatch.setattr(script, "load_tace", load)
    monkeypatch.setattr(script, "convert_cgtp", convert)
    monkeypatch.setattr(script, "export_tace", export)
    monkeypatch.setattr(sys, "argv", ["tace-convert-cgtp", "-m", str(path), *arguments])
    script.main()
    load.assert_called_once_with(str(path), device=device, dtype=dtype)
    convert.assert_called_once_with(load.return_value)
    export.assert_called_once_with(
        convert.return_value, str(tmp_path / "model-converted.pt")
    )

    output = tmp_path / "model-converted.pt"
    output.write_bytes(b"existing model")
    with pytest.raises(SystemExit) as error:
        script.main()
    assert error.value.code == 2
    assert output.read_bytes() == b"existing model"
    assert load.call_count == export.call_count == 1


@pytest.mark.parametrize(
    ("magnetic_scales", "expected_scales"),
    [
        ([None], [2.5]),
        ([None, None], [2.5, 3.7]),
        ([2.0, {26: 4.0}], [2.0, 4.0]),
        ([None, {"26": 4.0}], [2.5, 4.0]),
    ],
)
def test_magnetic_scale_is_resolved_per_fidelity(magnetic_scales, expected_scales):
    config = check_model_config(
        {
            "statistics": [
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 2.0,
                    "max_noncollinear_magmoms_norm_by_element": {26: 2.0},
                },
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 4.0,
                    "max_noncollinear_magmoms_norm_by_element": {26: 3.0},
                },
            ][: len(magnetic_scales)],
            "target_property": [],
            "fidelity": [
                {"name": name, "magnetic_scale": scale}
                for name, scale in zip(("PBE", "SCAN"), magnetic_scales)
            ],
        }
    )

    assert [scale[26] for scale in config["magnetic_scale"]] == pytest.approx(
        expected_scales
    )


@pytest.mark.parametrize("magnetic_scale", [2.0, {26: 2.0}])
def test_manual_magnetic_scale_is_used_without_rescaling(magnetic_scale):
    config = check_model_config(
        {
            "statistics": [
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 2.0,
                    "max_noncollinear_magmoms_norm_by_element": {26: 9.0},
                }
            ],
            "target_property": [],
            "fidelity": [
                {"name": "PBE", "magnetic_scale": magnetic_scale},
            ],
        }
    )

    assert config["magnetic_scale"] == [{26: 2.0}]


@pytest.mark.parametrize("num_nodes", [0, 4])
@pytest.mark.parametrize("num_fidelities", [1, 2])
def test_magnetic_basis_selects_fidelity_and_element(num_nodes, num_fidelities):
    basis = MagneticBasis(
        [2.0, {26: 4.0, 28: 3.0}][:num_fidelities],
        num_mag_radial_basis=4,
        Lmax=1,
        atomic_numbers=[26, 28],
    ).to(DEVICE, DTYPE)
    magmoms = (
        torch.tensor(
            [[0.3, -0.4, 0.5]] * num_nodes,
            dtype=DTYPE,
            device=DEVICE,
        )
        .reshape(num_nodes, 3)
        .requires_grad_()
    )
    node_attrs = torch.eye(2, dtype=DTYPE, device=DEVICE)[
        torch.tensor([1, 0, 0, 1], device=DEVICE)[:num_nodes]
    ]
    node_fidelity = (
        torch.tensor([0, 1, 0, 1], device=DEVICE)[:num_nodes] % num_fidelities
    )
    edge_index = torch.tensor([[0, 2, 1, 3], [2, 0, 3, 1]], device=DEVICE)[
        :, :num_nodes
    ]

    radial, node_attrs_out, edge_attrs_out = basis(
        magmoms, node_attrs, edge_index, node_fidelity
    )
    magnetic_scale = torch.tensor(
        [2.0] * 4 if num_fidelities == 1 else [2.0, 4.0, 2.0, 3.0],
        dtype=DTYPE,
        device=DEVICE,
    )[:num_nodes, None]
    squared_magnitude = (magmoms / magnetic_scale).square().sum(-1, keepdim=True)
    expected_radial = basis.radial_basis(1.0 - 2.0 * squared_magnitude.clamp(max=1.0))
    torch.testing.assert_close(radial, expected_radial)
    torch.testing.assert_close(
        torch.autograd.grad(radial.sum(), magmoms)[0],
        torch.autograd.grad(expected_radial.sum(), magmoms)[0],
    )
    torch.testing.assert_close(node_attrs_out, basis.angular_basis(magmoms))
    assert edge_attrs_out.shape == (num_nodes, basis.magnetic_edge_irreps_out.dim)
    assert basis.magnetic_scale.shape == (num_fidelities, 2)
    assert "magnetic_scale=[[" in repr(basis)


@pytest.mark.parametrize(
    ("num_fidelities", "legacy"), [(1, True), (1, False), (2, False)]
)
@pytest.mark.parametrize("nested", [False, True])
def test_magnetic_basis_loads_scale_fidelity_axis(num_fidelities, legacy, nested):
    basis = MagneticBasis(
        [1.0] * num_fidelities,
        num_mag_radial_basis=4,
        Lmax=1,
        atomic_numbers=[26, 28],
    ).to(DEVICE, DTYPE)
    module = torch.nn.ModuleDict({"magnetic_basis": basis}) if nested else basis
    key = "magnetic_basis.magnetic_scale" if nested else "magnetic_scale"
    expected = torch.arange(2, 2 + num_fidelities * 2, dtype=DTYPE, device=DEVICE).view(
        num_fidelities, 2
    )
    state_dict = module.state_dict()
    state_dict[key] = expected[0] if legacy else expected

    incompatible = module.load_state_dict(state_dict, strict=True)

    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []
    torch.testing.assert_close(basis.magnetic_scale, expected)


@pytest.mark.parametrize(
    "name,embedding_type",
    [
        ("linear_spin", LinearSpinNodeEmbedding),
        ("nonlinear_spin", NonLinearSpinNodeEmbedding),
    ],
)
def test_spin_node_embedding(name, embedding_type):
    embedding = embedding_type(
        num_elements=2,
        num_radial_basis=4,
        num_mag_radial_basis=3,
        num_channel=5,
        Lmax=1,
        lmax=1,
        avg_num_neighbors=2.0,
        bias=False,
    )

    node_attrs = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    magnetic_radial_basis = torch.tensor([[0.2, -0.1, 0.4], [0.3, 0.5, -0.2]])

    output = embedding(
        node_attrs,
        torch.empty(0),
        torch.empty(0, dtype=torch.long),
        torch.empty(0),
        None,
        None,
        None,
        magnetic_radial_basis,
    )
    expected = (
        embedding.element_embedding(node_attrs)
        + embedding.spin_embedding(magnetic_radial_basis)
    ) / (2.0**0.5)

    assert NODE_EMBEDDING[name] is embedding_type
    assert embedding.spin_embedding.irreps_in == o3.Irreps("3x0e")
    if name == "nonlinear_spin":
        expected = embedding.activation(expected)
    torch.testing.assert_close(output, expected)


def test_o2_tensor_node_embedding_is_equivariant():
    embedding = O2TensorNodeEmbedding(
        num_elements=2,
        num_radial_basis=4,
        num_mag_radial_basis=3,
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
        torch.empty(4, 0, dtype=DTYPE, device=DEVICE),
        edge_cutoff,
        wigner,
        wigner_inv,
    )

    rotation = o3.rand_matrix(dtype=DTYPE, device=DEVICE)
    rotated_wigner, rotated_wigner_inv = wigner_module(edge_vectors @ rotation.T)
    rotated_output = embedding(
        node_attrs,
        edge_feats,
        edge_index,
        torch.empty(4, 0, dtype=DTYPE, device=DEVICE),
        edge_cutoff,
        rotated_wigner,
        rotated_wigner_inv,
    )

    assert NODE_EMBEDDING["o2_tensor"] is O2TensorNodeEmbedding
    assert embedding.irreps_out == o3.Irreps("3x0e+3x1o+3x2e")
    assert torch.isfinite(
        embedding(
            node_attrs,
            edge_feats,
            edge_index,
            torch.empty(4, 0, dtype=DTYPE, device=DEVICE),
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


@pytest.mark.parametrize("interaction", ["o2", "uu_o2"])
@pytest.mark.parametrize(("Lmax", "lmax"), [(2, 3), (3, 2)])
def test_o2_representation_uses_common_angular_coverage(Lmax, lmax, interaction):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["node_embedding"]["type"] = "linear"
    config["atomic_basis"]["type"] = [interaction]
    config["atomic_basis"]["nonlinear"] = ["gate"]
    config["atomic_basis"]["edge_nonlinear"] = ["gate"]
    config["atomic_basis"]["use_radial_rotary_attention"] = interaction != "uu_o2"
    config["product_basis"]["type"] = ["cgtp"]
    config["product_basis"]["correlation"] = [2]

    representation = Representation(
        num_layers=1,
        atomic_numbers=[1],
        cutoff=3.0,
        avg_num_neighbors=2.0,
        magnetic_scale=None,
        mmax=2,
        Lmax=Lmax,
        lmax=lmax,
        angular_basis={
            "magnetic_Lmax": 1,
        },
        num_channel=2,
        target_irreps=o3.Irreps("0e"),
        node_embedding=config["node_embedding"],
        edge_embedding=config["edge_embedding"],
        edge_update=config["edge_update"],
        node_update=config["node_update"],
        radial_basis=config["radial_basis"],
        atomic_basis=config["atomic_basis"],
        resnet=config["resnet"],
        product_basis=config["product_basis"],
        invariant_property=[],
        equivariant_property=[],
        universal_embedding=config["universal_embedding"],
        layer_norm=config["layer_norm"],
        dropout=config["dropout"],
        parity=False,
    )

    common_lmax = max(Lmax, lmax)
    assert representation.use_o2
    assert not representation.use_so2
    assert not representation.use_time_reversal
    assert representation.o2_angular_basis.lmax == common_lmax
    rejector = representation.interactions[0].rejector
    assert rejector.local_frame_in.lmax == rejector.irreps_in.lmax

    model = torch.nn.ModuleDict({"representation": representation})
    state_dict = model.state_dict()
    assert not any("o2_angular_basis" in key for key in state_dict)
    for angular_basis in ("so2_angular_basis", "o2_angular_basis"):
        for name in ("wigner_index_to_m_array", "wigner_inv_rescale"):
            state_dict[f"representation.{angular_basis}.{name}"] = torch.empty(0)
    incompatible = model.load_state_dict(state_dict, strict=True)
    assert incompatible.missing_keys == []
    assert incompatible.unexpected_keys == []


@pytest.mark.parametrize(
    "magnetic_info_type",
    [None, "node"],
    ids=["default", "node"],
)
@pytest.mark.parametrize("magnetic_type", ["identity", "element", "element2"])
def test_magnetic_info_is_independent_per_interaction(
    magnetic_info_type,
    magnetic_type,
    monkeypatch,
):
    if magnetic_info_type is not None:
        monkeypatch.setattr(
            O2MagneticInteraction,
            "magnetic_info_type",
            magnetic_info_type,
        )
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["node_update"]["magnetic_type"] = magnetic_type
    config["atomic_basis"]["type"] = ["o2_mag", "o2_mag"]
    config["atomic_basis"]["nonlinear"] = ["gate", "gate"]
    config["atomic_basis"]["edge_nonlinear"] = ["gate", "gate"]
    config["product_basis"]["type"] = ["cgtp", "cgtp"]
    config["product_basis"]["correlation"] = [2, 2]

    representation = Representation(
        num_layers=2,
        atomic_numbers=[26],
        cutoff=3.0,
        avg_num_neighbors=2.0,
        magnetic_scale=[{26: 2.0}],
        mmax=1,
        Lmax=1,
        lmax=1,
        angular_basis={
            "magnetic_Lmax": 1,
            "use_spin_orbit_coupling": False,
        },
        num_channel=2,
        target_irreps=o3.Irreps("0e"),
        node_embedding=config["node_embedding"],
        edge_embedding=config["edge_embedding"],
        edge_update=config["edge_update"],
        node_update=config["node_update"],
        radial_basis=config["radial_basis"],
        atomic_basis=config["atomic_basis"],
        resnet=config["resnet"],
        product_basis=config["product_basis"],
        invariant_property=[],
        equivariant_property=[],
        universal_embedding=config["universal_embedding"],
        layer_norm=config["layer_norm"],
        dropout=config["dropout"],
        parity=True,
    )

    num_mag_radial_basis = config["radial_basis"]["num_mag_radial_basis"]
    assert len(representation.node_updates) == 2
    assert representation.magnetic_edge_irreps_out.num_irreps == 2
    assert all(
        ir.l == 0 and ir.p == 1 for _, ir in representation.magnetic_edge_irreps_out
    )
    assert representation.node_updates[0] is not (representation.node_updates[1])
    for update, interaction in zip(
        representation.node_updates,
        representation.interactions,
    ):
        assert isinstance(update, NodeUpdate)
        assert isinstance(update, NODE_UPDATE[magnetic_type])
        assert interaction.edge_info.dims[0] == representation.edge_updates[0].out_dim
        expected_info_dims = [
            update.out_dim,
            *config["radial_basis"]["hidden"],
            interaction.magnetic_linear.weight_numel,
        ]
        assert interaction.source_magnetic_info.dims == expected_info_dims
        assert interaction.target_magnetic_info.dims == expected_info_dims
        assert all(
            mul == representation.num_channel
            for mul, _ in interaction.magnetic_edge_irreps_out
        )
        assert interaction.magnetic_linear.bias is None

        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        magnetic_radial_basis = torch.randn(3, num_mag_radial_basis, requires_grad=True)
        node_attrs = torch.ones(3, 1)
        magnetic_info = update(
            magnetic_radial_basis,
            node_attrs,
        )
        magnetic_edge_attrs = torch.randn(
            edge_index.size(1),
            interaction.magnetic_edge_irreps.dim,
        )
        magnetic_weights = interaction._magnetic_weights(
            magnetic_info,
            edge_index,
        )
        source_info, target_info = magnetic_info
        source, target = edge_index
        expected_weights = (
            interaction.source_magnetic_info(source_info)[source]
            * interaction.target_magnetic_info(target_info)[target]
        )
        torch.testing.assert_close(magnetic_weights, expected_weights)
        projected = interaction.magnetic_linear(
            magnetic_edge_attrs,
            magnetic_weights,
        )
        assert projected.shape == (
            edge_index.size(1),
            interaction.magnetic_edge_irreps_out.dim,
        )
        projected.square().sum().backward()
        assert torch.isfinite(magnetic_radial_basis.grad).all()
        for num_nodes in (0, 3):
            empty_edge_weights = interaction._magnetic_weights(
                update(magnetic_radial_basis[:num_nodes], node_attrs[:num_nodes]),
                edge_index[:, :0],
            )
            assert empty_edge_weights.shape == (
                0,
                interaction.magnetic_linear.weight_numel,
            )


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


def _scatter_module(use_attention, linear_type="uv"):
    irreps = o3.Irreps("2x0e+2x0o+2x1o+2x1e")
    return O2ScatterTensorProduct(
        irreps,
        irreps,
        num_channel=2,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=1,
        num_radial_basis=4,
        use_radial_rotary_attention=use_attention,
        linear_type=linear_type,
    ).to(DEVICE, DTYPE)


@pytest.mark.parametrize(
    ("use_attention", "linear_type"), [(False, "uv"), (True, "uv"), (False, "uu")]
)
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_scatter_is_o3_equivariant(use_attention, linear_type, reflected):
    torch.manual_seed(7)
    module = _scatter_module(use_attention, linear_type)
    edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=DEVICE)
    node_features = module.irreps_in.randn(3, -1, dtype=DTYPE, device=DEVICE)
    edge_vectors = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(4, module.weight_numel, dtype=DTYPE, device=DEVICE)
    radial = torch.randn(4, 4, dtype=DTYPE, device=DEVICE)
    cutoff = torch.rand(4, 1, dtype=DTYPE, device=DEVICE)
    wigner_module = o2.WignerD(1, 1).to(DEVICE, DTYPE)
    wigner, wigner_inv = wigner_module(edge_vectors)
    output = module(
        node_features,
        weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_radial_basis=radial,
        edge_cutoff=cutoff,
    )
    if linear_type == "uu":
        torch.testing.assert_close(
            module(
                node_features,
                2 * weights,
                edge_index,
                wigner,
                wigner_inv,
                edge_radial_basis=radial,
                edge_cutoff=cutoff,
            ),
            2 * output,
        )

    rotation = o3.rand_matrix(dtype=DTYPE, device=DEVICE) * (-1 if reflected else 1)
    matrix = module.irreps_in.D_from_matrix(rotation.cpu()).to(node_features)
    rotated_features = node_features @ matrix.T
    rotated_vectors = edge_vectors @ rotation.T
    rotated_wigner, rotated_wigner_inv = wigner_module(rotated_vectors)
    rotated_output = module(
        rotated_features,
        weights,
        edge_index,
        rotated_wigner,
        rotated_wigner_inv,
        edge_radial_basis=radial,
        edge_cutoff=cutoff,
    )
    torch.testing.assert_close(
        rotated_output,
        output @ matrix.T,
        atol=2.0e-6,
        rtol=2.0e-5,
    )


def test_uu_o2_rejects_radial_rotary_attention():
    with pytest.raises(
        ValueError, match="uu_o2 does not support radial rotary attention"
    ):
        _scatter_module(True, "uu")


def test_uu_o2_scatter_uses_only_source_features(monkeypatch):
    torch.manual_seed(8)
    module = _scatter_module(False, "uu")
    assert module.linear.irreps_in == module.node_irreps
    assert module.attention is None
    assert (
        sum(isinstance(layer, (o2.Linear, o2.UuLinear)) for layer in module.modules())
        == 1
    )
    edge_index = torch.tensor([[0, 0], [1, 2]], device=DEVICE)
    node_features = module.irreps_in.randn(
        3, -1, dtype=DTYPE, device=DEVICE, requires_grad=True
    )
    weights = torch.randn(2, module.weight_numel, dtype=DTYPE, device=DEVICE)
    cutoff = torch.rand(2, 1, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(1, 1).to(DEVICE, DTYPE)(
        torch.randn(2, 3, dtype=DTYPE, device=DEVICE)
    )
    to_local = Mock(wraps=module.local_frame_in.to_local)
    monkeypatch.setattr(module.local_frame_in, "to_local", to_local)
    output = module(
        node_features,
        weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=cutoff,
    )
    to_local.assert_called_once()
    torch.testing.assert_close(
        to_local.call_args.args[0], module.reshape_in(node_features)[edge_index[0]]
    )
    changed_features = node_features.detach().clone()
    changed_features[1:] = torch.randn_like(changed_features[1:])
    torch.testing.assert_close(
        module(
            changed_features,
            weights,
            edge_index,
            wigner,
            wigner_inv,
            edge_cutoff=cutoff,
        ),
        output,
        atol=0,
        rtol=0,
    )
    gradients = torch.autograd.grad(output.square().sum(), node_features)[0]
    assert gradients[0].abs().sum() > 0
    torch.testing.assert_close(gradients[1:], torch.zeros_like(gradients[1:]))


@pytest.mark.parametrize("linear_type", ["uv", "uu"])
@pytest.mark.parametrize("num_nodes", [0, 3])
def test_o2_scatter_supports_empty_edges(linear_type, num_nodes):
    module = _scatter_module(False, linear_type)
    assert "reshape_in" not in repr(module)
    assert "reshape_out" not in repr(module)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(1, 1).to(DEVICE, DTYPE)(
        torch.empty(0, 3, dtype=DTYPE, device=DEVICE)
    )
    output = module(
        module.irreps_in.randn(num_nodes, -1, dtype=DTYPE, device=DEVICE),
        torch.empty(0, module.weight_numel, dtype=DTYPE, device=DEVICE),
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=torch.empty(0, 1, dtype=DTYPE, device=DEVICE),
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


def test_uu_o2_interaction_trains_forces_and_uses_external_weights(double_precision):
    from tace.models._e3nn.inter import UuO2Interaction
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=2,
        mmax=1,
        parity=True,
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["atomic_basis"].update(
        type="uu_o2",
        edge_nonlinear=None,
        use_radial_rotary_attention=False,
        num_head=1,
    )
    config["node_embedding"]["type"] = "linear"
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["readout_emlp"]["hidden"] = [2]
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["apply_cutoff"] = False
    config["scale_shift"]["enable"] = False
    model = TensorModel(e3nnTACE(**config)).train()
    for interaction in model.readout_fn.representation.interactions:
        assert isinstance(interaction, UuO2Interaction)
        assert isinstance(interaction.rejector.linear, o2.UuLinear)
        assert not hasattr(interaction.rejector, "nonlinearity")
        assert not hasattr(interaction.rejector, "linear_up")
        assert not hasattr(interaction.rejector, "linear_down")
        assert interaction.rejector.attention is None
        assert list(interaction.rejector.linear.parameters()) == []
        edge_features = torch.randn(3, interaction.edge_feats_channel)
        weights = interaction.edge_info(edge_features)
        assert weights.shape == (3, interaction.rejector.linear.weight_numel)

    data = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]]),
        node_attrs=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        edge_shifts=torch.zeros(6, 3),
        lattice=torch.eye(3).unsqueeze(0) * 8,
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
        fidelity_idx=torch.zeros(1, dtype=torch.long),
    )
    output = model(data)
    for name in ("energy", "forces", "stress", "virials"):
        assert torch.isfinite(output[name]).all()
    output["forces"].square().sum().backward()
    for interaction in model.readout_fn.representation.interactions:
        gradients = [p.grad for p in interaction.edge_info.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum() for g in gradients) > 0


@pytest.mark.parametrize("Lmax", [1, 2])
@pytest.mark.parametrize("parity", [False, True])
def test_magnetic_basis_builds_regrouped_edge_attrs(Lmax, parity):
    time_reversal = hasattr(o3.Irrep("0e"), "t")
    basis = MagneticBasis(
        [{26: 2.0}],
        num_mag_radial_basis=4,
        Lmax=Lmax,
        atomic_numbers=[26],
        time_reversal=time_reversal,
        parity=parity,
    ).to(DEVICE, DTYPE)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        device=DEVICE,
    )
    magmoms = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    node_attrs = torch.ones(4, 1, dtype=DTYPE, device=DEVICE)
    node_fidelity = torch.zeros(4, dtype=torch.long, device=DEVICE)

    radial, magnetic_node_attrs, magnetic_edge_attrs = basis(
        magmoms,
        node_attrs,
        edge_index,
        node_fidelity,
    )
    source, target = edge_index
    expected = basis.magnetic_edge_tensor_product(
        magnetic_node_attrs[target],
        magnetic_node_attrs[source],
    )
    scaled_magmoms = magmoms / basis.magnetic_scale[0]
    squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
    expected_radial_coordinate = 1.0 - 2.0 * torch.clamp(
        squared_magnitude,
        min=0.0,
        max=1.0,
    )

    assert basis.angular_basis.normalization == "integral"
    assert not basis.angular_basis.normalize
    assert basis.angular_basis.irreps_in[0].ir.p == (1 if parity else -1)
    assert basis.angular_basis.irreps_out == basis.magnetic_node_irreps_out
    for _, ir in basis.magnetic_node_irreps_out:
        assert ir.p == (1 if parity else (-1) ** ir.l)
        assert getattr(ir, "t", 1) == ((-1) ** ir.l if time_reversal else 1)
    if not parity:
        assert all(ir.p == (-1) ** ir.l for _, ir in basis.magnetic_edge_irreps_out)
    assert basis.radial_normalization == "clamp"
    torch.testing.assert_close(
        basis.magnetic_scale,
        torch.tensor([[2.0]], dtype=DTYPE, device=DEVICE),
    )
    assert basis.magnetic_edge_tensor_product.weight_numel == 0
    assert basis.magnetic_node_irreps_out.lmax == Lmax
    assert basis.magnetic_edge_irreps_out.lmax == Lmax
    assert basis.magnetic_edge_tensor_product.irreps_out.simplify() == (
        basis.magnetic_edge_irreps_out
    )
    assert radial.shape == (4, 4)
    assert magnetic_node_attrs.shape[-1] == basis.magnetic_node_irreps_out.dim
    assert magnetic_edge_attrs.shape == (
        edge_index.size(1),
        basis.magnetic_edge_irreps_out.dim,
    )
    torch.testing.assert_close(radial, basis.radial_basis(expected_radial_coordinate))
    torch.testing.assert_close(
        magnetic_node_attrs,
        basis.angular_basis(magmoms),
    )
    torch.testing.assert_close(magnetic_edge_attrs, expected)

    if time_reversal:
        reversed_radial, reversed_node, reversed_edge = basis(
            -magmoms,
            node_attrs,
            edge_index,
            node_fidelity,
        )
        torch.testing.assert_close(reversed_radial, radial)
        torch.testing.assert_close(
            reversed_node,
            _time_reverse(magnetic_node_attrs, basis.magnetic_node_irreps_out),
        )
        torch.testing.assert_close(
            reversed_edge,
            _time_reverse(magnetic_edge_attrs, basis.magnetic_edge_irreps_out),
        )


def test_magnetic_basis_clamps_radial_coordinate_smoothly_at_zero():
    basis = MagneticBasis(
        [{26: 2.0, 28: 4.0}],
        num_mag_radial_basis=4,
        Lmax=2,
        atomic_numbers=[26, 28],
    ).to(DEVICE, DTYPE)
    magmoms = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, -2.0, 3.0]],
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    node_attrs = torch.eye(2, dtype=DTYPE, device=DEVICE)
    edge_index = torch.tensor([[0, 1], [1, 0]], device=DEVICE)
    node_fidelity = torch.zeros(2, dtype=torch.long, device=DEVICE)

    radial, magnetic_node_attrs, _ = basis(
        magmoms, node_attrs, edge_index, node_fidelity
    )
    magnetic_scale = basis.magnetic_scale[0].unsqueeze(-1)
    scaled_magmoms = magmoms / magnetic_scale
    squared_magnitude = scaled_magmoms.square().sum(dim=-1, keepdim=True)
    radial_coordinate = 1.0 - 2.0 * torch.clamp(
        squared_magnitude,
        min=0.0,
        max=1.0,
    )

    assert basis.angular_basis.normalization == "integral"
    torch.testing.assert_close(radial, basis.radial_basis(radial_coordinate))
    torch.testing.assert_close(
        magnetic_node_attrs,
        basis.angular_basis(magmoms),
    )
    zero_gradient = torch.autograd.grad(radial[0].sum(), magmoms, create_graph=True)[0][
        0
    ]
    torch.testing.assert_close(zero_gradient, torch.zeros_like(zero_gradient))
    zero_hessian = torch.stack(
        [
            torch.autograd.grad(zero_gradient[axis], magmoms, retain_graph=True)[0][0]
            for axis in range(3)
        ]
    )
    orders = torch.arange(1, basis.num_mag_radial_basis + 1, dtype=DTYPE, device=DEVICE)
    expected_curvature = (
        -4.0 * orders.square().sum() / basis.magnetic_scale[0, 0].square()
    )
    torch.testing.assert_close(
        zero_hessian,
        expected_curvature * torch.eye(3, dtype=DTYPE, device=DEVICE),
    )


@pytest.mark.parametrize("Lmax", [1, 2, 3])
def test_magnetic_basis_without_soc_keeps_independent_spin_scalars(Lmax):
    basis = MagneticBasis(
        [{26: 2.0}],
        num_mag_radial_basis=4,
        Lmax=Lmax,
        atomic_numbers=[26],
        time_reversal=hasattr(o3.Irrep("0e"), "t"),
        use_spin_orbit_coupling=False,
    ).to(DEVICE, DTYPE)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        device=DEVICE,
    )
    magmoms = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    node_attrs = torch.ones(4, 1, dtype=DTYPE, device=DEVICE)
    node_fidelity = torch.zeros(4, dtype=torch.long, device=DEVICE)

    _, _, edge_scalars = basis(magmoms, node_attrs, edge_index, node_fidelity)
    rotation = o3.rand_matrix(dtype=DTYPE, device=DEVICE)
    _, _, rotated_edge_scalars = basis(
        magmoms @ rotation.T,
        node_attrs,
        edge_index,
        node_fidelity,
    )
    _, _, reversed_edge_scalars = basis(-magmoms, node_attrs, edge_index, node_fidelity)

    assert not basis.use_spin_orbit_coupling
    assert basis.magnetic_edge_irreps_out.num_irreps == Lmax + 1
    assert len(basis.magnetic_edge_tensor_product.instructions) == Lmax + 1
    assert all(
        ir.l == 0 and ir.p == 1 and getattr(ir, "t", 1) == 1
        for _, ir in basis.magnetic_edge_irreps_out
    )
    torch.testing.assert_close(rotated_edge_scalars, edge_scalars)
    torch.testing.assert_close(reversed_edge_scalars, edge_scalars)


def test_magnetic_radial_basis_is_bounded_and_has_no_constant_mode():
    basis = MagneticBasis(
        [{26: 2.0}],
        num_mag_radial_basis=4,
        Lmax=2,
        atomic_numbers=[26],
    ).to(DEVICE, DTYPE)
    node_attrs = torch.ones(2, 1, dtype=DTYPE, device=DEVICE)
    edge_index = torch.tensor([[0], [1]], device=DEVICE)
    node_fidelity = torch.zeros(2, dtype=torch.long, device=DEVICE)
    magmoms = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0e8, -2.0e8, 3.0e8]],
        dtype=DTYPE,
        device=DEVICE,
    )

    radial, angular, _ = basis(magmoms, node_attrs, edge_index, node_fidelity)

    assert not basis.radial_basis.include_constant
    torch.testing.assert_close(radial[0], torch.ones_like(radial[0]))
    assert torch.isfinite(angular).all()
    torch.testing.assert_close(angular, basis.angular_basis(magmoms))


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


def _magnetic_scatter_module(use_attention: bool):
    irreps = o3.Irreps("2x0ee+2x1oe")
    magnetic_edge_irreps = o3.Irreps("2x0ee+2x1eo+2x1ee")
    module = O2ScatterMagneticTensorProduct(
        irreps,
        irreps,
        magnetic_edge_irreps,
        num_channel=2,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=1,
        num_radial_basis=4,
        use_radial_rotary_attention=use_attention,
    ).to(DEVICE, DTYPE)
    return module, irreps, magnetic_edge_irreps


@pytest.mark.parametrize("use_attention", [False, True])
@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_o2_magnetic_scatter_is_time_reversal_invariant(use_attention):
    module, irreps, magnetic_edge_irreps = _magnetic_scatter_module(use_attention)
    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], device=DEVICE)
    num_edges = edge_index.size(1)
    node_features = torch.randn(4, irreps.dim, dtype=DTYPE, device=DEVICE)
    magnetic_edge_attrs = torch.randn(
        num_edges, magnetic_edge_irreps.dim, dtype=DTYPE, device=DEVICE
    )
    conv_weights = torch.randn(
        num_edges, module.weight_numel, dtype=DTYPE, device=DEVICE
    )
    edge_vectors = torch.randn(num_edges, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).to(DEVICE, DTYPE)(edge_vectors)
    edge_cutoff = torch.rand(num_edges, 1, dtype=DTYPE, device=DEVICE)
    edge_radial_basis = torch.randn(num_edges, 4, dtype=DTYPE, device=DEVICE)

    def apply(edge_attrs):
        return module(
            node_features,
            edge_attrs,
            conv_weights,
            edge_index,
            wigner,
            wigner_inv,
            edge_radial_basis=edge_radial_basis,
            edge_cutoff=edge_cutoff,
        )

    output = apply(magnetic_edge_attrs)
    torch.testing.assert_close(
        apply(_time_reverse(magnetic_edge_attrs, magnetic_edge_irreps)),
        output,
    )
    assert not torch.allclose(output, apply(torch.zeros_like(magnetic_edge_attrs)))


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_o2_magnetic_scatter_supports_empty_edges():
    module, irreps, magnetic_edge_irreps = _magnetic_scatter_module(False)
    assert "reshape_in" not in repr(module)
    assert "reshape_out" not in repr(module)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).to(DEVICE, DTYPE)(
        torch.empty(0, 3, dtype=DTYPE, device=DEVICE)
    )
    output = module(
        torch.randn(3, irreps.dim, dtype=DTYPE, device=DEVICE),
        torch.empty(
            0,
            magnetic_edge_irreps.dim,
            dtype=DTYPE,
            device=DEVICE,
        ),
        torch.empty(0, module.weight_numel, dtype=DTYPE, device=DEVICE),
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=torch.empty(0, 1, dtype=DTYPE, device=DEVICE),
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize(
    ("irreps_in", "irreps_sh", "irreps_out"),
    [
        ("2x3o", "0e+1o", "2x2e"),
        ("2x1o", "0e+1o+2e+3o", "2x2e"),
        ("2x1o", "0e+1o", "2x2e"),
    ],
)
def test_o2_cgtp_infers_degrees_and_accepts_larger_shared_wigner(
    double_precision, monkeypatch, irreps_in, irreps_sh, irreps_out
):
    from types import SimpleNamespace

    from tace.models._e3nn.fused import (
        O2CgtpScatterTensorProduct,
        O3ScatterTensorProduct,
    )

    for name in ("TACE_USE_OEQ", "TACE_USE_CUE"):
        monkeypatch.setenv(name, "0")
    module = O2CgtpScatterTensorProduct(irreps_in, irreps_sh, irreps_out)
    reference = O3ScatterTensorProduct(irreps_in, irreps_sh, irreps_out)
    lmax = max(o3.Irreps(irreps).lmax for irreps in (irreps_in, irreps_out))
    assert module.tp.lmax == lmax
    assert module.irreps_out == reference.irreps_out
    assert module.weight_numel == reference.weight_numel

    x = torch.randn(3, o3.Irreps(irreps_in).dim, requires_grad=True)
    r = torch.randn(6, 3, requires_grad=True)
    w = torch.randn(6, module.weight_numel, requires_grad=True)
    edge_index = torch.tensor([[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]])
    graph = SimpleNamespace(
        edge_vector=r,
        edge_length=r.square().sum(-1, keepdim=True).sqrt() + 1e-9,
    )
    d, di = o2.WignerD(lmax + 2, lmax + 2)(r)
    actual = module(x, w, edge_index, d, di, graph)
    expected = reference(
        x,
        o3.spherical_harmonics(
            o3.Irreps(irreps_sh), r / graph.edge_length, False, "component"
        ),
        w,
        edge_index,
    )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.square().sum(), (x, r, w), retain_graph=True),
        torch.autograd.grad(expected.square().sum(), (x, r, w)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad, atol=2e-8, rtol=2e-9)


@pytest.mark.parametrize(
    ("interaction", "node_embedding", "Lmax", "lmax"),
    [
        ("o2_cgtp", "linear", 2, 2),
        (["cgtp", "o2_cgtp"], "linear", 2, 2),
        ("o2_cgtp", "tensor", 2, 2),
        ("o2_cgtp", "linear", 4, 2),
        ("o2_cgtp", "linear", 1, 3),
    ],
)
def test_o2_cgtp_model_matches_energy_forces_stress_and_training(
    double_precision,
    monkeypatch,
    tmp_path,
    interaction,
    node_embedding,
    Lmax,
    lmax,
):
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    for name in ("TACE_USE_EQT", "TACE_USE_OEQ", "TACE_USE_CUE", "TACE_USE_EQX"):
        monkeypatch.setenv(name, "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=4.0,
        max_neighbors=None,
        num_layers=2,
        num_channel=3,
        Lmax=Lmax,
        lmax=lmax,
        mmax=0,
        parity=True,
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
        ],
        target_property=["energy", "forces", "stress", "virials"],
    )
    config["node_embedding"]["type"] = node_embedding
    config["atomic_basis"]["type"] = "cgtp"
    config["readout_emlp"]["use_one_body_magmoms"] = False
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [2]
    config["scale_shift"]["enable"] = False
    reference = TensorModel(e3nnTACE(**deepcopy(config))).train()
    config["atomic_basis"]["type"] = interaction
    module = TensorModel(e3nnTACE(**config)).train()
    reference_parameters = dict(reference.named_parameters())
    assert reference_parameters.keys() == dict(module.named_parameters()).keys()
    with torch.no_grad():
        for name, parameter in module.named_parameters():
            parameter.copy_(reference_parameters[name])
    representation = module.readout_fn.representation
    assert representation.o2_angular_basis.mmax == max(Lmax, lmax)
    if interaction == "o2_cgtp" and node_embedding == "linear":
        assert not representation.use_o3_angular_basis

        def no_spherical_harmonics(*args):
            raise AssertionError("o2_cgtp should not evaluate spherical harmonics")

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
            ]
        ),
        node_attrs=torch.ones(5, 1),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2, 3, 4], [1, 0, 2, 0, 2, 1, 4, 3]]),
        edge_shifts=torch.zeros(8, 3),
        lattice=torch.eye(3).repeat(2, 1, 1) * 8,
        batch=torch.tensor([0, 0, 0, 1, 1]),
        ptr=torch.tensor([0, 3, 5]),
        fidelity_idx=torch.zeros(2, dtype=torch.long),
    )
    expected = reference({key: value.clone() for key, value in data.items()})
    actual = module({key: value.clone() for key, value in data.items()})
    for key in ("energy", "forces", "stress", "virials"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-9, rtol=2e-8)
    for model, output in ((reference, expected), (module, actual)):
        sum(
            output[key].square().sum() for key in ("energy", "forces", "stress")
        ).backward()
    for name, parameter in module.named_parameters():
        expected_grad = reference_parameters[name].grad
        if expected_grad is not None:
            torch.testing.assert_close(
                parameter.grad, expected_grad, atol=2e-8, rtol=2e-7
            )

    if isinstance(interaction, list):
        from tace.lightning import convert_cgtp

        converted = convert_cgtp(module)
        assert converted.readout_fn.model_config["atomic_basis"]["type"] == [
            "o2_cgtp",
            "cgtp",
        ]
        output = converted({key: value.clone() for key, value in data.items()})
        for key in ("energy", "forces", "stress", "virials"):
            torch.testing.assert_close(output[key], expected[key], atol=2e-9, rtol=2e-8)

    if interaction == "o2_cgtp" and node_embedding == "linear" and Lmax == lmax == 2:
        from tace.models.compile.compile import trace_to_fx
        from tace.models.compile.wrapper import (
            CompileTensorModel,
            _FlatE3nnCompileModel,
        )

        compiled_model = CompileTensorModel(module.readout_fn).eval()
        input_keys = compiled_model._input_keys(data)
        output_keys = compiled_model._output_keys()
        flat_model = _FlatE3nnCompileModel(compiled_model, input_keys, output_keys)
        inputs = tuple(data[key] for key in input_keys)
        traced = trace_to_fx(flat_model, inputs)
        for key, output in zip(output_keys, traced(*inputs)):
            torch.testing.assert_close(output, actual[key], atol=2e-9, rtol=2e-8)

    if interaction == "o2_cgtp" and node_embedding == "linear" and Lmax == lmax == 2:
        from tace.lightning import convert_cgtp, export_tace, load_tace

        reference.eval()
        next(reference.parameters()).requires_grad_(False)
        reference.retain_graph = True
        converted = convert_cgtp(reference)
        assert not converted.training
        assert converted.retain_graph
        assert reference.readout_fn.model_config["atomic_basis"]["type"] == "cgtp"
        for (name, parameter), (other_name, other) in zip(
            reference.named_parameters(), converted.named_parameters()
        ):
            assert name == other_name
            assert parameter.requires_grad == other.requires_grad
            assert parameter.data_ptr() != other.data_ptr()
            torch.testing.assert_close(parameter, other, atol=0, rtol=0)
        restored = convert_cgtp(converted)
        path = str(tmp_path / "o2_cgtp.pt")
        export_tace(converted, path)
        reloaded = load_tace(path, device="cpu").eval()
        assert reloaded.readout_fn.model_config["atomic_basis"]["type"] == [
            "o2_cgtp",
            "o2_cgtp",
        ]
        for model in (converted, restored, reloaded):
            output = model({key: value.clone() for key, value in data.items()})
            for key in ("energy", "forces", "stress", "virials"):
                torch.testing.assert_close(
                    output[key], expected[key], atol=2e-9, rtol=2e-8
                )
        with pytest.raises(ValueError, match="implementation"):
            convert_cgtp(reference, "o2_linear")
        with pytest.raises(TypeError, match="eager"):
            convert_cgtp(torch.nn.Identity())

        from tace.scripts.convert_cgtp import main

        path = tmp_path / "cli.pt"
        export_tace(reference, str(path))
        source_bytes = path.read_bytes()
        for kind in ("o2_cgtp", "cgtp"):
            monkeypatch.setattr(
                sys,
                "argv",
                [
                    "tace-convert-cgtp",
                    "-m",
                    str(path),
                    "--dtype",
                    "float64",
                    "--device",
                    "cpu",
                ],
            )
            main()
            path = path.with_name(f"{path.stem}-converted.pt")
            converted = load_tace(path, device="cpu")
            assert converted.readout_fn.model_config["atomic_basis"]["type"] == [
                kind,
                kind,
            ]
            assert converted.get_model_dtype() == torch.float64
            for name, parameter in converted.named_parameters():
                torch.testing.assert_close(
                    parameter, reference_parameters[name], atol=0, rtol=0
                )
            output = converted({key: value.clone() for key, value in data.items()})
            for key in ("energy", "forces", "stress", "virials"):
                torch.testing.assert_close(
                    output[key], expected[key], atol=2e-9, rtol=2e-8
                )
        assert (tmp_path / "cli.pt").read_bytes() == source_bytes

    data["edge_index"] = torch.empty(2, 0, dtype=torch.long)
    data["edge_shifts"] = torch.empty(0, 3)
    expected = reference({key: value.clone() for key, value in data.items()})
    actual = module({key: value.clone() for key, value in data.items()})
    for key in ("energy", "forces", "stress", "virials"):
        torch.testing.assert_close(actual[key], expected[key], atol=2e-9, rtol=2e-8)

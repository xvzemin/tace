################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from copy import deepcopy
from pathlib import Path

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
from tace.models.layout import LayoutTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


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


def _spin_node_embedding(embedding_type):
    return embedding_type(
        num_elements=2,
        num_radial_basis=4,
        num_mag_radial_basis=3,
        num_channel=5,
        Lmax=1,
        lmax=1,
        avg_num_neighbors=2.0,
        bias=False,
    )


def test_linear_spin_node_embedding():
    embedding = _spin_node_embedding(LinearSpinNodeEmbedding)
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

    assert NODE_EMBEDDING["linear_spin"] is LinearSpinNodeEmbedding
    assert embedding.spin_embedding.irreps_in == o3.Irreps("3x0e")
    torch.testing.assert_close(output, expected)


def test_nonlinear_spin_node_embedding():
    embedding = _spin_node_embedding(NonLinearSpinNodeEmbedding)
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
    linear_output = (
        embedding.element_embedding(node_attrs)
        + embedding.spin_embedding(magnetic_radial_basis)
    ) / (2.0**0.5)

    assert NODE_EMBEDDING["nonlinear_spin"] is NonLinearSpinNodeEmbedding
    torch.testing.assert_close(output, embedding.activation(linear_output))


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
        output @ embedding.irreps_out.D_from_matrix(rotation).T,
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


def _transform(features, irreps, angle, reflected=False, time_reversal=False):
    matrix = irreps.D_from_angle(
        angle,
        reflected=reflected,
        time_reversal=time_reversal,
        dtype=features.dtype,
        device=features.device,
    )
    return torch.matmul(features, matrix.transpose(-1, -2))


def _time_reverse(features: torch.Tensor, irreps) -> torch.Tensor:
    output = features.clone()
    for ir_mul, ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir_mul.ir.t
    return output


@pytest.mark.parametrize(
    ("rotation_matrix", "axis"),
    [
        (o2.rotation_matrix_to_x_axis, 0),
        (o2.rotation_matrix_to_y_axis, 1),
        (o2.rotation_matrix_to_z_axis, 2),
    ],
)
def test_rotation_matrix_to_axis(rotation_matrix, axis):
    generator = torch.Generator().manual_seed(20260903)
    vectors = torch.randn(64, 3, dtype=DTYPE, generator=generator).to(DEVICE)
    rotation = rotation_matrix(vectors)
    rotated = torch.einsum("bij,bj->bi", rotation, vectors)
    expected = torch.zeros_like(vectors)
    expected[:, axis] = torch.linalg.vector_norm(vectors, dim=-1)

    torch.testing.assert_close(rotated, expected)
    identity = torch.eye(3, dtype=DTYPE, device=DEVICE).expand_as(rotation)
    torch.testing.assert_close(rotation @ rotation.transpose(-1, -2), identity)
    torch.testing.assert_close(
        torch.linalg.det(rotation),
        torch.ones(vectors.size(0), dtype=DTYPE, device=DEVICE),
    )


@pytest.mark.parametrize(("Lmax", "lmax"), [(2, 3), (3, 2)])
def test_o2_representation_uses_common_angular_coverage(Lmax, lmax):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["node_embedding"]["type"] = "linear"
    config["atomic_basis"]["type"] = ["o2"]
    config["atomic_basis"]["nonlinear"] = ["gate"]
    config["atomic_basis"]["edge_nonlinear"] = ["gate"]
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


def test_o2_does_not_import_tace():
    directory = Path(__file__).resolve().parents[1] / "eqx" / "o2"
    for source_path in directory.glob("*.py"):
        source = source_path.read_text()
        assert "from tace" not in source
        assert "import tace" not in source


@pytest.mark.parametrize("wigner_lmax", [2, 4])
def test_local_frame_roundtrip_flattened_ir_mul(wigner_lmax):
    irreps = o3.Irreps("2x0e+1x0o+3x1e+2x1o+1x2e+3x2o")
    frame = o2.LocalFrame(irreps).to(DEVICE, DTYPE)
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
    assert repr(frame) == (
        f"LocalFrame({frame.global_irreps} -> {frame.local_irreps})(mmax=2)"
    )
    reverse_frame = o2.LocalFrame(irreps, reverse=True)
    assert repr(reverse_frame) == (
        f"LocalFrame({frame.local_irreps} -> {frame.global_irreps})(mmax=2)"
    )
    assert local.shape == (7, frame.irreps_out.dim)
    torch.testing.assert_close(
        layout.inverse(frame.to_global(local, wigner_inv)),
        features,
    )


@pytest.mark.parametrize("mmax", [0, 1])
@pytest.mark.parametrize(
    ("wigner_mmax", "wigner_lmax"), [(1, 2), (1, 4), (2, 4), (4, 4)]
)
def test_local_frame_trailing_axes_and_empty_batch(mmax, wigner_mmax, wigner_lmax):
    irreps = o3.Irreps("2x0e+2x1o+2x2e")
    frame = o2.LocalFrame(irreps, mmax=mmax).to(DEVICE, DTYPE)
    vectors = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(wigner_mmax, wigner_lmax).to(DEVICE, DTYPE)(vectors)
    reference, reference_inv = o2.WignerD(mmax, 2).to(DEVICE, DTYPE)(vectors)
    features = torch.randn(4, 2, irreps.dim, dtype=DTYPE, device=DEVICE)

    local = frame.to_local(features, wigner)
    assert local.shape == (4, 2, frame.irreps_out.dim)
    torch.testing.assert_close(local, frame.to_local(features, reference))
    torch.testing.assert_close(
        frame.to_global(local, wigner_inv), frame.to_global(local, reference_inv)
    )

    empty = frame.to_local(features[:0], wigner[:0])
    assert empty.shape == (0, 2, frame.irreps_out.dim)
    assert frame.to_global(empty, wigner_inv[:0]).shape == (0, 2, irreps.dim)


@pytest.mark.parametrize(
    ("local_dim", "global_dim", "message"),
    [(4, 4, "degree"), (9, 10, "degree"), (5, 9, "orders"), (6, 9, "orders")],
)
def test_local_frame_rejects_incompatible_wigner_layout(local_dim, global_dim, message):
    frame = o2.LocalFrame("2x2e", mmax=4)
    assert frame.mmax == 2
    wigner = torch.zeros(3, local_dim, global_dim)
    with pytest.raises(ValueError, match=message):
        frame.to_local(torch.zeros(3, frame.input_dim), wigner)
    with pytest.raises(ValueError, match=message):
        frame.to_global(torch.zeros(3, frame.output_dim), wigner.transpose(1, 2))


def test_local_frame_empty_irreps():
    frame = o2.LocalFrame("")
    d, di = o2.WignerD(2, 2)(torch.randn(3, 3))
    for batch_size in (3, 0):
        features = torch.empty(batch_size, 2, 0)
        local = frame(features, d[:batch_size])
        assert local.shape == features.shape
        assert frame.to_global(local, di[:batch_size]).shape == features.shape


def test_local_frame_truncation_compiles_with_shared_wigner(o2_dtype):
    frame = o2.LocalFrame("2x0e+1x1e+2x2o", mmax=1)

    def roundtrip(features, wigner, wigner_inv):
        return frame.to_global(frame.to_local(features, wigner), wigner_inv)

    compiled = torch.compile(
        roundtrip, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for mmax, lmax in ((1, 2), (2, 4)):
        for batch_size in (3, 1, 0):
            x = torch.randn(batch_size, 2, frame.input_dim, requires_grad=True)
            r = torch.randn(batch_size, 3, requires_grad=True)
            d, di = o2.WignerD(mmax, lmax)(r)
            actual = compiled(x, d, di)
            x_ref = x.detach().requires_grad_()
            r_ref = r.detach().requires_grad_()
            d, di = o2.WignerD(mmax, lmax)(r_ref)
            expected = roundtrip(x_ref, d, di)
            torch.testing.assert_close(actual, expected)
            for actual_grad, expected_grad in zip(
                torch.autograd.grad(actual.square().sum(), (x, r)),
                torch.autograd.grad(expected.square().sum(), (x_ref, r_ref)),
            ):
                torch.testing.assert_close(actual_grad, expected_grad)


def test_o2_irrep_and_irreps_metadata():
    assert o2.Irrep("0e") == o2.Irrep("0ee") == o2.Irrep(0, 1)
    assert o2.Irrep("0o") == o2.Irrep("0oe") == o2.Irrep((0, -1))
    assert o2.Irrep("0eo") == o2.Irrep((0, 1, -1))
    assert o2.Irrep("1mo") == o2.Irrep(1, 0, -1)
    assert str(o2.Irrep("0eo")) == "0eo"
    assert str(o2.Irrep("2mo")) == "2mo"
    assert o2.Irrep("3m").dim == 2
    assert o2.Irrep("0e").is_invariant_scalar()

    angle = torch.tensor(0.37, dtype=DTYPE)
    time_odd = o2.Irrep("1mo")
    torch.testing.assert_close(
        time_odd.D_from_angle(angle, time_reversal=True),
        -time_odd.D_from_angle(angle),
    )

    irreps = o2.Irreps("2x0e+0o+3x1m+2m")
    assert irreps.dim == 11
    assert irreps.num_irreps == 7
    assert irreps.mmax == 2
    assert irreps.slices() == (
        slice(0, 2),
        slice(2, 3),
        slice(3, 9),
        slice(9, 11),
    )
    assert irreps.regroup() == o2.Irreps("2x0e+0o+3x1m+2m")


def test_o2_irreps_sort_and_serialization():
    import pickle

    irreps = o2.Irreps("2x1mo+0oo+3x0ee+1me+0eo")
    result = irreps.sort()
    assert result.irreps == o2.Irreps("3x0ee+0eo+0oo+1me+2x1mo")
    for i, ir_mul in enumerate(irreps):
        assert result.irreps[result.p[i]] == ir_mul
        assert result.inv[result.p[i]] == i
    assert deepcopy(irreps) == pickle.loads(pickle.dumps(irreps)) == irreps
    assert o2.Irrep("1mo") in irreps
    assert "2me" not in irreps
    assert 3 * o2.Irrep("1mo") == o2.Irreps("3x1mo")
    assert o2.Irrep("0ee") + o2.Irrep("1mo") == o2.Irreps("0ee+1mo")
    assert 0 * irreps == o2.Irreps()
    with pytest.raises(AttributeError, match="immutable"):
        irreps._irreps = ()

    module = o2.Linear(irreps, irreps)
    features = irreps.randn(4, -1)
    torch.testing.assert_close(deepcopy(module)(features), module(features))
    torch.testing.assert_close(
        pickle.loads(pickle.dumps(module))(features), module(features)
    )


def test_o2_irrep_products_and_restriction():
    assert o2.Irrep("0o") * o2.Irrep("0o") == (o2.Irrep("0e"),)
    assert o2.Irrep("1m") * o2.Irrep("2m") == (
        o2.Irrep("1m"),
        o2.Irrep("3m"),
    )
    assert o2.Irrep("2m") * o2.Irrep("2m") == (
        o2.Irrep("0e"),
        o2.Irrep("0o"),
        o2.Irrep("4m"),
    )
    assert o2.Irrep("0oo") * o2.Irrep("1mo") == (o2.Irrep("1me"),)
    assert o2.Irrep("1mo") * o2.Irrep("1mo") == (
        o2.Irrep("0ee"),
        o2.Irrep("0oe"),
        o2.Irrep("2me"),
    )
    assert o2.Irrep("1me") * o2.Irrep("2mo") == (
        o2.Irrep("1mo"),
        o2.Irrep("3mo"),
    )
    assert o2.LocalFrame.restrict("2x1e+0o") == o2.Irreps("3x0o+2x1m")


@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_o2_irreps_randn_uses_flattened_ir_mul(normalization):
    irreps = o2.Irreps("2x0e+0o+3x1m+2m")
    sample = irreps.randn(
        5,
        -1,
        4,
        normalization=normalization,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    assert sample.shape == (5, irreps.dim, 4)
    if normalization == "norm":
        for (ir, mul), ir_slice in zip(irreps, irreps.slices()):
            values = sample[:, ir_slice].reshape(5, ir.dim, mul, 4)
            torch.testing.assert_close(
                values.norm(dim=1),
                torch.ones(5, mul, 4, dtype=DTYPE, device=DEVICE),
            )


def test_o2_direct_sum_matrix_uses_ir_mul_layout():
    irreps = o2.Irreps("0e+0o+2x1m")
    angle = torch.tensor([0.2, -0.7], dtype=DTYPE, device=DEVICE)
    actual = irreps.D_from_angle(angle, reflected=True)
    one = o2.Irrep("1m").D_from_angle(angle, reflected=True)
    identity = torch.eye(2, dtype=DTYPE, device=DEVICE)
    expected = torch.einsum("bij,uv->biujv", one, identity).reshape(2, 4, 4)

    torch.testing.assert_close(actual[:, 0, 0], torch.ones_like(angle))
    torch.testing.assert_close(actual[:, 1, 1], -torch.ones_like(angle))
    torch.testing.assert_close(actual[:, 2:, 2:], expected)


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("reflected", [False, True])
def test_circular_harmonics_is_equivariant(normalize, reflected):
    module = o2.CircularHarmonics(4, normalize=normalize).to(DEVICE, DTYPE)
    vectors = torch.randn(9, 2, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    transformed = _transform(vectors, module.irreps_in, angle, reflected)
    expected = _transform(module(vectors), module.irreps_out, angle, reflected)
    torch.testing.assert_close(module(transformed), expected)


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_linear_is_equivariant(reflected):
    module = o2.Linear(
        "2x0e+3x1m",
        "4x0e+2x1m+0o",
        biases=True,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.41, dtype=DTYPE, device=DEVICE)
    expected = _transform(module(features), module.irreps_out, angle, reflected)
    actual = module(_transform(features, module.irreps_in, angle, reflected))
    torch.testing.assert_close(actual, expected)


def test_o2_linear_external_weights_broadcast_and_zero_pad():
    module = o2.Linear(
        "2x0e+3x1m",
        "4x0e+2x1m+0o",
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(5, module.weight_numel, dtype=DTYPE, device=DEVICE)
    output = module(features, weights)
    reference = torch.stack(
        [module(features[index], weights[index]) for index in range(5)]
    )
    torch.testing.assert_close(output, reference)
    torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]))

    singleton = torch.randn(1, module.weight_numel, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        module(features, singleton), module(features, singleton[0])
    )


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_gate_is_equivariant(reflected):
    module = o2.Gate(
        "2x0e+0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "3x0e",
        [torch.nn.Sigmoid()],
        "3x1m",
    ).to(DEVICE, DTYPE)
    features = module.irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.29, dtype=DTYPE, device=DEVICE)
    expected = _transform(module(features), module.irreps_out, angle, reflected)
    actual = module(_transform(features, module.irreps_in, angle, reflected))
    torch.testing.assert_close(actual, expected)


def test_o2_odd_scalar_activation_is_odd():
    activation = o2.Activation("2x0o", [torch.nn.Tanh()])
    features = torch.randn(4, 2, dtype=DTYPE)
    torch.testing.assert_close(activation(-features), -activation(features))


def _tensor_product_case(mode):
    if mode == "u1u":
        return "2x1m", "0e", "2x1m"
    if mode == "uuu":
        return "2x1m", "2x1m", "2x0e"
    return "2x1m", "3x1m", "4x0e"


@pytest.mark.parametrize("mode", ["u1u", "uuu", "uvw"])
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_tensor_product_is_equivariant(mode, reflected):
    irreps_in1, irreps_in2, irreps_out = _tensor_product_case(mode)
    module = o2.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        [(0, 0, 0, mode, True)],
        internal_weights=False,
        shared_weights=False,
    ).to(DEVICE, DTYPE)
    input1 = module.irreps_in1.randn(5, -1, dtype=DTYPE, device=DEVICE)
    input2 = module.irreps_in2.randn(5, -1, dtype=DTYPE, device=DEVICE)
    weights = torch.randn(5, module.weight_numel, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    output = module(input1, input2, weights)
    expected = _transform(output, module.irreps_out, angle, reflected)
    actual = module(
        _transform(input1, module.irreps_in1, angle, reflected),
        _transform(input2, module.irreps_in2, angle, reflected),
        weights,
    )
    torch.testing.assert_close(actual, expected)


def test_o2_tensor_product_zero_pads_missing_outputs():
    module = o2.TensorProduct(
        "2x1m",
        "0e",
        "2x1m+0o",
        [(0, 0, 0, "u1u", False)],
    )
    output = module(torch.randn(3, 4), torch.randn(3, 1))
    torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]))


@pytest.fixture
def o2_dtype():
    previous = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    yield
    torch.set_default_dtype(previous)


def _asymmetric_contractions(correlation=3, path_mode="sum"):
    irreps_in = o2.Irreps("2x0e+2x0o+2x1m")
    irreps_out = o2.Irreps("2x0e+2x0o+2x1m+2x2m")
    recursive = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="recursive",
        path_mode=path_mode,
    ).to(DEVICE, DTYPE)
    dense = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="dense",
        path_mode=path_mode,
    ).to(DEVICE, DTYPE)
    return recursive, dense


@pytest.mark.parametrize("correlation", [2, 3])
@pytest.mark.parametrize("path_mode", ["sum", "expand"])
@pytest.mark.parametrize("batch_size", [0, 4])
def test_o2_asymmetric_contraction_algorithms_match(correlation, path_mode, batch_size):
    recursive, dense = _asymmetric_contractions(correlation, path_mode)
    inputs = [
        recursive.irreps_in.randn(
            batch_size, -1, dtype=DTYPE, device=DEVICE, requires_grad=True
        )
        for _ in range(correlation)
    ]
    weights = torch.randn(
        batch_size,
        recursive.weight_numel,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    assert recursive.order_num_paths == dense.order_num_paths
    actual, expected = recursive(inputs, weights), dense(inputs, weights)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (*inputs, weights)),
        torch.autograd.grad(expected.square().sum(), (*inputs, weights)),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("algorithm", ["recursive", "dense"])
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_asymmetric_contraction_is_equivariant(algorithm, reflected):
    module = o2.AsymmetricContraction(
        "2x0e+2x0o+2x1m",
        "2x0e+2x0o+2x1m+2x2m",
        2,
        algorithm=algorithm,
    ).to(DEVICE, DTYPE)
    inputs = [
        module.irreps_in.randn(4, -1, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]
    weights = torch.randn(4, module.weight_numel, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.31, dtype=DTYPE, device=DEVICE)
    output = module(inputs, weights)
    expected = _transform(output, module.irreps_out, angle, reflected)
    transformed = [
        _transform(features, module.irreps_in, angle, reflected) for features in inputs
    ]
    torch.testing.assert_close(module(transformed, weights), expected)


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


def _scatter_module(use_attention):
    irreps = o3.Irreps("2x0e+2x1o")
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
    ).to(DEVICE, DTYPE)


@pytest.mark.parametrize("use_attention", [False, True])
def test_o2_scatter_is_o3_equivariant(use_attention):
    torch.manual_seed(7)
    module = _scatter_module(use_attention)
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

    rotation = o3.rand_matrix(dtype=DTYPE, device=DEVICE)
    matrix = module.irreps_in.D_from_matrix(rotation)
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


def test_o2_scatter_supports_empty_edges():
    module = _scatter_module(False)
    assert "reshape_in" not in repr(module)
    assert "reshape_out" not in repr(module)
    edge_index = torch.empty(2, 0, dtype=torch.long, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(1, 1).to(DEVICE, DTYPE)(
        torch.empty(0, 3, dtype=DTYPE, device=DEVICE)
    )
    output = module(
        module.irreps_in.randn(3, -1, dtype=DTYPE, device=DEVICE),
        torch.empty(0, module.weight_numel, dtype=DTYPE, device=DEVICE),
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=torch.empty(0, 1, dtype=DTYPE, device=DEVICE),
    )
    torch.testing.assert_close(output, torch.zeros_like(output))


@pytest.mark.parametrize("path_normalization", ["element", "path"])
@pytest.mark.parametrize("shared_weights", [False, True])
def test_o2_scalar_linear_matches_o3_normalization(
    o2_dtype, path_normalization, shared_weights
):
    irreps_in, irreps_out = "2x0e+3x0e+2x0o", "4x0e+2x0o"
    kwargs = dict(
        internal_weights=False,
        shared_weights=shared_weights,
        biases=shared_weights,
        path_normalization=path_normalization,
    )
    module = o2.Linear(irreps_in, irreps_out, **kwargs)
    reference = o3.Linear(irreps_in, irreps_out, **kwargs)
    assert module.weight_numel == reference.weight_numel
    x = torch.randn(5, module.irreps_in.dim, requires_grad=True)
    weight_shape = (
        (module.weight_numel,) if shared_weights else (5, module.weight_numel)
    )
    weight = torch.randn(weight_shape, requires_grad=True)
    bias = (
        torch.randn(module.bias_numel, requires_grad=True) if shared_weights else None
    )
    actual, expected = module(x, weight, bias), reference(x, weight, bias)
    torch.testing.assert_close(actual, expected)
    inputs = (x, weight, bias) if shared_weights else (x, weight)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), inputs),
        torch.autograd.grad(expected.square().sum(), inputs),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize("batch_size", [0, 3])
def test_o2_linear_broadcasts_bias_and_unconnected_outputs(o2_dtype, batch_size):
    module = o2.Linear(
        "2x0e+1m",
        "0e+1m+0o",
        internal_weights=False,
        shared_weights=False,
        biases=True,
    )
    x = torch.randn(1, 1, module.irreps_in.dim, requires_grad=True)
    weight = torch.randn(2, 1, module.weight_numel, requires_grad=True)
    bias = torch.randn(1, batch_size, module.bias_numel, requires_grad=True)
    output = module(x, weight, bias)
    expected = module(
        x.expand(2, batch_size, -1),
        weight.expand(2, batch_size, -1),
        bias.expand(2, batch_size, -1),
    )
    assert output.shape == (2, batch_size, module.irreps_out.dim)
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output[..., -1], torch.zeros_like(output[..., -1]))
    for grad in torch.autograd.grad(output.square().sum(), (x, weight, bias)):
        assert torch.isfinite(grad).all()


def test_o2_gated_linear_compiles_with_dynamic_batches(o2_dtype):
    gate = o2.Gate(
        "2x0ee+0oo",
        [torch.nn.SiLU(), torch.tanh],
        "4x0ee",
        [torch.sigmoid],
        "2x1me+2x1mo",
    )
    linear_up = o2.Linear(
        "2x0ee+0oo+2x1me+2x1mo",
        gate.irreps_in,
        internal_weights=False,
        shared_weights=False,
    )
    linear_down = o2.Linear(gate.irreps_out, linear_up.irreps_in)

    def forward(features, weight):
        return linear_down(gate(linear_up(features, weight)))

    compiled = torch.compile(forward, backend="aot_eager", fullgraph=True, dynamic=True)
    for batch_size in (3, 1, 0):
        features = linear_up.irreps_in.randn(batch_size, -1, requires_grad=True)
        weight = torch.randn(batch_size, linear_up.weight_numel, requires_grad=True)
        actual, expected = compiled(features, weight), forward(features, weight)
        torch.testing.assert_close(actual, expected)
        inputs = (features, weight, linear_down.weight)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), inputs),
            torch.autograd.grad(expected.square().sum(), inputs),
        ):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("irreps_out", ["", "0o"])
def test_o2_disconnected_operators_have_zero_gradients(o2_dtype, irreps_out):
    linear = o2.Linear("0e", irreps_out, instructions=[])
    tp = o2.TensorProduct("0e", "0e", irreps_out, [])
    for batch_size in (3, 0):
        x = torch.randn(batch_size, 1, requires_grad=True)
        y = torch.randn(batch_size, 1, requires_grad=True)
        for output, inputs in ((linear(x), (x,)), (tp(x, y), (x, y))):
            assert output.shape == (batch_size, linear.irreps_out.dim)
            torch.testing.assert_close(output, torch.zeros_like(output))
            for grad in torch.autograd.grad(output.sum(), inputs):
                torch.testing.assert_close(grad, torch.zeros_like(grad))


@pytest.mark.parametrize(
    ("scalars", "gates", "gated"),
    [("", "", ""), ("0o", "", ""), ("", "0o", "1m")],
)
def test_o2_gate_supports_empty_sectors(o2_dtype, scalars, gates, gated):
    module = o2.Gate(
        scalars,
        [torch.tanh] if scalars else [],
        gates,
        [torch.tanh] if gates else [],
        gated,
    )
    for batch_size in (4, 0):
        features = module.irreps_in.randn(batch_size, -1, requires_grad=True)
        output = module(features)
        assert output.shape == (batch_size, module.irreps_out.dim)
        transformed = _transform(features, module.irreps_in, torch.tensor(0.3), True)
        torch.testing.assert_close(
            module(transformed),
            _transform(output, module.irreps_out, torch.tensor(0.3), True),
        )
        (grad,) = torch.autograd.grad(output.sum(), (features,))
        assert torch.isfinite(grad).all()


@pytest.mark.parametrize("normalization", ["component", "norm", "none"])
@pytest.mark.parametrize(
    ("ir1", "ir2", "ir_out"),
    [
        ("0o", "1m", "1m"),
        ("1m", "1m", "0e"),
        ("1m", "1m", "0o"),
        ("1m", "2m", "1m"),
        ("1m", "2m", "3m"),
    ],
)
def test_o2_tensor_product_coupling_normalization(
    o2_dtype, normalization, ir1, ir2, ir_out
):
    ir1, ir2, ir_out = o2.Irrep(ir1), o2.Irrep(ir2), o2.Irrep(ir_out)
    module = o2.TensorProduct(
        ir1,
        ir2,
        ir_out,
        [(0, 0, 0, "uuu", False)],
        irrep_normalization=normalization,
        path_normalization="none",
    )
    coefficients = module(torch.eye(ir1.dim)[:, None], torch.eye(ir2.dim)[None, :])
    squared_norm = {"component": ir_out.dim, "norm": ir1.dim * ir2.dim, "none": 1}
    torch.testing.assert_close(
        coefficients.square().sum(), torch.tensor(float(squared_norm[normalization]))
    )


@pytest.mark.parametrize("batch_size", [0, 4])
@pytest.mark.parametrize(
    ("ir1", "ir2"), [("0oo", "1mo"), ("1me", "1mo"), ("1mo", "2me")]
)
def test_o2_uuu_matches_diagonal_uvw(o2_dtype, batch_size, ir1, ir2):
    channels = 16
    ir1, ir2 = o2.Irrep(ir1), o2.Irrep(ir2)
    irreps_out = o2.Irreps([(ir, channels) for ir in ir1 * ir2])
    kwargs = dict(
        internal_weights=False, shared_weights=False, path_normalization="none"
    )
    module = o2.TensorProduct(
        channels * ir1,
        channels * ir2,
        irreps_out,
        [(0, 0, i, "uuu", True) for i in range(len(irreps_out))],
        **kwargs,
    )
    reference = o2.TensorProduct(
        channels * ir1,
        channels * ir2,
        irreps_out,
        [(0, 0, i, "uvw", True) for i in range(len(irreps_out))],
        **kwargs,
    )
    x = torch.randn(batch_size, ir1.dim * channels, requires_grad=True)
    y = torch.randn(batch_size, ir2.dim * channels, requires_grad=True)
    weight = torch.randn(batch_size, module.weight_numel, requires_grad=True)
    reference_weight = torch.einsum(
        "bpu,uv,uw->bpuvw",
        weight.reshape(batch_size, len(irreps_out), channels),
        torch.eye(channels),
        torch.eye(channels),
    ).flatten(-4)
    actual, expected = module(x, y, weight), reference(x, y, reference_weight)
    torch.testing.assert_close(actual, expected)
    for a, b in zip(
        torch.autograd.grad(actual.square().sum(), (x, y, weight)),
        torch.autograd.grad(expected.square().sum(), (x, y, weight)),
    ):
        torch.testing.assert_close(a, b)


@pytest.mark.parametrize(("lmax", "mmax"), [(0, 0), (1, 1), (3, 1), (4, 2), (3, 3)])
@pytest.mark.parametrize("optimize", [False, True])
def test_wigner_matches_o3_rotation_matrices(o2_dtype, lmax, mmax, optimize):
    vectors = torch.randn(3, 3, generator=torch.Generator().manual_seed(7))
    rotation = o2.rotation_matrix_to_y_axis(vectors)
    module = o2.WignerD(mmax, lmax, use_opt_einsum_fx=optimize)
    actual, inverse = module(vectors)
    full = torch.stack(
        [
            torch.block_diag(
                *[o3.Irrep(l, 1).D_from_matrix(r) for l in range(lmax + 1)]
            )
            for r in rotation
        ]
    )
    expected = full.index_select(1, module.local_indices)
    # The reference converts the rotation matrix through Euler angles.
    torch.testing.assert_close(actual, expected, atol=2e-9, rtol=2e-9)
    torch.testing.assert_close(
        inverse, expected.transpose(1, 2) * module.inverse_scale, atol=2e-9, rtol=2e-9
    )
    assert not module.state_dict()
    empty, empty_inverse = module(vectors[:0])
    assert empty.shape == (0, *actual.shape[1:])
    assert empty_inverse.shape == (0, *inverse.shape[1:])


def test_time_odd_circular_harmonics_alternate_time_parity():
    harmonics = o2.CircularHarmonics(4, time_reversal=True).to(DEVICE, DTYPE)
    assert harmonics.irreps_in == o2.Irreps("1mo")
    assert harmonics.irreps_out == o2.Irreps("0ee+1mo+2me+3mo+4me")
    vectors = torch.randn(8, 2, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        harmonics(-vectors),
        _time_reverse(harmonics(vectors), harmonics.irreps_out),
    )


def test_o2_linear_preserves_time_parity():
    irreps = o2.Irreps("2x0ee+3x0eo+2x1me+3x1mo")
    linear = o2.Linear(irreps, irreps, biases=True).to(DEVICE, DTYPE)
    assert linear.bias_numel == 2
    assert all(
        linear.irreps_in[instruction.i_in].ir.t
        == linear.irreps_out[instruction.i_out].ir.t
        for instruction in linear._weight_instructions
    )
    features = irreps.randn(5, -1, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        linear(_time_reverse(features, irreps)),
        _time_reverse(linear(features), irreps),
    )


def test_o2_activation_and_gate_preserve_time_parity():
    odd_activation = o2.Activation("3x0eo", [torch.nn.Tanh()])
    features = torch.randn(4, 3, dtype=DTYPE)
    torch.testing.assert_close(odd_activation(-features), -odd_activation(features))

    gate = o2.Gate(
        "2x0ee",
        [torch.nn.SiLU()],
        "3x0oo",
        [torch.nn.Tanh()],
        "3x1mo",
    ).to(DEVICE, DTYPE)
    features = gate.irreps_in.randn(7, -1, dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(
        gate(_time_reverse(features, gate.irreps_in)),
        _time_reverse(gate(features), gate.irreps_out),
    )


def test_o2_tensor_product_preserves_time_parity():
    irreps_in = o2.Irreps("2x1mo")
    irreps_out = o2.Irreps("2x0ee+2x0oe+2x2me")
    tensor_product = o2.TensorProduct(
        irreps_in,
        irreps_in,
        irreps_out,
        [(0, 0, i_out, "uuu", False) for i_out in range(len(irreps_out))],
    ).to(DEVICE, DTYPE)
    first = irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    second = irreps_in.randn(6, -1, dtype=DTYPE, device=DEVICE)
    output = tensor_product(first, second)
    torch.testing.assert_close(
        tensor_product(
            _time_reverse(first, irreps_in),
            _time_reverse(second, irreps_in),
        ),
        _time_reverse(output, irreps_out),
    )


def test_o2_asymmetric_contraction_preserves_time_parity():
    irreps_in = o2.Irreps("2x0ee+2x0oo+2x1mo")
    contraction = o2.AsymmetricContraction(
        irreps_in,
        "2x0ee+2x1me",
        correlation=2,
        algorithm="recursive",
    ).to(DEVICE, DTYPE)
    inputs = [irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE) for _ in range(2)]
    weights = torch.randn(5, contraction.weight_numel, dtype=DTYPE, device=DEVICE)
    output = contraction(inputs, weights)
    torch.testing.assert_close(
        contraction([_time_reverse(value, irreps_in) for value in inputs], weights),
        _time_reverse(output, contraction.irreps_out),
    )


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_local_frame_preserves_time_parity():
    irreps = o3.Irreps("2x1eo+2x2ee")
    frame = o2.LocalFrame(irreps).to(DEVICE, DTYPE)
    edge_vectors = torch.randn(6, 3, dtype=DTYPE, device=DEVICE)
    wigner, _ = o2.WignerD(mmax=2, lmax=2).to(DEVICE, DTYPE)(edge_vectors)
    features = torch.randn(6, irreps.dim, dtype=DTYPE, device=DEVICE)
    local = frame.to_local(features, wigner)
    torch.testing.assert_close(
        frame.to_local(_time_reverse(features, irreps), wigner),
        _time_reverse(local, frame.irreps_out),
    )


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

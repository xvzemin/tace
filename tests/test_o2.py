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
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG, check_model_config
from tace.models._e3nn.edge import MAGNETIC_EDGE_UPDATE
from tace.models._e3nn.o2 import (
    O2ScatterMagneticTensorProduct,
    O2ScatterTensorProduct,
)
from tace.models._e3nn.representation import Representation
from tace.models.layout import LayoutTransform
from tace.models.magnetic import MagneticBasis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def test_magnetic_scale_type_selects_statistics():
    config = check_model_config(
        {
            "statistics": [
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 2.0,
                    "rms_initial_noncollinear_magmoms_norm_by_element": {26: 2.0},
                },
                {
                    "atomic_numbers": [26],
                    "avg_num_neighbors": 4.0,
                    "rms_initial_noncollinear_magmoms_norm_by_element": {26: 3.0},
                },
            ],
            "target_property": [],
            "scale_shift": {
                "magmoms_scale_type": (
                    "rms_initial_noncollinear_magmoms_norm_by_element"
                )
            },
        }
    )

    assert config["magmoms_scale_by_element"] == {26: 3.0}


def test_magnetic_basis_normalization_validation():
    basis = MagneticBasis(
        {26: 2.0},
        num_basis=4,
        Lmax=1,
        atomic_numbers=[26],
        num_elements=1,
    )
    assert basis.normalization == "integral"

    with pytest.raises(ValueError, match="'integral' or 'component'"):
        MagneticBasis(
            {26: 2.0},
            num_basis=4,
            Lmax=1,
            atomic_numbers=[26],
            num_elements=1,
            normalization="norm",
        )


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
        magmoms_scale_by_element=None,
        mmax=2,
        Lmax=Lmax,
        lmax=lmax,
        angular_basis={"magnetic_basis": {"Lmax": 1, "normalization": "integral"}},
        num_channel=2,
        target_irreps=o3.Irreps("0e"),
        node_embedding=config["node_embedding"],
        edge_embedding=config["edge_embedding"],
        edge_update=config["edge_update"],
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
        use_one_body_magmoms=False,
    )

    common_lmax = max(Lmax, lmax)
    assert representation.use_o2
    assert not representation.use_legacy_so2
    assert not representation.use_time_reversal
    assert representation.so2_angular_basis.lmax == common_lmax
    assert representation.interactions[0].rejector.local_frame_in.lmax == common_lmax


@pytest.mark.parametrize("update_type", ["identity", "element", "element2"])
def test_magnetic_edge_update_is_independent_per_interaction(update_type):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["magnetic_edge_update"]["type"] = update_type
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
        magmoms_scale_by_element={26: 2.0},
        mmax=1,
        Lmax=1,
        lmax=1,
        angular_basis={"magnetic_basis": {"Lmax": 1, "normalization": "integral"}},
        num_channel=2,
        target_irreps=o3.Irreps("0e"),
        node_embedding=config["node_embedding"],
        edge_embedding=config["edge_embedding"],
        edge_update=config["edge_update"],
        magnetic_edge_update=config["magnetic_edge_update"],
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
        use_one_body_magmoms=False,
    )

    num_mag_radial_basis = config["radial_basis"]["num_mag_radial_basis"] - 1
    assert len(representation.magnetic_edge_updates) == 2
    assert representation.magnetic_edge_updates[0] is not (
        representation.magnetic_edge_updates[1]
    )
    for update, interaction in zip(
        representation.magnetic_edge_updates,
        representation.interactions,
    ):
        assert isinstance(update, MAGNETIC_EDGE_UPDATE[update_type])
        assert interaction.edge_info.dims[0] == representation.edge_updates[0].out_dim
        assert interaction.magnetic_edge_info.dims == [
            update.out_dim,
            *config["radial_basis"]["hidden"],
            interaction.magnetic_linear.weight_numel,
        ]
        assert all(
            mul == representation.num_channel
            for mul, _ in interaction.magnetic_edge_irreps_out
        )
        assert interaction.magnetic_linear.bias is None

        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        magnetic_radial_basis = torch.randn(3, num_mag_radial_basis)
        node_attrs = torch.ones(3, 1)
        magnetic_edge_feats = update(
            magnetic_radial_basis,
            node_attrs,
            edge_index,
        )
        magnetic_edge_attrs = torch.randn(
            edge_index.size(1),
            interaction.magnetic_edge_irreps.dim,
        )
        magnetic_weights = interaction.magnetic_edge_info(magnetic_edge_feats)
        projected = interaction.magnetic_linear(
            magnetic_edge_attrs,
            magnetic_weights,
        )
        assert projected.shape == (
            edge_index.size(1),
            interaction.magnetic_edge_irreps_out.dim,
        )


def test_o2_does_not_import_tace():
    directory = Path(__file__).resolve().parents[1] / "eqx" / "o2"
    for source_path in directory.glob("*.py"):
        source = source_path.read_text()
        assert "from tace" not in source
        assert "import tace" not in source


def test_local_frame_roundtrip_flattened_ir_mul():
    irreps = o3.Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o")
    frame = o2.LocalFrame(irreps, lmax=2).to(DEVICE, DTYPE)
    layout = LayoutTransform(
        irreps,
        layout_in="flatten_mul_ir",
        layout_out="flatten_ir_mul",
    ).to(DEVICE)
    vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(2, 2).to(DEVICE, DTYPE).get_wigner(vectors)
    features = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)

    local = frame(layout(features), wigner)
    assert frame.irreps_out == o2.Irreps("6x0e+6x0o+8x1m+4x2m")
    assert repr(frame) == (
        f"LocalFrame({frame.global_irreps} -> {frame.local_irreps})(mmax=2)"
    )
    reverse_frame = o2.LocalFrame(irreps, lmax=2, reverse=True)
    assert repr(reverse_frame) == (
        f"LocalFrame({frame.local_irreps} -> {frame.global_irreps})(mmax=2)"
    )
    assert local.shape == (7, frame.irreps_out.dim)
    torch.testing.assert_close(
        layout.inverse(frame.to_global(local, wigner_inv)),
        features,
    )


def test_local_frame_trailing_axes_and_empty_batch():
    irreps = o3.Irreps("2x0e+2x1o+2x2e")
    frame = o2.LocalFrame(irreps, lmax=2, mmax=1).to(DEVICE, DTYPE)
    vectors = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(1, 2).to(DEVICE, DTYPE).get_wigner(vectors)
    features = torch.randn(4, 2, irreps.dim, dtype=DTYPE, device=DEVICE)

    local = frame.to_local(features, wigner)
    assert local.shape == (4, 2, frame.irreps_out.dim)
    assert frame.to_global(local, wigner_inv).shape == features.shape

    empty = frame.to_local(features[:0], wigner[:0])
    assert empty.shape == (0, 2, frame.irreps_out.dim)
    assert frame.to_global(empty, wigner_inv[:0]).shape == (0, 2, irreps.dim)


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
    torch.testing.assert_close(module(features, singleton), module(features, singleton[0]))


def test_o2_linear_bias_requires_invariant_scalar():
    with pytest.raises(ValueError, match="time-reversal-even"):
        o2.Linear("0eo", "0eo", biases=[True])
    with pytest.raises(ValueError, match="same irrep"):
        o2.Linear("0e", "0o", instructions=[(0, 0)])


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


def test_o2_activation_rejects_non_equivariant_odd_scalar_map():
    with pytest.raises(ValueError, match="must be either even or odd"):
        o2.Activation("0o", [torch.nn.SiLU()])
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


def test_o2_tensor_product_validates_paths_and_zero_pads():
    module = o2.TensorProduct(
        "2x1m",
        "0e",
        "2x1m+0o",
        [(0, 0, 0, "u1u", False)],
    )
    output = module(torch.randn(3, 4), torch.randn(3, 1))
    torch.testing.assert_close(output[:, -1], torch.zeros_like(output[:, -1]))
    with pytest.raises(ValueError, match=r"Illegal O\(2\)"):
        o2.TensorProduct("1m", "1m", "1m", [(0, 0, 0, "uuu", False)])


def _asymmetric_contractions(correlation=3):
    irreps_in = o2.Irreps("2x0e+2x0o+2x1m")
    irreps_out = o2.Irreps("2x0e+2x0o+2x1m+2x2m")
    edge = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="edge",
    ).to(DEVICE, DTYPE)
    node = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation,
        algorithm="node",
    ).to(DEVICE, DTYPE)
    return edge, node


def test_o2_asymmetric_contraction_algorithms_match():
    edge, node = _asymmetric_contractions()
    inputs = [
        edge.irreps_in.randn(4, -1, dtype=DTYPE, device=DEVICE)
        for _ in range(3)
    ]
    weights = torch.randn(4, edge.weight_numel, dtype=DTYPE, device=DEVICE)
    assert edge.order_num_paths == node.order_num_paths
    torch.testing.assert_close(edge(inputs, weights), node(inputs, weights))


@pytest.mark.parametrize("algorithm", ["edge", "node"])
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_asymmetric_contraction_is_equivariant(algorithm, reflected):
    module = o2.AsymmetricContraction(
        "2x0e+2x0o+2x1m",
        "2x0e+2x0o+2x1m+2x2m",
        2,
        algorithm=algorithm,
    ).to(DEVICE, DTYPE)
    inputs = [
        module.irreps_in.randn(4, -1, dtype=DTYPE, device=DEVICE)
        for _ in range(2)
    ]
    weights = torch.randn(4, module.weight_numel, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.31, dtype=DTYPE, device=DEVICE)
    output = module(inputs, weights)
    expected = _transform(output, module.irreps_out, angle, reflected)
    transformed = [
        _transform(features, module.irreps_in, angle, reflected)
        for features in inputs
    ]
    torch.testing.assert_close(module(transformed, weights), expected)


def _scatter_module(use_attention):
    irreps = o3.Irreps("2x0e+2x1o")
    return O2ScatterTensorProduct(
        irreps,
        irreps,
        num_channel=2,
        lmax=1,
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
    wigner, wigner_inv = wigner_module.get_wigner(edge_vectors)
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
    rotated_wigner, rotated_wigner_inv = wigner_module.get_wigner(rotated_vectors)
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
    wigner, wigner_inv = o2.WignerD(1, 1).to(DEVICE, DTYPE).get_wigner(
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
    with pytest.raises(ValueError, match="time-reversal-odd"):
        o2.Activation("0eo", [torch.nn.SiLU()])

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
        algorithm="edge",
    ).to(DEVICE, DTYPE)
    inputs = [
        irreps_in.randn(5, -1, dtype=DTYPE, device=DEVICE) for _ in range(2)
    ]
    weights = torch.randn(
        5, contraction.weight_numel, dtype=DTYPE, device=DEVICE
    )
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
    frame = o2.LocalFrame(irreps, lmax=2).to(DEVICE, DTYPE)
    edge_vectors = torch.randn(6, 3, dtype=DTYPE, device=DEVICE)
    wigner, _ = o2.WignerD(mmax=2, lmax=2).to(DEVICE, DTYPE).get_wigner(
        edge_vectors
    )
    features = torch.randn(6, irreps.dim, dtype=DTYPE, device=DEVICE)
    local = frame.to_local(features, wigner)
    torch.testing.assert_close(
        frame.to_local(_time_reverse(features, irreps), wigner),
        _time_reverse(local, frame.irreps_out),
    )


@pytest.mark.parametrize("Lmax", [1, 2])
@pytest.mark.parametrize("normalization", ["integral", "component"])
def test_magnetic_basis_builds_regrouped_edge_attrs(Lmax, normalization):
    time_reversal = hasattr(o3.Irrep("0e"), "t")
    basis = MagneticBasis(
        {26: 2.0},
        num_basis=4,
        Lmax=Lmax,
        atomic_numbers=[26],
        num_elements=1,
        time_reversal=time_reversal,
        normalization=normalization,
    ).to(DEVICE, DTYPE)
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        device=DEVICE,
    )
    magmoms = torch.randn(4, 3, dtype=DTYPE, device=DEVICE)
    node_attrs = torch.ones(4, 1, dtype=DTYPE, device=DEVICE)

    radial, magnetic_node_attrs, magnetic_edge_attrs = basis(
        magmoms,
        node_attrs,
        edge_index,
    )
    source, target = edge_index
    expected = basis.magnetic_edge_tensor_product(
        magnetic_node_attrs[target],
        magnetic_node_attrs[source],
    )

    assert basis.angular_basis.normalization == normalization
    assert not basis.angular_basis.normalize
    torch.testing.assert_close(
        basis.scale,
        torch.tensor([1.0 / 2.5], dtype=DTYPE, device=DEVICE),
    )
    assert basis.magnetic_edge_tensor_product.weight_numel == 0
    assert basis.magnetic_node_irreps_out.lmax == Lmax
    assert basis.magnetic_edge_irreps_out.lmax == Lmax
    assert repr(basis).splitlines() == [
        "MagneticBasis(",
        f"  scale={basis.scale.tolist()},",
        "  num_basis=4,",
        f"  Lmax={Lmax},",
        f"  normalization='{normalization}',",
        f"  magnetic_node_irreps_out={basis.magnetic_node_irreps_out},",
        f"  magnetic_edge_irreps_out={basis.magnetic_edge_irreps_out}",
        ")",
    ]
    assert basis.magnetic_edge_tensor_product.irreps_out.simplify() == (
        basis.magnetic_edge_irreps_out
    )
    assert radial.shape == (4, 4)
    assert magnetic_node_attrs.shape[-1] == basis.magnetic_node_irreps_out.dim
    assert magnetic_edge_attrs.shape == (
        edge_index.size(1),
        basis.magnetic_edge_irreps_out.dim,
    )
    torch.testing.assert_close(magnetic_edge_attrs, expected)

    if time_reversal:
        reversed_radial, reversed_node, reversed_edge = basis(
            -magmoms,
            node_attrs,
            edge_index,
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


@pytest.mark.parametrize("update_type", ["identity", "element", "element2"])
def test_magnetic_edge_update_gathers_source_and_target(update_type):
    update = MAGNETIC_EDGE_UPDATE[update_type](
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
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        device=DEVICE,
    )

    output = update(magnetic_radial_basis, node_attrs, edge_index)
    source, target = edge_index
    if update_type == "identity":
        source_features = target_features = magnetic_radial_basis
    elif update_type == "element":
        source_features = target_features = update.embedding(
            magnetic_radial_basis,
            node_attrs,
        )
    else:
        source_features = update.source_embedding(magnetic_radial_basis, node_attrs)
        target_features = update.target_embedding(magnetic_radial_basis, node_attrs)

    feature_dim = 3 if update_type == "identity" else 5
    assert output.shape == (edge_index.size(1), update.out_dim)
    torch.testing.assert_close(output[..., :feature_dim], source_features[source])
    torch.testing.assert_close(output[..., feature_dim:], target_features[target])


def _magnetic_scatter_module(use_attention: bool):
    irreps = o3.Irreps("2x0ee+2x1oe")
    magnetic_edge_irreps = o3.Irreps("2x0ee+2x1eo+2x1ee")
    module = O2ScatterMagneticTensorProduct(
        irreps,
        irreps,
        magnetic_edge_irreps,
        num_channel=2,
        lmax=1,
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
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]], device=DEVICE
    )
    num_edges = edge_index.size(1)
    node_features = torch.randn(4, irreps.dim, dtype=DTYPE, device=DEVICE)
    magnetic_edge_attrs = torch.randn(
        num_edges, magnetic_edge_irreps.dim, dtype=DTYPE, device=DEVICE
    )
    conv_weights = torch.randn(
        num_edges, module.weight_numel, dtype=DTYPE, device=DEVICE
    )
    edge_vectors = torch.randn(num_edges, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).to(
        DEVICE, DTYPE
    ).get_wigner(edge_vectors)
    edge_cutoff = torch.rand(num_edges, 1, dtype=DTYPE, device=DEVICE)
    edge_radial_basis = torch.randn(
        num_edges, 4, dtype=DTYPE, device=DEVICE
    )

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
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).to(
        DEVICE, DTYPE
    ).get_wigner(torch.empty(0, 3, dtype=DTYPE, device=DEVICE))
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

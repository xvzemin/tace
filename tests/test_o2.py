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
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.o2 import O2ScatterTensorProduct
from tace.models._e3nn.representation import Representation
from tace.models.layout import LayoutTransform

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def _transform(features, irreps, angle, reflected=False, time_reversal=False):
    matrix = irreps.D_from_angle(
        angle,
        reflected=reflected,
        time_reversal=time_reversal,
        dtype=features.dtype,
        device=features.device,
    )
    return torch.matmul(features, matrix.transpose(-1, -2))


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
        magmoms_norm_by_element=None,
        mmax=2,
        Lmax=Lmax,
        lmax=lmax,
        mag_Lmax=1,
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
    assert o2.Irrep("3m").dim == 2
    assert o2.Irrep("0e").is_invariant_scalar()

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

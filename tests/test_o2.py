import math

import pytest
import torch
from e3nn import o3

from tace.models import o2
from tace.models._e3nn.inter import O2MagneticInteraction
from tace.models._e3nn.o2 import _O3O2Layout
from tace.models.angular import SolidHarmonics
from tace.models.legacy_so2 import WignerD
from tace.models.mag import MagmomsNormalizer
from tace.models.o2 import (
    Irrep,
    Irreps,
    fully_connected_tensor_product_paths,
    has_tensor_product_path,
    o2_clebsch_gordan,
    o2_irreps_representation,
    o2_recoupling,
    o2_representation,
    restrict_o3_irrep,
    restrict_o3_irreps,
    tensor_product_irrep,
    tensor_product_irreps,
)
from tace.models.radial import MagneticChebyshevBasis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def test_o3_o2_layout_roundtrip_supports_both_parities_per_degree():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        irreps = o3.Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o")
        layout = _O3O2Layout(irreps, 2).to(device=DEVICE)
        wigner_module = WignerD(2, 2).to(device=DEVICE)
    finally:
        torch.set_default_dtype(previous_dtype)
    edge_vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = wigner_module.get_wigner(edge_vectors)
    input = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)

    blocks = layout(input, wigner)
    assert layout.local_irreps == o2.Irreps("3x0e+3x0o+4x1m+2x2m")
    assert tuple(block.shape for block in blocks) == (
        (7, 1, 6),
        (7, 1, 6),
        (7, 2, 8),
        (7, 2, 4),
    )
    torch.testing.assert_close(layout.inverse(blocks, wigner_inv), input)


def _contract(input1, input2, cg):
    return torch.einsum("...i,...j,ijk->...k", input1, input2, cg)


def _coupling_tensors(input1, input2, input3, output):
    recoupling = o2_recoupling(
        input1,
        input2,
        input3,
        output,
        dtype=DTYPE,
        device=DEVICE,
    )
    left = []
    for intermediate in recoupling.left_intermediates:
        first = o2_clebsch_gordan(
            input1,
            input2,
            intermediate,
            dtype=DTYPE,
            device=DEVICE,
        )
        second = o2_clebsch_gordan(
            intermediate,
            input3,
            output,
            dtype=DTYPE,
            device=DEVICE,
        )
        left.append(torch.einsum("abe,ecd->abcd", first, second))
    right = []
    for intermediate in recoupling.right_intermediates:
        first = o2_clebsch_gordan(
            input2,
            input3,
            intermediate,
            dtype=DTYPE,
            device=DEVICE,
        )
        second = o2_clebsch_gordan(
            input1,
            intermediate,
            output,
            dtype=DTYPE,
            device=DEVICE,
        )
        right.append(torch.einsum("bcf,afd->abcd", first, second))
    return recoupling, torch.stack(left), torch.stack(right)


def test_o2_irrep_and_irreps_metadata():
    assert Irrep("0e") == Irrep(0, 1)
    assert Irrep("0o") == Irrep((0, -1))
    assert Irrep("3m").dim == 2

    irreps = Irreps("2x0e + 0o + 3x1m + 2m")
    assert irreps.dim == 2 + 1 + 6 + 2
    assert irreps.num_irreps == 7
    assert irreps.m_max == 2
    assert irreps.expanded() == (
        Irrep("0e"),
        Irrep("0e"),
        Irrep("0o"),
        Irrep("1m"),
        Irrep("1m"),
        Irrep("1m"),
        Irrep("2m"),
    )
    assert irreps.expanded_slices()[-1] == slice(9, 11)
    with pytest.raises(AttributeError, match="immutable"):
        irreps._groups = ()


@pytest.mark.parametrize("value", ["0m", "1e", "1o", "-1m", "x"])
def test_o2_irrep_rejects_invalid_labels(value):
    with pytest.raises((TypeError, ValueError)):
        Irrep(value)


def test_o3_restriction_is_complete_and_parity_aware():
    assert restrict_o3_irrep(1, "o") == Irreps("0e+1m")
    assert restrict_o3_irrep(1, "e") == Irreps("0o+1m")
    assert restrict_o3_irrep(2, "e") == Irreps("0e+1m+2m")
    assert restrict_o3_irrep(2, "o") == Irreps("0o+1m+2m")
    assert restrict_o3_irreps([(2, 1, "e"), (1, 0, "o")]) == Irreps("2x0o+2x1m+0o")


def test_o2_direct_sum_representation_uses_complete_layout():
    irreps = Irreps("0e+0o+2x1m")
    angle = torch.tensor([0.2, -0.7], dtype=DTYPE, device=DEVICE)
    representation = o2_irreps_representation(irreps, angle, True)
    assert representation.shape == (2, irreps.dim, irreps.dim)
    torch.testing.assert_close(representation[:, 0, 0], torch.ones_like(angle))
    torch.testing.assert_close(representation[:, 1, 1], -torch.ones_like(angle))
    expected = o2_representation("1m", angle, True)
    torch.testing.assert_close(representation[:, 2:4, 2:4], expected)
    torch.testing.assert_close(representation[:, 4:6, 4:6], expected)


def test_complete_o2_tensor_product_rules_and_paths():
    assert tensor_product_irrep("0o", "0o") == (Irrep("0e"),)
    assert tensor_product_irrep("0o", "2m") == (Irrep("2m"),)
    assert tensor_product_irrep("1m", "1m") == (
        Irrep("0e"),
        Irrep("0o"),
        Irrep("2m"),
    )
    assert tensor_product_irrep("1m", "2m") == (
        Irrep("1m"),
        Irrep("3m"),
    )
    assert tensor_product_irreps("2x1m", "0o") == Irreps("2x1m")
    assert has_tensor_product_path("0o", "1m", "1m")
    assert not has_tensor_product_path("0o", "1m", "2m")

    paths = fully_connected_tensor_product_paths("0e+0o", "2x1m", "1m")
    assert len(paths) == 4


def test_all_small_o2_cg_paths_are_normalized_and_equivariant():
    torch.manual_seed(0)
    irreps = [Irrep("0e"), Irrep("0o")]
    irreps.extend(Irrep(m, "m") for m in range(1, 4))
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)

    for input1 in irreps:
        for input2 in irreps:
            for output in tensor_product_irrep(input1, input2):
                cg = o2_clebsch_gordan(
                    input1,
                    input2,
                    output,
                    dtype=DTYPE,
                    device=DEVICE,
                )
                gram = torch.einsum("ijk,ijl->kl", cg, cg)
                torch.testing.assert_close(
                    gram,
                    torch.eye(output.dim, dtype=DTYPE, device=DEVICE),
                )

                value1 = torch.randn(5, input1.dim, dtype=DTYPE, device=DEVICE)
                value2 = torch.randn(5, input2.dim, dtype=DTYPE, device=DEVICE)
                value_out = _contract(value1, value2, cg)
                for reflected in (False, True):
                    transform1 = o2_representation(input1, angle, reflected)
                    transform2 = o2_representation(input2, angle, reflected)
                    transform_out = o2_representation(output, angle, reflected)
                    transformed1 = torch.einsum("ij,bj->bi", transform1, value1)
                    transformed2 = torch.einsum("ij,bj->bi", transform2, value2)
                    actual = _contract(transformed1, transformed2, cg)
                    expected = torch.einsum(
                        "ij,bj->bi",
                        transform_out,
                        value_out,
                    )
                    torch.testing.assert_close(actual, expected)


def test_o2_recoupling_is_nontrivial_and_exact():
    recoupling, left, right = _coupling_tensors("1m", "1m", "1m", "1m")
    assert recoupling.left_intermediates == (
        Irrep("0e"),
        Irrep("0o"),
        Irrep("2m"),
    )
    assert recoupling.right_intermediates == recoupling.left_intermediates

    expected_matrix = torch.tensor(
        [
            [0.5, 0.5, math.sqrt(0.5)],
            [-0.5, -0.5, math.sqrt(0.5)],
            [math.sqrt(0.5), -math.sqrt(0.5), 0.0],
        ],
        dtype=DTYPE,
        device=DEVICE,
    )
    torch.testing.assert_close(recoupling.matrix, expected_matrix)
    torch.testing.assert_close(
        recoupling.matrix @ recoupling.matrix.T,
        torch.eye(3, dtype=DTYPE, device=DEVICE),
    )
    torch.testing.assert_close(
        left, torch.einsum("ij,jabcd->iabcd", recoupling.matrix, right)
    )


def test_all_small_o2_recoupling_matrices_are_orthogonal():
    irreps = [Irrep("0e"), Irrep("0o"), Irrep("1m"), Irrep("2m")]
    for input1 in irreps:
        for input2 in irreps:
            for input3 in irreps:
                left_product = tensor_product_irreps(input1, input2)
                outputs = tensor_product_irreps(left_product, input3).regroup()
                for _, output in outputs:
                    recoupling = o2_recoupling(
                        input1,
                        input2,
                        input3,
                        output,
                        dtype=DTYPE,
                        device=DEVICE,
                    )
                    identity = torch.eye(
                        recoupling.matrix.shape[0],
                        dtype=DTYPE,
                        device=DEVICE,
                    )
                    torch.testing.assert_close(
                        recoupling.matrix @ recoupling.matrix.T,
                        identity,
                    )


def test_o2_linear_is_exported_without_prefixed_class_name():
    assert o2.Linear.__name__ == "Linear"
    assert o2.Irrep.__name__ == "Irrep"
    assert o2.Irreps.__name__ == "Irreps"


def test_o2_linear_defaults_to_uv_path_mode():
    module = o2.Linear(
        "0e+1m",
        "0e+1m",
        channels_in=2,
        channels_out=3,
    )
    assert module.path_mode == "uv"
    assert module.weight_shape == (2, 2, 3)


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_is_equivariant_with_complete_irreps(reflected, path_mode):
    torch.manual_seed(0)
    irreps_in = o2.Irreps("2x0e+0o+2x1m+2m")
    irreps_out = o2.Irreps("0e+2x0o+1m+2x2m")
    module = o2.Linear(
        irreps_in,
        irreps_out,
        channels_in=2,
        channels_out=2 if path_mode == "uu" else 3,
        path_mode=path_mode,
        bias=True,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(5, irreps_in.dim, 2, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.41, dtype=DTYPE, device=DEVICE)
    input_transform = o2.o2_irreps_representation(
        irreps_in,
        angle,
        reflected,
    )
    output_transform = o2.o2_irreps_representation(
        irreps_out,
        angle,
        reflected,
    )

    transformed_input = torch.einsum("ij,bjc->bic", input_transform, input)
    actual = module(transformed_input)
    expected = torch.einsum("ij,bjc->bic", output_transform, module(input))
    torch.testing.assert_close(actual, expected)


def test_o2_linear_positive_order_components_share_one_real_matrix():
    module = o2.Linear(
        "1m",
        "1m",
        channels_in=2,
        channels_out=3,
        internal_weights=False,
        bias=False,
        path_norm=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(4, 2, 2, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(*module.weight_shape, dtype=DTYPE, device=DEVICE)
    actual = module(input, weight)
    expected = torch.matmul(input, weight[0])
    torch.testing.assert_close(actual, expected)


def _pack_o2_groups(irreps, input):
    blocks = []
    offset = 0
    for multiplicity, irrep in irreps:
        width = multiplicity * irrep.dim
        block = input[:, offset : offset + width]
        blocks.append(
            block.reshape(
                input.size(0),
                multiplicity,
                irrep.dim,
                input.size(-1),
            )
            .permute(0, 2, 1, 3)
            .reshape(
                input.size(0),
                irrep.dim,
                multiplicity * input.size(-1),
            )
        )
        offset += width
    return tuple(blocks)


def _unpack_o2_groups(irreps, blocks, channels):
    outputs = []
    for (multiplicity, irrep), block in zip(irreps, blocks):
        outputs.append(
            block.reshape(
                block.size(0),
                irrep.dim,
                multiplicity,
                channels,
            )
            .permute(0, 2, 1, 3)
            .reshape(block.size(0), multiplicity * irrep.dim, channels)
        )
    return torch.cat(outputs, dim=1)


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_grouped_gemm_matches_pathwise_forward(path_mode):
    torch.manual_seed(9)
    module = o2.Linear(
        "2x0e+0o+3x1m+2m",
        "0e+2x0o+2x1m+2x2m",
        channels_in=2,
        channels_out=2 if path_mode == "uu" else 3,
        path_mode=path_mode,
        bias=True,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        5,
        module.irreps_in.dim,
        module.channels_in,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    reference = module(input)
    grouped = module.forward_grouped(_pack_o2_groups(module.irreps_in, input))
    observed = _unpack_o2_groups(
        module.irreps_out,
        grouped,
        module.channels_out,
    )
    torch.testing.assert_close(observed, reference)

    probe = torch.randn_like(reference)
    reference_gradients = torch.autograd.grad(
        (reference * probe).sum(),
        (input, module.weight),
        retain_graph=True,
    )
    grouped_gradients = torch.autograd.grad(
        (observed * probe).sum(),
        (input, module.weight),
    )
    for observed_gradient, reference_gradient in zip(
        grouped_gradients,
        reference_gradients,
    ):
        torch.testing.assert_close(observed_gradient, reference_gradient)


def test_o2_linear_uu_is_channelwise_and_sums_paths():
    module = o2.Linear(
        "2x0e",
        "0e",
        channels_in=2,
        path_mode="uu",
        internal_weights=False,
        bias=False,
        path_norm=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0]]],
        dtype=DTYPE,
        device=DEVICE,
    )
    weight = torch.tensor(
        [[2.0, 3.0], [5.0, 7.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    actual = module(input, weight)
    expected = input[:, :1] * weight[0] + input[:, 1:] * weight[1]
    torch.testing.assert_close(actual, expected)


def test_o2_linear_uu_requires_matching_channels():
    with pytest.raises(ValueError, match="channels_in == channels_out"):
        o2.Linear(
            "0e",
            "0e",
            channels_in=2,
            channels_out=3,
            path_mode="uu",
        )


def test_o2_linear_rejects_unknown_path_mode():
    with pytest.raises(ValueError, match="path_mode"):
        o2.Linear("0e", "0e", channels_in=1, path_mode="invalid")


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_supports_batched_external_weights(path_mode):
    torch.manual_seed(1)
    module = o2.Linear(
        "2x0e+0o+2x1m",
        "0e+2x0o+1m",
        channels_in=2,
        channels_out=2 if path_mode == "uu" else 3,
        path_mode=path_mode,
        internal_weights=False,
        bias=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        4,
        module.irreps_in.dim,
        module.channels_in,
        dtype=DTYPE,
        device=DEVICE,
    )
    weight = torch.randn(
        4,
        *module.weight_shape,
        dtype=DTYPE,
        device=DEVICE,
    )
    actual = module(input, weight)
    expected = torch.stack(
        [module(input[index], weight[index]) for index in range(input.shape[0])]
    )
    torch.testing.assert_close(actual, expected)


def test_o2_linear_bias_is_restricted_to_even_scalars():
    module = o2.Linear(
        "0e",
        "0e+0o+1m",
        channels_in=1,
        channels_out=2,
        bias=True,
    ).to(device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        module.bias.copy_(torch.tensor([[2.0, -3.0]], device=DEVICE, dtype=DTYPE))
    input = torch.zeros(3, 1, 1, dtype=DTYPE, device=DEVICE)
    output = module(input)
    torch.testing.assert_close(
        output[:, 0],
        torch.tensor([2.0, -3.0], dtype=DTYPE, device=DEVICE).expand(3, -1),
    )
    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_external_weight_gradcheck_and_gradgradcheck(path_mode):
    module = o2.Linear(
        "0e+0o+1m",
        "2x0e+0o+1m",
        channels_in=2,
        channels_out=2,
        path_mode=path_mode,
        internal_weights=False,
        bias=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        2,
        module.irreps_in.dim,
        module.channels_in,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    weight = torch.randn(
        *module.weight_shape,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    def function(features, weights):
        return module(features, weights)

    assert torch.autograd.gradcheck(function, (input, weight), fast_mode=True)
    assert torch.autograd.gradgradcheck(function, (input, weight), fast_mode=True)


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_zero_pads_missing_output_irreps(path_mode):
    module = o2.Linear(
        "0e+1m",
        "0e+0o+2m",
        channels_in=2,
        channels_out=2,
        path_mode=path_mode,
        internal_weights=False,
        bias=False,
        path_norm=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        3,
        module.irreps_in.dim,
        module.channels_in,
        dtype=DTYPE,
        device=DEVICE,
    )
    weight = torch.randn(*module.weight_shape, dtype=DTYPE, device=DEVICE)
    output = module(input, weight)

    if path_mode == "uv":
        expected_scalar = torch.matmul(input[:, :1], weight[0])
    else:
        expected_scalar = input[:, :1] * weight[0]
    torch.testing.assert_close(output[:, :1], expected_scalar)
    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))


def test_o2_linear_rejects_non_equivariant_path():
    with pytest.raises(ValueError, match="identical irreps"):
        o2.Linear(
            "0e",
            "0o",
            channels_in=1,
            path=[(0, 0)],
        )


def test_o2_linear_without_paths_is_zero_and_differentiable():
    module = o2.Linear(
        "0e",
        "0o",
        channels_in=2,
        internal_weights=False,
        bias=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        3,
        1,
        2,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    weight = torch.empty(
        *module.weight_shape,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    output = module(input, weight)
    torch.testing.assert_close(output, torch.zeros_like(output))
    gradients = torch.autograd.grad(output.sum(), (input, weight))
    torch.testing.assert_close(gradients[0], torch.zeros_like(input))
    assert gradients[1].numel() == 0


def test_o2_linear_rejects_complex_input_and_weight():
    internal = o2.Linear("1m", "1m", channels_in=1)
    with pytest.raises(TypeError, match="real inputs"):
        internal(torch.ones(2, 1, dtype=torch.complex64))

    external = o2.Linear(
        "1m",
        "1m",
        channels_in=1,
        internal_weights=False,
    )
    with pytest.raises(TypeError, match="real weights"):
        external(
            torch.ones(2, 1),
            torch.ones(*external.weight_shape, dtype=torch.complex64),
        )


def _build_o2_magnetic_interaction(
    monkeypatch,
    path_mode,
    mag_Lmax=1,
    angular_max=None,
):
    monkeypatch.setattr(O2MagneticInteraction, "path_mode", path_mode)
    angular_max = mag_Lmax if angular_max is None else angular_max
    module = O2MagneticInteraction(
        layer=0,
        num_layers=1,
        num_elements=2,
        avg_num_neighbors=4.0,
        mmax=angular_max,
        Lmax=angular_max,
        lmax=angular_max,
        mag_Lmax=mag_Lmax,
        correlation=[1],
        num_channel=2,
        use_temperature=False,
        edge_feats_channel=4,
        target_irreps=o3.Irreps("0e"),
        num_radial_basis=4,
        num_mag_radial_basis=3,
        magnetic_irreps=o3.Irreps.spherical_harmonics(mag_Lmax, p=1),
        radial_mlp=[8],
        radial_bias=True,
        irreps_in=o3.Irreps("2x0e + 2x1o"),
        scalar_act=None,
        tensor_act=None,
        edge_ace_hidden=None,
        parity=True,
        nonlinear=None,
    )
    return module.to(device=DEVICE, dtype=DTYPE)


def _o2_magnetic_inputs(module):
    num_nodes = 5
    num_edges = 9
    edge_index = torch.randint(num_nodes, (2, num_edges), device=DEVICE)
    edge_vectors = torch.randn(num_edges, 3, dtype=DTYPE, device=DEVICE)
    node_feats = torch.randn(
        num_nodes,
        module.irreps_in.dim,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    node_attrs = torch.nn.functional.one_hot(
        torch.randint(2, (num_nodes,), device=DEVICE),
        2,
    ).to(node_feats)
    edge_feats = torch.randn(
        num_edges,
        4,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    initial_noncollinear_magmoms = torch.randn(
        num_nodes,
        3,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    cutoff = torch.rand(num_edges, 1, dtype=DTYPE, device=DEVICE)
    return (
        node_feats,
        node_attrs,
        edge_feats,
        edge_vectors,
        edge_index,
        cutoff,
        initial_noncollinear_magmoms,
    )


def _evaluate_o2_magnetic_interaction(module, inputs):
    (
        node_feats,
        node_attrs,
        edge_feats,
        edge_vectors,
        edge_index,
        cutoff,
        initial_noncollinear_magmoms,
    ) = inputs
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        wigner_module = WignerD(module.mmax, module.Lmax)
    finally:
        torch.set_default_dtype(previous_dtype)
    wigner, wigner_inv = wigner_module.to(device=DEVICE).get_wigner(edge_vectors)
    magnetic_radial_basis = MagneticChebyshevBasis(num_basis=3).to(
        device=DEVICE,
        dtype=DTYPE,
    )(
        MagmomsNormalizer([4.0, 4.0], num_elements=2).to(device=DEVICE, dtype=DTYPE)(
            initial_noncollinear_magmoms, node_attrs
        )
    )
    magnetic_node_attrs = SolidHarmonics(
        module.magnetic_irreps,
        normalization="integral",
        irreps_in=o3.Irreps("1e"),
    ).to(device=DEVICE, dtype=DTYPE)(initial_noncollinear_magmoms)
    return module._compute_messages(
        node_feats,
        node_attrs,
        edge_feats,
        None,
        edge_index,
        cutoff,
        initial_noncollinear_magmoms,
        wigner,
        wigner_inv,
        magnetic_radial_basis,
        magnetic_node_attrs,
    )


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_magnetic_interaction_uses_standard_linear_paths(
    path_mode,
    monkeypatch,
):
    torch.manual_seed(7)
    module = _build_o2_magnetic_interaction(monkeypatch, path_mode)
    assert module.rejector.path_mode == path_mode
    assert module.edge_info.dims[0] == module.edge_feats_channel + 2 * 3
    if path_mode == "uv":
        assert module.rejector.linear.internal_weights
    else:
        assert not module.rejector.linear.internal_weights

    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_magnetic_interaction(module, inputs)
    assert output.shape == (inputs[0].size(0), module.rejector.irreps_out.dim)
    gradients = torch.autograd.grad(
        output.square().sum(),
        (inputs[0], inputs[2], inputs[-1]),
        create_graph=True,
    )
    assert all(gradient.isfinite().all() for gradient in gradients)


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
@pytest.mark.parametrize("mag_Lmax", [1, 2])
@pytest.mark.parametrize("improper", [False, True])
def test_o2_magnetic_interaction_is_globally_o3_equivariant(
    path_mode,
    mag_Lmax,
    improper,
    monkeypatch,
):
    torch.manual_seed(8)
    module = _build_o2_magnetic_interaction(monkeypatch, path_mode, mag_Lmax)
    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_magnetic_interaction(module, inputs)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        rotation = o3.rand_matrix(dtype=DTYPE)
        if improper:
            rotation = -rotation
        node_rotation = module.irreps_in.D_from_matrix(rotation)
        magnetic_rotation = o3.Irreps("1e").D_from_matrix(rotation)
        output_rotation = module.rejector.irreps_out.D_from_matrix(rotation)
    finally:
        torch.set_default_dtype(previous_dtype)
    rotation = rotation.to(DEVICE)
    node_rotation = node_rotation.to(DEVICE)
    magnetic_rotation = magnetic_rotation.to(DEVICE)
    output_rotation = output_rotation.to(DEVICE)
    rotated_inputs = (
        inputs[0] @ node_rotation.T,
        inputs[1],
        inputs[2],
        inputs[3] @ rotation.T,
        inputs[4],
        inputs[5],
        inputs[6] @ magnetic_rotation.T,
    )
    rotated_output = _evaluate_o2_magnetic_interaction(module, rotated_inputs)
    torch.testing.assert_close(
        rotated_output,
        output @ output_rotation.T,
        atol=3.0e-10,
        rtol=3.0e-10,
    )


def test_o2_magnetic_interaction_restricts_all_magnetic_degrees(monkeypatch):
    module = _build_o2_magnetic_interaction(monkeypatch, "uu", mag_Lmax=2)
    assert module.magnetic_irreps == o3.Irreps("0e + 1e + 2e")
    assert module.rejector.magnetic_layout.local_irreps == o2.Irreps(
        "2x0e + 0o + 2x1m + 2m"
    )


@pytest.mark.parametrize("mag_Lmax", [0, 2])
def test_o2_magnetic_interaction_validates_mag_Lmax(mag_Lmax, monkeypatch):
    with pytest.raises(
        ValueError,
        match="mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax",
    ):
        _build_o2_magnetic_interaction(
            monkeypatch,
            "uu",
            mag_Lmax=mag_Lmax,
            angular_max=1,
        )

import pytest
import torch
from e3nn import o3

from tace.models import o2
from tace.models._e3nn.inter import O2MagneticInteraction
from tace.models.angular import SolidHarmonics
from tace.models.legacy_so2 import WignerD
from tace.models.mag import MagmomsNormalizer
from tace.models.mlp import (
    ACTIVATION,
    ScaledSigmoid,
    ScaledSiLU,
    ScaledTanh,
    get_activation_scale_factor,
    get_scaled_activation,
)
from tace.models.o2 import (
    Irrep,
    Irreps,
    O2Gate,
    o2_irreps_representation,
    o2_representation,
    restrict_o3_irrep,
    restrict_o3_irreps,
)
from tace.models.radial import (
    MagneticChebyshevBasis,
    j0SincSphericalBesselBasis,
    j0SphericalBesselBasis,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


def test_j0_sinc_preserves_nonzero_values_and_defines_origin():
    reference = j0SphericalBesselBasis(cutoff=1.0, num_basis=3).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    sinc = j0SincSphericalBesselBasis(cutoff=1.0, num_basis=3).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    x = torch.tensor([[0.2], [0.7]], device=DEVICE, dtype=DTYPE)
    torch.testing.assert_close(sinc(x), reference(x, None, None))

    at_origin = sinc(torch.zeros(1, 1, device=DEVICE, dtype=DTYPE))
    torch.testing.assert_close(
        at_origin,
        sinc.prefactor * sinc.bessel_weights.unsqueeze(0),
    )
    assert torch.isfinite(at_origin).all()


def test_o3_o2_layout_roundtrip_supports_both_parities_per_degree():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        irreps = o3.Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o")
        layout = o2.O3O2Layout(irreps, 2).to(device=DEVICE)
        wigner_module = WignerD(2, 2).to(device=DEVICE)
    finally:
        torch.set_default_dtype(previous_dtype)
    edge_vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = wigner_module.get_wigner(edge_vectors)
    input = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)

    blocks = layout(input, wigner)
    channel_major_blocks = layout.forward_channel_major(input, wigner)
    assert layout.local_irreps == o2.Irreps("3x0e+3x0o+4x1m+2x2m")
    assert tuple(block.shape for block in blocks) == (
        (7, 1, 6),
        (7, 1, 6),
        (7, 2, 8),
        (7, 2, 4),
    )
    for (multiplicity, irrep), block, channel_major_block in zip(
        layout.local_irreps,
        blocks,
        channel_major_blocks,
    ):
        assert channel_major_block.shape == (7, 2, irrep.dim, multiplicity)
        torch.testing.assert_close(
            channel_major_block.permute(0, 2, 3, 1).reshape(
                7,
                irrep.dim,
                multiplicity * 2,
            ),
            block,
        )
    torch.testing.assert_close(layout.inverse(blocks, wigner_inv), input)
    torch.testing.assert_close(
        layout.inverse_channel_major(channel_major_blocks, wigner_inv),
        input,
    )


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


@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("reflected", [False, True])
def test_circular_harmonics_is_native_o2_equivariant(normalize, reflected):
    module = o2.CircularHarmonics(4, normalize=normalize).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    vectors = torch.randn(9, 2, dtype=DTYPE, device=DEVICE)
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    input_transform = o2.o2_representation("1m", angle, reflected)
    output_transform = o2.o2_irreps_representation(
        module.irreps_out,
        angle,
        reflected,
    )

    transformed_vectors = torch.einsum("ij,bj->bi", input_transform, vectors)
    actual = module(transformed_vectors)
    expected = torch.einsum("ij,bj->bi", output_transform, module(vectors))
    torch.testing.assert_close(actual, expected)


def test_circular_harmonics_matches_analytic_angles_and_zero():
    angles = torch.tensor([0.2, -0.7], dtype=DTYPE, device=DEVICE)
    vectors = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
    actual = o2.circular_harmonics(vectors, 3)
    expected = torch.cat(
        [torch.ones_like(angles).unsqueeze(-1)]
        + [
            torch.stack((torch.cos(order * angles), torch.sin(order * angles)), dim=-1)
            for order in range(1, 4)
        ],
        dim=-1,
    )
    torch.testing.assert_close(actual, expected)

    at_zero = o2.CircularHarmonics(3)(torch.zeros(1, 2, dtype=DTYPE, device=DEVICE))
    torch.testing.assert_close(at_zero[:, :1], torch.ones_like(at_zero[:, :1]))
    torch.testing.assert_close(at_zero[:, 1:], torch.zeros_like(at_zero[:, 1:]))


def test_circular_harmonics_gradcheck():
    vectors = torch.randn(
        4,
        2,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    module = o2.CircularHarmonics(3, normalize=True).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    assert torch.autograd.gradcheck(module, (vectors,))


def test_o2_linear_is_exported_without_prefixed_class_name():
    assert o2.CircularHarmonics.__name__ == "CircularHarmonics"
    assert o2.Linear.__name__ == "Linear"
    assert o2.O2Gate.__name__ == "O2Gate"
    assert o2.O3O2Layout.__name__ == "O3O2Layout"
    assert o2.TensorProduct.__name__ == "TensorProduct"
    assert o2.Irrep.__name__ == "Irrep"
    assert o2.Irreps.__name__ == "Irreps"


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_gate_is_equivariant_and_matches_grouped_forward(reflected):
    irreps_out = o2.Irreps("2x0e+0o+2x1m+2m")
    module = O2Gate(
        irreps_out,
        act_0e=torch.nn.SiLU(),
        act_0o=None,
        act_lm=torch.nn.Sigmoid(),
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        4,
        module.irreps_in.dim,
        3,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    output = module(input)
    grouped = module.forward_grouped(_pack_o2_groups(module.irreps_in, input))
    grouped_output = _unpack_o2_groups(irreps_out, grouped, input.size(-1))
    torch.testing.assert_close(grouped_output, output)

    angle = torch.tensor(0.41, dtype=DTYPE, device=DEVICE)
    input_transform = o2.o2_irreps_representation(
        module.irreps_in,
        angle,
        reflected,
    )
    output_transform = o2.o2_irreps_representation(
        module.irreps_out,
        angle,
        reflected,
    )
    transformed_input = torch.einsum("ij,bjc->bic", input_transform, input)
    transformed_output = module(transformed_input)
    expected = torch.einsum("ij,bjc->bic", output_transform, output)
    torch.testing.assert_close(transformed_output, expected)


@pytest.mark.parametrize("direct_0o", [False, True])
def test_o2_gate_gradcheck_and_gradgradcheck(direct_0o):
    module = O2Gate(
        "0e+0o+1m",
        act_0e=torch.nn.SiLU(),
        act_0o=torch.nn.Tanh() if direct_0o else None,
        act_lm=torch.nn.Sigmoid(),
    ).to(device=DEVICE, dtype=DTYPE)
    input = torch.randn(
        2,
        module.irreps_in.dim,
        2,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    assert torch.autograd.gradcheck(module, (input,), fast_mode=True)
    assert torch.autograd.gradgradcheck(module, (input,), fast_mode=True)


@pytest.mark.parametrize("reflected", [False, True])
def test_o2_gate_direct_0o_activation_is_equivariant(reflected):
    irreps_out = o2.Irreps("2x0e+0o+2x1m+2m")
    module = O2Gate(
        irreps_out,
        act_0e=torch.nn.SiLU(),
        act_0o=torch.nn.Tanh(),
        act_lm=torch.nn.Sigmoid(),
    ).to(device=DEVICE, dtype=DTYPE)
    assert module.num_gates == 3

    input = torch.randn(
        4,
        module.irreps_in.dim,
        3,
        dtype=DTYPE,
        device=DEVICE,
    )
    output = module(input)
    grouped = module.forward_grouped(_pack_o2_groups(module.irreps_in, input))
    torch.testing.assert_close(
        _unpack_o2_groups(irreps_out, grouped, input.size(-1)),
        output,
    )

    angle = torch.tensor(0.41, dtype=DTYPE, device=DEVICE)
    input_transform = o2.o2_irreps_representation(
        module.irreps_in,
        angle,
        reflected,
    )
    output_transform = o2.o2_irreps_representation(
        module.irreps_out,
        angle,
        reflected,
    )
    transformed_input = torch.einsum("ij,bjc->bic", input_transform, input)
    transformed_output = module(transformed_input)
    expected = torch.einsum("ij,bjc->bic", output_transform, output)
    torch.testing.assert_close(transformed_output, expected)


def test_o2_gate_rejects_nonodd_direct_0o_activation():
    with pytest.raises(ValueError, match="act_0o must be an odd function"):
        O2Gate(
            "0e+0o+1m",
            act_0e=torch.nn.SiLU(),
            act_0o=torch.nn.SiLU(),
            act_lm=torch.nn.Sigmoid(),
        )


def test_o2_gate_accepts_passed_scaled_tanh_for_0o():
    module = O2Gate(
        "0e+0o+1m",
        act_0e=torch.nn.SiLU(),
        act_0o=ScaledTanh(),
        act_lm=torch.nn.Sigmoid(),
    )
    assert isinstance(module.act_0o, ScaledTanh)
    assert module.num_gates == 1

    gated_0o = O2Gate(
        "0e+0o+1m",
        act_0e=torch.nn.SiLU(),
        act_0o=None,
        act_lm=torch.nn.Sigmoid(),
    )
    assert gated_0o.act_0o is None
    assert gated_0o.num_gates == 2


@pytest.mark.parametrize(
    ("activation", "scaled_activation"),
    [
        (torch.nn.functional.silu, ScaledSiLU),
        (torch.sigmoid, ScaledSigmoid),
        (torch.tanh, ScaledTanh),
    ],
)
def test_activation_scale_factor_matches_e3nn_without_changing_rng(
    activation,
    scaled_activation,
):
    from e3nn.math import normalize2mom

    torch.manual_seed(1729)
    random_state = torch.random.get_rng_state().clone()
    actual = get_activation_scale_factor(activation)
    torch.testing.assert_close(torch.random.get_rng_state(), random_state)

    expected = normalize2mom(activation).cst
    assert actual == expected
    assert scaled_activation().scale_factor == expected
    assert (
        ACTIVATION[f"scaled_{scaled_activation.__name__[6:].lower()}"]
        is scaled_activation
    )


def test_scaled_activation_factory_uses_fixed_common_constants_and_general_fallback():
    from e3nn.math import normalize2mom

    assert isinstance(get_scaled_activation("silu"), ScaledSiLU)
    assert isinstance(get_scaled_activation("scaled_silu"), ScaledSiLU)
    assert isinstance(get_scaled_activation("sigmoid"), ScaledSigmoid)
    assert isinstance(get_scaled_activation("tanh"), ScaledTanh)

    torch.manual_seed(1729)
    random_state = torch.random.get_rng_state().clone()
    scaled_relu = get_scaled_activation("relu")
    torch.testing.assert_close(torch.random.get_rng_state(), random_state)
    assert scaled_relu.scale_factor == normalize2mom(torch.nn.ReLU()).cst


def test_o2_tensor_product_irrep_rules_are_complete():
    assert Irrep("0e") * Irrep("0o") == (Irrep("0o"),)
    assert Irrep("0o") * Irrep("0o") == (Irrep("0e"),)
    assert Irrep("0o") * Irrep("2m") == (Irrep("2m"),)
    assert Irrep("1m") * Irrep("2m") == (Irrep("1m"), Irrep("3m"))
    assert Irrep("2m") * Irrep("2m") == (
        Irrep("0e"),
        Irrep("0o"),
        Irrep("4m"),
    )


@pytest.mark.parametrize("reflected", [False, True])
@pytest.mark.parametrize("path_mode", ["u1u", "uuu", "uvw"])
def test_o2_tensor_product_is_equivariant(reflected, path_mode):
    torch.manual_seed(11)
    irreps_in1 = o2.Irreps("0e+0o+1m+2m")
    irreps_in2 = o2.Irreps("0e+0o+1m")
    irreps_out = o2.Irreps("2x0e+2x0o+2x1m+2x2m+3m")
    module = o2.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        channels_in1=2,
        channels_in2=1 if path_mode == "u1u" else 2,
        channels_out=2 if path_mode in ("u1u", "uuu") else 3,
        path_mode=path_mode,
    ).to(device=DEVICE, dtype=DTYPE)
    input1 = torch.randn(4, irreps_in1.dim, 2, dtype=DTYPE, device=DEVICE)
    input2 = torch.randn(
        4,
        irreps_in2.dim,
        module.channels_in2,
        dtype=DTYPE,
        device=DEVICE,
    )
    angle = torch.tensor(0.37, dtype=DTYPE, device=DEVICE)
    representation1 = o2.o2_irreps_representation(
        irreps_in1,
        angle,
        reflected,
    )
    representation2 = o2.o2_irreps_representation(
        irreps_in2,
        angle,
        reflected,
    )
    representation_out = o2.o2_irreps_representation(
        irreps_out,
        angle,
        reflected,
    )

    transformed1 = torch.einsum("ij,bjc->bic", representation1, input1)
    transformed2 = torch.einsum("ij,bjc->bic", representation2, input2)
    actual = module(transformed1, transformed2)
    expected = torch.einsum(
        "ij,bjc->bic",
        representation_out,
        module(input1, input2),
    )
    torch.testing.assert_close(actual, expected)


def test_o2_tensor_product_u1u_matches_dense_uvw():
    irreps_in1 = o2.Irreps("0e+1m")
    irreps_in2 = o2.Irreps("0o+1m")
    irreps_out = o2.Irreps("0e+0o+1m+2m")
    u1u = o2.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        channels_in1=3,
        path_mode="u1u",
        internal_weights=False,
    ).to(device=DEVICE, dtype=DTYPE)
    uvw = o2.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        channels_in1=3,
        channels_in2=1,
        channels_out=3,
        path_mode="uvw",
        internal_weights=False,
    ).to(device=DEVICE, dtype=DTYPE)
    assert u1u.path == uvw.path
    input1 = torch.randn(5, irreps_in1.dim, 3, dtype=DTYPE, device=DEVICE)
    input2 = torch.randn(5, irreps_in2.dim, 1, dtype=DTYPE, device=DEVICE)
    weight = torch.randn(*u1u.weight_shape, dtype=DTYPE, device=DEVICE)
    dense_weight = torch.zeros(*uvw.weight_shape, dtype=DTYPE, device=DEVICE)
    channel_indices = torch.arange(3, device=DEVICE)
    dense_weight[:, channel_indices, 0, channel_indices] = weight
    torch.testing.assert_close(
        u1u(input1, input2, weight),
        uvw(input1, input2, dense_weight),
    )


def test_o2_tensor_product_u1u_validates_channels():
    with pytest.raises(ValueError, match="channels_in2 == 1"):
        o2.TensorProduct(
            "1m",
            "1m",
            "0e",
            channels_in1=2,
            channels_in2=2,
            path_mode="u1u",
        )
    with pytest.raises(ValueError, match="channels_in1 == channels_out"):
        o2.TensorProduct(
            "1m",
            "1m",
            "0e",
            channels_in1=2,
            channels_in2=1,
            channels_out=3,
            path_mode="u1u",
        )


def test_o2_tensor_product_pseudoscalar_rotates_positive_order_by_quarter_turn():
    module = o2.TensorProduct(
        "0o",
        "1m",
        "1m",
        channels_in1=1,
        path_mode="uuu",
        internal_weights=False,
        path_norm=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input1 = torch.tensor([[[2.0]]], dtype=DTYPE, device=DEVICE)
    input2 = torch.tensor([[[3.0], [5.0]]], dtype=DTYPE, device=DEVICE)
    weight = torch.ones(module.weight_shape, dtype=DTYPE, device=DEVICE)
    expected = torch.tensor([[[-10.0], [6.0]]], dtype=DTYPE, device=DEVICE)
    torch.testing.assert_close(module(input1, input2, weight), expected)


@pytest.mark.parametrize("path_mode", ["u1u", "uuu"])
def test_o2_tensor_product_zero_pads_missing_output_irreps(path_mode):
    module = o2.TensorProduct(
        "0e",
        "0e",
        "0e+0o+1m",
        channels_in1=2,
        path_mode=path_mode,
    ).to(device=DEVICE, dtype=DTYPE)
    input1 = torch.randn(3, 1, 2, dtype=DTYPE, device=DEVICE)
    input2 = torch.randn(
        3,
        1,
        module.channels_in2,
        dtype=DTYPE,
        device=DEVICE,
    )
    output = module(input1, input2)
    torch.testing.assert_close(output[:, 1:], torch.zeros_like(output[:, 1:]))


@pytest.mark.parametrize("path_mode", ["u1u", "uvw"])
def test_o2_tensor_product_external_weight_gradcheck_and_gradgradcheck(path_mode):
    module = o2.TensorProduct(
        "0e+1m",
        "0o+1m",
        "0e+0o+1m+2m",
        channels_in1=2,
        channels_in2=1 if path_mode == "u1u" else 2,
        channels_out=2,
        path_mode=path_mode,
        internal_weights=False,
    ).to(device=DEVICE, dtype=DTYPE)
    input1 = torch.randn(
        2,
        module.irreps_in1.dim,
        module.channels_in1,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    input2 = torch.randn(
        2,
        module.irreps_in2.dim,
        module.channels_in2,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    weight = torch.randn(
        module.weight_shape,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    def function(x, y, w):
        return module(x, y, w)

    assert torch.autograd.gradcheck(function, (input1, input2, weight))
    assert torch.autograd.gradgradcheck(function, (input1, input2, weight))


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


def test_o2_linear_grouped_gemm_matches_pathwise_forward():
    torch.manual_seed(9)
    module = o2.Linear(
        "2x0e+0o+3x1m+2m",
        "0e+2x0o+2x1m+2x2m",
        channels_in=2,
        channels_out=3,
        path_mode="uv",
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
    mag_Lmax=1,
    angular_max=None,
    nonlinear=None,
    scalar_act="silu",
):
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
        edge_feats_channel=4,
        target_irreps=o3.Irreps("0e"),
        num_radial_basis=4,
        num_mag_radial_basis=3,
        magnetic_irreps=o3.Irreps.spherical_harmonics(mag_Lmax, p=1),
        radial_mlp=[8],
        radial_bias=True,
        irreps_in=o3.Irreps("2x0e + 2x1o"),
        scalar_act=scalar_act,
        tensor_act=None,
        edge_ace_hidden=None,
        parity=True,
        nonlinear=nonlinear,
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
        None,
        edge_feats,
        None,
        edge_index,
        cutoff,
        wigner,
        wigner_inv,
        magnetic_radial_basis,
        magnetic_node_attrs,
    )


def test_o2_magnetic_interaction_uses_uv_gate_uv():
    torch.manual_seed(7)
    module = _build_o2_magnetic_interaction()
    assert module.edge_info.dims[0] == module.edge_feats_channel + 2 * 3
    assert module.rejector.linear_in.path_mode == "uv"
    assert module.rejector.linear_in.internal_weights
    assert isinstance(module.rejector.gate, O2Gate)
    assert isinstance(module.rejector.gate.act_0o, ScaledTanh)
    assert module.rejector.linear_out.path_mode == "uv"
    assert module.rejector.linear_out.internal_weights
    assert not isinstance(module.linear_down, torch.nn.Identity)

    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_magnetic_interaction(module, inputs)
    assert output.shape == (inputs[0].size(0), module.rejector.irreps_out.dim)
    trainable = tuple(module.rejector.parameters()) + tuple(
        module.edge_info.parameters()
    )
    gradients = torch.autograd.grad(
        output.square().sum(),
        (inputs[0], inputs[2], inputs[-1], *trainable),
        create_graph=True,
    )
    assert all(gradient.isfinite().all() for gradient in gradients)


def test_o2_magnetic_interaction_parses_scalar_activations():
    single = _build_o2_magnetic_interaction(scalar_act="scaled_silu")
    assert isinstance(single.rejector.gate.act_0e, ScaledSiLU)
    assert isinstance(single.rejector.gate.act_0o, ScaledTanh)

    separate = _build_o2_magnetic_interaction(scalar_act=["scaled_silu", "tanh"])
    assert isinstance(separate.rejector.gate.act_0e, ScaledSiLU)
    assert isinstance(separate.rejector.gate.act_0o, ScaledTanh)


@pytest.mark.parametrize(
    "scalar_act",
    [
        ["silu"],
        ["silu", "tanh", "sigmoid"],
        ("silu", "tanh"),
        1,
        None,
    ],
)
def test_o2_magnetic_interaction_rejects_invalid_scalar_activations(scalar_act):
    with pytest.raises(TypeError, match="string or a list of two strings"):
        _build_o2_magnetic_interaction(scalar_act=scalar_act)


def test_o2_magnetic_interaction_rejects_nonstring_scalar_activation_entries():
    with pytest.raises(TypeError, match="entries must be strings"):
        _build_o2_magnetic_interaction(scalar_act=["silu", None])


def test_o2_magnetic_interaction_preserves_outer_o3_gate_and_linears():
    torch.manual_seed(11)
    module = _build_o2_magnetic_interaction(nonlinear="sigmoid_gate")
    assert not isinstance(module.linear_up, torch.nn.Identity)
    assert not isinstance(module.linear_down, torch.nn.Identity)
    assert not isinstance(module.nonlinearity, torch.nn.Identity)
    assert not isinstance(module.linear_nonlinearity, torch.nn.Identity)

    messages = _evaluate_o2_magnetic_interaction(
        module,
        _o2_magnetic_inputs(module),
    )
    output = module.linear_nonlinearity(
        module.nonlinearity(module.linear_down(messages))
    )
    assert output.shape == (messages.size(0), module.irreps_out.dim)
    parameters = (
        *module.linear_down.parameters(),
        *module.linear_nonlinearity.parameters(),
    )
    gradients = torch.autograd.grad(output.square().sum(), (messages, *parameters))
    assert all(gradient.isfinite().all() for gradient in gradients)


@pytest.mark.parametrize("mag_Lmax", [1, 2])
@pytest.mark.parametrize("improper", [False, True])
def test_o2_magnetic_interaction_is_globally_o3_equivariant(
    mag_Lmax,
    improper,
):
    torch.manual_seed(8)
    module = _build_o2_magnetic_interaction(mag_Lmax)
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


def test_o2_magnetic_interaction_restricts_all_magnetic_degrees():
    module = _build_o2_magnetic_interaction(mag_Lmax=2)
    assert module.magnetic_irreps == o3.Irreps("0e + 1e + 2e")
    assert module.rejector.magnetic_layout.local_irreps == o2.Irreps(
        "2x0e + 0o + 2x1m + 2m"
    )


@pytest.mark.parametrize("mag_Lmax", [0, 2])
def test_o2_magnetic_interaction_validates_mag_Lmax(mag_Lmax):
    with pytest.raises(
        ValueError,
        match="mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax",
    ):
        _build_o2_magnetic_interaction(mag_Lmax=mag_Lmax, angular_max=1)

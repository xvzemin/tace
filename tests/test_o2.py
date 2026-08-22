import ast
from copy import deepcopy
from pathlib import Path

import pytest
import torch
from e3nn import o3

from eqx import o2
from eqx.o2 import (
    Irrep,
    Irreps,
    O2Gate,
    WignerD,
)

from tace.models._e3nn.inter import (
    INTERACTION,
    O2Interaction,
    O2MagneticInteraction,
)
from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG
from tace.models._e3nn.o2 import (
    O2MagneticScatterLinear,
    O2ScatterLinear,
    RadialRotaryComplexAttention,
)
from tace.models._e3nn.representation import Representation
from tace.models.angular import SolidHarmonics
from tace.models.linear import torchLinear
from tace.models.mag import MagneticBasis
from tace.models.mlp import (
    ACTIVATION,
    ScaledSigmoid,
    ScaledSiLU,
    ScaledTanh,
    get_activation_scale_factor,
    get_scaled_activation,
)
from tace.models.radial import j0SincSphericalBesselBasis, j0SphericalBesselBasis

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


@pytest.mark.parametrize(("Lmax", "lmax"), [(2, 3), (3, 2)])
def test_o2_representation_uses_common_angular_coverage(Lmax, lmax):
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["node_embedding"]["type"] = "linear"
    config["atomic_basis"]["type"] = ["o2"]
    config["atomic_basis"]["nonlinear"] = ["sigmoid_gate"]
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
    assert representation.so2_angular_basis.lmax == common_lmax
    rejector = representation.interactions[0].rejector
    assert rejector.reshape_in.lmax == common_lmax
    assert rejector.reshape_out.lmax == common_lmax


def test_o2_does_not_import_other_tace_model_modules():
    o2_directory = Path(__file__).resolve().parents[1] / "eqx" / "o2"
    for source_path in o2_directory.glob("*.py"):
        tree = ast.parse(source_path.read_text(), filename=str(source_path))
        absolute_imports = {
            node.module
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level == 0
        }
        absolute_imports.update(
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )
        assert not any(
            module == "tace" or module.startswith("tace.")
            for module in absolute_imports
        ), source_path


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


def test_o3_o2_layout_mmax_restricts_local_blocks():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        irreps = o3.Irreps("2x0e+2x0o+2x1e+2x1o+2x2e+2x2o")
        layout = o2.O3O2Layout(irreps, lmax=2, mmax=1).to(device=DEVICE)
        wigner_module = WignerD(1, 2).to(device=DEVICE)
    finally:
        torch.set_default_dtype(previous_dtype)

    edge_vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner, wigner_inv = wigner_module.get_wigner(edge_vectors)
    input = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)
    blocks = layout(input, wigner)

    assert layout.local_irreps == o2.Irreps("3x0e+3x0o+4x1m")
    assert tuple(block.shape for block in blocks) == (
        (7, 1, 6),
        (7, 1, 6),
        (7, 2, 8),
    )
    assert layout.inverse(blocks, wigner_inv).shape == input.shape


def test_o3_o2_layout_inverse_rescales_shared_higher_mmax_wigner():
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        irreps = o3.Irreps("2x0e+2x1o+2x2e+2x3o")
        layout = o2.O3O2Layout(irreps, lmax=3, mmax=0).to(device=DEVICE)
        active_wigner = WignerD(0, 3).to(device=DEVICE)
        shared_wigner = WignerD(3, 3).to(device=DEVICE)
    finally:
        torch.set_default_dtype(previous_dtype)

    edge_vectors = torch.randn(7, 3, dtype=DTYPE, device=DEVICE)
    wigner_active, wigner_inv_active = active_wigner.get_wigner(edge_vectors)
    wigner_shared, wigner_inv_shared = shared_wigner.get_wigner(edge_vectors)
    input = torch.randn(7, irreps.dim, dtype=DTYPE, device=DEVICE)

    active_blocks = layout(input, wigner_active)
    shared_blocks = layout(input, wigner_shared)
    for active_block, shared_block in zip(active_blocks, shared_blocks):
        torch.testing.assert_close(shared_block, active_block)
    torch.testing.assert_close(
        layout.inverse(shared_blocks, wigner_inv_shared),
        layout.inverse(active_blocks, wigner_inv_active),
    )


def test_o3_o2_layout_repr_shows_irreps_conversion():
    layout = o2.O3O2Layout(
        o3.Irreps("2x0e+2x1o+2x1e+2x2e"),
        lmax=2,
        mmax=1,
    )

    assert repr(layout) == (
        f"O3O2Layout({layout.irreps} -> "
        f"{layout.channels * layout.local_irreps})(mmax=1)"
    )


def test_o3_o2_layout_validates_mmax():
    irreps = o3.Irreps("0e+1o+2e")
    with pytest.raises(ValueError, match="0 <= mmax <= lmax"):
        o2.O3O2Layout(irreps, lmax=2, mmax=3)


@pytest.mark.parametrize("lmax,mmax", [(0, 0), (2, 0), (3, 1), (4, 2), (4, 4)])
def test_o2_wigner_layout_matches_legacy_so2_mapping(lmax, mmax):
    from tace.models._e3nn.legacy.oam_2026_07_05 import (
        CoefficientMappingModule as LegacyCoefficientMappingModule,
    )

    full_mapping = LegacyCoefficientMappingModule(
        lmax=lmax,
        mmax=lmax,
        use_rotate_inv_rescale=True,
    )
    retained = full_mapping.coefficient_idx(lmax, mmax)
    expected_inverse_scale = full_mapping.get_rotate_inv_rescale(lmax, mmax)

    truncated_mapping = LegacyCoefficientMappingModule(lmax=lmax, mmax=mmax)
    to_m = truncated_mapping.to_m
    expected_inverse_scale = torch.einsum(
        "nia,ba->nib", expected_inverse_scale, to_m
    )
    expected_layout = torch.zeros(to_m.shape[0], (lmax + 1) ** 2)
    expected_layout[:, retained] = to_m

    wigner = WignerD(mmax=mmax, lmax=lmax)
    torch.testing.assert_close(wigner.wigner_index_to_m_array, expected_layout)
    torch.testing.assert_close(wigner.wigner_inv_rescale, expected_inverse_scale)


@pytest.mark.parametrize("lmax,mmax", [(2, 0), (3, 1), (4, 2), (4, 4)])
def test_o2_wigner_mmax_truncation_preserves_degree_variance(lmax, mmax):
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        module = WignerD(mmax=mmax, lmax=lmax)
    finally:
        torch.set_default_dtype(previous_dtype)
    module = module.to(device=DEVICE, dtype=DTYPE)
    edge_vectors = torch.randn(17, 3, device=DEVICE, dtype=DTYPE)
    wigner, wigner_inv = module.get_wigner(edge_vectors)
    raw_inverse = wigner.transpose(1, 2)

    identity = torch.eye(wigner.shape[1], device=DEVICE, dtype=DTYPE).expand(
        edge_vectors.shape[0], -1, -1
    )
    torch.testing.assert_close(
        wigner @ wigner.transpose(1, 2),
        identity,
        rtol=1e-10,
        atol=1e-10,
    )

    for degree in range(lmax + 1):
        degree_slice = slice(degree**2, (degree + 1) ** 2)
        degree_dim = 2 * degree + 1
        retained_dim = 2 * min(degree, mmax) + 1
        raw_variance = raw_inverse[:, degree_slice].square().sum((1, 2)) / degree_dim
        scaled_variance = wigner_inv[:, degree_slice].square().sum((1, 2)) / degree_dim
        torch.testing.assert_close(
            raw_variance,
            torch.full_like(raw_variance, retained_dim / degree_dim),
            rtol=1e-10,
            atol=1e-10,
        )
        torch.testing.assert_close(
            scaled_variance,
            torch.ones_like(scaled_variance),
            rtol=1e-10,
            atol=1e-10,
        )


def test_o2_wigner_validates_limits_and_hides_legacy_mapping():
    assert not hasattr(o2, "CoefficientMappingModule")
    with pytest.raises(ValueError, match="0 <= mmax <= lmax"):
        WignerD(mmax=3, lmax=2)
    with pytest.raises(ValueError, match="non-negative"):
        WignerD(mmax=0, lmax=-1)
    with pytest.raises(TypeError, match="integer"):
        WignerD(mmax=True, lmax=2)


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
    assert o2.O3O2Layout.restrict(o3.Irreps("1o")) == Irreps("0e+1m")
    assert o2.O3O2Layout.restrict(o3.Irreps("1e")) == Irreps("0o+1m")
    assert o2.O3O2Layout.restrict(o3.Irreps("2e")) == Irreps("0e+1m+2m")
    assert o2.O3O2Layout.restrict(o3.Irreps("2o")) == Irreps("0o+1m+2m")
    assert o2.O3O2Layout.restrict(o3.Irreps("2x1e+0o")) == Irreps("2x0o+1m")


def test_o2_direct_sum_representation_uses_complete_layout():
    irreps = Irreps("0e+0o+2x1m")
    angle = torch.tensor([0.2, -0.7], dtype=DTYPE, device=DEVICE)
    representation = irreps.D_from_angle(angle, True)
    assert representation.shape == (2, irreps.dim, irreps.dim)
    torch.testing.assert_close(representation[:, 0, 0], torch.ones_like(angle))
    torch.testing.assert_close(representation[:, 1, 1], -torch.ones_like(angle))
    expected = Irrep("1m").D_from_angle(angle, True)
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
    input_transform = o2.Irrep("1m").D_from_angle(angle, reflected)
    output_transform = module.irreps_out.D_from_angle(angle, reflected)

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


@pytest.mark.parametrize("path_mode", ["uv", "uu"])
def test_o2_linear_repr_only_shows_shape_bias_and_weight_count(path_mode):
    channels_out = 5 if path_mode == "uv" else None
    module = o2.Linear(
        "2x0e+1m",
        "0e+2x1m",
        channels_in=3,
        channels_out=channels_out,
        path_mode=path_mode,
        bias=True,
    )

    assert repr(module) == (
        f"Linear({module.channels_in * module.irreps_in} -> "
        f"{module.channels_out * module.irreps_out} | "
        f"{module.weight_numel} weights)(bias=True)"
    )


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
    input_transform = module.irreps_in.D_from_angle(angle, reflected)
    output_transform = module.irreps_out.D_from_angle(angle, reflected)
    transformed_input = torch.einsum("ij,bjc->bic", input_transform, input)
    transformed_output = module(transformed_input)
    expected = torch.einsum("ij,bjc->bic", output_transform, output)
    torch.testing.assert_close(transformed_output, expected)


def test_o2_gate_repr_uses_its_own_irreps():
    module = O2Gate(
        "2x0e+0o+2x1m",
        act_0e=torch.nn.SiLU(),
        act_0o=None,
        act_lm=torch.nn.Sigmoid(),
    )

    assert repr(module) == f"O2Gate({module.irreps_in} -> {module.irreps_out})"


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
    input_transform = module.irreps_in.D_from_angle(angle, reflected)
    output_transform = module.irreps_out.D_from_angle(angle, reflected)
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


def _build_o2_asymmetric_contractions(correlation=3, channels=2):
    kwargs = {
        "irreps_in": "0e+0o+1m",
        "irreps_out": "0e+0o+1m+2m",
        "channels": channels,
        "correlation": correlation,
    }
    edge = o2.O2AsymmetricContraction(**kwargs, algorithm="edge").to(
        device=DEVICE,
        dtype=DTYPE,
    )
    node = o2.O2AsymmetricContraction(**kwargs, algorithm="node").to(
        device=DEVICE,
        dtype=DTYPE,
    )
    assert edge.order_num_paths == node.order_num_paths
    assert edge.weight_numel == node.weight_numel
    return edge, node


def test_o2_asymmetric_contraction_algorithms_match_with_batch_weights():
    edge, node = _build_o2_asymmetric_contractions()
    assert not tuple(edge.buffers())
    assert any(name.startswith("generalized_cg_") for name, _ in node.named_buffers())
    inputs = [
        torch.randn(5, edge.irreps_in.dim, 2, dtype=DTYPE, device=DEVICE)
        for _ in range(edge.correlation)
    ]
    weights = torch.randn(
        5,
        edge.weight_numel,
        dtype=DTYPE,
        device=DEVICE,
    )
    edge_output = edge(inputs, weights)
    node_output = node(inputs, weights)
    torch.testing.assert_close(edge_output, node_output)

    sample_outputs = torch.cat(
        [
            node(
                [input[index : index + 1] for input in inputs],
                weights[index : index + 1],
            )
            for index in range(weights.size(0))
        ],
        dim=0,
    )
    torch.testing.assert_close(node_output, sample_outputs)


@pytest.mark.parametrize("algorithm", ["edge", "node"])
@pytest.mark.parametrize("reflected", [False, True])
def test_o2_asymmetric_contraction_is_equivariant(algorithm, reflected):
    module = o2.O2AsymmetricContraction(
        "0e+0o+1m",
        "0e+0o+1m+2m",
        channels=2,
        correlation=3,
        algorithm=algorithm,
    ).to(device=DEVICE, dtype=DTYPE)
    inputs = [
        torch.randn(4, module.irreps_in.dim, 2, dtype=DTYPE, device=DEVICE)
        for _ in range(module.correlation)
    ]
    weights = torch.randn(
        4,
        module.weight_numel,
        dtype=DTYPE,
        device=DEVICE,
    )
    angle = torch.tensor(0.43, dtype=DTYPE, device=DEVICE)
    input_representation = module.irreps_in.D_from_angle(angle, reflected)
    output_representation = module.irreps_out.D_from_angle(angle, reflected)
    transformed_inputs = [
        torch.einsum("ij,bjc->bic", input_representation, input) for input in inputs
    ]
    actual = module(transformed_inputs, weights)
    expected = torch.einsum(
        "ij,bjc->bic",
        output_representation,
        module(inputs, weights),
    )
    torch.testing.assert_close(actual, expected)


def test_o2_asymmetric_contraction_algorithms_match_first_and_second_derivatives():
    edge, node = _build_o2_asymmetric_contractions(correlation=2, channels=1)
    edge_inputs = [
        torch.randn(
            2,
            edge.irreps_in.dim,
            1,
            dtype=DTYPE,
            device=DEVICE,
            requires_grad=True,
        )
        for _ in range(edge.correlation)
    ]
    node_inputs = [input.detach().clone().requires_grad_() for input in edge_inputs]
    edge_weights = torch.randn(
        2,
        edge.weight_numel,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )
    node_weights = edge_weights.detach().clone().requires_grad_()
    output_probe = torch.randn(
        2,
        edge.irreps_out.dim,
        1,
        dtype=DTYPE,
        device=DEVICE,
        requires_grad=True,
    )

    edge_tensors = (*edge_inputs, edge_weights)
    node_tensors = (*node_inputs, node_weights)
    edge_first = torch.autograd.grad(
        edge(edge_inputs, edge_weights),
        edge_tensors,
        grad_outputs=output_probe,
        create_graph=True,
    )
    node_first = torch.autograd.grad(
        node(node_inputs, node_weights),
        node_tensors,
        grad_outputs=output_probe,
        create_graph=True,
    )
    for edge_gradient, node_gradient in zip(edge_first, node_first):
        torch.testing.assert_close(edge_gradient, node_gradient)

    gradient_probes = [torch.randn_like(gradient) for gradient in edge_first]
    edge_target = sum(
        (gradient * probe).sum() for gradient, probe in zip(edge_first, gradient_probes)
    )
    node_target = sum(
        (gradient * probe).sum() for gradient, probe in zip(node_first, gradient_probes)
    )
    edge_second = torch.autograd.grad(edge_target, edge_tensors + (output_probe,))
    node_second = torch.autograd.grad(node_target, node_tensors + (output_probe,))
    for edge_gradient, node_gradient in zip(edge_second, node_second):
        torch.testing.assert_close(edge_gradient, node_gradient)


def test_o2_asymmetric_contraction_has_stable_variance_for_independent_inputs():
    torch.manual_seed(19)
    batch = 20_000
    module = o2.O2AsymmetricContraction(
        "0e+0o+1m",
        "0e+0o+1m",
        channels=1,
        correlation=3,
        algorithm="node",
    )
    inputs = [
        torch.randn(batch, module.irreps_in.dim, 1) for _ in range(module.correlation)
    ]
    weights = torch.randn(batch, module.weight_numel)
    output = module(inputs, weights).squeeze(-1)
    variance = output.var(dim=0, unbiased=False)
    torch.testing.assert_close(
        variance, torch.ones_like(variance), rtol=0.08, atol=0.08
    )


def test_o2_asymmetric_contraction_validates_algorithm_inputs_and_weights():
    with pytest.raises(ValueError, match="algorithm"):
        o2.O2AsymmetricContraction("0e", "0e", 1, 2, algorithm="invalid")

    module = o2.O2AsymmetricContraction("0e", "0e", 1, 2, algorithm="edge")
    input = torch.randn(3, 1, 1)
    weights = torch.randn(3, module.weight_numel)
    with pytest.raises(ValueError, match="Expected 2 independent"):
        module([input], weights)
    with pytest.raises(ValueError, match="trailing dimension"):
        module([input, input.clone()], weights[:, :-1])


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
    representation1 = irreps_in1.D_from_angle(angle, reflected)
    representation2 = irreps_in2.D_from_angle(angle, reflected)
    representation_out = irreps_out.D_from_angle(angle, reflected)

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
    input_transform = irreps_in.D_from_angle(angle, reflected)
    output_transform = irreps_out.D_from_angle(angle, reflected)

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
    assert module.path == ((0, 0),)

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


def test_magnetic_basis_keeps_radial_scaling_formula():
    magmoms = torch.tensor(
        [[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    node_attrs = torch.eye(2, dtype=DTYPE, device=DEVICE)
    basis = MagneticBasis(
        [4.0, 2.0],
        num_basis=4,
        magnetic_irreps="0e",
        atomic_numbers=[1, 2],
        num_elements=2,
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    observed, _ = basis(magmoms, node_attrs)
    element_max = torch.tensor(
        [[1.2 * 4.0 + 0.1], [1.2 * 2.0 + 0.1]],
        dtype=DTYPE,
        device=DEVICE,
    )
    expected = 1.0 - 2.0 * torch.clamp(
        magmoms.norm(dim=-1, keepdim=True) / element_max,
        min=0.0,
        max=1.0,
    ).square()
    torch.testing.assert_close(observed, basis.radial_basis(expected))


def test_magnetic_basis_normalizes_solid_harmonics_by_element_scale():
    irreps = o3.Irreps("0e + 1e + 2e")
    magmoms = torch.tensor(
        [[1.0, 2.0, 3.0], [-2.0, 1.0, 0.5]],
        dtype=DTYPE,
        device=DEVICE,
    )
    node_attrs = torch.eye(2, dtype=DTYPE, device=DEVICE)
    magnetic_basis = MagneticBasis(
        [4.0, 2.0],
        num_basis=4,
        magnetic_irreps=irreps,
        atomic_numbers=[1, 2],
        num_elements=2,
        normalize=True,
    ).to(device=DEVICE, dtype=DTYPE)
    _, observed = magnetic_basis(magmoms, node_attrs)

    element_scale = torch.tensor(
        [[1.2 * 4.0 + 0.1], [1.2 * 2.0 + 0.1]],
        dtype=DTYPE,
        device=DEVICE,
    )
    solid_harmonics = SolidHarmonics(
        irreps,
        irreps_in=o3.Irreps("1e"),
    ).to(device=DEVICE, dtype=DTYPE)
    assert not solid_harmonics.normalize
    torch.testing.assert_close(
        observed,
        solid_harmonics(magmoms / element_scale),
    )


def _build_o2_magnetic_interaction(
    mag_Lmax=1,
    angular_max=None,
    edge_lmax=None,
    mmax=None,
    nonlinear=None,
    scalar_act=None,
    tensor_act=None,
    correlation=1,
    use_asymmetric_contraction=False,
    use_radial_rotary_attention=False,
):
    angular_max = mag_Lmax if angular_max is None else angular_max
    edge_lmax = angular_max if edge_lmax is None else edge_lmax
    mmax = max(angular_max, edge_lmax) if mmax is None else mmax
    module = O2MagneticInteraction(
        layer=0,
        num_layers=1,
        num_elements=2,
        avg_num_neighbors=4.0,
        mmax=mmax,
        Lmax=angular_max,
        lmax=edge_lmax,
        mag_Lmax=mag_Lmax,
        correlation=[correlation],
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
        tensor_act=tensor_act,
        edge_ace_hidden=None,
        parity=True,
        nonlinear=nonlinear,
        num_head=2,
        use_o2_asymmetric_contraction=use_asymmetric_contraction,
        use_radial_rotary_attention=use_radial_rotary_attention,
    )
    return module.to(device=DEVICE, dtype=DTYPE)


def _build_o2_interaction(
    *,
    angular_max=1,
    edge_lmax=None,
    mmax=None,
    correlation=1,
    irreps_in=None,
    scalar_act=None,
    tensor_act=None,
    scatter_norm="avg_num_neighbors",
    use_asymmetric_contraction=False,
    use_radial_rotary_attention=False,
):
    edge_lmax = angular_max if edge_lmax is None else edge_lmax
    mmax = max(angular_max, edge_lmax) if mmax is None else mmax
    irreps_in = o3.Irreps("2x0e + 2x1o") if irreps_in is None else irreps_in
    module = O2Interaction(
        layer=0,
        num_layers=1,
        num_elements=2,
        avg_num_neighbors=4.0,
        mmax=mmax,
        Lmax=angular_max,
        lmax=edge_lmax,
        correlation=[correlation],
        num_channel=2,
        edge_feats_channel=4,
        target_irreps=o3.Irreps("0e"),
        num_radial_basis=4,
        radial_mlp=[8],
        radial_bias=True,
        irreps_in=irreps_in,
        scalar_act=scalar_act,
        tensor_act=tensor_act,
        edge_ace_hidden=None,
        scatter_norm=scatter_norm,
        parity=True,
        num_head=2,
        use_o2_asymmetric_contraction=use_asymmetric_contraction,
        use_radial_rotary_attention=use_radial_rotary_attention,
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
        wigner_module = WignerD(module.mmax, max(module.Lmax, module.lmax))
    finally:
        torch.set_default_dtype(previous_dtype)
    wigner, wigner_inv = wigner_module.to(device=DEVICE).get_wigner(edge_vectors)
    magnetic_basis = MagneticBasis(
        [4.0, 4.0],
        num_basis=4,
        magnetic_irreps=module.magnetic_irreps,
        atomic_numbers=[1, 2],
        num_elements=2,
    ).to(
        device=DEVICE,
        dtype=DTYPE,
    )
    magnetic_radial_basis, magnetic_node_attrs = magnetic_basis(
        initial_noncollinear_magmoms,
        node_attrs,
    )
    magnetic_radial_basis = magnetic_radial_basis[..., 1:]
    return module._compute_messages(
        node_feats,
        node_attrs,
        edge_feats,
        edge_feats,
        None,
        edge_index,
        cutoff,
        wigner,
        wigner_inv,
        magnetic_radial_basis,
        magnetic_node_attrs,
    )


def _evaluate_o2_interaction(module, inputs):
    node_feats, node_attrs, edge_feats, edge_vectors, edge_index, cutoff, _ = inputs
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        wigner_module = WignerD(module.mmax, max(module.Lmax, module.lmax))
    finally:
        torch.set_default_dtype(previous_dtype)
    wigner, wigner_inv = wigner_module.to(device=DEVICE).get_wigner(edge_vectors)
    return module._compute_messages(
        node_feats,
        node_attrs,
        edge_feats,
        edge_feats,
        None,
        edge_index,
        cutoff,
        wigner,
        wigner_inv,
    )


def test_o2_interaction_is_nonmagnetic_base_for_o2_mag():
    module = _build_o2_interaction(
        correlation=2,
        use_asymmetric_contraction=True,
        use_radial_rotary_attention=True,
    )
    assert INTERACTION["o2"] is O2Interaction
    assert issubclass(O2MagneticInteraction, O2Interaction)
    assert type(module.rejector) is O2ScatterLinear
    assert not isinstance(module.rejector, O2MagneticScatterLinear)
    assert module.edge_info.dims[0] == module.edge_feats_channel

    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_interaction(module, inputs)
    parameters = (
        *module.rejector.parameters(),
        *module.edge_info.parameters(),
    )
    gradients = torch.autograd.grad(
        output.square().sum(),
        (inputs[0], inputs[2], *parameters),
        create_graph=True,
    )
    assert all(gradient.isfinite().all() for gradient in gradients)


def test_o2_scatter_requires_cutoff_before_wigner():
    module = _build_o2_interaction()
    inputs = _o2_magnetic_inputs(module)
    radial_weights = module.edge_info(inputs[2])
    with pytest.raises(ValueError, match="requires edge_cutoff"):
        module.rejector(
            inputs[0],
            radial_weights,
            inputs[4],
            None,
            None,
            edge_cutoff=None,
        )


@pytest.mark.parametrize("use_radial_rotary_attention", [False, True])
def test_o2_cutoff_scales_messages_not_radial_weights(
    use_radial_rotary_attention,
):
    torch.manual_seed(17)
    module = _build_o2_interaction(
        use_radial_rotary_attention=use_radial_rotary_attention,
    )
    inputs = _o2_magnetic_inputs(module)
    unit_cutoff = torch.ones_like(inputs[5])
    scale = 0.37
    scaled_cutoff = torch.full_like(inputs[5], scale)
    unit_output = _evaluate_o2_interaction(
        module,
        (*inputs[:5], unit_cutoff, inputs[6]),
    )
    scaled_output = _evaluate_o2_interaction(
        module,
        (*inputs[:5], scaled_cutoff, inputs[6]),
    )
    torch.testing.assert_close(
        scaled_output,
        scale * unit_output,
        atol=5.0e-10,
        rtol=5.0e-10,
    )


def test_o2_radial_rotary_attention_skips_scatter_normalization():
    torch.manual_seed(18)
    module = _build_o2_interaction(
        scatter_norm="density",
        use_radial_rotary_attention=True,
    )
    assert module.rejector.attention is not None
    assert module.scatter_norm is None
    assert not hasattr(module, "edge_density")
    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_interaction(module, inputs)
    module.avg_num_neighbors *= 10.0
    unnormalized_output = _evaluate_o2_interaction(module, inputs)
    torch.testing.assert_close(
        unnormalized_output,
        output,
        atol=5.0e-10,
        rtol=5.0e-10,
    )
    assert not module._uses_edge_density()

    module_without_attention = _build_o2_interaction(scatter_norm="density")
    assert module_without_attention.scatter_norm == "density"
    assert hasattr(module_without_attention, "edge_density")


def test_o2_interaction_mmax_restricts_internal_paths():
    irreps_in = o3.Irreps("2x0e+2x1o+2x2e")
    truncated = _build_o2_interaction(
        angular_max=2,
        mmax=1,
        irreps_in=irreps_in,
    )
    complete = _build_o2_interaction(
        angular_max=2,
        mmax=2,
        irreps_in=irreps_in,
    )

    assert truncated.rejector.irreps_out_local.m_max == 1
    assert complete.rejector.irreps_out_local.m_max == 2
    assert len(truncated.rejector.linear_down.path) < len(
        complete.rejector.linear_down.path
    )
    assert truncated.rejector.linear_down.weight_numel < (
        complete.rejector.linear_down.weight_numel
    )


@pytest.mark.parametrize(("Lmax", "lmax"), [(2, 1), (1, 2)])
@pytest.mark.parametrize("magnetic", [False, True])
def test_o2_interaction_allows_Lmax_and_lmax_to_differ(magnetic, Lmax, lmax):
    common_lmax = max(Lmax, lmax)
    if magnetic:
        module = _build_o2_magnetic_interaction(
            mag_Lmax=1,
            angular_max=Lmax,
            edge_lmax=lmax,
            mmax=common_lmax,
            correlation=2,
        )
        inputs = _o2_magnetic_inputs(module)
        output = _evaluate_o2_magnetic_interaction(module, inputs)
    else:
        module = _build_o2_interaction(
            angular_max=Lmax,
            edge_lmax=lmax,
            mmax=common_lmax,
            correlation=2,
        )
        inputs = _o2_magnetic_inputs(module)
        output = _evaluate_o2_interaction(module, inputs)

    assert module.Lmax == Lmax
    assert module.lmax == lmax
    assert module.rejector.lmax == common_lmax
    assert module.rejector.reshape_in.lmax == common_lmax
    assert module.rejector.reshape_out.lmax == common_lmax
    assert output.shape == (inputs[0].size(0), module.rejector.irreps_out.dim)


@pytest.mark.parametrize("improper", [False, True])
def test_o2_first_layer_only_registers_input_irreps_before_zero_padding(improper):
    module = _build_o2_interaction(
        angular_max=3,
        mmax=3,
        irreps_in=o3.Irreps("2x0e"),
    )

    assert module.rejector.active_mmax == 0
    assert module.rejector.reshape_in.mmax == 0
    assert module.rejector.reshape_out.mmax == 0
    assert module.rejector.irreps_out_local.m_max == 0
    assert module.rejector.irreps_hidden_local.m_max == 0
    assert module.rejector.nonlinearity.irreps_out.m_max == 0
    assert module.rejector.linear_up.irreps_out.m_max == 0
    assert module.rejector.linear_down.irreps_in.m_max == 0
    assert module.rejector.linear_down.irreps_out.m_max == 0
    assert module.rejector.irreps_out.lmax == 3

    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_interaction(module, inputs)
    assert output.isfinite().all()

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        rotation = o3.rand_matrix(dtype=DTYPE)
        if improper:
            rotation = -rotation
        input_rotation = module.irreps_in.D_from_matrix(rotation)
        output_rotation = module.rejector.irreps_out.D_from_matrix(rotation)
    finally:
        torch.set_default_dtype(previous_dtype)
    rotation = rotation.to(DEVICE)
    input_rotation = input_rotation.to(DEVICE)
    output_rotation = output_rotation.to(DEVICE)
    rotated_inputs = (
        inputs[0] @ input_rotation.T,
        inputs[1],
        inputs[2],
        inputs[3] @ rotation.T,
        inputs[4],
        inputs[5],
        inputs[6],
    )
    rotated_output = _evaluate_o2_interaction(module, rotated_inputs)
    torch.testing.assert_close(
        rotated_output,
        output @ output_rotation.T,
        atol=3.0e-10,
        rtol=3.0e-10,
    )


@pytest.mark.parametrize(("angular_max", "mmax"), [(1, 1), (2, 1), (2, 0)])
@pytest.mark.parametrize("improper", [False, True])
def test_o2_interaction_is_globally_o3_equivariant(
    improper,
    angular_max,
    mmax,
):
    torch.manual_seed(16)
    module = _build_o2_interaction(
        correlation=2,
        angular_max=angular_max,
        mmax=mmax,
        use_asymmetric_contraction=True,
        use_radial_rotary_attention=True,
    )
    assert module.rejector.reshape_in.local_irreps.m_max == mmax
    assert module.rejector.reshape_out.local_irreps.m_max == mmax
    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_interaction(module, inputs)

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(DTYPE)
    try:
        rotation = o3.rand_matrix(dtype=DTYPE)
        if improper:
            rotation = -rotation
        node_rotation = module.irreps_in.D_from_matrix(rotation)
        output_rotation = module.rejector.irreps_out.D_from_matrix(rotation)
    finally:
        torch.set_default_dtype(previous_dtype)
    rotation = rotation.to(DEVICE)
    node_rotation = node_rotation.to(DEVICE)
    output_rotation = output_rotation.to(DEVICE)
    rotated_inputs = (
        inputs[0] @ node_rotation.T,
        inputs[1],
        inputs[2],
        inputs[3] @ rotation.T,
        inputs[4],
        inputs[5],
        inputs[6],
    )
    rotated_output = _evaluate_o2_interaction(module, rotated_inputs)
    torch.testing.assert_close(
        rotated_output,
        output @ output_rotation.T,
        atol=5.0e-10,
        rtol=5.0e-10,
    )


def test_o2_magnetic_interaction_uses_uv_gate_uv():
    torch.manual_seed(7)
    module = _build_o2_magnetic_interaction()
    assert module.edge_info.dims[0] == module.edge_feats_channel + 2 * 3
    assert module.rejector.linear_up.path_mode == "uv"
    assert module.rejector.linear_up.internal_weights
    assert isinstance(module.rejector.nonlinearity, O2Gate)
    assert isinstance(module.rejector.nonlinearity.act_0e, ScaledSiLU)
    assert isinstance(module.rejector.nonlinearity.act_0o, ScaledTanh)
    assert isinstance(module.rejector.nonlinearity.act_lm, ScaledSigmoid)
    assert module.rejector.nonlinearity.irreps_out == (
        module.num_channel * module.rejector.irreps_hidden_local
    )
    assert module.rejector.linear_down.path_mode == "uv"
    assert module.rejector.linear_down.internal_weights
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


def test_o2_magnetic_interaction_uses_linear_asymmetric_weights():
    torch.manual_seed(13)
    module = _build_o2_magnetic_interaction(
        correlation=2,
        use_asymmetric_contraction=True,
    )
    assert isinstance(
        module.rejector.asymmetric_contraction,
        o2.O2AsymmetricContraction,
    )
    assert module.rejector.asymmetric_contraction.algorithm == "edge"
    assert module.rejector.contraction_weight_numel > 0
    assert module.rejector.nonlinearity is None
    assert not hasattr(module, "contraction_info")
    assert isinstance(module.rejector.scalar_act, ScaledSiLU)
    assert module.rejector.projection_irreps.count("0e") == (
        module.rejector.irreps_out_local.count("0e") * module.correlation
        + module.rejector.asymmetric_contraction.num_paths
    )
    assert not tuple(module.rejector.asymmetric_contraction.parameters())

    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_magnetic_interaction(module, inputs)
    parameters = tuple(module.rejector.linear_up.parameters())
    gradients = torch.autograd.grad(
        output.square().sum(),
        (inputs[0], inputs[2], inputs[-1], *parameters),
        create_graph=True,
    )
    assert all(gradient.isfinite().all() for gradient in gradients)


def test_o2_magnetic_interaction_uses_real_radial_rotary_attention():
    torch.manual_seed(14)
    module = _build_o2_magnetic_interaction(use_radial_rotary_attention=True)
    assert module.rejector.use_radial_rotary_attention
    assert not hasattr(module.rejector, "radial_phase")
    assert isinstance(module.rejector.attention, RadialRotaryComplexAttention)
    assert sum(
        isinstance(child, o2.Linear)
        for child in module.rejector.attention.modules()
    ) == 1
    radial_basis = torch.randn(
        7,
        module.num_radial_basis,
        device=DEVICE,
        dtype=DTYPE,
    )
    radial_projection = module.rejector.attention.radial_scale_shift
    assert isinstance(radial_projection, torchLinear)
    assert radial_projection.out_features == 2 * module.num_head
    assert torch.count_nonzero(radial_projection.weight) == 0
    assert torch.count_nonzero(radial_projection.bias) == 0
    radial_scale, radial_shift = radial_projection(radial_basis).chunk(
        2,
        dim=-1,
    )
    assert torch.all(torch.sigmoid(radial_scale) == 0.5)
    assert torch.count_nonzero(radial_shift) == 0

    inputs = _o2_magnetic_inputs(module)
    output = _evaluate_o2_magnetic_interaction(module, inputs)
    parameters = tuple(radial_projection.parameters())
    gradients = torch.autograd.grad(
        output.square().sum(),
        (inputs[0], inputs[2], inputs[-1], *parameters),
        create_graph=True,
    )
    assert all(gradient.isfinite().all() for gradient in gradients)


def test_o2_magnetic_interaction_parses_scalar_activations():
    single = _build_o2_magnetic_interaction(scalar_act="scaled_silu")
    assert isinstance(single.rejector.nonlinearity.act_0e, ScaledSiLU)
    assert isinstance(single.rejector.nonlinearity.act_0o, ScaledTanh)

    separate = _build_o2_magnetic_interaction(scalar_act=["scaled_silu", "tanh"])
    assert isinstance(separate.rejector.nonlinearity.act_0e, ScaledSiLU)
    assert isinstance(separate.rejector.nonlinearity.act_0o, ScaledTanh)

    tensor = _build_o2_magnetic_interaction(tensor_act="tanh")
    assert isinstance(tensor.rejector.nonlinearity.act_lm, ScaledTanh)


def test_o2_magnetic_interaction_rejects_invalid_tensor_activation():
    with pytest.raises(TypeError, match="tensor_act must be None or a string"):
        _build_o2_magnetic_interaction(tensor_act=["scaled_sigmoid"])


def test_o2_asymmetric_contraction_defaults_to_scaled_silu():
    module = _build_o2_magnetic_interaction(
        correlation=2,
        use_asymmetric_contraction=True,
    )
    assert module._o2_act_0e_name == "scaled_silu"
    assert module._o2_act_0o_name == "scaled_silu"
    assert module._o2_act_lm_name == "scaled_silu"


def test_o2_asymmetric_contraction_requires_identical_activations():
    with pytest.raises(ValueError, match="act_0e, act_0o, and act_lm"):
        _build_o2_magnetic_interaction(
            correlation=2,
            scalar_act=["scaled_silu", "scaled_tanh"],
            tensor_act="scaled_silu",
            use_asymmetric_contraction=True,
        )


@pytest.mark.parametrize(
    "scalar_act",
    [
        ["silu"],
        ["silu", "tanh", "sigmoid"],
        ("silu", "tanh"),
        1,
    ],
)
def test_o2_magnetic_interaction_rejects_invalid_scalar_activations(scalar_act):
    with pytest.raises(TypeError, match="None, a string, or a list of two strings"):
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


@pytest.mark.parametrize("improper", [False, True])
def test_o2_magnetic_optional_contraction_and_attention_are_o3_equivariant(
    improper,
):
    torch.manual_seed(15)
    module = _build_o2_magnetic_interaction(
        correlation=2,
        use_asymmetric_contraction=True,
        use_radial_rotary_attention=True,
    )
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
        atol=5.0e-10,
        rtol=5.0e-10,
    )


def test_o2_magnetic_interaction_restricts_all_magnetic_degrees():
    module = _build_o2_magnetic_interaction(mag_Lmax=2)
    assert module.magnetic_irreps == o3.Irreps("0e + 1e + 2e")
    assert module.rejector.magnetic_layout.local_irreps == o2.Irreps(
        "2x0e + 0o + 2x1m + 2m"
    )

    truncated = _build_o2_magnetic_interaction(mag_Lmax=2, mmax=1)
    assert truncated.rejector.magnetic_layout.local_irreps == o2.Irreps(
        "2x0e + 0o + 2x1m"
    )


@pytest.mark.parametrize("mag_Lmax", [0, 2])
def test_o2_magnetic_interaction_validates_mag_Lmax(mag_Lmax):
    with pytest.raises(
        ValueError,
        match="mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax",
    ):
        _build_o2_magnetic_interaction(mag_Lmax=mag_Lmax, angular_max=1)

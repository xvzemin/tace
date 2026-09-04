################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import pytest
import torch
from e3nn import o3

from eqx import o2
from tace.models._e3nn.o2 import O2ScatterMagneticTensorProduct

DTYPE = torch.float64


def _time_reverse(features: torch.Tensor, irreps: o2.Irreps) -> torch.Tensor:
    output = features.clone()
    for (ir, _), ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir.t
    return output


def _time_reverse_o3(features: torch.Tensor, irreps: o3.Irreps) -> torch.Tensor:
    output = features.clone()
    for (_, ir), ir_slice in zip(irreps, irreps.slices()):
        output[..., ir_slice] *= ir.t
    return output


def test_o2_time_reversal_irrep_construction_and_products():
    assert o2.Irrep("0e") == o2.Irrep("0ee") == o2.Irrep(0, 1, 1)
    assert o2.Irrep("0eo") == o2.Irrep((0, 1, -1))
    assert o2.Irrep("1mo") == o2.Irrep(1, 0, -1)
    assert str(o2.Irrep("0eo")) == "0eo"
    assert str(o2.Irrep("2mo")) == "2mo"

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


def test_o2_time_reversal_representation_matrix():
    angle = torch.tensor(0.37, dtype=DTYPE)
    for name in ("0ee", "0eo", "0oe", "0oo", "1me", "1mo"):
        ir = o2.Irrep(name)
        expected = ir.D_from_angle(angle) * ir.t
        actual = ir.D_from_angle(angle, time_reversal=True)
        torch.testing.assert_close(actual, expected)


def test_time_odd_circular_harmonics_alternate_time_parity():
    harmonics = o2.CircularHarmonics(4, time_reversal=True).to(dtype=DTYPE)
    assert harmonics.irreps_in == o2.Irreps("1mo")
    assert harmonics.irreps_out == o2.Irreps("0ee+1mo+2me+3mo+4me")
    vectors = torch.randn(8, 2, dtype=DTYPE)
    torch.testing.assert_close(
        harmonics(-vectors),
        _time_reverse(harmonics(vectors), harmonics.irreps_out),
    )


def test_o2_linear_separates_time_parities_and_restricts_bias():
    torch.manual_seed(1)
    irreps = o2.Irreps("2x0ee+3x0eo+2x1me+3x1mo")
    linear = o2.Linear(irreps, irreps, biases=True).to(dtype=DTYPE)
    assert linear.bias_numel == 2
    assert all(
        linear.irreps_in[instruction.i_in].ir.t
        == linear.irreps_out[instruction.i_out].ir.t
        for instruction in linear._weight_instructions
    )

    features = irreps.randn(5, -1, dtype=DTYPE)
    torch.testing.assert_close(
        linear(_time_reverse(features, irreps)),
        _time_reverse(linear(features), irreps),
    )

    with pytest.raises(ValueError, match="time-reversal-even"):
        o2.Linear("0eo", "0eo", biases=[True])


def test_o2_activation_and_gate_track_time_parity():
    odd_activation = o2.Activation("3x0eo", [torch.nn.Tanh()])
    assert odd_activation.irreps_out == o2.Irreps("3x0eo")
    features = torch.randn(4, 3, dtype=DTYPE)
    torch.testing.assert_close(
        odd_activation(-features),
        -odd_activation(features),
    )

    with pytest.raises(ValueError, match="time-reversal-odd"):
        o2.Activation("0eo", [torch.nn.SiLU()])

    gate = o2.Gate(
        "2x0ee",
        [torch.nn.SiLU()],
        "3x0oo",
        [torch.nn.Tanh()],
        "3x1mo",
    ).to(dtype=DTYPE)
    assert gate.irreps_out == o2.Irreps("2x0ee+3x1me")
    features = gate.irreps_in.randn(7, -1, dtype=DTYPE)
    torch.testing.assert_close(
        gate(_time_reverse(features, gate.irreps_in)),
        _time_reverse(gate(features), gate.irreps_out),
    )


def test_o2_tensor_product_tracks_time_parity():
    irreps_in = o2.Irreps("2x1mo")
    irreps_out = o2.Irreps("2x0ee+2x0oe+2x2me")
    tensor_product = o2.TensorProduct(
        irreps_in,
        irreps_in,
        irreps_out,
        [(0, 0, i_out, "uuu", False) for i_out in range(len(irreps_out))],
    )
    first = irreps_in.randn(6, -1, dtype=DTYPE)
    second = irreps_in.randn(6, -1, dtype=DTYPE)
    output = tensor_product(first, second)
    torch.testing.assert_close(
        tensor_product(
            _time_reverse(first, irreps_in),
            _time_reverse(second, irreps_in),
        ),
        _time_reverse(output, irreps_out),
    )

    with pytest.raises(ValueError, match=r"Illegal O\(2\) TensorProduct"):
        o2.TensorProduct(
            "1mo",
            "1mo",
            "0eo",
            [(0, 0, 0, "uuu", False)],
        )


def test_o2_asymmetric_contraction_tracks_time_parity():
    torch.manual_seed(2)
    irreps_in = o2.Irreps("2x0ee+2x0oo+2x1mo")
    irreps_out = o2.Irreps("2x0ee+2x1me")
    contraction = o2.AsymmetricContraction(
        irreps_in,
        irreps_out,
        correlation=2,
        algorithm="edge",
    )
    inputs = [irreps_in.randn(5, -1, dtype=DTYPE) for _ in range(2)]
    weights = torch.randn(5, contraction.weight_numel, dtype=DTYPE)
    output = contraction(inputs, weights)
    reversed_inputs = [_time_reverse(value, irreps_in) for value in inputs]
    torch.testing.assert_close(
        contraction(reversed_inputs, weights),
        _time_reverse(output, contraction.irreps_out),
    )


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_local_frame_preserves_time_parity():
    irreps = o3.Irreps("2x1eo+2x2ee")
    frame = o2.LocalFrame(irreps, lmax=2)
    assert frame.irreps_out == o2.Irreps(
        "2x0ee+2x0oo+2x1me+2x1mo+2x2me"
    )

    edge_vectors = torch.randn(6, 3, dtype=DTYPE)
    wigner, _ = o2.WignerD(mmax=2, lmax=2).to(dtype=DTYPE).get_wigner(
        edge_vectors
    )
    features = torch.randn(6, irreps.dim, dtype=DTYPE)
    local = frame.to(dtype=DTYPE).to_local(features, wigner)
    torch.testing.assert_close(
        frame.to_local(_time_reverse_o3(features, irreps), wigner),
        _time_reverse(local, frame.irreps_out),
    )


@pytest.mark.parametrize("use_radial_rotary_attention", [False, True])
@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_o2_magnetic_interaction_is_time_reversal_invariant(
    use_radial_rotary_attention,
):
    torch.manual_seed(3)
    irreps = o3.Irreps("2x0ee+2x1oe")
    magnetic_irreps = o3.Irreps("0ee+1eo")
    module = O2ScatterMagneticTensorProduct(
        irreps,
        irreps,
        magnetic_irreps,
        num_channel=2,
        lmax=1,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=1,
        num_radial_basis=4,
        use_radial_rotary_attention=use_radial_rotary_attention,
    ).to(dtype=DTYPE)

    edge_index = torch.tensor([[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]])
    num_edges = edge_index.size(1)
    node_features = torch.randn(4, irreps.dim, dtype=DTYPE)
    magnetic = torch.randn(4, magnetic_irreps.dim, dtype=DTYPE)
    conv_weights = torch.randn(num_edges, module.weight_numel, dtype=DTYPE)
    edge_vectors = torch.randn(num_edges, 3, dtype=DTYPE)
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).to(
        dtype=DTYPE
    ).get_wigner(edge_vectors)
    edge_cutoff = torch.rand(num_edges, 1, dtype=DTYPE)
    edge_radial_basis = torch.randn(num_edges, 4, dtype=DTYPE)

    output = module(
        node_features,
        magnetic,
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_radial_basis=edge_radial_basis,
        edge_cutoff=edge_cutoff,
    )
    reversed_output = module(
        node_features,
        _time_reverse_o3(magnetic, magnetic_irreps),
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_radial_basis=edge_radial_basis,
        edge_cutoff=edge_cutoff,
    )
    torch.testing.assert_close(reversed_output, output)

    zero_magnetic_output = module(
        node_features,
        torch.zeros_like(magnetic),
        conv_weights,
        edge_index,
        wigner,
        wigner_inv,
        edge_radial_basis=edge_radial_basis,
        edge_cutoff=edge_cutoff,
    )
    assert not torch.allclose(output, zero_magnetic_output)


@pytest.mark.skipif(
    not hasattr(o3.Irrep("0e"), "t"),
    reason="The installed e3nn does not expose time-reversal irreps.",
)
def test_o2_time_reversal_magnetic_interaction_supports_empty_edges():
    irreps = o3.Irreps("2x0ee+2x1oe")
    magnetic_irreps = o3.Irreps("0ee+1eo")
    module = O2ScatterMagneticTensorProduct(
        irreps,
        irreps,
        magnetic_irreps,
        num_channel=2,
        lmax=1,
        mmax=1,
        even_scalar_act=torch.nn.SiLU(),
        odd_scalar_act=torch.nn.Tanh(),
        tensor_act=torch.nn.Sigmoid(),
        num_head=1,
        num_radial_basis=4,
        use_radial_rotary_attention=False,
    )
    edge_index = torch.empty(2, 0, dtype=torch.long)
    wigner, wigner_inv = o2.WignerD(mmax=1, lmax=1).get_wigner(
        torch.empty(0, 3)
    )
    output = module(
        torch.randn(3, irreps.dim),
        torch.randn(3, magnetic_irreps.dim),
        torch.empty(0, module.weight_numel),
        edge_index,
        wigner,
        wigner_inv,
        edge_cutoff=torch.empty(0, 1),
    )
    torch.testing.assert_close(output, torch.zeros_like(output))

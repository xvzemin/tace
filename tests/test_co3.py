import pytest
import torch

import eqx
from eqx import co3

DTYPE = torch.float64


def _orthogonal(*, improper: bool) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=DTYPE))
    if bool(torch.linalg.det(matrix) < 0) != improper:
        matrix[:, 0] *= -1
    return matrix


def _irreducible(irreps: co3.Irreps, batch: int) -> torch.Tensor:
    outputs = []
    for ir, mul in irreps:
        values = torch.randn(batch, ir.dim, mul, dtype=DTYPE)
        outputs.append(co3.project(values, ir.l).reshape(batch, ir.dim * mul))
    return torch.cat(outputs, dim=-1)


def _transform(
    features: torch.Tensor,
    irreps: co3.Irreps,
    matrix: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("ij,bj->bi", irreps.D_from_matrix(matrix), features)


def _instructions(
    irreps_in1: co3.Irreps,
    irreps_in2: co3.Irreps,
    irreps_out: co3.Irreps,
    mode: str,
    train: bool = True,
) -> list[tuple]:
    return [
        (i_in1, i_in2, i_out, mode, train)
        for i_in1, (ir1, _) in enumerate(irreps_in1)
        for i_in2, (ir2, _) in enumerate(irreps_in2)
        for i_out, (ir_out, _) in enumerate(irreps_out)
        if ir_out in ir1 * ir2
    ]


def test_public_namespace_and_irrep_metadata() -> None:
    assert hasattr(eqx, "co3")
    assert not hasattr(eqx, "o3")
    assert co3.Irrep("2e").dim == 9
    assert co3.Irrep("2e").dof == 5
    assert not co3.Irrep("2e").pseudo
    assert co3.Irrep("2o").pseudo
    assert co3.Irrep("0e").is_even_scalar()
    assert co3.Irrep("0o").is_odd_scalar()
    assert tuple(map(str, co3.Irrep("1e") * co3.Irrep("2o"))) == (
        "1o",
        "2o",
        "3o",
    )
    with pytest.raises(TypeError):
        co3.Irrep(1)


def test_irreps_use_ir_mul_entries_and_flattened_slices() -> None:
    irreps = co3.Irreps("2x0e+1o+2x2e")
    assert irreps.dim == 23
    assert tuple(irreps) == (
        (co3.Irrep("0e"), 2),
        (co3.Irrep("1o"), 1),
        (co3.Irrep("2e"), 2),
    )
    assert irreps.slices() == (slice(0, 2), slice(2, 5), slice(5, 23))
    assert co3.Irreps("0e+0e+1o").simplify() == co3.Irreps("2x0e+1o")
    assert co3.Irreps("1o+0o+0e+1o").regroup() == co3.Irreps("0e+0o+2x1o")
    assert irreps.filter(lmax=1) == co3.Irreps("2x0e+1o")
    assert irreps.filter(keep="2e") == co3.Irreps("2x2e")


@pytest.mark.parametrize("normalization", ["component", "norm"])
def test_irreps_randn_is_irreducible(normalization: str) -> None:
    irreps = co3.Irreps("2x0e+3x1o+2x2e")
    features = irreps.randn(7, -1, normalization=normalization, dtype=DTYPE)
    assert features.shape == (7, irreps.dim)
    for (ir, mul), ir_slice in zip(irreps, irreps.slices()):
        values = features[..., ir_slice].reshape(7, ir.dim, mul)
        torch.testing.assert_close(co3.project(values, ir.l), values)
        if normalization == "norm":
            torch.testing.assert_close(
                values.norm(dim=-2), torch.ones(7, mul, dtype=DTYPE)
            )


def test_direct_sum_matrix_matches_ir_mul_layout() -> None:
    irreps = co3.Irreps("2x1o")
    matrix = _orthogonal(improper=False)
    representation = irreps.D_from_matrix(matrix)
    expected = torch.einsum("ij,uv->iujv", matrix, torch.eye(2, dtype=DTYPE))
    torch.testing.assert_close(representation, expected.reshape(6, 6))


@pytest.mark.parametrize("input_irrep", ["1o", "1e"])
@pytest.mark.parametrize("improper", [False, True])
def test_cartesian_harmonics_equivariance(
    input_irrep: str,
    improper: bool,
) -> None:
    harmonics = co3.CartesianHarmonics(
        3,
        irreps_in=input_irrep,
        normalize=True,
    ).to(dtype=DTYPE)
    vectors = torch.randn(7, 3, dtype=DTYPE)
    matrix = _orthogonal(improper=improper)
    transformed_vectors = torch.einsum(
        "ij,bj->bi", co3.Irrep(input_irrep).D_from_matrix(matrix), vectors
    )
    actual = harmonics(transformed_vectors)
    expected = _transform(harmonics(vectors), harmonics.irreps_out, matrix)
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


@pytest.mark.parametrize(
    ("normalization", "expected_scale"),
    [("component", 1.0), ("norm", None), ("integral", 1.0 / (4.0 * torch.pi))],
)
def test_cartesian_harmonics_normalization(
    normalization: str,
    expected_scale,
) -> None:
    harmonics = co3.CartesianHarmonics(4, normalization=normalization).to(dtype=DTYPE)
    values = harmonics(torch.tensor([[1.0, 2.0, 3.0]], dtype=DTYPE))
    for (ir, _), ir_slice in zip(harmonics.irreps_out, harmonics.irreps_out.slices()):
        norm = values[..., ir_slice].square().sum().item()
        expected = 1.0 if expected_scale is None else ir.dof * expected_scale
        assert norm == pytest.approx(expected, abs=2.0e-12)


@pytest.mark.parametrize("improper", [False, True])
def test_linear_equivariance(improper: bool) -> None:
    irreps_in = co3.Irreps("2x0e+0o+2x1o+1e+2e+2o")
    irreps_out = co3.Irreps("0e+2x0o+1o+2x1e+2e+2o")
    linear = co3.Linear(irreps_in, irreps_out, biases=True).to(dtype=DTYPE)
    features = _irreducible(irreps_in, 5)
    matrix = _orthogonal(improper=improper)
    actual = linear(_transform(features, irreps_in, matrix))
    expected = _transform(linear(features), irreps_out, matrix)
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_linear_external_batched_weights_and_variance() -> None:
    linear = co3.Linear(
        "4x0e",
        "5x0e",
        internal_weights=False,
        shared_weights=False,
    ).to(dtype=DTYPE)
    features = torch.randn(4096, 4, dtype=DTYPE)
    shared = torch.randn(linear.weight_numel, dtype=DTYPE)
    singleton = shared.unsqueeze(0)
    torch.testing.assert_close(linear(features, shared), linear(features, singleton))

    torch.manual_seed(0)
    internal = co3.Linear("128x0e", "128x0e", biases=False).to(dtype=DTYPE)
    values = torch.randn(4096, 128, dtype=DTYPE)
    assert internal(values).var().item() == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize("improper", [False, True])
def test_gate_equivariance(improper: bool) -> None:
    gate = co3.Gate(
        "2x0e+0o",
        [torch.nn.SiLU(), torch.nn.Tanh()],
        "8x0e",
        [torch.nn.Sigmoid()],
        "2x1o+2x1e+2x2e+2x2o",
    ).to(dtype=DTYPE)
    features = _irreducible(gate.irreps_in, 5)
    matrix = _orthogonal(improper=improper)
    actual = gate(_transform(features, gate.irreps_in, matrix))
    expected = _transform(gate(features), gate.irreps_out, matrix)
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_invariant_cartesian_tensors_and_projector() -> None:
    kronecker = co3.delta(dtype=DTYPE)
    epsilon = co3.levi_civita(dtype=DTYPE)
    torch.testing.assert_close(kronecker, torch.eye(3, dtype=DTYPE))
    assert epsilon.square().sum().item() == 6
    values = torch.randn(4, 9, 3, dtype=DTYPE)
    projected = co3.project(values, 2)
    torch.testing.assert_close(co3.project(projected, 2), projected)
    matrix = co3.path_matrix(2, dtype=DTYPE)
    expected = torch.einsum("dm,mn,...nc->...dc", matrix, matrix.T, values)
    torch.testing.assert_close(projected, expected)


def test_tensor_product_can_defer_projection() -> None:
    instructions = [(0, 0, 0, "uuu", False)]
    projected_product = co3.TensorProduct(
        "1o",
        "1o",
        "2e",
        instructions,
        project=True,
    ).to(dtype=DTYPE)
    unprojected_product = co3.TensorProduct(
        "1o",
        "1o",
        "2e",
        instructions,
        project=False,
    ).to(dtype=DTYPE)
    first = torch.randn(8, 3, dtype=DTYPE)
    second = torch.randn(8, 3, dtype=DTYPE)
    unprojected = unprojected_product(first, second)
    expected = projected_product(first, second)
    torch.testing.assert_close(
        unprojected_product.project_output(unprojected),
        expected,
    )
    assert not torch.allclose(unprojected, expected)


def test_tensor_product_simplifies_path_entries_internally() -> None:
    path_irreps = co3.Irreps([(co3.Irrep("1o"), 2), (co3.Irrep("1o"), 2)])
    tensor_product = co3.TensorProduct(
        path_irreps,
        "0e",
        path_irreps,
        [(0, 0, 0, "u1u", False), (1, 0, 1, "u1u", False)],
        project=True,
        simplify=True,
    ).to(dtype=DTYPE)
    assert tensor_product.irreps_in1 == path_irreps.simplify()
    assert tensor_product.irreps_out == path_irreps.simplify()
    assert tensor_product.path_irreps_out == path_irreps

    features = path_irreps.simplify().randn(6, -1, dtype=DTYPE)
    scalars = torch.ones(6, 1, dtype=DTYPE)
    torch.testing.assert_close(tensor_product(features, scalars), features)


def test_tensor_product_parallel_uuu_matches_generic_paths() -> None:
    irreps_in1 = co3.Irreps(
        [
            (co3.Irrep("1o"), 3),
            (co3.Irrep("1o"), 3),
            (co3.Irrep("2e"), 3),
        ]
    )
    irreps_in2 = co3.Irreps("3x1o")
    irreps_out = co3.Irreps(
        [
            (co3.Irrep("0e"), 3),
            (co3.Irrep("0e"), 3),
            (co3.Irrep("1o"), 3),
        ]
    )
    instructions = [
        (0, 0, 0, "uuu", True),
        (1, 0, 1, "uuu", True),
        (2, 0, 2, "uuu", True),
    ]
    reference = co3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        project=True,
        simplify=True,
        internal_weights=False,
        shared_weights=False,
    ).to(dtype=DTYPE)
    reference._contraction_groups = ()
    contraction = co3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        instructions,
        project=True,
        simplify=True,
        internal_weights=False,
        shared_weights=False,
    ).to(dtype=DTYPE)
    assert contraction._contraction_groups == ((0, 1), (2,))

    input1 = _irreducible(irreps_in1.simplify(), 5)
    input2 = _irreducible(irreps_in2, 5)
    weights = torch.randn(5, contraction.weight_numel, dtype=DTYPE)
    torch.testing.assert_close(
        contraction(input1, input2, weights),
        reference(input1, input2, weights),
    )
    exported = torch.export.export(
        contraction,
        (input1, input2, weights),
        strict=True,
    )
    torch.testing.assert_close(
        exported.module()(input1, input2, weights),
        reference(input1, input2, weights),
    )


@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize(
    ("irreps_in1", "irreps_in2", "irreps_out"),
    [
        ("3x1o", "3x1o", "3x0e+3x1e+3x2e"),
        ("3x1e", "3x1o", "3x0o+3x1o+3x2o"),
        ("3x2e", "3x1e", "3x1e+3x2e+3x3e"),
        ("3x2o", "3x2e", "3x0o+3x1o+3x2o+3x3o+3x4o"),
    ],
)
def test_tensor_product_equivariance(
    improper: bool,
    irreps_in1: str,
    irreps_in2: str,
    irreps_out: str,
) -> None:
    first_irreps = co3.Irreps(irreps_in1)
    second_irreps = co3.Irreps(irreps_in2)
    output_irreps = co3.Irreps(irreps_out)
    tensor_product = co3.TensorProduct(
        first_irreps,
        second_irreps,
        output_irreps,
        _instructions(first_irreps, second_irreps, output_irreps, "uuu"),
        project=True,
    ).to(dtype=DTYPE)
    first = _irreducible(first_irreps, 4)
    second = _irreducible(second_irreps, 4)
    matrix = _orthogonal(improper=improper)
    actual = tensor_product(
        _transform(first, first_irreps, matrix),
        _transform(second, second_irreps, matrix),
    )
    expected = _transform(tensor_product(first, second), output_irreps, matrix)
    torch.testing.assert_close(actual, expected, atol=3.0e-12, rtol=3.0e-12)


@pytest.mark.parametrize("mode", ["u1u", "uuu", "uvw"])
def test_tensor_product_connection_modes(mode: str) -> None:
    mul1, mul2, mul_out = {
        "u1u": (3, 1, 3),
        "uuu": (3, 3, 3),
        "uvw": (2, 3, 4),
    }[mode]
    irreps_in1 = co3.Irreps([(co3.Irrep("1o"), mul1)])
    irreps_in2 = co3.Irreps([(co3.Irrep("1o"), mul2)])
    irreps_out = co3.Irreps([(co3.Irrep("1e"), mul_out)])
    tensor_product = co3.TensorProduct(
        irreps_in1,
        irreps_in2,
        irreps_out,
        [(0, 0, 0, mode, True)],
        project=True,
    ).to(dtype=DTYPE)
    output = tensor_product(_irreducible(irreps_in1, 5), _irreducible(irreps_in2, 5))
    assert output.shape == (5, irreps_out.dim)


@pytest.mark.parametrize(
    ("irrep1", "irrep2", "irrep_out"),
    [("1o", "1o", "1e"), ("1o", "2e", "1o"), ("2e", "2e", "2e")],
)
def test_tensor_product_component_normalization_preserves_variance(
    irrep1: str,
    irrep2: str,
    irrep_out: str,
) -> None:
    tensor_product = co3.TensorProduct(
        irrep1,
        irrep2,
        irrep_out,
        [(0, 0, 0, "uuu", False)],
        project=True,
        irrep_normalization="component",
    ).to(dtype=DTYPE)
    first = co3.Irreps(irrep1).randn(32768, -1, dtype=DTYPE)
    second = co3.Irreps(irrep2).randn(32768, -1, dtype=DTYPE)
    output = tensor_product(first, second)
    ir = co3.Irrep(irrep_out)
    second_moment = output.square().sum(dim=-1).mean() / ir.dof
    assert second_moment.item() == pytest.approx(1.0, abs=0.04)


def test_tensor_product_external_weight_derivatives() -> None:
    tensor_product = co3.TensorProduct(
        "1o",
        "1o",
        "1e",
        [(0, 0, 0, "uuu", True)],
        project=True,
        internal_weights=False,
        shared_weights=False,
    ).to(dtype=DTYPE)
    first = torch.randn(2, 3, dtype=DTYPE, requires_grad=True)
    second = torch.randn(2, 3, dtype=DTYPE, requires_grad=True)
    weight = torch.randn(tensor_product.weight_numel, dtype=DTYPE, requires_grad=True)

    def function(input1, input2, external_weight):
        return tensor_product(input1, input2, external_weight)

    assert torch.autograd.gradcheck(function, (first, second, weight))
    assert torch.autograd.gradgradcheck(function, (first, second, weight))

import pytest
import torch

from eqx import co3


DTYPE = torch.float64


def _orthogonal(*, improper: bool) -> torch.Tensor:
    matrix, _ = torch.linalg.qr(torch.randn(3, 3, dtype=DTYPE))
    determinant_is_negative = torch.linalg.det(matrix) < 0
    if bool(determinant_is_negative) != improper:
        matrix[:, 0] *= -1
    return matrix


def _irreducible(irreps: co3.Irreps, batch: int, channels: int) -> torch.Tensor:
    blocks = []
    for irrep in irreps.expanded():
        value = torch.randn(batch, irrep.dim, channels, dtype=DTYPE)
        blocks.append(co3.project(value, irrep.l))
    return torch.cat(blocks, dim=-2)


def _transform(
    input: torch.Tensor,
    irreps: co3.Irreps,
    matrix: torch.Tensor,
) -> torch.Tensor:
    return torch.einsum("ij,bjc->bic", irreps.D_from_matrix(matrix), input)


def test_irrep_metadata_and_complete_parity() -> None:
    assert co3.Irrep("2e").dim == 9
    assert co3.Irrep("2e").dof == 5
    assert not co3.Irrep("2e").pseudo
    assert co3.Irrep("2o").pseudo
    assert tuple(map(str, co3.Irrep("1e") * co3.Irrep("2o"))) == (
        "1o",
        "2o",
        "3o",
    )
    assert co3.Irreps("2x0e+1o+2e").dim == 14
    with pytest.raises(TypeError):
        co3.Irrep(1)


def test_layout_round_trip() -> None:
    layout = co3.Layout("2x0e+1o+2x2e", channels=3)
    input = torch.randn(5, layout.irreps.dim, 3)
    grouped = layout.to_grouped(input)
    assert [block.shape for block in grouped] == [
        (5, 1, 6),
        (5, 3, 3),
        (5, 9, 6),
    ]
    torch.testing.assert_close(layout.from_grouped(grouped), input)


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
        "ij,bj->bi",
        co3.Irrep(input_irrep).D_from_matrix(matrix),
        vectors,
    )
    actual = harmonics(transformed_vectors).unsqueeze(-1)
    expected = _transform(
        harmonics(vectors).unsqueeze(-1),
        harmonics.irreps_out,
        matrix,
    )
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


@pytest.mark.parametrize("improper", [False, True])
def test_linear_complete_o3_equivariance(improper: bool) -> None:
    irreps_in = co3.Irreps("2x0e+0o+2x1o+1e+2e+2o")
    irreps_out = co3.Irreps("0e+2x0o+1o+2x1e+2e+2o")
    linear = co3.Linear(irreps_in, irreps_out, 3, 4).to(dtype=DTYPE)
    input = _irreducible(irreps_in, 5, 3)
    matrix = _orthogonal(improper=improper)
    actual = linear(_transform(input, irreps_in, matrix))
    expected = _transform(linear(input), irreps_out, matrix)
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_linear_is_uv_only() -> None:
    linear = co3.Linear("2x0e+1o", "0e+2x1o", 3, 5)
    assert linear.weight_shape == (4, 3, 5)
    assert linear.weight_scale == pytest.approx(3**-0.5)
    with pytest.raises(TypeError, match="path_mode"):
        co3.Linear("0e", "0e", 3, path_mode="uu")


def test_linear_default_initialization_preserves_variance() -> None:
    torch.manual_seed(0)
    channels = 128
    linear = co3.Linear("4x0e", "5x0e", channels, channels, bias=False)
    input = torch.randn(4096, 4, channels)
    output = linear(input)
    assert input.var().item() == pytest.approx(1.0, abs=0.02)
    assert output.var().item() == pytest.approx(1.0, abs=0.08)


@pytest.mark.parametrize("improper", [False, True])
def test_gate_complete_o3_equivariance(improper: bool) -> None:
    gate = co3.Gate(
        "0e+0o+1o+1e+2e+2o",
        act_0e=torch.nn.SiLU(),
        act_0o=torch.nn.Tanh(),
        act_tensor=torch.nn.Sigmoid(),
    )
    input = _irreducible(gate.irreps_in, 5, 3)
    matrix = _orthogonal(improper=improper)
    actual = gate(_transform(input, gate.irreps_in, matrix))
    expected = _transform(gate(input), gate.irreps_out, matrix)
    torch.testing.assert_close(actual, expected, atol=2.0e-12, rtol=2.0e-12)


def test_tensor_product_requires_explicit_projection() -> None:
    with pytest.raises(TypeError, match="project"):
        co3.TensorProduct("1o", "1o", "0e", 2)


def test_levi_civita_is_a_fixed_buffer() -> None:
    tensor_product = co3.TensorProduct("1o", "1o", "1e", 2, project=True)
    buffers = dict(tensor_product.named_buffers())
    assert "levi_civita" in buffers
    assert "levi_civita" not in tensor_product.state_dict()
    assert tensor_product.levi_civita.shape == (3, 3, 3)
    assert tensor_product.levi_civita.square().sum().item() == 6
    assert not tensor_product.levi_civita.requires_grad
    assert tensor_product.to(dtype=DTYPE).levi_civita.dtype == DTYPE


def test_public_cartesian_invariant_tensors() -> None:
    kronecker = co3.delta(dtype=DTYPE)
    epsilon = co3.levi_civita(dtype=DTYPE)
    torch.testing.assert_close(kronecker, torch.eye(3, dtype=DTYPE))
    assert epsilon.shape == (3, 3, 3)
    assert epsilon.square().sum().item() == 6
    assert epsilon[0, 1, 2].item() == 1
    assert epsilon[0, 2, 1].item() == -1


@pytest.mark.parametrize("improper", [False, True])
@pytest.mark.parametrize(
    ("irreps_in1", "irreps_in2", "irreps_out"),
    [
        ("1o", "1o", "0e+1e+2e"),
        ("1e", "1o", "0o+1o+2o"),
        ("2e", "1e", "1e+2e+3e"),
        ("2o", "2e", "0o+1o+2o+3o+4o"),
    ],
)
def test_tensor_product_complete_o3_equivariance(
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
        3,
        project=True,
        path_mode="uuu",
    ).to(dtype=DTYPE)
    first = _irreducible(first_irreps, 4, 3)
    second = _irreducible(second_irreps, 4, 3)
    matrix = _orthogonal(improper=improper)
    actual = tensor_product(
        _transform(first, first_irreps, matrix),
        _transform(second, second_irreps, matrix),
    )
    expected = _transform(
        tensor_product(first, second), output_irreps, matrix
    )
    torch.testing.assert_close(actual, expected, atol=3.0e-12, rtol=3.0e-12)


@pytest.mark.parametrize("improper", [False, True])
def test_tensor_product_all_complete_o3_paths_through_l2(improper: bool) -> None:
    matrix = _orthogonal(improper=improper)
    for degree1 in range(3):
        for parity1 in (-1, 1):
            irrep1 = co3.Irrep(degree1, parity1)
            first = _irreducible(co3.Irreps(irrep1), 2, 1)
            transformed_first = torch.einsum(
                "ij,bjc->bic", irrep1.D_from_matrix(matrix), first
            )
            for degree2 in range(3):
                for parity2 in (-1, 1):
                    irrep2 = co3.Irrep(degree2, parity2)
                    second = _irreducible(co3.Irreps(irrep2), 2, 1)
                    transformed_second = torch.einsum(
                        "ij,bjc->bic", irrep2.D_from_matrix(matrix), second
                    )
                    for irrep_out in irrep1 * irrep2:
                        tensor_product = co3.TensorProduct(
                            irrep1,
                            irrep2,
                            irrep_out,
                            1,
                            project=True,
                        ).to(dtype=DTYPE)
                        actual = tensor_product(
                            transformed_first,
                            transformed_second,
                        )
                        expected = torch.einsum(
                            "ij,bjc->bic",
                            irrep_out.D_from_matrix(matrix),
                            tensor_product(first, second),
                        )
                        torch.testing.assert_close(
                            actual,
                            expected,
                            atol=3.0e-12,
                            rtol=3.0e-12,
                        )


def test_projection_can_be_deferred_until_after_aggregation() -> None:
    kwargs = {
        "irreps_in1": "1o+2e",
        "irreps_in2": "1e+2o",
        "irreps_out": "0o+1o+2o+3o+4o",
        "channels_in1": 2,
        "path_mode": "uuu",
    }
    raw = co3.TensorProduct(**kwargs, project=False).to(dtype=DTYPE)
    direct = co3.TensorProduct(**kwargs, project=True).to(dtype=DTYPE)
    direct.load_state_dict(raw.state_dict())
    first = _irreducible(raw.irreps_in1, 11, 2)
    second = _irreducible(raw.irreps_in2, 11, 2)
    batch = torch.randint(0, 4, (11,))
    raw_nodes = torch.zeros(4, raw.irreps_out.dim, 2, dtype=DTYPE)
    raw_nodes.index_add_(0, batch, raw(first, second))
    direct_nodes = torch.zeros_like(raw_nodes)
    direct_nodes.index_add_(0, batch, direct(first, second))
    torch.testing.assert_close(
        raw.project_output(raw_nodes),
        direct_nodes,
        atol=3.0e-12,
        rtol=3.0e-12,
    )


def test_tensor_product_first_and_second_derivatives() -> None:
    tensor_product = co3.TensorProduct(
        "1o",
        "1o",
        "1e",
        1,
        project=True,
        path_mode="uuu",
        internal_weights=False,
    ).to(dtype=DTYPE)
    first = torch.randn(2, 3, 1, dtype=DTYPE, requires_grad=True)
    second = torch.randn(2, 3, 1, dtype=DTYPE, requires_grad=True)
    weight = torch.randn(
        tensor_product.weight_shape,
        dtype=DTYPE,
        requires_grad=True,
    )

    def function(input1, input2, external_weight):
        return tensor_product(input1, input2, external_weight)

    assert torch.autograd.gradcheck(function, (first, second, weight))
    assert torch.autograd.gradgradcheck(function, (first, second, weight))

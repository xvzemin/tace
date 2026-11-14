"""Tests for the corresponding EQX source module."""

from copy import deepcopy

import pytest
import torch

from equivariantx import o2

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.float64


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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rotation_precision(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    representations = (o2.Irrep("0e"), o2.Irrep("2m"), o2.Irreps("2x2m"), o2.Irreps())
    for representation in representations:
        for dtype in (torch.float16, torch.bfloat16, torch.complex64):
            angle = torch.tensor(0.3, device=device, dtype=dtype)
            with pytest.raises(TypeError, match="float32 or float64"):
                representation.D_from_angle(angle)
        for dtype in (torch.float32, torch.float64):
            angle = torch.tensor(0.3, device=device, dtype=dtype)
            assert representation.D_from_angle(angle).dtype == dtype
            with torch.autocast(device, dtype=torch.bfloat16):
                with pytest.raises(TypeError, match="autocast"):
                    representation.D_from_angle(angle)
        assert representation.D_from_angle(1).dtype == torch.get_default_dtype()


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_rotation_derivatives_and_compile(device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    angle = torch.randn(3, device=device, requires_grad=True)
    representation = o2.Irrep("2m")
    actual = representation.D_from_angle(angle, reflected=True)
    expected = torch.stack(
        (
            (2 * angle).cos(),
            -(2 * angle).sin(),
            -(2 * angle).sin(),
            -(2 * angle).cos(),
        ),
        -1,
    ).reshape(3, 2, 2)
    for _ in range(4):
        torch.testing.assert_close(actual, expected)
        actual, expected = [
            torch.autograd.grad(value.sin().sum(), angle, create_graph=True)[0]
            for value in (actual, expected)
        ]
    torch.testing.assert_close(actual, expected)
    compiled = torch.compile(
        representation.D_from_angle, backend="aot_eager", fullgraph=True
    )
    torch.testing.assert_close(compiled(angle), representation.D_from_angle(angle))
    torch.testing.assert_close(
        torch.func.vmap(representation.D_from_angle)(angle),
        representation.D_from_angle(angle),
    )

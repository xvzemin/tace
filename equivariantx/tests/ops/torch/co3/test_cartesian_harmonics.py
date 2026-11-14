"""Exact high-rank Cartesian polynomials and their complete derivative paths."""

import io
from fractions import Fraction

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx

from equivariantx import co3
from equivariantx.ops import config
from equivariantx.ops.torch.co3 import cartesian_harmonics as kernel


def close(actual, expected, dtype):
    for value, reference in zip(actual, expected):
        assert value.shape == reference.shape
        delta = value.detach().double() - reference.detach().double()
        relative = delta.norm() / reference.detach().double().norm().clamp_min(1e-30)
        assert torch.isfinite(value).all()
        assert relative < (3e-5 if dtype == torch.float32 else 3e-10), relative.item()


@pytest.mark.parametrize("degree", [4, 5, 6])
def test_exact_coefficients(degree):
    powers = kernel.monomial_powers(degree)
    rows = [dict(row) for row in kernel.polynomial_coefficients(degree)]
    assert all(isinstance(value, Fraction) for row in rows for value in row.values())
    for indices in kernel.monomial_powers(degree - 2):
        trace = {}
        for axis in range(3):
            position = powers.index(
                tuple(value + 2 * (i == axis) for i, value in enumerate(indices))
            )
            for power, value in rows[position].items():
                trace[power] = trace.get(power, Fraction(0)) + value
        assert not any(trace.values())
    for row in rows:
        laplacian = {}
        for index, value in row.items():
            for axis, count in enumerate(powers[index]):
                if count >= 2:
                    power = tuple(
                        number - 2 * (i == axis)
                        for i, number in enumerate(powers[index])
                    )
                    laplacian[power] = (
                        laplacian.get(power, Fraction(0)) + count * (count - 1) * value
                    )
        assert not any(laplacian.values())


@pytest.mark.parametrize("normalization", ["norm", "component", "integral"])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_layout_and_degrees(normalization, normalize, dtype):
    orders = [6, 4, 2, 4, 0, 3]
    layer = co3.CartesianHarmonics(orders, normalize, normalization).to(dtype)
    value = (torch.randn(2, 3, 6, dtype=dtype) * 0.1 + 0.3)[..., ::2]
    value = value.transpose(0, 1)
    assert not value.is_contiguous() and value.stride(-1) == 2
    precise = co3.CartesianHarmonics(orders, normalize, normalization).double()
    close(
        (kernel.polynomial(layer, value),),
        (kernel.reference(precise, value.double()),),
        dtype,
    )
    assert not hasattr(layer, "polynomial_5")
    empty = value[:0]
    assert kernel.polynomial(layer, empty).shape == (
        *empty.shape[:-1],
        layer.irreps_out.dim,
    )


@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_five_derivatives(degree, normalize, dtype):
    layer = co3.CartesianHarmonics(degree, normalize, "component").to(dtype)
    precise = co3.CartesianHarmonics(degree, normalize, "component").double()
    value = (
        torch.randn(2, 3, dtype=dtype) * 0.03
        + torch.tensor([0.3, 0.5, 0.7], dtype=dtype)
    ).requires_grad_()
    reference_value = value.detach().double().requires_grad_()
    actual = kernel.polynomial(layer, value)
    expected = kernel.reference(precise, reference_value)
    close((actual,), (expected,), dtype)
    for _ in range(5):
        seed = torch.randn_like(actual) * 0.1
        actual = torch.autograd.grad(
            (actual.sin() * seed).sum(), value, create_graph=True
        )[0]
        expected = torch.autograd.grad(
            (expected.sin() * seed.double()).sum(), reference_value, create_graph=True
        )[0]
        close((actual,), (expected,), dtype)


@pytest.mark.parametrize("degree", [4, 5, 6])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_functional_transforms(degree, dtype):
    layer = co3.CartesianHarmonics(degree, True, "norm").to(dtype)
    precise = co3.CartesianHarmonics(degree, True, "norm").double()

    def function(x):
        return kernel.polynomial(layer, x)

    def reference(x):
        return kernel.reference(precise, x)

    value = torch.randn(2, 3, dtype=dtype) * 0.03 + torch.tensor(
        [0.3, 0.5, 0.7], dtype=dtype
    )
    tangent = torch.randn_like(value) * 0.1
    close(
        torch.func.jvp(function, (value,), (tangent,)),
        torch.func.jvp(reference, (value.double(),), (tangent.double(),)),
        dtype,
    )
    with torch.autograd.forward_ad.dual_level():
        actual = torch.autograd.forward_ad.unpack_dual(
            function(torch.autograd.forward_ad.make_dual(value, tangent))
        )
        expected = torch.autograd.forward_ad.unpack_dual(
            reference(
                torch.autograd.forward_ad.make_dual(value.double(), tangent.double())
            )
        )
        assert actual.tangent is not None
        close(actual, expected, dtype)

    def first(x):
        return torch.func.jvp(function, (x,), (tangent,))[1]

    def first_reference(x):
        return torch.func.jvp(reference, (x,), (tangent.double(),))[1]

    close(
        torch.func.jvp(first, (value,), (tangent,)),
        torch.func.jvp(first_reference, (value.double(),), (tangent.double(),)),
        dtype,
    )
    close((torch.vmap(function)(value),), (reference(value.double()),), dtype)
    gradient = torch.func.grad(lambda x: function(x).sin().sum())
    reference_gradient = torch.func.grad(lambda x: reference(x).sin().sum())
    close(
        torch.func.jvp(gradient, (value,), (tangent,)),
        torch.func.jvp(reference_gradient, (value.double(),), (tangent.double(),)),
        dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_zero_without_normalization(dtype):
    layer = co3.CartesianHarmonics([4, 6, 4], False, "norm").to(dtype)
    value = torch.zeros(2, 3, dtype=dtype, requires_grad=True)
    result = kernel.polynomial(layer, value)
    assert torch.count_nonzero(result) == 0
    gradient = torch.autograd.grad(result.sin().sum(), value, create_graph=True)[0]
    assert torch.count_nonzero(gradient) == 0
    assert torch.isfinite(torch.autograd.grad(gradient.sum(), value)[0]).all()
    tangent = torch.ones_like(value)
    close(
        torch.func.jvp(lambda x: kernel.polynomial(layer, x), (value,), (tangent,)),
        (result, result),
        dtype,
    )


def test_capture_policy_and_shared_degrees(monkeypatch):
    layer = co3.CartesianHarmonics([6, 4, 2, 4, 0, 3], True, "norm").double()
    value = torch.randn(3, 3, dtype=torch.float64)
    calls = []
    original = kernel.polynomial

    def counted(module, value):
        calls.append(module)
        return original(module, value)

    monkeypatch.setattr(kernel, "polynomial", counted)
    layer(value)
    assert not calls
    graph = make_fx(layer)(value)
    assert len(calls) == 1
    assert (
        sum(node.target == torch.ops.aten.mm.default for node in graph.graph.nodes) == 3
    )
    close((graph(value),), (kernel.reference(layer, value),), value.dtype)
    for switch in ("TORCH_CARTESIAN_HARMONIC_POLYNOMIAL", "TORCH_CARTESIAN_HARMONICS"):
        with monkeypatch.context() as flags:
            flags.setattr(config, switch, False)
            make_fx(layer)(value)
        assert len(calls) == 1
    with monkeypatch.context() as flags:
        flags.setattr(config, "TORCH_REFERENCE", True)
        make_fx(layer)(value)
    assert len(calls) == 1
    for orders in ([], [3], [4, 7]):
        other = co3.CartesianHarmonics(orders, True, "norm").double()
        if other.lmax > 6:
            assert not any(name.startswith("polynomial_") for name in other._buffers)
        graph = make_fx(other)(value)
        close((graph(value),), (kernel.reference(other, value),), value.dtype)
    assert len(calls) == 1


@pytest.mark.parametrize("normalization", ["norm", "component", "integral"])
def test_dtype_and_legacy_pickle(normalization):
    layer = co3.CartesianHarmonics([4, 6, 4], True, normalization)
    expected = {
        degree: kernel.polynomial_matrix(degree, normalization).clone()
        for degree in (4, 6)
    }
    layer.float().double()
    for degree in (4, 6):
        torch.testing.assert_close(
            getattr(layer, f"polynomial_{degree}"), expected[degree], atol=0, rtol=0
        )
    assert not any("polynomial" in name for name in layer.state_dict())
    for degree in (4, 6):
        delattr(layer, f"polynomial_{degree}")
    stream = io.BytesIO()
    torch.save(layer, stream)
    stream.seek(0)
    legacy = torch.load(stream, weights_only=False)
    legacy.float().double()
    assert not hasattr(legacy, "polynomial_4") and not hasattr(legacy, "polynomial_6")
    value = torch.randn(3, 3, dtype=torch.float64)
    graph = make_fx(legacy)(value)
    close((graph(value),), (kernel.reference(legacy, value),), value.dtype)


def test_public_dynamo_graph():
    layer = co3.CartesianHarmonics([4, 2, 6, 4, 3, 0], True, "component").double()
    graphs = []

    def backend(graph, values):
        graphs.append(graph)
        return graph.forward

    run = torch.compile(layer, backend=backend, fullgraph=True, dynamic=True)
    for count in (3, 5, 7):
        value = torch.randn(count, 3, dtype=torch.float64)
        torch._dynamo.mark_dynamic(value, 0)
        torch._dynamo.mark_static(value, 1)
        close((run(value),), (kernel.reference(layer, value),), value.dtype)
    assert len(graphs) == 1

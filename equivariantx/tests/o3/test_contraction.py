"""Path and backend checks for independent and repeated factors."""

import pytest
import torch

import equivariantx as eqx


@pytest.mark.parametrize("space", ["o3", "co3", "o2", "co2"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_independent_axial_path(space, backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if backend == "cuda" else "cpu"
    group = getattr(eqx, space)
    vector, axial, dim = (
        ("2x1o", "2x1e", 6) if space.endswith("3") else ("2x1m", "2x0o", 4)
    )
    op = (
        group.AsymmetricContraction(vector, axial, 2, backend=backend)
        .double()
        .to(device)
    )
    ref = (
        group.AsymmetricContraction(vector, axial, 2, backend="torch")
        .double()
        .to(device)
    )
    x, y = (
        torch.randn(4, dim, device=device, dtype=torch.float64, requires_grad=True)
        for _ in range(2)
    )
    w = torch.randn(
        op.weight_numel, device=device, dtype=torch.float64, requires_grad=True
    )
    actual, expected = op((x, y), w), ref((x, y), w)
    assert actual.abs().max() > 1e-8
    torch.testing.assert_close(actual, -op((y, x), w))
    torch.testing.assert_close(op((x, x), w), torch.zeros_like(actual))
    torch.testing.assert_close(actual, expected)
    for _ in range(2):
        ga, gb = (
            torch.autograd.grad(v.sin().sum(), (x, y, w), create_graph=True)
            for v in (actual, expected)
        )
        for a, b in zip(ga, gb):
            torch.testing.assert_close(a, b)
        actual, expected = sum(v.sum() for v in ga), sum(v.sum() for v in gb)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("path_mode", ["sum", "expand"])
@pytest.mark.parametrize(
    "spherical,cartesian,ins,outs",
    [
        (eqx.o3, eqx.co3, "2x0e+2x1o", "2x0e+2x1o+2x2e"),
        (eqx.o2, eqx.co2, "2x0eo+2x1m+2x2mo", "2x0e+2x1mo+2x2m"),
    ],
)
def test_spherical_cartesian_contraction(
    spherical, cartesian, ins, outs, shared, path_mode
):
    ctor = "SymmetricContraction" if shared else "AsymmetricContraction"
    ops = [
        getattr(group, ctor)(
            ins, outs, 3, path_mode=path_mode, backend="torch"
        ).double()
        for group in (spherical, cartesian)
    ]
    embed = cartesian.ChangeOfBasis(ins, backend="torch").double()
    to_spherical = cartesian.ChangeOfBasis(
        ops[1].irreps_out, inverse=True, backend="torch"
    ).double()
    xs = [
        torch.randn(*head, ops[0].irreps_in.dim, dtype=torch.float64)
        for head in ((2, 1), (1, 3), (1, 1))
    ]
    w = torch.randn(1, 3, ops[0].weight_numel, dtype=torch.float64)
    assert ops[0].weight_numel == ops[1].weight_numel
    expected = ops[0](xs[0] if shared else xs, w)
    actual = ops[1](embed(xs[0]) if shared else tuple(map(embed, xs)), w)
    torch.testing.assert_close(to_spherical(actual), expected, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("space", ["o3", "co3", "o2", "co2"])
@pytest.mark.parametrize("path_mode", ["sum", "expand"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_empty_symmetric_path(space, path_mode, backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cpu" if backend == "torch" else "cuda"
    ins, outs, din, dout = (
        ("2x1o", "2x1e", 6, 6) if space.endswith("3") else ("2x1m", "2x0o", 4, 2)
    )
    layer = (
        getattr(eqx, space)
        .SymmetricContraction(ins, outs, 2, path_mode=path_mode, backend=backend)
        .to(device)
    )
    assert layer.weight_numel == 0
    x = torch.randn(5, din, device=device, requires_grad=True)
    weight = x.new_empty(0, requires_grad=True)
    value = layer(x, weight)
    torch.testing.assert_close(
        value, x.new_zeros(5, 0 if path_mode == "expand" else dout)
    )
    dx, dw = torch.autograd.grad(value.sum(), (x, weight))
    torch.testing.assert_close(dx, torch.zeros_like(x))
    assert dw.shape == weight.shape

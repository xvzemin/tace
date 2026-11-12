"""Tests for the corresponding EQX source module."""

import pytest
import torch
from torch.fx.experimental.proxy_tensor import make_fx


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native CUDA angular products"
)
def test_crossed_angular_product(double_precision):
    from e3nn import o3
    from equivariantx.ops.channel_product import crossed_angular_product

    cg = o3.wigner_3j(1, 1, 2, device="cuda")
    entries = tuple((i, j, k, float(cg[i, j, k])) for i, j, k in cg.nonzero().tolist())
    values = (
        torch.randn(3, 5, 1, 3, device="cuda", requires_grad=True),
        torch.randn(1, 1, 7, 3, device="cuda", requires_grad=True),
    )

    def reference(x, y):
        return torch.einsum("...i,...j,ijk->...k", x, y, cg)

    def function(x, y):
        return crossed_angular_product(x, y, entries, 5)

    actual, expected = function(*values), reference(*values)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    for _ in range(4):
        ga, gb = [
            torch.autograd.grad(value.sin().sum(), values, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(ga, gb, atol=2e-10, rtol=2e-10)
        actual, expected = [
            torch.cat([g.flatten() for g in gradients]) / 30 for gradients in (ga, gb)
        ]
    tangents = tuple(torch.randn_like(value) for value in values)
    torch.testing.assert_close(
        torch.func.jvp(function, values, tangents),
        torch.func.jvp(reference, values, tangents),
        atol=2e-11,
        rtol=2e-11,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Native CUDA products")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
def test_native_functional_transforms(double_precision, compiler):
    from equivariantx.ops.channel_product import local_product

    metadata = repr((3, (2, 1, 2), (((0, 0, 0), 2.0), ((1, 0, 1), -1.0))))
    values = tuple(
        torch.randn(*shape, device="cuda") for shape in ((5, 2, 3), (1, 1, 3))
    )

    def function(x, w):
        return local_product(metadata, x, w).reshape_as(x)

    def reference(x, w):
        return x * w * x.new_tensor((2.0, -1.0))[None, :, None]

    tangents = tuple(torch.randn_like(value) for value in values)

    def jvp(*values):
        return torch.func.jvp(function, values, tangents)

    torch.compiler.reset()
    run = torch.compile(jvp, backend=compiler, fullgraph=True) if compiler else jvp
    torch.testing.assert_close(
        run(*values), torch.func.jvp(reference, values, tangents)
    )
    batched = tuple(torch.stack((value, value * 0.8)) for value in values)
    torch.testing.assert_close(
        torch.vmap(function)(*batched), torch.vmap(reference)(*batched)
    )
    gradient = torch.func.grad(lambda x, w: function(x, w).sin().sum(), argnums=(0, 1))
    reference_gradient = torch.func.grad(
        lambda x, w: reference(x, w).sin().sum(), argnums=(0, 1)
    )

    def gradient_jvp(*values):
        return torch.func.jvp(gradient, values, tangents)

    run = (
        torch.compile(gradient_jvp, backend=compiler, fullgraph=True)
        if compiler
        else gradient_jvp
    )
    torch.testing.assert_close(
        run(*values), torch.func.jvp(reference_gradient, values, tangents)
    )

    def nested(function, *inputs):
        return torch.func.jvp(
            lambda *primals: torch.func.jvp(function, primals, tangents)[1],
            inputs,
            tangents,
        )

    nested_jvp = lambda *inputs: nested(function, *inputs)
    run = nested_jvp
    if compiler:
        graph = make_fx(nested_jvp)(*values)
        graph.graph.eliminate_dead_code()
        graph.recompile()
        run = torch.compile(graph, backend=compiler, fullgraph=True)
    torch.testing.assert_close(run(*values), nested(reference, *values))


def test_local_channel_scaling():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.channel_product import local_product

    indices = [0, 1, 2, 1, 2]
    metadata = repr(
        (4, (5, 3, 5), tuple(((i, j, i), 1.0) for i, j in enumerate(indices)))
    )
    x = torch.randn(3, 5, 8, device="cuda", dtype=torch.float64)[
        ..., ::2
    ].requires_grad_()
    w = torch.randn(3, 3, 4, device="cuda", dtype=torch.float64, requires_grad=True)
    actual = local_product(metadata, x, w).view_as(x)
    expected = x * w[:, indices]
    torch.testing.assert_close(actual, expected)
    losses = [z.square().sum() for z in (actual, expected)]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(loss, (x, w), create_graph=True) for loss in losses
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        losses = [sum(g.square().sum() for g in gs) for gs in gradients]

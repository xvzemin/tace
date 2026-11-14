"""Contraction-path estimates must not constrain symbolic batch dimensions."""

import builtins

import pytest
import torch

from equivariantx.ops.common.torch.einsum import contraction_path, einsum


@pytest.mark.parametrize(
    "equation,shapes",
    [
        ("eij,ejk,ekl->eil", ((0, 2, 3), (1, 3, 4), (0, 4, 5))),
        ("...a,aij,...j->...i", ((0, 3), (3, 4, 5), (0, 5))),
    ],
)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_dynamic_derivatives(equation, shapes, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.compiler.reset()

    @torch.compiler.allow_in_graph
    def operation(*values):
        return einsum(equation, *values)

    graphs = []

    def compiler(graph, inputs):
        graphs.append(graph)
        return graph.forward

    compiled = torch.compile(operation, backend=compiler, dynamic=True, fullgraph=True)
    for edges in (7, 11, 9):
        values = tuple(
            torch.randn(
                *(size or edges for size in shape), device=device, requires_grad=True
            )
            for shape in shapes
        )
        for value, shape in zip(values, shapes):
            if shape[0] == 0:
                torch._dynamo.mark_dynamic(value, 0, min=2)
        actual, expected = compiled(*values), torch.einsum(equation, *values)
        torch.testing.assert_close(actual, expected, atol=1e-9, rtol=1e-9)
        actual, expected = actual.sin().sum(), expected.sin().sum()
        directions = tuple(torch.randn_like(value) * 0.03 for value in values)
        for _ in range(3):
            gradients = [
                torch.autograd.grad(y, values, create_graph=True)
                for y in (actual, expected)
            ]
            torch.testing.assert_close(*gradients, atol=1e-8, rtol=1e-8)
            actual, expected = [
                sum((g * d).sum() for g, d in zip(gs, directions)) for gs in gradients
            ]
    assert len(graphs) == 1


def test_eager_path_selection(monkeypatch):
    original = torch.einsum
    calls = []

    def observed(*args, **kwargs):
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(torch, "einsum", observed)
    values = tuple(torch.randn(3, 3) for _ in range(3))
    torch.testing.assert_close(
        einsum("ij,jk,kl->il", *values), original("ij,jk,kl->il", *values)
    )
    assert calls == ["ij,jk,kl->il"]


def test_optional_planner(monkeypatch):
    original = builtins.__import__

    def without_planner(name, *args, **kwargs):
        if name == "opt_einsum":
            raise ImportError("Optional planner is unavailable")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_planner)
    assert (
        contraction_path.__wrapped__("ij,jk,kl->il", ((2, 3), (3, 4), (4, 5))) is None
    )

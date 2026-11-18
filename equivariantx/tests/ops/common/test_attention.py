"""Native segmented normalization with arbitrary functional derivatives."""

import pytest
import torch

from equivariantx.nn import graph_softmax
from equivariantx.ops import get_kernel
from equivariantx.ops.common.attention import evaluate_adjoint
from equivariantx.ops.common.torch.attention import normalized_exp_reference
from equivariantx.tests.ops.testing import assert_functional_derivatives
from equivariantx.tests.utils.testing import check_forward_ad


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph attention")
@pytest.mark.parametrize("weight_heads", [1, 3])
def test_forward_ad(weight_heads, double_precision):
    values = tuple(
        (torch.randn(7, width, device="cuda") * 0.2).requires_grad_()
        for width in (3, 3, weight_heads)
    )
    target = torch.arange(7, device="cuda") % 3
    check_forward_ad(
        lambda scores, weight: graph_softmax(scores, target, 5, weight, backend="cuda"),
        lambda scores, weight: (
            normalized_exp_reference(scores, weight, target, 5, 1e-16) * weight
        ),
        (values[0], values[2] + 1),
    )
    check_forward_ad(
        lambda *args: evaluate_adjoint(*args, target, 5),
        lambda *args: get_kernel("graph_normalized_exp_adjoint", "torch")(
            *args, target, 5
        ),
        values,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph attention")
@pytest.mark.parametrize("compiler", [None, "aot_eager", "inductor"])
@pytest.mark.parametrize("edges,weight_heads", [(0, 1), (7, 1), (7, 3)])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_functional_transforms(
    compiler, edges, weight_heads, backend, double_precision
):
    values = (
        torch.randn(edges, 3, device="cuda"),
        torch.rand(edges, weight_heads, device="cuda") + 0.5,
        torch.arange(edges, device="cuda") % 3,
    )

    def function(scores, weight, target):
        return graph_softmax(scores, target, 5, weight, 1e-5, backend=backend)

    def reference(scores, weight, target):
        return normalized_exp_reference(scores, weight, target, 5, 1e-5) * weight

    assert_functional_derivatives(function, reference, values, compiler)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graph attention")
@pytest.mark.parametrize("weight_heads", [1, 3])
def test_adjoint_and_four_derivatives(weight_heads, double_precision):
    values = tuple(
        torch.randn(7, width, device="cuda", requires_grad=True) * 0.2
        for width in (3, 3, weight_heads)
    )
    target = torch.arange(7, device="cuda") % 3
    actual = evaluate_adjoint(*values, target, 5)
    expected = get_kernel("graph_normalized_exp_adjoint", "torch")(*values, target, 5)
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    actual, expected = [
        sum(value.sin().sum() for value in output) for output in (actual, expected)
    ]
    for _ in range(4):
        gradients = [
            torch.autograd.grad(value, values, create_graph=True)
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [sum(value.sum() for value in grads) for grads in gradients]


@pytest.mark.parametrize("edges", [0, 9])
def test_rotary_interaction_providers(edges, double_precision, monkeypatch):
    from equivariantx.models.tace.tece_oam_rra.expressions import first_adjoint
    from equivariantx.ops import config, get_kernel
    from equivariantx.ops.common import tece
    from equivariantx.utils.program import Program

    device = "cuda" if torch.cuda.is_available() else "cpu"
    p = Program()
    specs = (
        (4, "source"),
        (2, "edge"),
        (4, "shared"),
        (4, "shared"),
        (1, "edge"),
        (1, "edge"),
        (1, "edge"),
    )
    x, score, scale, *_ = [
        p.input(i, kind, width) for i, (width, kind) in enumerate(specs)
    ]
    v = p.binary("add", p.binary("mul", x, scale), p.input(0, "target", 4))
    base = (tuple(p.nodes), score, v, specs, 2, 4, 1e-6)
    meta = repr((repr((0, 0, 4, 2)), base))
    inputs = [
        torch.randn(*shape, device=device, dtype=torch.float64).requires_grad_()
        for shape in (
            (5, 1, 4),
            (edges, 2),
            (2, 2),
            (4,),
            (edges, 1),
            (edges, 1),
            (edges, 1),
        )
    ]
    inputs[4] = inputs[4].sigmoid()
    src = torch.arange(edges, device=device) % 5
    dst = (src + 1) % 4
    results = []
    for ordinary in (True, False):
        with monkeypatch.context() as switch:
            switch.setattr(config, "TORCH_REFERENCE", ordinary)
            results.append(
                get_kernel("tece_interaction", "torch")(meta, src, dst, inputs)
            )
    if device == "cuda":
        results.append(tece.interaction(meta, src, dst, inputs))
    selected = (0, 1, 2, 4)
    for result in results[1:]:
        for a, b in zip(result, results[0]):
            torch.testing.assert_close(a, b, atol=1e-11, rtol=1e-11)
        if edges:
            actual, expected = [
                v[0].sin().sum() + v[1].square().sum() for v in (result, results[0])
            ]
            for _ in range(3):
                actual, expected = [
                    torch.cat(
                        [
                            g.flatten()
                            for g in torch.autograd.grad(
                                v.sin().sum(),
                                [inputs[i] for i in selected],
                                create_graph=True,
                                retain_graph=True,
                            )
                        ]
                    )
                    for v in (actual, expected)
                ]
                torch.testing.assert_close(actual, expected, rtol=1e-9, atol=1e-10)
    expanded = [
        *inputs,
        *results[0],
        torch.randn_like(results[0][0]),
        torch.randn_like(results[0][1]),
    ]
    adjoint = first_adjoint(base, selected)
    contractions = []
    for ordinary in (True, False):
        with monkeypatch.context() as switch:
            switch.setattr(config, "TORCH_REFERENCE", ordinary)
            contractions.append(
                get_kernel("tece_contraction", "torch")(adjoint, src, dst, expanded)
            )
    if device == "cuda":
        contractions.append(tece.contraction(adjoint, src, dst, expanded))
    for result in contractions[1:]:
        torch.testing.assert_close(result, contractions[0], rtol=1e-10, atol=1e-10)

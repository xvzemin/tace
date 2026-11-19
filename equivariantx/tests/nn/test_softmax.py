"""Segmented softmax values and derivatives."""

import pytest
import torch

from equivariantx.nn import GraphSoftmax, graph_softmax


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_graph_softmax_precision(backend, dtype):

    with pytest.raises(TypeError, match="float32 or float64"):
        graph_softmax(
            torch.ones(3, 1, dtype=dtype), torch.tensor([0, 0, 1]), 2, backend=backend
        )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            graph_softmax(torch.ones(3, 1), torch.tensor([0, 0, 1]), 2, backend=backend)


def test_graph_softmax_backend():

    inputs = torch.randn(3, 1), torch.tensor([0, 0, 1], dtype=torch.int32), 2
    torch.testing.assert_close(
        graph_softmax(*inputs), graph_softmax(*inputs, fused=False)
    )
    torch.testing.assert_close(
        graph_softmax(*inputs, backend="cuda"), graph_softmax(*inputs, fused=True)
    )
    with pytest.raises(ValueError, match="backend"):
        graph_softmax(*inputs, backend="unknown")


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_graph_softmax_unit_weights(backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scores = torch.randn(13, 4, device=device, dtype=torch.float64, requires_grad=True)
    index = torch.arange(13, device=device) % 3
    actual = graph_softmax(scores, index, 5, eps=1e-3, backend=backend)
    expected = graph_softmax(
        scores, index, 5, scores.new_ones(13, 1), eps=1e-3, backend="torch"
    )
    for _ in range(3):
        torch.testing.assert_close(actual, expected)
        seed = torch.randn_like(scores)
        actual, expected = (
            torch.autograd.grad(y, scores, seed, create_graph=True)[0]
            for y in (actual, expected)
        )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("heads,weight_heads", [(1, 1), (3, 1), (3, 3), (8, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graph_softmax_derivatives(heads, weight_heads, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.cuda.attention import normalized_exp_forward

    target = torch.randint(4, (146,), device="cuda")[::2]
    scores = torch.randn(73, heads, device="cuda", dtype=dtype, requires_grad=True)
    weight = torch.rand(73, weight_heads, device="cuda", dtype=dtype)
    weight[::7] = 0
    weight.requires_grad_()
    actual = graph_softmax(scores, target, 6, weight, eps=1e-8, backend="cuda")
    expected = graph_softmax(scores, target, 6, weight, eps=1e-8, fused=False)
    tolerance = 3e-5 if dtype == torch.float32 else 2e-10
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
        seed = torch.randn_like(actual)
        a = torch.autograd.grad(
            (actual.sin() * seed).sum(), (scores, weight), create_graph=True
        )
        b = torch.autograd.grad(
            (expected.sin() * seed).sum(), (scores, weight), create_graph=True
        )
        actual, expected = (
            torch.cat([x.flatten() for x in values]) for values in (a, b)
        )
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
    denominator = scores.new_empty(6, heads)
    maximum = torch.empty_like(denominator)
    with torch.no_grad():
        normalized_exp_forward(
            scores[:0], weight[:0], target[:0], 6, 1e-8, denominator, maximum
        )
    torch.testing.assert_close(denominator, torch.full_like(denominator, 1e-8))
    torch.testing.assert_close(maximum, torch.zeros_like(maximum))


@pytest.mark.parametrize("heads,weight_heads", [(3, 1), (8, 8), (32, 1)])
@pytest.mark.parametrize("output", [0, 1, 2])
def test_graph_softmax_adjoint(heads, weight_heads, output, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.common.attention import evaluate_adjoint

    target = torch.randint(4, (73,), device="cuda")
    inputs = [
        (0.1 * torch.randn(73, width, device="cuda")).requires_grad_()
        for width in (heads, heads, weight_heads)
    ]
    values, grad, weight = inputs
    total = values.new_zeros(6, heads).index_add(0, target, values * grad)
    expected = (
        values * (grad - weight * total[target]),
        (-values * total[target]).sum_to_size(weight.shape),
        total,
    )[output]
    actual = evaluate_adjoint(*inputs, target, 6)[output]
    for _ in range(3):
        torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)
        seed = torch.randn_like(actual)
        results = [
            torch.autograd.grad(
                (x.sin() * seed).sum(), inputs, create_graph=True, allow_unused=True
            )
            for x in (actual, expected)
        ]
        actual, expected = (
            torch.cat(
                [
                    (torch.zeros_like(x) if value is None else value).flatten()
                    for x, value in zip(inputs, result)
                ]
            )
            for result in results
        )
    torch.testing.assert_close(actual, expected, atol=2e-10, rtol=2e-10)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_graph_softmax_compile(backend, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")

    function = torch.compile(
        graph_softmax, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for edges in (7, 13, 0):
        scores = torch.randn(edges, 3, device="cuda", requires_grad=True)
        weights = torch.rand(edges, 1, device="cuda", requires_grad=True)
        target = torch.randint(4, (edges,), device="cuda")
        actual = function(scores, target, 6, weights, backend=backend)
        expected = graph_softmax(scores, target, 6, weights, fused=False)
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (scores, weights)),
            torch.autograd.grad(expected.square().sum(), (scores, weights)),
        ):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize(
    ("shape", "dim", "use_ptr"),
    [
        ((5, 3), 0, False),
        ((3, 5), 1, False),
        ((3, 5), -1, False),
        ((5, 3), 0, True),
        ((5,), 0, True),
        ((2, 5, 3), -2, True),
        ((2, 5, 3), 1, False),
        ((0, 3), 0, False),
        ((0, 3), 0, True),
    ],
)
def test_graph_softmax_groups(shape, dim, use_ptr, device):

    module = GraphSoftmax()
    src = torch.randn(shape, dtype=torch.float64, device=device, requires_grad=True)
    scale = torch.rand_like(src, requires_grad=True)
    counts = [2, 0, 3, 0] if src.size(dim) else [0, 0, 0, 0]
    count = torch.tensor(counts, device=device)
    index = torch.arange(len(counts), device=device).repeat_interleave(count)
    ptr = torch.cat((count.new_zeros(1), count.cumsum(0)))
    actual = module(
        src,
        dim=dim,
        exp_rescale=scale,
        **({"ptr": ptr} if use_ptr else {"index": index, "num_nodes": len(counts)}),
    )
    expected = []
    for value, weight in zip(src.split(counts, dim), scale.split(counts, dim)):
        if value.size(dim):
            value = (value - value.detach().amax(dim, keepdim=True)).exp()
        value = value * weight
        expected.append(value / (value.sum(dim, keepdim=True) + module.eps))
    expected = torch.cat(expected, dim=dim)
    torch.testing.assert_close(actual, expected)
    for actual_grad, expected_grad in zip(
        torch.autograd.grad(actual.square().sum(), (src, scale)),
        torch.autograd.grad(expected.square().sum(), (src, scale)),
    ):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 13])
def test_graph_attention_derivatives(device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    target = torch.arange(edges, device=device) % 4
    scores = torch.randn(
        edges, 3, dtype=torch.float64, device=device, requires_grad=True
    )
    weight = torch.rand(edges, 1, dtype=torch.float64, device=device)
    weight[::3] = 0
    weight.requires_grad_()
    actual = graph_softmax(scores, target, 5, weight, eps=1e-3)
    expected = GraphSoftmax(eps=1e-3, backend="torch")(
        scores, target, num_nodes=5, exp_rescale=weight
    )
    torch.testing.assert_close(actual, expected)
    losses = [x.square().sum() for x in (actual, expected)]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, (scores, weight), create_graph=True)
            for loss in losses
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-8)
        losses = [sum(g.square().sum() for g in values) for values in grads]

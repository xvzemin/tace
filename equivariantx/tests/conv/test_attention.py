"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx.tests.conv.testing import assert_native_autograd


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.complex64])
def test_graph_softmax_precision(backend, dtype):
    from equivariantx.conv.attention import graph_softmax

    with pytest.raises(TypeError, match="float32 or float64"):
        graph_softmax(
            torch.ones(3, 1, dtype=dtype), torch.tensor([0, 0, 1]), 2, backend=backend
        )
    with torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(TypeError, match="autocast"):
            graph_softmax(torch.ones(3, 1), torch.tensor([0, 0, 1]), 2, backend=backend)


def test_graph_softmax_backend():
    from equivariantx.conv.attention import graph_softmax

    inputs = torch.randn(3, 1), torch.tensor([0, 0, 1], dtype=torch.int32), 2
    torch.testing.assert_close(
        graph_softmax(*inputs), graph_softmax(*inputs, fused=False)
    )
    torch.testing.assert_close(
        graph_softmax(*inputs, backend="cuda"), graph_softmax(*inputs, fused=True)
    )
    with pytest.raises(ValueError, match="backend"):
        graph_softmax(*inputs, backend="unknown")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_online_attention_merge(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from equivariantx.ops.kernels.cuda.attention import merge_attention

    torch.manual_seed(6)
    scores = torch.randn(3, 5, 7, 2, device="cuda", dtype=dtype) * 20 + 1000
    cutoff = torch.rand(3, 5, 7, 1, device="cuda", dtype=dtype)
    value = torch.randn(3, 5, 7, 4, 2, 3, device="cuda", dtype=dtype)
    scores[0] = -torch.inf
    scores[1, 0] = -torch.inf
    cutoff[2, 1:] = 0
    maximum = scores.amax(2)
    shift = torch.where(torch.isfinite(maximum), maximum, 0)
    exponential = (scores - shift[:, :, None]).exp()
    denominator = (exponential * cutoff).sum(2)
    partial = (
        exponential[:, :, :, None, :, None]
        * cutoff[:, :, :, None, :, None].square()
        * value
    ).sum(2)
    eps = 0.03
    outputs = [
        value.new_empty(3, 4, 6),
        value.new_empty(3, 2),
        value.new_empty(3, 2),
    ]
    merge_attention(partial.reshape(3, 5, 4, 6), denominator, maximum, 6, eps, outputs)
    global_maximum = scores.amax((1, 2))
    global_maximum = torch.where(torch.isfinite(global_maximum), global_maximum, 0)
    exponential = (scores - global_maximum[:, None, None]).exp()
    total = (exponential * cutoff).sum((1, 2)) + eps
    expected = (
        exponential[:, :, :, None, :, None]
        * cutoff[:, :, :, None, :, None].square()
        * value
    ).sum((1, 2))
    expected = (expected / total[:, None, :, None]).reshape(3, 4, 6)
    for actual, reference in zip(outputs, (expected, total, global_maximum)):
        torch.testing.assert_close(actual, reference)


@pytest.mark.parametrize("tile_size", [1, 4, 16])
@pytest.mark.parametrize("edges", [0, 7])
def test_streaming_attention_torch(tile_size, edges, double_precision):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.conv.attention import StreamingGraphAttention, graph_softmax

    def shapes(inputs):
        return tuple(
            inputs[0].new_empty(shape) for shape in ((4, 3, 6), (4, 2), (4, 2))
        )

    module = StreamingGraphAttention(
        lambda inputs: (inputs[1], inputs[2]),
        shapes,
        {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        3,
        4,
        5,
        6,
        tile_size,
        1e-16,
        backend="torch",
    )
    scores = torch.randn(edges, 2, device="cuda", requires_grad=True)
    values = torch.randn(edges, 3, 6, device="cuda", requires_grad=True)
    normalizer = torch.rand(edges, 1, device="cuda", requires_grad=True)
    weight = torch.rand(edges, 1, device="cuda", requires_grad=True)
    target = torch.arange(edges, device="cuda") % 3
    valid = torch.arange(edges, device="cuda") % 4 != 0
    actual, _, _ = module(
        scores.new_empty(4, 1), scores, values, target, normalizer, weight, valid
    )
    scale = (
        graph_softmax(
            torch.where(valid[:, None], scores, -torch.inf),
            target,
            4,
            normalizer,
            fused=False,
        )
        * weight
    )
    message = (values.reshape(edges, 3, 2, 3) * scale[:, None, :, None]).flatten(2)
    expected = torch.zeros_like(actual).index_add(0, target, message)
    inputs = scores, values, normalizer, weight
    for order in range(4 if edges else 2):
        assert_native_autograd(actual)
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
        if order < (3 if edges else 1):
            actual, expected = [
                torch.cat(
                    [
                        g.flatten()
                        for g in torch.autograd.grad(
                            value.sin().sum(), inputs, create_graph=True
                        )
                    ]
                )
                / 20
                for value in (actual, expected)
            ]


@pytest.mark.parametrize("heads,weight_heads", [(1, 1), (3, 1), (3, 3), (8, 8)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_graph_softmax_derivatives(heads, weight_heads, dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.ops.kernels.cuda.attention import normalized_exp_forward
    from equivariantx.conv.attention import graph_softmax

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
    from equivariantx.ops.attention import evaluate_adjoint

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
    from equivariantx.conv.attention import graph_softmax

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

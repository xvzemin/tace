"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx.tests.utils.convolution import assert_native_autograd


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_online_attention_merge(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from equivariantx.ops.common.cuda.attention import merge_attention

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
def test_streaming_attention_torch(tile_size, edges, double_precision, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from equivariantx.nn import StreamingGraphAttention, graph_softmax
    from equivariantx.ops import config

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
    args = (scores.new_empty(4, 1), scores, values, target, normalizer, weight, valid)
    actual, denominator, maximum = module(*args)
    with monkeypatch.context() as flags:
        flags.setattr(config, "TORCH_REFERENCE", True)
        reference = module(*args)
    torch.testing.assert_close((actual, denominator, maximum), reference)
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


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("nodes,tile", [(0, 4), (5, 4), (5, 16)])
def test_streaming_graph_attention_weights(device, nodes, tile):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    from equivariantx.nn import GraphSoftmax, StreamingGraphAttention

    edges, heads = (9 if nodes else 0), 2
    target = torch.arange(edges, device=device) % 3
    inputs = [
        torch.zeros(nodes, 1, device=device, dtype=torch.float64),
        torch.randn(edges, heads, device=device, dtype=torch.float64).requires_grad_(),
        torch.randn(edges, 3, 4, device=device, dtype=torch.float64).requires_grad_(),
        torch.rand(edges, 1, device=device, dtype=torch.float64).requires_grad_(),
        torch.rand(edges, heads, device=device, dtype=torch.float64).requires_grad_(),
        target,
        torch.ones(edges, device=device, dtype=torch.bool),
    ]
    with torch.no_grad():
        inputs[3][::3] = 0

    def fake(v):
        return (
            v[0].new_empty(nodes, 3, 4),
            v[0].new_empty(nodes, heads),
            v[0].new_empty(nodes, heads),
        )

    attention = StreamingGraphAttention(
        lambda v: (v[1], v[2]),
        fake,
        {i: 0 for i in range(1, 7)},
        target_slot=5,
        normalizer_slot=3,
        value_weight_slot=4,
        valid_slot=6,
        tile_size=tile,
        eps=1e-3,
    )
    actual, denominator, maximum = attention(*inputs)
    weight = (
        GraphSoftmax(eps=1e-3, backend="torch")(
            inputs[1], target, num_nodes=nodes, exp_rescale=inputs[3]
        )
        * inputs[4]
    )
    value = (inputs[2].view(edges, 3, heads, 2) * weight[:, None, :, None]).reshape(
        edges, 3, 4
    )
    expected = value.new_zeros(nodes, 3, 4).index_add(0, target, value)
    torch.testing.assert_close(actual, expected)
    scores = inputs[1]
    peak = scores.new_full((nodes, heads), -torch.inf).scatter_reduce(
        0, target[:, None].expand_as(scores), scores.detach(), reduce="amax"
    )
    peak = torch.where(torch.isneginf(peak), 0, peak)
    total = (
        torch.zeros_like(peak).index_add(
            0, target, (scores - peak[target]).exp() * inputs[3]
        )
        + 1e-3
    )
    torch.testing.assert_close((denominator, maximum), (total, peak))
    losses = [
        actual.square().sum() + denominator.sin().sum(),
        expected.square().sum() + total.sin().sum(),
    ]
    for _ in range(3):
        grads = [
            torch.autograd.grad(loss, inputs[1:5], create_graph=True) for loss in losses
        ]
        for a, b in zip(*grads):
            torch.testing.assert_close(a, b, atol=1e-7, rtol=1e-8)
        losses = [sum(g.square().sum() for g in values) for values in grads]

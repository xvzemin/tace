"""Segmented normalization against independent node-wise sums."""

import pytest
import torch

from equivariantx.conv.attention import graph_softmax


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("edges", [0, 7])
def test_normalization_derivatives(device, edges):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    scores = torch.randn(
        edges, 3, device=device, dtype=torch.float64, requires_grad=True
    )
    weight = torch.rand(edges, 1, device=device, dtype=torch.float64)
    weight[:1] = 0
    weight.requires_grad_()
    target = torch.arange(edges, device=device) % 3
    eps = 1e-5

    def reference(scores, weight):
        result = scores * 0 + weight * 0
        for node in range(5):
            index = torch.where(target == node)[0]
            if index.numel():
                value = scores[index]
                value = (value - value.detach().amax(0, keepdim=True)).exp()
                value = value * weight[index]
                result = result.index_copy(
                    0, index, value / (value.sum(0, keepdim=True) + eps)
                )
        return result

    def optimized(scores, weight):
        return graph_softmax(scores, target, 5, weight, eps)

    actual, expected = optimized(scores, weight), reference(scores, weight)
    torch.testing.assert_close(actual, expected)
    for _ in range(4):
        seed = torch.randn_like(actual) * 0.1
        gradients = [
            torch.autograd.grad(
                (value.sin() * seed).sum(), (scores, weight), create_graph=True
            )
            for value in (actual, expected)
        ]
        torch.testing.assert_close(*gradients, atol=1e-9, rtol=1e-9)
        actual, expected = [
            torch.cat([value.flatten() for value in values]) for values in gradients
        ]
    inputs = scores, weight
    tangents = tuple(torch.randn_like(value) for value in inputs)
    torch.testing.assert_close(
        torch.func.jvp(optimized, inputs, tangents),
        torch.func.jvp(reference, inputs, tangents),
    )
    batched = tuple(torch.stack((value, value * 0.8)) for value in inputs)
    expected = torch.stack([reference(*values) for values in zip(*batched)])
    torch.testing.assert_close(torch.vmap(optimized)(*batched), expected)

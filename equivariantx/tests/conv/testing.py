"""Tests for the corresponding EQX source module."""

import pytest
import torch
from e3nn import o3

from equivariantx import conv as eqx_conv


def assert_native_autograd(value):
    """Reject custom autograd functions in a PyTorch reference graph."""
    pending, visited = [value.grad_fn], set()
    while pending:
        node = pending.pop()
        if node is None or node in visited:
            continue
        visited.add(node)
        assert not isinstance(node, torch.autograd.function.BackwardCFunction), type(
            node
        ).__name__
        pending.extend(parent for parent, _ in node.next_functions)


def layout(features, irreps, inverse=False):
    values = []
    for (mul, ir), section in zip(irreps, irreps.slices()):
        shape = (ir.dim, mul) if inverse else (mul, ir.dim)
        values.append(
            features[..., section]
            .reshape(*features.shape[:-1], *shape)
            .transpose(-1, -2)
            .flatten(-2)
        )
    return torch.cat(values, dim=-1)


def tensor_product(channels=3, merge=False, unweighted=False):
    return o3.TensorProduct(
        f"{channels}x0e+{channels}x1o+2x2e",
        "0e+2x1o+1x2e",
        f"{channels}x1o" + ("" if merge else f"+{channels}x1o") + "+2x2e",
        [
            (0, 1, 0, "uvu", True),
            (1, 0, 0 if merge else 1, "uvu", not unweighted),
            (2, 0, 1 if merge else 2, "uvu", True),
        ],
        internal_weights=False,
        shared_weights=False,
    )


def reference(tp, x, attrs, radial, projection, edges):
    weights = radial @ projection if projection.numel() else radial
    message = tp(
        layout(x, tp.irreps_in1, True)[edges[0]],
        layout(attrs, tp.irreps_in2, True),
        weights,
    )
    result = x.new_zeros(x.size(0), tp.irreps_out.dim).index_add(0, edges[1], message)
    return layout(result, tp.irreps_out)


def check_product_and_convolution_compile(cls, device, double_precision):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    tp = cls("0e+1o", "0e+1o", "0e+1o+2e", shared_weights=False)
    convolution = getattr(eqx_conv, cls.__name__ + "Conv")(tp).to(device)
    compiled = torch.compile(
        convolution, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for edges in (3, 6, 0):
        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(edges, 4, device=device, requires_grad=True)
        weight = torch.randn(edges, tp.weight_numel, device=device, requires_grad=True)
        index = torch.randint(4, (2, edges), device=device)
        actual, expected = (
            compiled(x, y, index, weight),
            convolution(x, y, index, weight),
        )
        torch.testing.assert_close(actual, expected)
        for a, b in zip(
            torch.autograd.grad(actual.square().sum(), (x, y, weight)),
            torch.autograd.grad(expected.square().sum(), (x, y, weight)),
        ):
            torch.testing.assert_close(a, b)

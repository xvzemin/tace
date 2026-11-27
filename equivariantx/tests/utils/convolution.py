"""Tests for the corresponding EQX source module."""

import torch
from e3nn import o3


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

"""Fixed sparse bilinear maps with recursively transposed exact derivatives."""

import torch

from ....utils.autograd import forward_ad_active


def reference(x, y, tables, role=2):
    """Gather all coefficient terms using ordinary Torch autograd."""
    left, right, coefficient = tables[3 * role : 3 * role + 3]
    if not x.shape[-1] or not y.shape[-1]:
        shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
        zero = x[..., :0].sum(-1, keepdim=True) + y[..., :0].sum(-1, keepdim=True)
        return x.new_zeros((*shape, left.shape[0])) + zero
    return (x[..., left] * y[..., right] * coefficient).sum(-1)


def primal(x, y, tables, role=2):
    """Expose short sparse sums without retaining expanded term arrays."""
    left, right, coefficient = tables[3 * role : 3 * role + 3]
    if left.shape[-1] > 8 or not x.shape[-1] or not y.shape[-1]:
        return reference(x, y, tables, role)
    terms = [
        x.index_select(-1, left[:, k])
        * y.index_select(-1, right[:, k])
        * coefficient[:, k]
        for k in range(left.shape[-1])
    ]
    return sum(terms[1:], terms[0])


def adjoints(operation, x, y, gradient, tables, role, needed=(True, True)):
    """Transpose a fixed three-axis coefficient tensor without atomics."""
    slots = [x, y]
    slots.insert(role, gradient)
    values = []
    for index, operand in enumerate((x, y)):
        target = index + (index >= role)
        others = [value for i, value in enumerate(slots) if i != target]
        values.append(
            operation(*others, tables, target).sum_to_size(operand.shape)
            if needed[index]
            else None
        )
    return tuple(values)


@torch.compiler.allow_in_graph
class SparseBilinear(torch.autograd.Function):
    """Save the two operands, not their expanded products."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x, y, role, *tables):
        return primal(x, y, tables, role)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, ctx.role, *tables = inputs
        ctx.save_for_backward(x, y, *tables)
        ctx.save_for_forward(x, y, *tables)

    @staticmethod
    def backward(ctx, gradient):
        x, y, *tables = ctx.saved_tensors
        dx, dy = adjoints(
            evaluate, x, y, gradient, tables, ctx.role, ctx.needs_input_grad[:2]
        )
        return dx, dy, None, *((None,) * len(tables))

    @staticmethod
    def jvp(ctx, dx, dy, unused_role, *unused_tables):
        x, y, *tables = ctx.saved_tensors
        dx = torch.zeros_like(x) if dx is None else dx
        dy = torch.zeros_like(y) if dy is None else dy
        return primal(dx, y, tables, ctx.role) + primal(x, dy, tables, ctx.role)


@torch.compiler.allow_in_graph
def evaluate(x, y, tables, role=2):
    """Apply a bilinear map; transformed or trainable coefficients use Torch AD."""
    if forward_ad_active() or any(value.requires_grad for value in tables):
        return primal(x, y, tables, role)
    return SparseBilinear.apply(x, y, role, *tables)

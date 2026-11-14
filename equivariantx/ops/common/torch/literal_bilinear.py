"""Short fixed bilinear expressions with recursive exact transposes."""

import torch

from ....utils.autograd import torch_gradient_mask


def primal(x, y, dimensions, entries, role=2, dim=-1):
    """Evaluate short literal sums without gather/scatter intermediates."""
    inputs = [x.unbind(dim), y.unbind(dim)]
    parts = [[] for _ in range(dimensions[role])]
    products = {}
    for *indices, coefficient in entries:
        a, b = [indices[k] for k in range(3) if k != role]
        if (a, b) not in products:
            products[a, b] = inputs[0][a] * inputs[1][b]
        parts[indices[role]].append(products[a, b] * coefficient)
    if parts and all(parts):
        return torch.stack([sum(terms[1:], terms[0]) for terms in parts], dim=dim)
    shapes = [list(value.shape) for value in (x, y)]
    for shape in shapes:
        shape.pop(dim)
    shape = torch.broadcast_shapes(*shapes)
    zero = x.narrow(dim, 0, 0).sum() + y.narrow(dim, 0, 0).sum()
    if parts:
        return torch.stack(
            [
                sum(terms[1:], terms[0]) if terms else x.new_zeros(shape) + zero
                for terms in parts
            ],
            dim=dim,
        )
    shape = list(shape)
    shape.insert(dim % (len(shape) + 1), 0)
    return x.new_zeros(shape) + zero


@torch.compiler.allow_in_graph
class LiteralBilinear(torch.autograd.Function):
    """Use the same short real formulas for every transposed derivative."""

    generate_vmap_rule = True

    @staticmethod
    def forward(x, y, dimensions, entries, role, dim):
        return primal(x, y, dimensions, entries, role, dim)

    @staticmethod
    def setup_context(ctx, inputs, output):
        x, y, ctx.dimensions, ctx.entries, ctx.role, ctx.dim = inputs
        ctx.save_for_backward(x, y)
        ctx.save_for_forward(x, y)

    @staticmethod
    def backward(ctx, gradient):
        x, y = ctx.saved_tensors
        values = [x, y]
        values.insert(ctx.role, gradient)
        gradients = []
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[:2])
        for i, operand in enumerate((x, y)):
            target = i + (i >= ctx.role)
            other = [value for j, value in enumerate(values) if j != target]
            gradients.append(
                evaluate(
                    *other, ctx.dimensions, ctx.entries, target, ctx.dim
                ).sum_to_size(operand.shape)
                if needed[i]
                else None
            )
        return *gradients, None, None, None, None

    @staticmethod
    def jvp(ctx, dx, dy, unused_dimensions, unused_entries, unused_role, unused_dim):
        x, y = ctx.saved_tensors
        dx = torch.zeros_like(x) if dx is None else dx
        dy = torch.zeros_like(y) if dy is None else dy
        return primal(dx, y, ctx.dimensions, ctx.entries, ctx.role, ctx.dim) + primal(
            x, dy, ctx.dimensions, ctx.entries, ctx.role, ctx.dim
        )


@torch.compiler.allow_in_graph
def evaluate(x, y, dimensions, entries, role=2, dim=-1):
    if torch._C._are_functorch_transforms_active():
        return primal(x, y, dimensions, entries, role, dim)
    return LiteralBilinear.apply(x, y, dimensions, entries, role, dim)

"""Shared gate broadcasts with reduction adjoints."""

import torch

from ....utils.autograd import nested_forward_ad
from ....utils.layout import permute
from ... import config
from ...precision import validate_precision


@torch.compiler.allow_in_graph
def expand_gates(gates, index, *layout):
    """Broadcast gates while preserving nested forward derivatives."""
    if config.TORCH_REFERENCE or nested_forward_ad():
        return gates.index_select(-1, index)
    return GateExpand.apply(gates, index, *layout)


@torch.compiler.allow_in_graph
class GateExpand(torch.autograd.Function):
    """Broadcast gates and sum component gradients in the backward pass."""

    generate_vmap_rule = True

    @staticmethod
    def forward(
        gates, index, output_index, output_inverse, gate_index, gate_inverse, layout
    ):
        validate_precision(
            (
                gates,
                index,
                output_index,
                output_inverse,
                gate_index,
                gate_inverse,
                layout,
            )
        )
        return gates.index_select(-1, index)

    @staticmethod
    def setup_context(ctx, inputs, output):
        _, index, output_index, output_inverse, gate_index, gate_inverse, layout = (
            inputs
        )
        ctx.save_for_backward(output_index, output_inverse, gate_index, gate_inverse)
        ctx.save_for_forward(index)
        ctx.layout = layout

    @staticmethod
    def backward(ctx, gradient):
        output_index, output_inverse, gate_index, gate_inverse = ctx.saved_tensors
        sizes, dimensions, output_identity, gate_identity = ctx.layout
        if not output_identity:
            gradient = permute(gradient, output_inverse, output_index)
        widths = tuple((dim * mul for dim, mul in zip(dimensions, sizes)))
        values = [
            value.view(*gradient.shape[:-1], dim, mul).sum(-2)
            for value, dim, mul in zip(
                gradient.split(widths, dim=-1), dimensions, sizes
            )
        ]
        gradient = values[0] if len(values) == 1 else torch.cat(values, dim=-1)
        if not gate_identity:
            gradient = permute(gradient, gate_inverse, gate_index)
        return (gradient, None, None, None, None, None, None)

    @staticmethod
    def jvp(ctx, tangent, *unused):
        (index,) = ctx.saved_tensors
        return tangent.index_select(-1, index)

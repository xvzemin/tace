"""Torch angular products, grouped channel GEMMs, and reference paths."""

import torch

from equivariantx.o2.clebsch_gordan import clebsch_gordan_products, coupling_entries

from ... import config
from ...common.torch import channel_product


def evaluate(module, input1, input2, weight):
    """Select the ordinary reference or optimized channel schedule."""
    if config.TORCH_REFERENCE:
        return module.forward_paths(input1, input2, weight, reference=True)
    if channel_product.use_slices(module, input1, input2, weight):
        return channel_product.tensor_product(module, input1, input2, weight)
    if module.contraction.indexed:
        return module.contraction(input1, input2, weight, module.path_scales.to(input1))
    return module.forward_paths(input1, input2, weight)


def forward_paths(module, input1, input2, weight, *, reference=False, native=False):
    """Reuse angular products, with ordinary or native sparse contractions."""
    shape = torch.broadcast_shapes(
        input1.shape[:-1], input2.shape[:-1], weight.shape[:-1]
    )
    inputs = [
        [
            block.reshape(*value.shape[:-1], ir.dim, mul)
            for block, (ir, mul) in zip(
                value.split([entry.dim for entry in irreps], -1), irreps
            )
        ]
        for value, irreps in ((input1, module.irreps_in1), (input2, module.irreps_in2))
    ]
    weights = weight.split(module._weight_sizes, -1)
    outputs = [None] * len(module.irreps_out)
    for i, j, elementwise, irreps, paths in (
        module.reference_couplings if reference else module.couplings
    ):
        if native:
            from equivariantx.ops.common.channel_product import angular_product

            x, y = (inputs[0][i].transpose(-1, -2), inputs[1][j].transpose(-1, -2))
            if not elementwise:
                x, y = (x.unsqueeze(-2), y.unsqueeze(-3))
            values = tuple(
                (
                    angular_product(
                        x,
                        y,
                        coupling_entries(
                            module.irreps_in1[i].ir, module.irreps_in2[j].ir, ir
                        ),
                        ir.dim,
                    ).movedim(-1, -2 if elementwise else -3)
                    for ir in irreps
                )
            )
        else:
            values = clebsch_gordan_products(
                inputs[0][i],
                module.irreps_in1[i].ir,
                inputs[1][j],
                module.irreps_in2[j].ir,
                irreps,
                elementwise=elementwise,
            )
        for index, angular in paths:
            ins = module.instructions[index]
            value = values[angular]
            if ins.has_weight:
                w = weights[index] * ins.path_weight
                if ins.connection_mode == "uvw":
                    value = torch.einsum(
                        "...duv,...uvw->...dw",
                        value,
                        w.reshape(*weight.shape[:-1], *ins.path_shape),
                    )
                else:
                    value = value * w.unsqueeze(-2)
            else:
                value = value * ins.path_weight
            previous = outputs[ins.i_out]
            outputs[ins.i_out] = value if previous is None else previous + value
    zero = (
        input1[..., :0].sum() + input2[..., :0].sum() + weight[..., :0].sum()
        if not module.contraction.complete or not outputs
        else None
    )
    if not outputs:
        return input1.new_zeros((*shape, 0)) + zero
    result = torch.cat(
        [
            (
                input1.new_zeros((*shape, ir.dim, mul))
                if value is None
                else value.expand(*shape, ir.dim, mul)
            ).flatten(-2)
            for value, (ir, mul) in zip(outputs, module.irreps_out)
        ],
        -1,
    )
    return result if zero is None else result + zero

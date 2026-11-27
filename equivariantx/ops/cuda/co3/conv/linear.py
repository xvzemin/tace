"""Cartesian layout, geometry and compact coefficient execution."""

import torch

from .....utils.layout import permute


def evaluate(module, features, weight, bias):
    """Evaluate validated Cartesian operands."""
    shape = torch.broadcast_shapes(
        features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
    )
    zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
    grouped = (
        module.backend == "cuda"
        and features.is_cuda
        and (features.ndim == 2)
        and module.input_permutation
    )
    if grouped:
        from equivariantx.ops.common.layout import grouped_permute

        features = grouped_permute(
            features, module.input_index, module.input_inverse, module.group_widths
        )
    elif not module.input_identity:
        features = (
            permute(features, module.input_index, module.input_inverse)
            if module.input_permutation
            else features.index_select(-1, module.input_index)
        )
    outputs = [None] * len(module.cartesian_out)
    for i, (dest, dim, mul_in, mul_out) in enumerate(module.compact_groups):
        section = module.input_sections[i]
        x = (
            features.flatten()[
                features.size(0) * section.start : features.size(0) * section.stop
            ]
            if grouped
            else features[..., section]
        )
        x = x.reshape(*features.shape[:-1], dim, mul_in)
        w = weight[..., module.weight_sections[i]]
        if module.weight_rows[i]:
            rows = (
                module.weight_sections[i].stop - module.weight_sections[i].start
            ) // mul_out
            w = w.reshape(*weight.shape[:-1], rows, mul_out)
        if not module.weight_identity[i]:
            w = w.index_select(
                -2 if module.weight_rows[i] else -1,
                getattr(module, f"weight_index_{i}"),
            )
        w = w.reshape(*weight.shape[:-1], mul_in, mul_out)
        if not module.uniform_scales[i]:
            w = w * getattr(module, f"weight_scale_{i}")
        mixed = x @ w
        value = mixed.transpose(-1, -2) @ getattr(module, f"basis_{i}")
        outputs[dest] = value if outputs[dest] is None else outputs[dest] + value
    for i, ((mul, ir), group) in enumerate(zip(module.cartesian_out, module.groups)):
        if outputs[i] is None:
            dim = ir.spherical_dim if module.output_basis == "spherical" else ir.dim
            outputs[i] = features.new_zeros((*shape, mul, dim)) + zero
        for ins, section in group:
            if ins.i_in < 0:
                outputs[i] = outputs[i] + bias[..., section].unsqueeze(-1)
        outputs[i] = outputs[i].expand(*shape, *outputs[i].shape[-2:])
    return (
        torch.cat([value.flatten(-2) for value in outputs], -1) + zero
        if outputs
        else features[..., :0].expand(*shape, 0) + zero
    )

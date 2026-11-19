"""Channel scheduling for independent Cartesian angular couplings."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode


def mix_channels(weight, features):
    """Contract narrow Cartesian blocks without a batched GEMM workspace."""
    if (
        features.is_cuda
        and features.shape[-1] <= 3
        and not torch.compiler.is_compiling()
        and get_proxy_mode() is None
        and weight.numel() * weight.element_size() <= 8 << 20
    ):
        return torch.stack(
            [(weight * col.unsqueeze(-2)).sum(-1) for col in features.unbind(-1)],
            dim=-1,
        )
    return weight @ features


def grouped(
    module, x, y, weight, shape, *, packed_degrees=(), projector=None, order_for=None
):
    """Preserve all paths and scales, changing only eligible contraction trees."""
    outputs = [None] * len(module.irreps_out)
    inputs = [
        [
            value[..., section].reshape(*value.shape[:-1], mul, ir.dim)
            for (mul, ir), section in zip(irreps, sections)
        ]
        for value, irreps, sections in (
            (x, module.irreps_in1, module.slices_in1),
            (y, module.irreps_in2, module.slices_in2),
        )
    ]
    for group in module.groups:
        couple = (
            module.symmetric_product
            if group.degrees[2] in packed_degrees
            else module.couple
        )
        operands = []
        for blocks, indices in ((inputs[0], group.i_in1), (inputs[1], group.i_in2)):
            operands.append(
                blocks[indices[0]].unsqueeze(-3)
                if len(indices) == 1
                else torch.stack([blocks[i] for i in indices], dim=-3)
            )
        left, right = operands
        mode = group.connection_mode
        w = None
        if group.weight_slices:
            if group.weight_slice is not None:
                w = weight[..., group.weight_slice].reshape(
                    *weight.shape[:-1], len(group.i_out), *group.path_shape
                )
            else:
                w = torch.stack(
                    [
                        weight[..., section].reshape(
                            *weight.shape[:-1], *group.path_shape
                        )
                        for section in group.weight_slices
                    ],
                    dim=-len(group.path_shape) - 1,
                )
        order = order_for(group) if order_for is not None else None
        if mode == "uvu" and right.shape[-2] > 1:
            # Each output u shares its left operand across v.
            right = (
                module.mix_channels(w, right)
                if w is not None
                else right.sum(-2, keepdim=True)
            )
            value = couple(left, right, *group.degrees)
        elif mode == "uvv" and left.shape[-2] > 1:
            left = (
                module.mix_channels(w.transpose(-1, -2), left)
                if w is not None
                else left.sum(-2, keepdim=True)
            )
            value = couple(left, right, *group.degrees)
        elif order == "left":
            value = torch.einsum("...pui,...puvw->...pvwi", left, w)
            value = torch.einsum("...pvwi,...pvj->...pwij", value, right).flatten(-2)
        elif order == "right":
            value = torch.einsum("...pvj,...puvw->...puwj", right, w)
            value = torch.einsum("...pui,...puwj->...pwij", left, value).flatten(-2)
        else:
            if mode not in ("uuu", "uuw"):
                left, right = left.unsqueeze(-2), right.unsqueeze(-3)
            value = couple(left, right, *group.degrees)
            if w is not None:
                if mode == "uvw":
                    value = torch.einsum("...puvd,...puvw->...pwd", value, w)
                elif mode == "uuw":
                    value = torch.einsum("...pud,...puw->...pwd", value, w)
                else:
                    value = value * w.unsqueeze(-1)
            elif mode == "uuw":
                value = value.sum(-2, keepdim=True)
            if mode == "uvu":
                value = value.sum(-2)
            elif mode == "uvv":
                value = value.sum(-3)
            elif mode == "uvuv":
                value = value.flatten(-3, -2)
        value = value * module.path_scales[group.scale_slice, None, None]
        for dest, block in zip(group.i_out, value.unbind(-3)):
            previous = outputs[dest]
            outputs[dest] = block if previous is None else previous + block
    for i, (mul, ir) in enumerate(module.irreps_out):
        size = (ir.l + 2) * (ir.l + 1) // 2 if ir.l in packed_degrees else ir.dim
        block = outputs[i]
        outputs[i] = (
            x.new_zeros((*shape, mul, size))
            if block is None
            else block.expand(*shape, mul, size)
        )
    if projector is not None and projector.groups:
        outputs = projector.forward_blocks(outputs, packed_degrees=packed_degrees)
    flat = [v.flatten(-2) for v in outputs]
    result = (
        (flat[0] if len(flat) == 1 else torch.cat(flat, -1))
        if flat
        else x.new_zeros((*shape, 0))
    )
    if not module.groups:
        result = result + x[..., :0].sum() + y[..., :0].sum()
    if not module.weight_numel:
        result = result + weight[..., :0].sum()
    return result

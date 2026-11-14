"""Torch angular products, grouped channel GEMMs, and reference paths."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config


def evaluate(module, x, y, weight):
    """Choose the independent reference or a compiler-visible contraction."""
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
    if module.backend == "torch" and config.TORCH_REFERENCE:
        zero = x[..., :0].sum() + y[..., :0].sum() + weight[..., :0].sum()
        return (
            module.projection(module.reference(x, y, weight, module.path_scales)) + zero
        )
    if (
        module.backend == "torch"
        and x.is_cuda
        and (module.kernel_metadata is not None)
        and (module.product_degree <= config.CARTESIAN_POLYNOMIAL_DEGREE)
        and (not config.TORCH_RECOMPUTE)
        and (
            torch.compiler.is_compiling()
            or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active()
        )
    ):
        from equivariantx.ops.common.channel_product import tensor_product

        return module.projection(
            tensor_product(
                module.kernel_metadata, x, y, weight, module.output_dim, backend="torch"
            )
        )
    return grouped(module, x, y, weight, shape)


def channel_order(group):
    """Choose a bounded contraction tree using static dimensions only."""
    a, b, c = group.degrees
    if (
        group.connection_mode != "uvw"
        or not group.weight_slices
        or not (1 <= a <= 2 and 1 <= b <= 2 and c == a + b <= 3)
    ):
        return None
    u, v, w = group.path_shape
    if u * v * w < config.CARTESIAN_REORDER_MIN_WORK:
        return None
    da, db, dc = 3**a, 3**b, 3**c
    current_cost, current_size = u * v * w * dc, u * v * dc
    alternatives = (
        (u * v * w * da + v * w * dc, v * w * da, "left"),
        (u * v * w * db + u * w * dc, u * w * db, "right"),
    )
    candidates = [
        (cost, order)
        for cost, size, order in alternatives
        if cost < current_cost and size <= current_size
    ]
    return min(candidates)[1] if candidates else None


def grouped(module, x, y, weight, shape):
    """Preserve all paths and scales, changing only eligible contraction trees."""
    zero = x[..., :0].sum() + y[..., :0].sum() + weight[..., :0].sum()
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
    active = (
        module.backend == "torch"
        and not x.is_cuda
        and not config.TORCH_REFERENCE
        and config.TORCH_CARTESIAN_REORDER
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )
    for group in module.groups:
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
        order = channel_order(group) if active else None
        if order == "left":
            value = torch.einsum("...pui,...puvw->...pvwi", left, w)
            value = torch.einsum("...pvwi,...pvj->...pwij", value, right).flatten(-2)
        elif order == "right":
            value = torch.einsum("...pvj,...puvw->...puwj", right, w)
            value = torch.einsum("...pui,...puwj->...pwij", left, value).flatten(-2)
        else:
            if mode not in ("uuu", "uuw"):
                left, right = left.unsqueeze(-2), right.unsqueeze(-3)
            value = module.couple(left, right, *group.degrees)
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
    outputs = [
        x.new_zeros((*shape, mul, ir.dim))
        if block is None
        else block.expand(*shape, mul, ir.dim)
        for block, (mul, ir) in zip(outputs, module.irreps_out)
    ]
    outputs = module.projection.forward_blocks(outputs)
    return (
        torch.cat([v.flatten(-2) for v in outputs], -1)
        if outputs
        else x.new_zeros((*shape, 0))
    ) + zero

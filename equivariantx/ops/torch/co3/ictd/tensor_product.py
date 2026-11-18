"""Torch angular products, grouped channel GEMMs, and reference paths."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config
from ..channels import grouped


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
    return grouped(
        module,
        x,
        y,
        weight,
        shape,
        packed_degrees=module.packed_degrees,
        projector=module.projection,
        order_for=channel_order,
    )


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

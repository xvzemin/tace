"""Indexed frame rotations and their multilinear adjoints."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ...utils.autograd import forward_ad_active
from ...utils.outputs import EDGE, MATRIX, NODE, indexed_frame_output
from ..precision import validate_module_precision, validate_uniform


def transform(frame, features, wigner, index, num_nodes, inverse):
    """Evaluate an indexed LocalFrame map using compact degree matrices."""
    validate_module_precision(frame, (features, wigner))
    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Frame kernels require CUDA float32 or float64 features.")
    if wigner.device != features.device:
        raise ValueError("Features and Wigner matrices must share a device.")
    wigner = wigner.to(dtype=features.dtype)
    if index.device != features.device or index.dtype != torch.int64 or index.ndim != 1:
        raise ValueError(
            "index must be a one-dimensional int64 tensor on the feature device."
        )
    width = frame.output_dim if inverse else frame.input_dim
    if features.ndim != 2 or features.size(1) != width:
        raise ValueError(f"features must have shape (batch, {width}).")
    if wigner.ndim not in (2, 3) or wigner.size(0) != index.numel():
        raise ValueError("Each edge requires one Wigner matrix.")
    if num_nodes < 0 or (inverse and features.size(0) != index.numel()):
        raise ValueError("Invalid node count or edge feature count.")
    if (
        forward_ad_active()
        or torch.compiler.is_compiling()
        or get_proxy_mode() is not None
    ):
        from ..torch.o2 import local_frame

        if inverse:
            return local_frame.scatter(frame, features, wigner, index, num_nodes)
        return local_frame.gather(frame, features, wigner, index)
    blocks = frame._rotation_matrices(wigner, inverse=inverse)
    matrix = (
        torch.cat(
            [
                (block.transpose(1, 2) if inverse else block).flatten(1)
                for block in blocks
            ],
            dim=1,
        )
        if blocks
        else wigner.flatten(1)[:, :0]
    )
    node = features.new_empty((0, frame.input_dim)) if inverse else features
    edge = features if inverse else features.new_empty((0, frame.output_dim))
    return contract(
        node,
        matrix,
        edge,
        index,
        num_nodes,
        frame.rotation_layout,
        NODE if inverse else EDGE,
    )


@torch.library.custom_op("eqx::indexed_frame", mutates_args=(), device_types="cuda")
def contract_cuda(
    nodes: torch.Tensor,
    matrix: torch.Tensor,
    edges: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
    layout: str,
    role: int,
) -> torch.Tensor:
    """Contract the two factors other than ``role`` in the trilinear pairing.

    ``NODE`` sums transposed matrix products, ``MATRIX`` sums channel products,
    and ``EDGE`` gathers and rotates node features. The selected input is unused;
    each adjoint applies the same contraction with a different output role.
    """
    from .. import get_kernel

    return get_kernel("indexed_frame", "cuda")(
        nodes, matrix, edges, index, num_nodes, layout, role
    )


contract_cuda.register_fake(indexed_frame_output)


def setup_context(ctx, inputs, output):
    nodes, matrix, edges, index, ctx.num_nodes, ctx.layout, ctx.role = inputs
    ctx.save_for_backward(nodes, matrix, edges, index)


def backward(ctx, grad):
    nodes, matrix, edges, index = ctx.saved_tensors
    values = [nodes, matrix, edges]
    values[ctx.role] = grad
    gradients = [
        contract(*values, index, ctx.num_nodes, ctx.layout, role)
        if role != ctx.role and ctx.needs_input_grad[role]
        else None
        for role in (NODE, MATRIX, EDGE)
    ]
    return *gradients, None, None, None, None


contract_cuda.register_autograd(backward, setup_context=setup_context)


@torch.compiler.allow_in_graph
def contract(nodes, matrix, edges, index, num_nodes, layout, role):
    """Expose frame algebra during joint capture and forward differentiation."""
    validate_uniform((nodes, matrix, edges))
    if (
        forward_ad_active()
        or torch.compiler.is_compiling()
        or get_proxy_mode() is not None
    ):
        from .torch.frame import indexed_frame

        return indexed_frame(nodes, matrix, edges, index, num_nodes, layout, role)
    return torch.ops.eqx.indexed_frame(
        nodes, matrix, edges, index, num_nodes, layout, role
    )

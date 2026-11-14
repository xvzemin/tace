"""Normalized symmetric storage with bounded contraction intermediates."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ....utils.autograd import nested_forward_ad
from ... import config
from ...common.torch import sparse_axis, tensor_axis


def pack_factorized(module, tensor):
    """Map successive Cartesian index groups in normalized storage."""
    if module.rank < 2:
        return tensor
    for k in range(2, module.rank + 1):
        tensor = tensor.unflatten(-1, (3 * k * (k + 1) // 2, 3 ** (module.rank - k)))
        tensor = (getattr(module, f"output_{k}") @ tensor).flatten(-2)
    return tensor


def unpack_factorized(module, tensor):
    """Evaluate normalized symmetric tensor coordinates."""
    if module.rank < 2:
        return tensor
    if module.rank >= 6:
        for k in range(module.rank, 1, -1):
            tensor = tensor.unflatten(
                -1, ((k + 1) * (k + 2) // 2, 3 ** (module.rank - k))
            )
            tensor = (getattr(module, f"output_{k}").T @ tensor).flatten(-2)
        return tensor
    return (tensor / module.norm).index_select(-1, module.indices)


def contract_reference(module, tensor, vector):
    """Evaluate normalized symmetric tensor coordinates."""
    if module.rank == 1:
        return (tensor * vector).sum(-1, keepdim=True)
    value = tensor[..., module.contraction_indices]
    return (value * (module.contraction_weights * vector.unsqueeze(-2))).sum(-1)


def trace_reference(module, tensor):
    """Evaluate normalized symmetric tensor coordinates."""
    return (tensor[..., module.trace_indices] * module.trace_weights).sum(-1)


def multiply_reference(module, tensor, matrix):
    """Evaluate normalized symmetric tensor coordinates."""
    if module.rank == 2:
        return tensor * (matrix * module.norm)
    return (
        tensor[..., module.product_indices]
        * (module.product_weights * matrix.unsqueeze(-2))
    ).sum(-1)


def transform_compact(module, tensor, matrix):
    """Evaluate normalized symmetric tensor coordinates."""
    if module.rank == 1:
        return torch.einsum("...ij,...j->...i", matrix, tensor)
    for k in range(1, module.rank + 1):
        rank = module.rank - k
        tensor = tensor.unflatten(-1, (k * (k + 1) // 2, (rank + 2) * (rank + 3) // 2))
        if rank == 0:
            tensor = torch.einsum("...ij,...aj->...ai", matrix, tensor).flatten(-2)
            tensor = torch.nn.functional.linear(tensor, getattr(module, f"output_{k}"))
        else:
            indices = getattr(module, f"input_{k}_indices")
            weights = getattr(module, f"input_{k}_weights")
            tensor = tensor[..., indices] * weights
            tensor = torch.einsum("...ij,...aqj->...aiq", matrix, tensor)
            if k == 1:
                tensor = tensor.flatten(-3)
            else:
                tensor = (
                    getattr(module, f"output_{k}") @ tensor.flatten(-3, -2)
                ).flatten(-2)
    return tensor


def pack_reference(module, tensor):
    """Ordinary scatter reduction over all Cartesian index permutations."""
    shape = tensor.shape[:-1]
    indices = module.indices.expand(*shape, module.indices.numel())
    result = tensor.new_zeros((*shape, module.dim)).scatter_add(-1, indices, tensor)
    return result / module.norm


def transform_reference(module, tensor, matrix):
    """Expand the tensor, transform each Cartesian index, and symmetrize."""
    value = (tensor / module.norm).index_select(-1, module.indices)
    for axis in range(module.rank):
        value = torch.einsum(
            "...ij,...ajb->...aib",
            matrix,
            value.unflatten(-1, (3**axis, 3, 3 ** (module.rank - axis - 1))),
        ).flatten(-3)
    return pack_reference(module, value)


def sparse_pack(module, tensor, operation):
    """Project successive index groups with bounded three-term rows."""
    for k in range(2, module.rank + 1):
        tensor = operation(
            tensor,
            getattr(module, f"output_{k}_indices"),
            getattr(module, f"output_{k}_weights"),
            1,
            3 ** (module.rank - k),
        )
    return tensor


def sparse_contract(module, tensor, vector, operation):
    """Fuse the normalized contraction coefficients with vector components."""
    if module.rank == 1:
        return (tensor * vector).sum(-1, keepdim=True)
    return operation(
        tensor,
        module.contraction_indices,
        module.contraction_weights * vector.unsqueeze(-2),
    )


def sparse_trace(module, tensor, operation):
    """Sum the three diagonal contractions in normalized storage."""
    return operation(tensor, module.trace_indices, module.trace_weights)


def sparse_multiply(module, tensor, matrix, operation):
    """Fuse normalized symmetric insertion of six matrix components."""
    if module.rank == 2:
        return tensor * (matrix * module.norm)
    return operation(
        tensor, module.product_indices, module.product_weights * matrix.unsqueeze(-2)
    )


def sparse_transform(module, tensor, matrix, operation, axis):
    """Transform compact left/right symmetric groups with sparse index maps."""
    if module.rank == 0:
        return tensor
    for k in range(1, module.rank + 1):
        rank = module.rank - k
        left = k * (k + 1) // 2
        right = (rank + 1) * (rank + 2) // 2
        if rank:
            indices = getattr(module, f"input_{k}_indices").reshape(-1, 1)
            weights = getattr(module, f"input_{k}_weights").reshape(-1, 1)
            tensor = operation(tensor, indices, weights, left, 1)
        tensor = axis(tensor, matrix, left * right, 1)
        tensor = tensor.unflatten(-1, (left, right, 3)).transpose(-1, -2).flatten(-3)
        tensor = operation(
            tensor,
            getattr(module, f"output_{k}_indices"),
            getattr(module, f"output_{k}_weights"),
            1,
            right,
        )
    return tensor


def selected():
    """Choose the handwritten sparse primal or its operand-only adjoint."""
    return (
        sparse_axis.evaluate if config.TORCH_SYMMETRIC_ADJOINT else sparse_axis.primal
    )


def compiling():
    """Recognize Dynamo and explicit joint-gradient tracing."""
    return torch.compiler.is_compiling() or get_proxy_mode() is not None


def pack(module, tensor):
    """Use a direct normalized map while retaining the ordinary reference."""
    if module.rank < 2:
        return tensor
    if config.TORCH_REFERENCE:
        return pack_reference(module, tensor)
    if not config.TORCH_SYMMETRIC_KERNELS:
        return pack_factorized(module, tensor)
    if module.rank >= 8:
        return (
            sparse_pack(module, tensor, sparse_axis.evaluate)
            if compiling()
            else pack_factorized(module, tensor)
        )
    if not tensor.is_cuda and not compiling():
        return pack_reference(module, tensor)
    return symmetry_map(tensor, module.indices, module.norm, False)


def unpack(module, tensor):
    """Expand directly instead of introducing repeated compact GEMMs."""
    if module.rank < 2:
        return tensor
    if config.TORCH_REFERENCE:
        return map_reference(tensor, module.indices, module.norm, True)
    if not config.TORCH_SYMMETRIC_KERNELS:
        return unpack_factorized(module, tensor)
    if module.rank >= 8:
        return unpack_factorized(module, tensor)
    if not tensor.is_cuda and not compiling():
        return map_reference(tensor, module.indices, module.norm, True)
    return symmetry_map(tensor, module.indices, module.norm, True)


def contract(module, tensor, vector):
    """Keep the measured fused gather/reduce schedule for vector contractions."""
    return contract_reference(module, tensor, vector)


def trace(module, tensor):
    """Keep the compact ordinary three-term trace for joint fusion."""
    return trace_reference(module, tensor)


def multiply(module, tensor, matrix):
    """Expose short insertion sums only where the joint compiler improves them."""
    if (
        config.TORCH_REFERENCE
        or not config.TORCH_SYMMETRIC_KERNELS
        or not compiling()
        or module.rank > 4
    ):
        return multiply_reference(module, tensor, matrix)
    return sparse_multiply(module, tensor, matrix, sparse_axis.primal)


def use_shared_matrix(module, tensor, matrix):
    """Recognize a bounded channel axis shared by the geometry matrix."""
    if (
        not config.TORCH_SYMMETRIC_MATRIX
        or tensor.is_cuda
        or module.rank not in (2, 4, 6)
        or not hasattr(module, "transform_identity")
        or tensor.ndim < 3
        or not statically_known_true(tensor.shape[-2] >= 64)
        or not statically_known_true(tensor.shape[-2] <= 1024)
    ):
        return False
    return matrix.ndim == 2 or (
        matrix.ndim >= 3 and statically_known_true(matrix.shape[-3] == 1)
    )


def transform(module, tensor, matrix):
    """Compile sparse tensor-axis adjoints; retain compact eager execution."""
    if config.TORCH_REFERENCE:
        return transform_reference(module, tensor, matrix)
    if not config.TORCH_SYMMETRIC_KERNELS or not compiling():
        return transform_compact(module, tensor, matrix)
    if use_shared_matrix(module, tensor, matrix):
        representation = transform_compact(module, module.transform_identity, matrix)
        return tensor @ representation
    return sparse_transform(module, tensor, matrix, selected(), tensor_axis.primal)


def map_reference(tensor, indices, norm, expand):
    """Independent scatter/gather pair for normalized symmetric coordinates."""
    if expand:
        return (tensor / norm).index_select(-1, indices)
    result = tensor.new_zeros((*tensor.shape[:-1], norm.numel()))
    return result.index_add(-1, indices, tensor) / norm


def map_gradients(tensor, indices, norm, gradient, expand, need_norm, apply):
    """Use the exact mutual adjoints of packing and normalized expansion."""
    dx = apply(gradient, not expand)
    dn = None
    if need_norm:
        value = dx * tensor if expand else gradient * apply(tensor, False)
        dn = -(value / norm).sum_to_size(norm.shape)
    return dx, dn


@torch.compiler.allow_in_graph
class SymmetryMap(torch.autograd.Function):
    """Retain constant metadata and transpose the normalized permutation map."""

    generate_vmap_rule = True

    @staticmethod
    def forward(tensor, indices, norm, expand):
        return map_reference(tensor, indices, norm, expand)

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, indices, norm, ctx.expand = inputs
        ctx.save_for_backward(
            tensor if ctx.needs_input_grad[2] else tensor.new_empty(0), indices, norm
        )
        ctx.save_for_forward(tensor, indices, norm)

    @staticmethod
    def backward(ctx, gradient):
        tensor, indices, norm = ctx.saved_tensors
        dx, dn = map_gradients(
            tensor,
            indices,
            norm,
            gradient,
            ctx.expand,
            ctx.needs_input_grad[2],
            lambda x, expand: symmetry_map(x, indices, norm, expand),
        )
        return dx, None, dn, None

    @staticmethod
    def jvp(ctx, dt, unused_indices, dn, unused_expand):
        tensor, indices, norm = ctx.saved_tensors
        dt = torch.zeros_like(tensor) if dt is None else dt
        dn = torch.zeros_like(norm) if dn is None else dn
        result = map_reference(dt, indices, norm, ctx.expand)
        if ctx.expand:
            return result - map_reference(tensor * (dn / norm), indices, norm, True)
        return result - map_reference(tensor, indices, norm, False) * (dn / norm)


@torch.compiler.allow_in_graph
def symmetry_map(tensor, indices, norm, expand):
    """Keep arbitrarily nested forward transforms in exact ordinary algebra."""
    if nested_forward_ad():
        return map_reference(tensor, indices, norm, expand)
    return SymmetryMap.apply(tensor, indices, norm, expand)

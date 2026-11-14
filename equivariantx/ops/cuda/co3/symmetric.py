"""Native normalized symmetric maps built from reusable tensor-axis kernels."""

import math

import torch
import triton
import triton.language as tl
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true
from torch.library import triton_op, wrap_triton

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda import sparse_axis, tensor_axis
from ...common.polynomial import native_polynomial
from ...common.symmetric import matrix_inputs, matrix_metadata
from ...common.torch import sparse_axis as torch_sparse
from ...common.torch import tensor_axis as torch_axis
from ...precision import validate_uniform
from ...torch.co3 import symmetric as torch_kernel


def inline():
    """Expose exact Torch algebra for transforms or optional joint fusion."""
    return (
        config.SYMMETRIC_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ) or (forward_ad_active())


def selected():
    return torch_sparse.primal if inline() else sparse_axis.evaluate


def pack(module, tensor):
    return map_tensor(module, tensor, False)


def unpack(module, tensor):
    return map_tensor(module, tensor, True)


def contract(module, tensor, vector):
    if inline():
        return torch_kernel.contract_reference(module, tensor, vector)
    return torch_kernel.sparse_contract(module, tensor, vector, sparse_axis.evaluate)


def trace(module, tensor):
    if inline():
        return torch_kernel.trace_reference(module, tensor)
    return torch_kernel.sparse_trace(module, tensor, sparse_axis.evaluate)


def multiply(module, tensor, matrix):
    if inline() and module.rank > 4:
        return torch_kernel.multiply_reference(module, tensor, matrix)
    return torch_kernel.sparse_multiply(module, tensor, matrix, selected())


def transform(module, tensor, matrix):
    if use_shared_matrix(module, tensor, matrix):
        from ...common.cuda.matmul import matmul

        value = matrix if matrix.ndim == 2 else matrix.squeeze(-3)
        program, types, operands = matrix_inputs(value, module.rank)
        representation = native_polynomial(
            "symmetric_matrix", matrix_metadata(module.rank), program, types, *operands
        )[0].reshape(*value.shape[:-2], module.dim, module.dim)
        return matmul(tensor, representation)
    return torch_kernel.sparse_transform(
        module,
        tensor,
        matrix,
        torch_sparse.evaluate if inline() else sparse_axis.evaluate,
        torch_axis.primal if inline() else tensor_axis.evaluate,
    )


def use_shared_matrix(module, tensor, matrix):
    """Select the optional matrix schedule only for measured channel widths."""
    if (
        not config.SYMMETRIC_MATRIX_NATIVE
        or not tensor.is_cuda
        or tensor.dtype != torch.float32
        or tensor.ndim < 3
        or module.rank not in (4, 6)
        or not statically_known_true(
            tensor.shape[-2] == (128 if module.rank == 4 else 1024)
        )
        or not (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return False
    return matrix.ndim == 2 or (
        matrix.ndim >= 3 and statically_known_true(matrix.shape[-3] == 1)
    )


@triton.jit
def pack_kernel(
    features,
    permutation,
    offsets,
    norm,
    output,
    rows,
    INNER: tl.constexpr,
    OUTER: tl.constexpr,
    ROWS: tl.constexpr,
    TERMS: tl.constexpr,
):
    row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    component = tl.program_id(1)
    start = tl.load(offsets + component)
    stop = tl.load(offsets + component + 1)
    k = tl.arange(0, TERMS)[None, :]
    index = tl.load(permutation + start + k, start + k < stop, 0)
    value = tl.load(
        features + row * INNER + index, (row < rows) & (start + k < stop), 0
    )
    result = tl.sum(value, 1) * tl.load(norm + component)
    tl.store(
        output + tl.reshape(row, (ROWS,)) * OUTER + component,
        result,
        tl.reshape(row, (ROWS,)) < rows,
    )


@triton.jit
def unpack_kernel(
    features,
    indices,
    norm,
    output,
    size,
    INNER: tl.constexpr,
    OUTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    source = tl.load(indices + index % OUTER, index < size, 0)
    value = tl.load(features + (index // OUTER) * INNER + source, index < size, 0)
    scale = tl.load(norm + source, index < size, 1)
    tl.store(output + index, value * scale, index < size)


@triton_op("eqx::symmetric_map", mutates_args=())
def map_operator(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    norm: torch.Tensor,
    permutation: torch.Tensor,
    offsets: torch.Tensor,
    terms: int,
    expand: bool,
) -> torch.Tensor:
    """Evaluate a complete normalized permutation map in a single native launch."""
    validate_uniform((tensor, norm))
    expected = norm.numel() if expand else indices.numel()
    if not tensor.is_cuda or tensor.shape[-1] != expected:
        raise ValueError("Native symmetric maps require matching CUDA features.")
    tensor = tensor.contiguous()
    inverse = norm.reciprocal()
    outer = indices.numel() if expand else norm.numel()
    result = tensor.new_empty((*tensor.shape[:-1], outer))
    rows = tensor.numel() // expected
    if not statically_known_true(result.numel() == 0):
        if expand:
            wrap_triton(unpack_kernel)[(triton.cdiv(result.numel(), 256),)](
                tensor, indices, inverse, result, result.numel(), expected, outer, 256
            )
        else:
            size = triton.next_power_of_2(terms)
            block = min(64, max(1, 8192 // size))
            wrap_triton(pack_kernel)[(triton.cdiv(rows, block), outer)](
                tensor,
                permutation,
                offsets,
                inverse,
                result,
                rows,
                expected,
                outer,
                block,
                size,
            )
    return result


def map_context(ctx, inputs, output):
    tensor, indices, norm, permutation, offsets, ctx.terms, ctx.expand = inputs
    ctx.save_for_backward(
        tensor if ctx.needs_input_grad[2] else tensor.new_empty(0),
        indices,
        norm,
        permutation,
        offsets,
    )


def map_backward(ctx, gradient):
    tensor, indices, norm, permutation, offsets = ctx.saved_tensors
    dx, dn = torch_kernel.map_gradients(
        tensor,
        indices,
        norm,
        gradient,
        ctx.expand,
        ctx.needs_input_grad[2],
        lambda x, expand: map_operator(
            x, indices, norm, permutation, offsets, ctx.terms, expand
        ),
    )
    return dx, None, dn, None, None, None, None


map_operator.register_autograd(map_backward, setup_context=map_context)


def map_tensor(module, tensor, expand):
    """Use a single native map with recursively native mutual adjoints."""
    if module.rank < 2:
        return tensor
    if inline():
        return (torch_kernel.unpack if expand else torch_kernel.pack)(module, tensor)
    if not hasattr(module, "permutation"):
        return torch_kernel.symmetry_map(tensor, module.indices, module.norm, expand)
    terms = math.factorial(module.rank) // math.prod(
        tuple(math.factorial((module.rank + i) // 3) for i in range(3))
    )
    return map_operator(
        tensor,
        module.indices,
        module.norm,
        module.permutation,
        module.offsets,
        terms,
        expand,
    )

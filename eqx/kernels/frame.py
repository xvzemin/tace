"""Indexed frame rotations and their multilinear adjoints."""

from functools import lru_cache

import torch

from ..utils.metadata import parse_metadata
from .cuda import kernels, runtime

# Tensor slots in the trilinear pairing <edges, matrix @ nodes>.
NODE, MATRIX, EDGE = 0, 1, 2


def transform(frame, features, wigner, index, num_nodes, inverse):
    """Evaluate an indexed LocalFrame map using compact degree matrices."""
    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Frame kernels require CUDA float32 or float64 features.")
    if wigner.device != features.device or wigner.dtype != features.dtype:
        raise ValueError("Features and Wigner matrices must share dtype and device.")
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


@lru_cache(maxsize=128)
def source(dtype, layout, role):
    """Generate node, compact-matrix or edge contractions for a frame map."""
    ni, no, nw, entries = parse_metadata(layout)
    scalar = "double" if dtype == torch.float64 else "float"
    branches = []
    if role == EDGE:
        for start, mul, dim, offset, rows in entries:
            for local, row, sign in rows:
                branches.append(f"""
                if (col >= {local} && col < {local + mul}) {{
                    const int c = col - {local}; scalar value = 0;
                    #pragma unroll
                    for (int k = 0; k < {dim}; ++k)
                        value += matrix[e * {nw} + {offset + row * dim} + k]
                            * nodes[n * {ni} + {start} + k * {mul} + c];
                    output[e * {no} + col] = {sign} * value;
                    return;
                }}""")
        body = f"""
        const int64_t e = blockIdx.x, n = index[e];
        const int col = blockIdx.y * blockDim.x + threadIdx.x;
        if (col >= {no}) return;
        {"".join(branches)}
        """
    elif role == NODE:
        for start, mul, dim, offset, rows in entries:
            terms = "\n".join(
                f"value += {sign} * matrix[e * {nw} + {offset + row * dim} + k]"
                f" * edges[e * {no} + {local} + c];"
                for local, row, sign in rows
            )
            branches.append(f"""
            if (col >= {start} && col < {start + dim * mul}) {{
                const int k = (col - {start}) / {mul}, c = (col - {start}) % {mul};
                scalar value = 0;
                for (int64_t j = bounds[0]; j < bounds[1]; ++j) {{
                    const int64_t e = order[j];
                    {terms}
                }}
                output[n * {ni} + col] = value;
                return;
            }}""")
        body = f"""
        const int64_t n = blockIdx.x;
        __shared__ int64_t bounds[2];
        if (threadIdx.x < 2) {{
            int64_t lo = 0, hi = num_edges;
            const int64_t node = n + threadIdx.x;
            while (lo < hi) {{
                const int64_t mid = lo + (hi - lo) / 2;
                if (index[order[mid]] < node) lo = mid + 1;
                else hi = mid;
            }}
            bounds[threadIdx.x] = lo;
        }}
        __syncthreads();
        const int col = blockIdx.y * blockDim.x + threadIdx.x;
        if (col >= {ni}) return;
        {"".join(branches)}
        """
    else:
        offsets = sorted({entry[3] for entry in entries})
        for offset in offsets:
            selected = [entry for entry in entries if entry[3] == offset]
            dim, count = selected[0][2], len(selected[0][4])
            terms = []
            for start, mul, _, _, rows in selected:
                for local, row, sign in rows:
                    terms.append(f"""
                    if (r == {row})
                        for (int c = lane; c < {mul}; c += 32)
                            value += {sign} * nodes[n * {ni} + {start} + k * {mul} + c]
                                * edges[e * {no} + {local} + c];
                    """)
            branches.append(f"""
            if (col >= {offset} && col < {offset + dim * count}) {{
                const int r = (col - {offset}) / {dim}, k = (col - {offset}) % {dim};
                {"".join(terms)}
            }}""")
        body = f"""
        const int64_t e = blockIdx.x, n = index[e];
        const int col = blockIdx.y * 4 + threadIdx.x / 32, lane = threadIdx.x % 32;
        if (col >= {nw}) return;
        scalar value = 0;
        {"".join(branches)}
        for (int shift = 16; shift; shift >>= 1)
            value += __shfl_down_sync(0xffffffff, value, shift);
        if (lane == 0) output[e * {nw} + col] = value;
        """
    return f"""
    using scalar = {scalar};
    using int64_t = long long;
    extern "C" __global__ void run(const scalar* nodes, const scalar* matrix,
        const scalar* edges, const int64_t* index, const int64_t* order,
        scalar* output, int64_t num_edges) {{ {body} }}
    """


@torch.library.custom_op("eqx::indexed_frame", mutates_args=(), device_types="cuda")
def contract(
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
    ni, no, nw, _ = parse_metadata(layout)
    nodes, matrix, edges, index = (
        x.contiguous() for x in (nodes, matrix, edges, index)
    )
    output = contract_fake(nodes, matrix, edges, index, num_nodes, layout, role)
    if not output.numel():
        return output
    if not index.numel():
        return output.zero_()
    order = index
    if role == NODE:
        from ..conv.graph import prepare_graph

        order = prepare_graph(index, index, 0)
    code = source(nodes.dtype, layout, role)
    kernel = kernels([code], nodes.device)[code]
    width = (ni, nw, no)[role]
    columns = 4 if role == MATRIX else 128
    runtime().launch(
        [
            (
                kernel,
                [x.data_ptr() for x in (nodes, matrix, edges, index, order, output)]
                + [index.numel()],
                num_nodes if role == NODE else index.numel(),
                (width + columns - 1) // columns,
                128,
                0,
            )
        ],
        torch.cuda.current_stream(nodes.device).cuda_stream,
    )
    return output


@contract.register_fake
def contract_fake(nodes, matrix, edges, index, num_nodes, layout, role):
    ni, no, nw, _ = parse_metadata(layout)
    shape = ((num_nodes, ni), (index.numel(), nw), (index.numel(), no))[role]
    return nodes.new_empty(shape)


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


contract.register_autograd(backward, setup_context=setup_context)

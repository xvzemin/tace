"""CUDA evaluation of indexed frame transformations."""

from functools import lru_cache

import torch

from ....utils.metadata import parse_metadata
from ....utils.outputs import EDGE, MATRIX, NODE, indexed_frame_output
from .runtime import kernels, runtime


@lru_cache(maxsize=128)
def matrix_sum_source(dtype, layouts):
    """Generate a channel reduction for a sum of rotation-matrix adjoints."""
    scalar = "double" if dtype == torch.float64 else "float"
    arguments, terms = [], []
    for slot, layout in enumerate(layouts):
        ni, no, width, entries = parse_metadata(layout)
        arguments.extend(
            (
                f"const scalar* nodes{slot}",
                f"const scalar* edges{slot}",
                f"const int64_t* index{slot}",
            )
        )
        for start, mul, dim, offset, rows in entries:
            for local, row, sign in rows:
                terms.append(f"""
                if(col>={offset + row * dim} && col<{offset + (row + 1) * dim}) {{
                  const int k=col-{offset + row * dim};
                  const int64_t n=index{slot}[e];
                  for(int c=lane;c<{mul};c+=32)
                    value+=scalar({sign})*nodes{slot}[n*{ni}+{start}+k*{mul}+c]
                           *edges{slot}[e*{no}+{local}+c];
                }}""")
    return f"""
    using scalar={scalar}; using int64_t=long long;
    extern "C" __global__ void run({", ".join(arguments)},scalar* output) {{
      const int64_t e=blockIdx.x;
      const int col=blockIdx.y*4+threadIdx.x/32, lane=threadIdx.x%32;
      if(col>={width}) return;
      scalar value=0;
      {"".join(terms)}
      for(int shift=16;shift;shift>>=1)
        value+=__shfl_down_sync(0xffffffffu,value,shift);
      if(lane==0) output[e*{width}+col]=value;
    }}"""


def matrix_sum(layouts, operands):
    """Sum rotation adjoints using one output buffer and channel reduction."""
    reference = operands[0][0]
    width = parse_metadata(layouts[0])[2]
    count = operands[0][2].numel()
    output = reference.new_empty((count, width))
    if not count or not width:
        return output
    values = [value.contiguous() for term in operands for value in term]
    code = matrix_sum_source(reference.dtype, layouts)
    kernel = kernels([code], reference.device)[code]
    runtime().launch(
        [
            (
                kernel,
                [value.data_ptr() for value in values] + [output.data_ptr()],
                count,
                (width + 3) // 4,
                128,
                0,
            )
        ],
        torch.cuda.current_stream(reference.device).cuda_stream,
    )
    return output


@lru_cache(maxsize=128)
def source(dtype, layout, role, rowwise=False, accumulate=False):
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
            reduction = (
                terms
                if rowwise
                else f"""
                for (int64_t j = bounds[0]; j < bounds[1]; ++j) {{
                    const int64_t e = order[j];
                    {terms}
                }}"""
            )
            branches.append(f"""
            if (col >= {start} && col < {start + dim * mul}) {{
                const int k = (col - {start}) / {mul}, c = (col - {start}) % {mul};
                scalar value = 0;
                {reduction}
                output[n * {ni} + col] {"+=" if accumulate else "="} value;
                return;
            }}""")
        bounds = (
            "const int64_t e = n;"
            if rowwise
            else """
        __shared__ int64_t bounds[2];
        if (threadIdx.x < 2) {
            int64_t lo = 0, hi = num_edges;
            const int64_t node = n + threadIdx.x;
            while (lo < hi) {
                const int64_t mid = lo + (hi - lo) / 2;
                if (index[order[mid]] < node) lo = mid + 1;
                else hi = mid;
            }
            bounds[threadIdx.x] = lo;
        }
        __syncthreads();
        """
        )
        body = f"""
        const int64_t n = blockIdx.x;
        {bounds}
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


def contract(
    nodes: torch.Tensor,
    matrix: torch.Tensor,
    edges: torch.Tensor,
    index: torch.Tensor,
    num_nodes: int,
    layout: str,
    role: int,
    *,
    rowwise: bool = False,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Contract the two factors other than ``role`` in the trilinear pairing.

    ``NODE`` sums transposed matrix products, ``MATRIX`` sums channel products,
    and ``EDGE`` gathers and rotates node features. The selected input is unused;
    each adjoint applies the same contraction with a different output role.
    Supplying ``output`` accumulates a node reduction into existing storage.
    """
    ni, no, nw, _ = parse_metadata(layout)
    nodes, matrix, edges, index = (
        x.contiguous() for x in (nodes, matrix, edges, index)
    )
    accumulate = output is not None
    if accumulate and (role != NODE or rowwise):
        raise ValueError("An output buffer requires an indexed node reduction.")
    if output is None:
        output = indexed_frame_output(
            nodes, matrix, edges, index, num_nodes, layout, role
        )
    if not output.numel():
        return output
    if not index.numel():
        return output if accumulate else output.zero_()
    order = index
    if role == NODE and not rowwise:
        from .reduction import prepare_graph

        order = prepare_graph(index, index, 0)
    code = source(nodes.dtype, layout, role, rowwise, accumulate)
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

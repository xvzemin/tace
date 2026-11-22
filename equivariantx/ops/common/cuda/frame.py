"""CUDA evaluation of indexed frame transformations."""

from functools import lru_cache

import torch

from ....utils.metadata import parse_metadata
from ....utils.outputs import EDGE, MATRIX, NODE, indexed_frame_output
from .graph import capture_constant
from .runtime import kernels, runtime

INDEX_MATRIX_WIDTH = 64


@lru_cache(maxsize=64)
def matrix_index(layouts, device):
    """Index the channel dots contributing to each packed matrix entry."""
    import numpy as np

    columns, records, strides = [], [], []
    width = parse_metadata(layouts[0])[2]
    for slot, layout in enumerate(layouts):
        ni, no, nw, entries = parse_metadata(layout)
        if nw != width:
            raise ValueError("Summed frame adjoints must share a matrix width.")
        strides.append((ni, no))
        for start, mul, dim, offset, rows in entries:
            k = np.arange(dim, dtype=np.int32)
            for local, row, sign in rows:
                if sign not in (-1, 1):
                    return None
                columns.append(offset + row * dim + k)
                data = np.empty((dim, 4), dtype=np.int32)
                data[:, 0], data[:, 1] = slot, start + k * mul
                data[:, 2], data[:, 3] = local, sign * mul
                records.append(data)
    if not columns:
        return None
    col = np.concatenate(columns)
    order = np.argsort(col, kind="stable")
    counts = np.bincount(col, minlength=width)
    fan = int(counts[0]) if (counts == counts[0]).all() else 0
    offsets = np.empty(0 if fan else width + 1, dtype=np.int32)
    if not fan:
        offsets[0] = 0
        np.cumsum(counts, out=offsets[1:])
    return (
        tuple(strides),
        fan,
        torch.from_numpy(np.concatenate(records)[order]).to(device),
        torch.from_numpy(offsets).to(device),
    )


@lru_cache(maxsize=128)
def indexed_matrix_source(dtype, strides, fan):
    """Generate one warp per matrix entry without enumerating angular rows in code."""
    from .codegen import HEADER

    header = HEADER.replace("SCALAR", "double" if dtype == torch.float64 else "float")
    arguments, terms = [], []
    for i, (ni, no) in enumerate(strides):
        arguments.extend((f"const T* n{i}", f"const T* m{i}", f"const I* idx{i}"))
        dot = f"""
          const T *x=n{i}+idx{i}[e]*{ni}+p.start, *y=m{i}+e*{no}+p.local;
          for(int c=lane;c<mul;c+=32) s=fma(T(p.width<0?-1:1)*x[c],y[c],s);"""
        terms.append(f"case {i}:{{{dot}break;}}" if len(strides) > 1 else dot)
    body = "".join(terms)
    if len(strides) > 1:
        body = "switch(p.slot){" + body + "}"
    begin, end = (f"col*{fan}", f"lo+{fan}") if fan else ("ptr[col]", "ptr[col+1]")
    return (
        header
        + f"""
    using I=long long; struct alignas(16) Row {{int slot,start,local,width;}};
    extern "C" __global__ void run({",".join(arguments)},const Row* rows,
      const int* ptr,T* out,I width,I count){{
      const int lane=threadIdx.x%32; const I col=blockIdx.x*4+threadIdx.x/32;
      if(col>=width) return;
      const int lo={begin}, hi={end};
      for(I e=blockIdx.y;e<count;e+=gridDim.y){{
        T s=0;
        for(int j=lo;j<hi;++j){{Row p=rows[j]; int mul=p.width<0?-p.width:p.width; {body} }}
        s=warp_sum(s);
        if(!lane) out[e*width+col]=s;
      }}
    }}"""
    )


def indexed_matrix(layouts, operands, output):
    """Write a sum of frame adjoints using a compact shared channel index."""
    plan = capture_constant(matrix_index, layouts, output.device)
    if plan is None:
        return False
    strides, fan, rows, ptr = plan
    code = indexed_matrix_source(output.dtype, strides, fan)
    kernel = kernels([code], output.device)[code]
    values = [value.contiguous() for term in operands for value in term]
    count, width = output.shape
    args = [value.data_ptr() for value in (*values, rows, ptr, output)] + [width, count]
    runtime().launch(
        [(kernel, args, (width + 3) // 4, min(count, 65535), 128, 0)],
        torch.cuda.current_stream(output.device).cuda_stream,
    )
    return True


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
    if width >= INDEX_MATRIX_WIDTH and indexed_matrix(layouts, operands, output):
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
                for (int64_t j = ptr[n]; j < ptr[n+1]; ++j) {{
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
        body = f"""
        const int64_t n = blockIdx.x;
        {"const int64_t e = n;" if rowwise else ""}
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
        const int64_t* ptr, scalar* output) {{ {body} }}
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
    if role == MATRIX and nw >= INDEX_MATRIX_WIDTH:
        if indexed_matrix((layout,), ((nodes, edges, index),), output):
            return output
    order = ptr = index
    if role == NODE and not rowwise:
        from .reduction import prepare_graph

        order, ptr = prepare_graph(index, index, 0, num_nodes)
    code = source(nodes.dtype, layout, role, rowwise, accumulate)
    kernel = kernels([code], nodes.device)[code]
    width = (ni, nw, no)[role]
    columns = 4 if role == MATRIX else 128
    runtime().launch(
        [
            (
                kernel,
                [
                    x.data_ptr()
                    for x in (nodes, matrix, edges, index, order, ptr, output)
                ],
                num_nodes if role == NODE else index.numel(),
                (width + columns - 1) // columns,
                128,
                0,
            )
        ],
        torch.cuda.current_stream(nodes.device).cuda_stream,
    )
    return output

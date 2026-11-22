"""Stable node ownership for explicit matrix index-adds, including density.

These post-differentiation rewrites preserve all summands. They remove atomic
update order from early model values, where TF32 and a force-loss derivative
can amplify otherwise tiny differences between identical executions.
"""

from functools import lru_cache

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from equivariantx.ops.common.cuda.runtime import kernels, runtime

from .graph import prepare_graph
from .row_schedule import prepare_segments


@lru_cache(maxsize=64)
def index_sum_source(dtype, width, alpha):
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
using T={scalar};
extern "C" __global__ void run(const T* base,const T* values,
    const long long* order,const long long* ptr,T* result,long long nodes) {{
    long long node=blockIdx.x;
    int column=blockIdx.y*128+threadIdx.x;
    if(node>=nodes || column>={width}) return;
    T sum=0;
    for(long long j=ptr[node];j<ptr[node+1];++j)
        sum+=values[order[j]*{width}+column];
    result[node*{width}+column]=base[node*{width}+column]+T({float(alpha):.17g})*sum;
}}
"""


@torch.library.custom_op(
    "eqx_oam_xl::ordered_index_add", mutates_args=(), device_types="cuda"
)
def ordered_index_add(
    base: torch.Tensor, index: torch.Tensor, values: torch.Tensor, alpha: float = 1.0
) -> torch.Tensor:
    """Add rows in stable edge order, with a single owner per result element."""
    if (
        base.ndim != 2
        or values.ndim != 2
        or index.ndim != 1
        or values.shape != (index.numel(), base.shape[1])
        or base.dtype != values.dtype
        or base.dtype not in (torch.float32, torch.float64)
        or index.dtype != torch.int64
    ):
        raise ValueError(
            "Ordered index-add requires compatible floating matrices and int64 rows"
        )
    if not base.numel() or not index.numel():
        return base.clone()
    base, values, index = base.contiguous(), values.contiguous(), index.contiguous()
    nodes, width = base.shape
    order, pointers = prepare_graph(index, index, 0, nodes)
    result = torch.empty_like(base)
    code = index_sum_source(base.dtype, width, alpha)
    kernel = kernels([code], base.device)[code]
    arguments = [value.data_ptr() for value in (base, values, order, pointers, result)]
    arguments += [nodes]
    runtime().launch(
        [(kernel, arguments, nodes, (width + 127) // 128, 128, 0)],
        torch.cuda.current_stream(base.device).cuda_stream,
    )
    return result


@ordered_index_add.register_fake
def ordered_index_outputs(base, index, values, alpha=1.0):
    return torch.empty_like(base, memory_format=torch.contiguous_format)


def order_reductions(graph):
    """Replace confirmed matrix index-adds after all derivatives are explicit."""
    from .program import refresh_inputs

    counts = dict(index_adds=0)
    for node in graph.nodes:
        if node.target == torch.ops.aten.index_add.default:
            base, dim, index, values = node.args[:4]
            base_meta, value_meta = base.meta.get("val"), values.meta.get("val")
            if (
                dim != 0
                or base_meta.device.type != "cuda"
                or base_meta.ndim != 2
                or value_meta.ndim != 2
                or base_meta.dtype not in (torch.float32, torch.float64)
                or index.meta["val"].dtype != torch.int64
            ):
                continue
            alpha = (
                node.args[4] if len(node.args) > 4 else node.kwargs.get("alpha", 1.0)
            )
            if not isinstance(alpha, (int, float)):
                continue
            node.target = torch.ops.eqx_oam_xl.ordered_index_add.default
            node.args, node.kwargs = (base, index, values, float(alpha)), {}
            counts["index_adds"] += 1
            refresh_inputs(node)
    graph.lint()
    counts["element_index_adds"] = replace_index_put(graph)
    return counts


@lru_cache(maxsize=4)
def grouped_sum_source(dtype, final):
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
using T = {scalar};
using I = long long;
extern "C" __global__ void run(const T* src, const I* order, const I* ptr,
    const I* jobs, const T* base, T* dst, I groups, I edges, I width, I segments) {{
    constexpr bool final = {int(final)};
    const I task = blockIdx.x, tiles = (width + 31) / 32;
    const I layer = blockIdx.y / tiles, col = (blockIdx.y % tiles) * 32 + threadIdx.x % 32;
    I begin, end;
    if (final) {{
        begin = jobs[task]; end = jobs[task + 1];
    }} else {{
        if (task >= jobs[groups]) return;
        I lo = 0, hi = groups;
        while (lo < hi) {{
            I mid = (lo + hi) / 2;
            if (jobs[mid + 1] <= task) lo = mid + 1; else hi = mid;
        }}
        begin = ptr[lo] + (task - jobs[lo]) * 128;
        end = min(begin + 128, ptr[lo + 1]);
    }}
    T sum = 0;
    if (col < width) {{
        for (I row = begin + threadIdx.x / 32; row < end; row += 8) {{
            I at = final ? layer * segments + row : layer * edges + order[row];
            sum += src[at * width + col];
        }}
    }}
    __shared__ T partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    if (threadIdx.x < 32 && col < width) {{
        #pragma unroll
        for (int warp = 1; warp < 8; ++warp) sum += partial[warp * 32 + threadIdx.x];
        I at = (layer * (final ? groups : segments) + task) * width + col;
        dst[at] = final ? base[at] + sum : sum;
    }}
}}
"""


@torch.library.custom_op(
    "eqx_oam_xl::grouped_index_sum", mutates_args=(), device_types="cuda"
)
def grouped_sum(
    base: torch.Tensor, index: torch.Tensor, values: torch.Tensor
) -> torch.Tensor:
    """Accumulate indexed rows independently for each layer and element."""
    base, index, values = base.contiguous(), index.contiguous(), values.contiguous()
    if not index.numel() or not base.numel():
        return base.clone()
    layers, groups, width = base.shape
    index = torch.where(index < 0, index + groups, index)
    order, pointers = prepare_graph(index, index, 0, groups)
    jobs = prepare_segments(pointers, 128)
    segments = (index.numel() + 127) // 128 + groups
    partial = base.new_empty((layers, segments, width))
    output = torch.empty_like(base)
    sources = [grouped_sum_source(base.dtype, final) for final in (False, True)]
    compiled = kernels(sources, base.device)
    plan = []
    for code, src, dst, tasks in zip(
        sources, (values, partial), (partial, output), (segments, groups)
    ):
        args = [x.data_ptr() for x in (src, order, pointers, jobs, base, dst)]
        args.extend((groups, index.numel(), width, segments))
        plan.append(
            (compiled[code], args, tasks, layers * ((width + 31) // 32), 256, 0)
        )
    runtime().launch(plan, torch.cuda.current_stream(base.device).cuda_stream)
    return output


@grouped_sum.register_fake
def fake_grouped_sum(base, index, values):
    return torch.empty_like(base, memory_format=torch.contiguous_format)


def replace_index_put(graph):
    from .program import refresh_inputs

    count = 0
    for node in graph.nodes:
        if node.target != torch.ops.aten.index_put.default:
            continue
        base, indices, values = node.args[:3]
        accumulate = (
            node.args[3] if len(node.args) > 3 else node.kwargs.get("accumulate", False)
        )
        if (
            not accumulate
            or len(indices) != 2
            or indices[0] is not None
            or indices[1] is None
        ):
            continue
        index = indices[1]
        a, b, i = (value.meta.get("val") for value in (base, values, index))
        if (
            a is None
            or b is None
            or i is None
            or a.device.type != "cuda"
            or a.ndim != 3
            or b.ndim != 3
            or i.ndim != 1
            or a.dtype not in (torch.float32, torch.float64)
            or b.dtype != a.dtype
            or i.dtype != torch.int64
            or not statically_known_true(a.shape[0] == b.shape[0])
            or not statically_known_true(a.shape[2] == b.shape[2])
            or not statically_known_true(b.shape[1] == i.shape[0])
        ):
            continue
        node.target = torch.ops.eqx_oam_xl.grouped_index_sum.default
        node.args, node.kwargs = (base, index, values), {}
        refresh_inputs(node)
        count += 1
    graph.lint()
    return count

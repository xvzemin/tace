"""Stable node ownership for explicit matrix index-adds, including density.

These post-differentiation rewrites preserve all summands. They remove atomic
update order from early model values, where TF32 and a force-loss derivative
can amplify otherwise tiny differences between identical executions.
"""

from functools import lru_cache

import torch

from .graph import prepare_graph
from equivariantx.ops.common.cuda.runtime import kernels, runtime


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
    return counts

"""Species-grouped FP32 maps on native OAM-L features, without packing rows.

One warp computes 32 output channels on four magnetic rows. Features stay
in atom-major order; adjacent lanes reuse input channels via warp shuffles.
The per-step permutation includes every species and is shared by all layers.
"""

from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.ops.common.torch.grouped import element_grouping
from equivariantx.utils.metadata import parse_metadata

_WIDTH = 16


def grouping(types, elements):
    slots, owners, chunks = element_grouping(types, elements, 1, _WIDTH)
    rows = slots.new_full((chunks * _WIDTH,), types.shape[0]).scatter_(
        0, slots, torch.arange(types.shape[0], device=types.device)
    )
    return rows, owners


@lru_cache(maxsize=128)
def _source(sx, sw, sy, ci, co, dim, scale, din, dout, weights):
    return f"""
    extern "C" __global__ void run(const float* __restrict__ X,
        const float* __restrict__ W, float* __restrict__ Y,
        const long long* __restrict__ rows, const long long* __restrict__ owners, int n) {{
        int lane=threadIdx.x%32, warp=threadIdx.x/32, group=blockIdx.y, tile=blockIdx.x;
        int v=tile%({co}/32)*32+lane, base=tile/({co}/32)*16;
        if(rows[(long long)group*{_WIDTH}]>=n) return;
        int species=owners[group];
        float acc[4]={{}};
        long long ix[4], iy[4];
        #pragma unroll
        for(int a=0;a<4;++a) {{
            int m=base+warp+a*4;
            long long row=m<{_WIDTH * dim} ? rows[(long long)group*{_WIDTH}+m/{dim}]:n;
            ix[a]=row<n ? row*{din}+{sx}+m%{dim} : -1;
            iy[a]=row<n ? row*{dout}+{sy}+m%{dim}+v*{dim} : -1;
        }}
        #pragma unroll 1
        for(int kk=0;kk<{ci};kk+=32) {{
            float av[4];
            #pragma unroll
            for(int a=0;a<4;++a)
                av[a]=ix[a]<0 ? 0.f:X[ix[a]+(kk+lane)*{dim}];
            #pragma unroll
            for(int k=0;k<32;++k) {{
                float w=W[(long long)species*{weights}+{sw}+(kk+k)*{co}+v];
                #pragma unroll
                for(int a=0;a<4;++a) acc[a]=fmaf(__shfl_sync(0xffffffff,av[a],k),w,acc[a]);
            }}
        }}
        #pragma unroll
        for(int a=0;a<4;++a) if(iy[a]>=0) Y[iy[a]]=acc[a]*float({float(scale).hex()});
    }}
    """


@torch.library.custom_op(
    "eqx_oam_l_element::grouped", mutates_args=(), device_types="cuda"
)
def grouped(
    metadata: str,
    x: torch.Tensor,
    table: torch.Tensor,
    types: torch.Tensor,
    rows: torch.Tensor,
    owners: torch.Tensor,
) -> torch.Tensor:
    din, dout, weights, experts, paths, layout = parse_metadata(metadata)
    intervals = sorted((sy, sy + co * d) for _, _, sy, _, co, d, _ in paths)
    complete = sum(end - start for start, end in intervals) == dout
    # Check at execution time, not while tracing a small export example.
    if (
        x.shape[0] < (1024 if max(din, dout) <= 2304 else 512)
        or owners.numel() > 65535
        or x.dtype != torch.float32
        or table.dtype != torch.float32
        or layout != "mul_ir"
        or experts != 1
        or not paths
        or any(ci % 32 or co % 32 for _, _, _, ci, co, _, _ in paths)
        or any(right[0] < left[1] for left, right in zip(intervals, intervals[1:]))
    ):
        return torch.ops.eqx.element_linear(
            metadata, 2, types, x, table.unsqueeze(1), x.new_empty((0, dout))
        )
    x, table = x.contiguous(), table.contiguous()
    y = x.new_empty((x.shape[0], dout)) if complete else x.new_zeros((x.shape[0], dout))
    sources = [_source(*p, din, dout, weights) for p in paths]
    launchers = kernels(sources, x.device)
    ptrs = [v.data_ptr() for v in (x, table, y, rows, owners)] + [x.shape[0]]
    calls = [
        (
            launchers[src],
            ptrs,
            ((_WIDTH * p[5] + 15) // 16) * (p[4] // 32),
            owners.numel(),
            128,
            0,
        )
        for src, p in zip(sources, paths)
    ]
    runtime().launch(calls, torch.cuda.current_stream(x.device).cuda_stream)
    return y


@grouped.register_fake
def _fake(metadata, x, table, types, rows, owners):
    return x.new_empty((x.shape[0], parse_metadata(metadata)[1]))

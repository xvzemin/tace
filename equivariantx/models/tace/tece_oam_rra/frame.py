"""Frame transformations for frozen rotary attention."""

from functools import lru_cache

import torch

from ....ops.common.cuda.runtime import kernels, runtime
from ....utils.frame_program import rotate
from ....utils.metadata import compact_metadata, parse_metadata
from ....utils.program import Program
from .expressions import packed_layout, paired_features
from .layout import local_orders

OUTPUT_FRAME = (1, (1600, 165, 1600), ("frozen_output_rotation",))
ENDPOINT_MATRIX = (1, (3200, 165), ("frozen_endpoint_matrix",))


@lru_cache(maxsize=1)
def local_positions():
    order = [l * l + row for l, row, _ in local_orders()]
    return tuple(order.index(i) for i in range(25))


def lower_output(nodes, value):
    """Replace an exact local-to-global value rotation in a frozen program."""
    if nodes[value][:2] != ("concat", 1600):
        return nodes
    program = Program(nodes)
    matrix = program.input(6, "edge", 165)
    if matrix >= len(nodes):
        return nodes
    layout = packed_layout(5, 4, 64)
    for i, (op, width, args, _) in enumerate(nodes[:value]):
        if (op, width) != ("concat", 1600) or tuple(nodes[j][1] for j in args) != (
            320,
            512,
            384,
            256,
            128,
        ):
            continue
        if (
            rotate(program, layout, i, matrix, 25, 25, inverse=True, packed=True)
            == value
        ):
            rows = list(nodes)
            rows[value] = ("product", 1600, (i, matrix, i), (OUTPUT_FRAME, 2, 0))
            return tuple(rows)
    return nodes


@lru_cache(maxsize=3)
def output_source(role):
    positions = local_positions()
    cases = []
    for l in range(5):
        d = 2 * l + 1
        off = l * (4 * l * l - 1) // 3
        local = positions[l * l : (l + 1) * (l + 1)]
        if role == 1:
            select = "".join(f"case {r}:row={j};break;" for r, j in enumerate(local))
            body = f"""
                for(int k=c/32;k<{d * d};k+=4){{
                    int row=0;switch(k/{d}){{{select}}}
                    int lane=c%32;
                    const float* x=a+(e*25+row)*64;
                    const float* g=b+(e*25+{l * l}+k%{d})*64;
                    float v=x[lane]*g[lane]+x[lane+32]*g[lane+32];
                    for(int s=16;s;s/=2)v+=__shfl_down_sync(0xffffffffu,v,s);
                    if(!lane)y[e*165+{off}+k]=v;
                }}"""
        else:
            reads = local if role == 2 else range(l * l, (l + 1) * (l + 1))
            writes = range(l * l, (l + 1) * (l + 1)) if role == 2 else local
            body = "".join(
                f"float x{k}=a[(e*25+{j})*64+c];" for k, j in enumerate(reads)
            )
            for j, col in enumerate(writes):
                terms = "+".join(
                    f"b[e*165+{off + (k * d + j if role == 2 else j * d + k)}]*x{k}"
                    for k in range(d)
                )
                body += f"y[(e*25+{col})*64+c]={terms};"
        cases.append(f"case {l}:{{{body}break;}}")
    return f"""
    extern "C" __global__ void run(const float* a,const float* b,float* y){{
        long long e=blockIdx.x; int c=threadIdx.x;
        switch(blockIdx.y){{{"".join(cases)}}}
    }}"""


def output_frame(a, b, role, destination=None):
    """Apply the output rotation or one of its two input adjoints to an edge tile."""
    rows = a.shape[0]
    width = 165 if role == 1 else 1600
    if (
        role not in (0, 1, 2)
        or not a.is_cuda
        or a.dtype != torch.float32
        or b.device != a.device
        or b.dtype != a.dtype
        or a.shape != (rows, 1600)
        or b.shape != (rows, 1600 if role == 1 else 165)
    ):
        raise ValueError("Invalid operands for a frozen output frame.")
    if destination is not None and (
        destination.shape != (rows, width)
        or destination.device != a.device
        or destination.dtype != a.dtype
    ):
        raise ValueError("Invalid frozen output frame destination.")
    a, b = a.contiguous(), b.contiguous()
    out = (
        destination
        if destination is not None and destination.is_contiguous()
        else a.new_empty((rows, width))
    )
    if rows:
        code = output_source(role)
        kernel = kernels((code,), a.device)[code]
        runtime().launch(
            [
                (
                    kernel,
                    [v.data_ptr() for v in (a, b, out)],
                    rows,
                    5,
                    128 if role == 1 else 64,
                    0,
                )
            ],
            torch.cuda.current_stream(a.device).cuda_stream,
        )
    return (
        destination.copy_(out)
        if destination is not None and out is not destination
        else out
    )


def lower_endpoint(nodes, outputs):
    """Replace the paired transposed frame rotations of matching endpoint adjoints."""
    endpoints = {kind: i for i, slot, kind in outputs if slot == 0}
    if set(endpoints) != {"source", "target"}:
        return nodes
    left, right = (nodes[endpoints[k]] for k in ("source", "target"))
    if left[:2] != ("slice", 1600) or len(left[2]) != 1:
        return nodes
    (root,) = left[2]
    if left[3] != 0 or right != ("slice", 1600, (root,), 1600):
        return nodes
    op, size, args, indices = nodes[root]
    if op != "scatter" or size != 3200 or len(args) != 1:
        return nodes
    if indices != tuple(
        side * 1600 + angular * 64 + c
        for angular in range(25)
        for side in range(2)
        for c in range(64)
    ):
        return nodes
    (joined,) = args
    op, _, legs, _ = nodes[joined]
    if op != "concat" or len(legs) != 5:
        return nodes
    op, _, vals, shape = nodes[legs[0]]
    if op != "matmul" or shape != (1, 1, 128):
        return nodes
    d0, x0 = (nodes[i] for i in vals)
    if d0[:2] != ("slice", 1) or x0[:2] != ("slice", 128):
        return nodes
    (wigner,), (local,) = d0[2], x0[2]
    if nodes[wigner] != ("input", 165, (), (5, "edge")) or nodes[local][1] != 3200:
        return nodes
    p = Program(nodes)
    positions = local_positions()
    for l, leg in enumerate(legs):
        d = 2 * l + 1
        offset = l * (4 * l * l - 1) // 3
        matrix = p.transpose(p.slice(wigner, offset, d * d), d, d)
        x = p.concatenate(
            [p.slice(local, positions[l * l + k] * 128, 128) for k in range(d)]
        )
        if p.matmul(matrix, x, d, d, 128) != leg:
            return nodes
    rows = list(nodes)
    desc = (1, (3200, 165, 3200), ("frozen_endpoint_adjoint",))
    rows[root] = ("product", 3200, (local, wigner, local), (desc, 2, 0))
    return tuple(rows)


def lower_endpoint_matrix(metadata):
    """Replace the input rotation-matrix pullback of a frozen endpoint pair."""
    nodes, outputs = parse_metadata(metadata)
    roots = [i for i, slot, kind in outputs if (slot, kind) == (5, "edge")]
    pairs = [
        args[:2]
        for op, _, args, desc in nodes
        if op == "product" and desc[0][2] == ("frozen_endpoint_adjoint",)
    ]
    if len(roots) != 1 or len(pairs) != 1:
        return metadata
    (root,), ((local, wigner),) = roots, pairs
    op, size, legs, _ = nodes[root]
    if (
        op != "concat"
        or size != 165
        or len(legs) != 5
        or nodes[local][1] != 3200
        or nodes[wigner] != ("input", 165, (), (5, "edge"))
    ):
        return metadata
    p = Program(nodes)
    pair = paired_features(
        p, p.input(0, "source", 1600), p.input(0, "target", 1600), 25, 64
    )
    pos = local_positions()
    for l, leg in enumerate(legs):
        d = 2 * l + 1
        x = p.concatenate([p.slice(local, pos[l * l + k] * 128, 128) for k in range(d)])
        y = p.transpose(p.slice(pair, l * l * 128, d * 128), d, 128)
        if p.matmul(x, y, d, 128, d) != leg:
            return metadata
    rows = list(nodes)
    rows[root] = ("product", 165, (local, wigner), (ENDPOINT_MATRIX, 1, 0))
    return compact_metadata(repr((tuple(rows), outputs)))


@lru_cache(maxsize=2)
def matrix_source(aligned):
    local, global_rows = [], []
    pos = local_positions()
    for l in range(5):
        for i in range(2 * l + 1):
            for j in range(2 * l + 1):
                local.append(str(pos[l * l + i]))
                global_rows.append(str(l * l + j))
    load = (
        "return reinterpret_cast<const float4*>(x)[i];"
        if aligned
        else "x+=4*i;return make_float4(x[0],x[1],x[2],x[3]);"
    )
    return f"""
    __device__ __constant__ int local[165]={{{",".join(local)}}},global_row[165]={{{",".join(global_rows)}}};
    __device__ __forceinline__ float4 read4(const float* x,long long i){{{load}}}
    extern "C" __global__ void run(const float* x,const float* g,const long long* src,
                                   const long long* dst,float* out){{
        long long e=blockIdx.x; int lane=threadIdx.x%32;
        long long a=lane<16?src[e]:dst[e];
        for(int m=threadIdx.x/32;m<165;m+=blockDim.x/32){{
            float4 h=read4(g,(e*25+local[m])*32+lane);
            float4 v=read4(x,(a*25+global_row[m])*16+lane%16);
            float sum=fmaf(h.w,v.w,fmaf(h.z,v.z,fmaf(h.y,v.y,h.x*v.x)));
            #pragma unroll
            for(int d=16;d>0;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
            if(!lane)out[e*165+m]=sum;
        }}
    }}"""


def endpoint_matrix_adjoint(nodes, local, source, target, destination=None):
    """Pull back two rotated endpoint features to the packed rotation matrices.

    Parameters
    ----------
    nodes : torch.Tensor
        Node features of shape ``(N, 25, 64)`` or ``(N, 1600)``.
    local : torch.Tensor
        Local endpoint cotangents of shape ``(E, 3200)``.
    source, target : torch.Tensor
        Source and target node indices of shape ``(E,)``.
    destination : torch.Tensor, optional
        Output buffer of shape ``(E, 165)``.
    """
    n = local.shape[0]
    if (
        nodes.shape[1:] not in ((1600,), (25, 64))
        or local.shape != (n, 3200)
        or nodes.dtype != torch.float32
        or local.dtype != nodes.dtype
        or not nodes.is_cuda
        or any(v.device != nodes.device for v in (source, target, local))
        or any(v.dtype != torch.int64 or v.shape != (n,) for v in (source, target))
        or (
            destination is not None
            and (
                destination.shape != (n, 165)
                or destination.dtype != nodes.dtype
                or destination.device != nodes.device
            )
        )
    ):
        raise ValueError("Invalid inputs to the frozen FP32 endpoint matrix adjoint.")
    x, g = nodes.flatten(1).contiguous(), local.contiguous()
    out = (
        destination
        if destination is not None and destination.is_contiguous()
        else g.new_empty((n, 165))
    )
    if n:
        src, dst = source.contiguous(), target.contiguous()
        code = matrix_source(((x.data_ptr() | g.data_ptr()) & 15) == 0)
        kernel = kernels((code,), x.device)[code]
        runtime().launch(
            [(kernel, [a.data_ptr() for a in (x, g, src, dst, out)], n, 1, 256, 0)],
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    return (
        destination.copy_(out)
        if destination is not None and destination is not out
        else out
    )


@lru_cache(maxsize=1)
def endpoint_source():
    cases = []
    positions = local_positions()
    for l in range(5):
        d = 2 * l + 1
        off = l * (4 * l * l - 1) // 3
        reads = ",".join(f"v{k}=x[{positions[l * l + k]}*128+c]" for k in range(d))
        body = [f"case {l}:{{const float {reads};"]
        for j in range(d):
            body.append("{float a=0;")
            body.extend(f"a=fmaf(w[{off + k * d + j}],v{k},a);" for k in range(d))
            body.append(f"out[{l * l + j}*64+lane]=a;}}")
        cases.extend((*body, "break;}"))
    return (
        """
    extern "C" __global__ void run(const float *x,const float *w,float *out){
        long long e=blockIdx.x;int c=threadIdx.x,lane=c&63;
        x+=e*3200;w+=e*165;out+=e*3200+(c/64)*1600;
        switch(blockIdx.y){"""
        + "".join(cases)
        + "}}"
    )


def endpoint_adjoint(features, wigner, destination=None):
    """Rotate and split the two local endpoint cotangents of a frozen FP32 tile.

    Parameters
    ----------
    features : torch.Tensor
        Local endpoint cotangents of shape ``(E, 3200)``.
    wigner : torch.Tensor
        Packed rotation matrices of shape ``(E, 165)``.
    destination : torch.Tensor, optional
        Output buffer of shape ``(E, 3200)``.
    """
    if (
        features.ndim != 2
        or features.shape[1] != 3200
        or wigner.shape != (len(features), 165)
        or features.device != wigner.device
        or not features.is_cuda
        or features.dtype != torch.float32
        or wigner.dtype != features.dtype
        or (
            destination is not None
            and (
                destination.shape != features.shape
                or destination.dtype != features.dtype
                or destination.device != features.device
            )
        )
    ):
        raise ValueError("Invalid inputs to the frozen FP32 endpoint adjoint.")
    x, w = features.contiguous(), wigner.contiguous()
    out = torch.empty_like(x) if destination is None else destination
    if not out.is_contiguous():
        out = torch.empty_like(x)
    if len(x):
        code = endpoint_source()
        kernel = kernels((code,), x.device)[code]
        runtime().launch(
            [(kernel, [t.data_ptr() for t in (x, w, out)], len(x), 5, 128, 0)],
            torch.cuda.current_stream(x.device).cuda_stream,
        )
    if destination is not None and out is not destination:
        return destination.copy_(out)
    return out

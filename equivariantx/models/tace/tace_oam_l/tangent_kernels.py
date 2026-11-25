"""CUDA kernels for frozen radial-tangent TACE-OAM-L inference."""

from collections import defaultdict
from functools import lru_cache

import torch

from equivariantx.ops.common.cuda.pointwise import launch, scalar_type
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata

from .radial import _kernel


def heads(p, feat, tangent, src, dst):
    e = len(feat)
    h = feat.new_empty(e, 128)
    hdot = torch.empty_like(h)
    d, ddot = (feat.new_empty(e, 1) for _ in range(2))
    code = _kernel(
        scalar_type(src.dtype), p.info_eps, p.density_eps, False, 128, tangent=True
    )
    for start in range(0, e, 32768):
        sl = slice(start, start + 32768)
        pre = feat[sl] @ p.edge
        pre_dot = tangent[sl] @ p.edge
        args = [
            pre,
            src[sl],
            dst[sl],
            *p[1:8],
            h[sl],
            d[sl],
            pre_dot,
            hdot[sl],
            ddot[sl],
        ]
        launch(code, args, (len(pre),), 32 * len(pre), threads=128)
    return h, d, hdot, ddot


def encode_tangent(length, roots, prefactor, weight, scale):
    width = weight.shape[1]
    if width not in (64, 128):
        raise ValueError("Expected an eight-mode embedding of width 64 or 128.")
    feat = weight.new_empty(len(length), width)
    tangent = torch.empty_like(feat)
    code = f"""
    extern "C" __global__ void run(const double* r, const double* k,
        const double* prefactor,const float* weight,float* feat,float* tangent,long long n) {{
      int lane=threadIdx.x%32;
      for(long long tile=((long long)blockIdx.x*blockDim.x+threadIdx.x)/32;tile<n*{width // 64};tile+=((long long)blockDim.x*gridDim.x)/32) {{
        long long e=tile/{width // 64};
        int channel=(tile%{width // 64})*64+lane;
        float basis=0.f,dbasis=0.f;
        if(lane<8) {{
          double a=k[lane],z=r[e]*a,s,c;
          sincos(z,&s,&c);
          double sinc,dsinc;
          if(fabs(z)<1.e-3) {{
            double z2=z*z;
            sinc=1.+z2*(-1./6.+z2*(1./120.-z2/5040.));
            dsinc=z*(-1./3.+z2*(1./30.-z2/840.));
          }} else {{sinc=s/z;dsinc=(c-sinc)/z;}}
          basis=float(*prefactor*a*sinc);
          dbasis=float(*prefactor*a*a*dsinc);
        }}
        float u0=0.f,u1=0.f,du0=0.f,du1=0.f;
        #pragma unroll
        for(int j=0;j<8;++j) {{
          float bv=__shfl_sync(0xffffffff,basis,j),v=__shfl_sync(0xffffffff,dbasis,j);
          float w0=weight[{width}*j+channel],w1=weight[{width}*j+32+channel];
          u0=fmaf(bv,w0,u0);u1=fmaf(bv,w1,u1);du0=fmaf(v,w0,du0);du1=fmaf(v,w1,du1);
        }}
        float s0=1.f/(1.f+expf(-u0)),s1=1.f/(1.f+expf(-u1)), gain=float({float(scale).hex()});
        feat[e*{width}+channel]=(u0*s0)*gain;feat[e*{width}+32+channel]=(u1*s1)*gain;
        tangent[e*{width}+channel]=du0*s0*(1.f+u0*(1.f-s0))*gain;
        tangent[e*{width}+32+channel]=du1*s1*(1.f+u1*(1.f-s1))*gain;
      }}
    }}"""
    launch(
        code,
        [length.contiguous(), roots, prefactor, weight, feat, tangent],
        (len(length),),
        (width // 2) * len(length),
        threads=128,
    )
    return feat, tangent


def expression(terms):
    result = "0.f"
    for a, b in terms:
        result = f"fmaf({a},{b},{result})"
    return result


def harmonic_reduction(columns):
    """Generate channel sums with consecutive angular outputs in consecutive lanes."""
    code = []
    while columns:
        start = columns[0]
        n = 1
        while n < min(32, len(columns)) and columns[n] == start + n:
            n += 1
        size = 1 << (n - 1).bit_length()
        values = [f"dy{b}" for b in columns[:n]] + ["0.f"] * (size - n)
        columns = columns[n:]
        code.append("{")
        stride = 1
        while len(values) > 1:
            next_values = []
            for i, (a, b) in enumerate(zip(values[::2], values[1::2])):
                name = f"v{stride}_{i}"
                code.append(
                    f"float {name}=((lane & {stride}) ? {b} : {a}) + "
                    f"__shfl_xor_sync(0xffffffff, ((lane & {stride}) ? {a} : {b}), {stride});"
                )
                next_values.append(name)
            values = next_values
            stride *= 2
        code.append(f"float acc={values[0]};")
        while stride < 32:
            code.append(f"acc+=__shfl_xor_sync(0xffffffff,acc,{stride});")
            stride *= 2
        condition = f"if(lane<{n}) " if n < 32 else ""
        code += [f"{condition}atomicAdd(dy+{start}+lane,acc);", "}"]
    return code


def adjoint_source(paths, columns, tangent=True, node_grad=True, *, paired=False):
    features, weights, harmonics, outputs = columns
    offset, size = paths[0][0], paths[0][5]
    channels = paths[0][3]
    row = "pair[e]" if paired else "e"
    pair_arg = ", const long long* pair" if paired else ""
    body = [
        f"""
extern "C" __global__ void run(const float* x, const float* y, const float* w,
 const float* g, float* dx, float* dy, float* dw, const long long* s,
 const long long* t, long long n, const float* wdot{pair_arg}) {{
 const long long e = (long long)blockIdx.x*(blockDim.x/32) + threadIdx.x/32;
 if(e>=n) return;
 const int c = threadIdx.x%32 + blockIdx.y*32, lane = threadIdx.x%32;
 const long long i=s[e], j=t[e];
 x += i*{features} + {offset} + c; {f"dx += i*{features} + {offset} + c;" if node_grad else ""}
 g += j*{outputs} + c; w+={row}*{weights}+c;
 {f"float dr=0.f; wdot+={row}*{weights}+c;" if tangent else f"dw+=e*{weights}+c;"}
 y += e*{harmonics}; dy += e*{harmonics};
"""
    ]
    for a in range(size):
        body += [f"const float x{a}=x[{channels * a}]; float dx{a}=0.f;"]
    attr_cols = sorted({p[1] + b for p in paths for b in range(p[6])})
    body.extend(f"float dy{b}=0.f;" for b in attr_cols)
    for _, attr, out, _, _, _, ds, do, weight, factor, cg in paths:
        body.append("{")
        body.extend(f"const float z{d}=g[{out + channels * d}];" for d in range(do))
        body.extend(f"const float y{b}=y[{attr + b}];" for b in range(ds))
        body.append(f"const float k=float({factor:.17g}), wf=k*w[{weight}];")
        # The same coefficient times the node cotangent appears in both adjoints.
        coeffs = sorted({(z, c) for _, _, z, c in cg})
        names = {key: f"p{j}" for j, key in enumerate(coeffs)}
        body.extend(
            f"const float {name}=float({c:.17g})*z{z};"
            for (z, c), name in names.items()
        )
        for row in range(size):
            expr = expression([(names[z, c], f"y{b}") for a, b, z, c in cg if a == row])
            body.append(f"const float u{row}={expr};")
            if node_grad:
                body.append(f"dx{row}=fmaf(wf,u{row},dx{row});")
        qw = f"k*{expression([(f'u{i}', f'x{i}') for i in range(size)])}"
        body.append(
            f"dr=fmaf({qw},wdot[{weight}],dr);" if tangent else f"dw[{weight}]={qw};"
        )
        for local in range(ds):
            expr = expression(
                [(names[z, c], f"x{a}") for a, b, z, c in cg if b == local]
            )
            body.append(f"dy{attr + local}=fmaf(wf,{expr},dy{attr + local});")
        body.append("}")
    if node_grad:
        body.extend(f"atomicAdd(dx+{channels * i},dx{i});" for i in range(size))
    if tangent:
        body.extend(
            f"dr+=__shfl_down_sync(0xffffffff,dr,{s});" for s in (16, 8, 4, 2, 1)
        )
        body.append("if(lane==0) atomicAdd(dw+e,dr);")
    body.extend(harmonic_reduction(attr_cols))
    body.append("}")
    return "\n".join(body)


@lru_cache(maxsize=32)
def adjoint_plan(metadata, columns, device, tangent, node_grad, *, paired=False):
    grouped = defaultdict(list)
    for path in parse_metadata(metadata)[0]:
        grouped[path[0]].append(path)
    groups = [
        group[i : i + 16]
        for group in grouped.values()
        for i in range(0, len(group), 16)
    ]
    code = [adjoint_source(p, columns, tangent, node_grad, paired=paired) for p in groups]
    compiled = kernels(code, device)
    return tuple((compiled[src], p[0][3] // 32) for src, p in zip(code, groups, strict=True))


def coordinate_adjoint(metadata, src, dst, x, y, w, g, dx, dy, radial_grad, wdot=None):
    """Return path-weight adjoints or contract them with the radial tangent."""
    count = src.numel()
    if not count:
        return
    dims = (x.shape[1], w.shape[1], y.shape[1], g.shape[1])
    ptrs = [v.data_ptr() for v in (x, y, w, g, dx, dy, radial_grad, src, dst)]
    args = [*ptrs, count, wdot.data_ptr() if wdot is not None else 0]
    runtime().launch(
        [
            (k, args, (count + 3) // 4, warps, 128, 0)
            for k, warps in adjoint_plan(
                metadata, dims, x.device, wdot is not None, dx.numel() != 0
            )
        ],
        torch.cuda.current_stream(x.device).cuda_stream,
    )

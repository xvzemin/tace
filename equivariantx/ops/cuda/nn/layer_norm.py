"""Weighted row normalization with optional scalar-block centering."""

from functools import lru_cache
from math import isfinite

import torch

from ....utils.metadata import parse_metadata
from ...common.cuda.pointwise import launch, scalar_type


@lru_cache(maxsize=128)
def source(key, dtype, adjoint=False, incoming=True, radial=False):
    widths, factors, centered, p = parse_metadata(key)
    moments, variances, outputs = [], [], []
    start = 0
    for i, (width, factor) in enumerate(zip(widths, factors)):
        loop = f"for(int j={start}+tid;j<{start + width};j+=blockDim.x)"
        value = (
            f"({'g[j]' if incoming else 'T(0)'}-v[j]*T({factor!r})*dot)"
            if adjoint
            else "v[j]"
        )
        if i in centered:
            moments.append(f"""
            T s{i}=0; {loop} s{i}+={value};
            T m{i}=reduce(s{i},buf)/T({width});
            """)
        z = f"({value}-m{i})" if i in centered else value
        variances.append(
            f"{loop} {{T z={'v[j]' if adjoint else z}; sum+=T({factor!r})*z*z;}}"
        )
        outputs.append(f"{loop} u[j]={z}*r;")
        start += width
    declarations = "const T* x,T* y,T* inv"
    scale = "T r=T(1)/sqrt(q+T(EPSILON));"
    if p != 1.0:
        scale = f"""
        T unit=fmax(q,T(EPSILON));
        T r=pow(pow(q/unit,T({p!r}))+pow(T(EPSILON)/unit,T({p!r})),T({-0.5 / p!r}))/sqrt(unit);
        """
    norm = f"""
        {"".join(moments)}
        T sum=0; {"".join(variances)}
        T q=reduce(sum,buf); {scale}
        if(!tid)inv[row]=r;
    """
    if adjoint:
        declarations = "const T* x,const T* inv,const T* dy,const T* dr,T* y"
        correction = (
            f"sum=0; {''.join(variances)} T q=reduce(sum,buf); "
            f"dot*=q>T(0)?pow(q,T({p - 1!r})):T(0);"
            if p != 1.0
            else ""
        )
        norm = f"""
        const T* g=dy+{"row*" + str(start) if incoming else "0"}; T r=inv[row];
        T sum=0;
        {f"for(int j=tid;j<{start};j+=blockDim.x) sum+=g[j]*v[j];" if incoming else ""}
        T dot={"reduce(sum,buf)" if incoming else "T(0)"} {"+dr[row]*r" if radial else ""};
        {correction}
        {"".join(moments)}
        """
    shuffle = dtype == torch.float32 and bool(centered) and start >= 128
    if shuffle:
        reduction = """
        #pragma unroll
        for(int d=16;d>0;d>>=1) a+=__shfl_down_sync(0xffffffffu,a,d);
        if(!(threadIdx.x&31)) buf[threadIdx.x>>5]=a;
        __syncthreads();
        T value=(buf[0]+buf[2])+(buf[1]+buf[3]);
        __syncthreads(); return value;
        """
    else:
        reduction = """
        int tid=threadIdx.x; buf[tid]=a; __syncthreads();
        for(int k=blockDim.x/2;k;k/=2){
            if(tid<k) buf[tid]+=buf[tid+k]; __syncthreads();
        }
        T value=buf[0]; __syncthreads(); return value;
        """
    return f"""
    using T={scalar_type(dtype)}; using I=long long;
    __device__ T reduce(T a,T* buf){{ {reduction} }}
    extern "C" __global__ void run({declarations},I rows){{
        __shared__ T buf[{4 if shuffle else 128}];
        int tid=threadIdx.x;
        for(I row=blockIdx.x;row<rows;row+=gridDim.x){{
            const T* v=x+row*{start}; T* u=y+row*{start};
            {norm} {"".join(outputs)}
        }}
    }}"""


def validate(key, x):
    widths, factors, centered, p = parse_metadata(key)
    if (
        x.ndim != 2
        or x.dtype not in (torch.float32, torch.float64)
        or len(widths) != len(factors)
        or not widths
        or any(type(k) is not int or k <= 0 for k in widths)
        or sum(widths) != x.shape[1]
        or any(type(f) is not float or not isfinite(f) for f in factors)
        or any(type(i) is not int or not 0 <= i < len(widths) for i in centered)
        or not isfinite(p)
        or p <= 0
    ):
        raise ValueError("Invalid normalization layout or input.")


@torch.library.custom_op("eqx::normalized_rows", mutates_args=(), device_types="cuda")
def normalize(
    key: str, x: torch.Tensor, eps: float
) -> tuple[torch.Tensor, torch.Tensor]:
    validate(key, x)
    if not isfinite(eps) or eps <= 0:
        raise ValueError("Expected a finite positive epsilon.")
    x = x.contiguous()
    y, inv = torch.empty_like(x), x.new_empty((x.shape[0], 1))
    code = source(key, x.dtype).replace("EPSILON", repr(eps))
    launch(code, (x, y, inv), (x.shape[0],), x.shape[0] * 128, threads=128)
    return y, inv


@normalize.register_fake
def fake(key, x, eps):
    return torch.empty_like(x, memory_format=torch.contiguous_format), x.new_empty(
        (x.shape[0], 1)
    )


def setup_context(ctx, inputs, output):
    ctx.key = inputs[0]
    ctx.save_for_backward(*output)
    ctx.set_materialize_grads(False)


@torch.library.custom_op(
    "eqx::normalized_rows_adjoint", mutates_args=(), device_types="cuda"
)
def adjoint(
    key: str,
    y: torch.Tensor,
    r: torch.Tensor,
    dy: torch.Tensor | None,
    dr: torch.Tensor | None,
) -> torch.Tensor:
    validate(key, y)
    if (
        r.shape != (y.shape[0], 1)
        or (dy is not None and dy.shape != y.shape)
        or (dr is not None and dr.shape != r.shape)
        or any(
            v.dtype != y.dtype or v.device != y.device
            for v in (r, dy, dr)
            if v is not None
        )
    ):
        raise ValueError("Normalization gradients must match their outputs.")
    code = source(key, y.dtype, True, dy is not None, dr is not None)
    dy = dy.contiguous() if dy is not None else r
    dr = dr.contiguous() if dr is not None else r
    dx = torch.empty_like(y, memory_format=torch.contiguous_format)
    launch(
        code,
        (y.contiguous(), r.contiguous(), dy, dr, dx),
        (y.shape[0],),
        y.shape[0] * 128,
        threads=128,
    )
    return dx


@adjoint.register_fake
def fake_adjoint(key, y, r, dy, dr):
    return torch.empty_like(y, memory_format=torch.contiguous_format)


def backward(ctx, dy, dr):
    y, r = ctx.saved_tensors
    return None, adjoint(ctx.key, y, r, dy, dr), None


def setup_adjoint(ctx, inputs, output):
    ctx.key, y, r, dy, dr = inputs
    ctx.save_for_backward(y, r, dy, dr)


def transpose_adjoint(ctx, x):
    y, r, dy, dr = ctx.saved_tensors
    widths, factors, centered, p = parse_metadata(ctx.key)
    if centered:
        bands = list(x.split(widths, -1))
        for i in centered:
            bands[i] = bands[i] - bands[i].mean(-1, keepdim=True)
        x = torch.cat(bands, -1)
    qx = torch.cat([v * f for v, f in zip(x.split(widths, -1), factors)], -1)
    a = (qx * y).sum(-1, keepdim=True)
    b = (dy * y).sum(-1, keepdim=True) if dy is not None else 0
    if dr is not None:
        b = b + dr * r
    needs = ctx.needs_input_grad
    # The adjoint is r P(dy - Q y k (<dy,y> + r dr)), k = (y^T Q y)^(p-1).
    dk = None
    if p != 1.0:
        qy = torch.cat([v * f for v, f in zip(y.split(widths, -1), factors)], -1)
        u = (qy * y).sum(-1, keepdim=True)
        if p.is_integer():
            k = u.pow(p - 1)
            if needs[1]:
                dk = 2 * (p - 1) * qy * u.pow(p - 2)
        else:
            positive = u > 0
            safe = u.masked_fill(~positive, 1)
            k = torch.where(positive, safe.pow(p - 1), 0)
            if needs[1]:
                dk = 2 * (p - 1) * qy * torch.where(positive, safe.pow(p - 2), 0)
        if needs[1]:
            dk = dk * (a * b)
        a = a * k
        qx = qx * k
    gy = qx * b + (dy * a if dy is not None else 0) if needs[1] else None
    if gy is not None:
        gy = -r * (gy if dk is None else gy + dk)
    gr = (
        ((x * dy).sum(-1, keepdim=True) if dy is not None else 0)
        - a * (b + (dr * r if dr is not None else 0))
        if needs[2]
        else None
    )
    gdy = r * (x - y * a) if dy is not None and needs[3] else None
    gdr = -r.square() * a if dr is not None and needs[4] else None
    return None, gy, gr, gdy, gdr


adjoint.register_autograd(transpose_adjoint, setup_context=setup_adjoint)
normalize.register_autograd(backward, setup_context=setup_context)

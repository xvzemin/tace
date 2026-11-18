"""Small symmetrized epsilon contractions and their transposes."""

import math
from functools import lru_cache
from itertools import permutations

import torch

from ....common.cuda.cartesian import packed_contract
from ....common.cuda.pointwise import broadcast_offset, launch, scalar_type
from ....common.torch.sparse_bilinear import adjoints
from ....precision import validate_uniform


@lru_cache(maxsize=2)
def permutations_code(rank):
    digits = [f"((c/{3 ** (rank - 1 - j)})%3)" for j in range(rank)]
    values = [
        "+".join(f"{3 ** (rank - 1 - j)}*{v}" for j, v in enumerate(order))
        for order in permutations(digits)
    ]
    cases = "".join(f"case {i}: return {v};" for i, v in enumerate(values[1:], 1))
    return f"__device__ __forceinline__ I perm(I c,int s) {{switch(s){{{cases} default:return c;}}}}"


@torch.library.custom_op(
    "eqx::ictp_symmetric_epsilon", mutates_args=(), device_types="cuda"
)
def contract(
    x: torch.Tensor, y: torch.Tensor, degrees: list[int], role: int
) -> torch.Tensor:
    """Apply a symmetrized epsilon product or either adjoint without coefficient tables."""
    if len(degrees) != 3 or role not in (0, 1, 2):
        raise ValueError("Expected three ICTP ranks and an output leg.")
    l1, l2, rank = degrees
    if min(l1, l2) < 1 or rank not in (2, 3) or rank != abs(l1 - l2) + 1:
        raise ValueError("Expected a rank-two or rank-three symmetrization-only path.")
    validate_uniform((x, y))
    widths = [3**v for v in degrees]
    width = widths.pop(role)
    if (x.shape[-1], y.shape[-1]) != tuple(widths):
        raise ValueError("Operand widths must match the ICTP ranks.")
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    out = x.new_empty((*shape, width))
    if not out.numel():
        return out
    source, lanes = epsilon_source(
        tuple(degrees),
        role,
        x.dtype,
        x.stride(-1),
        y.stride(-1),
        broadcast_offset(x, shape, 1),
        broadcast_offset(y, shape, 1),
    )
    launch(source, (x, y, out), (out.numel(),), out.numel() * lanes)
    return out


@lru_cache(maxsize=128)
def epsilon_source(degrees, role, dtype, stride_x, stride_y, row_x, row_y):
    l1, l2, rank = degrees
    width = 3 ** degrees[role]
    big = 0 if l1 > l2 else 1
    small = 1 - big
    inner = 3 ** (min(l1, l2) - 1)
    free = 3 ** (rank - 1)
    nperm = math.factorial(rank)
    slots = [i for i in range(3) if i != role]
    pointers = {s: p for s, p in zip(slots, ("px", "py"))}
    strides = dict(zip(slots, (stride_x, stride_y)))
    functions = []
    if big != role:
        address = "(f*3+a)*K+k" if big == 0 else "(k*3+a)*F+f"
        functions.append(
            f"auto M=[&](I f,I a,I k){{return {pointers[big]}[({address})*{strides[big]}];}};"
        )
    if small != role:
        address = "k*3+b" if big == 0 else "b*K+k"
        functions.append(
            f"auto V=[&](I k,I b){{return {pointers[small]}[({address})*{strides[small]}];}};"
        )
    if role != 2:
        functions.append(f"""auto G=[&](I f,I h){{T g=0;
            #pragma unroll
            for(int s=0;s<S;++s) g+={pointers[2]}[perm(3*f+h,s)*{strides[2]}];return g;}};""")
    if role == 2:
        extent = nperm * inner
        calculation = """for(I t=lane;t<S*K;t+=lanes){
            I j=perm(c,t/K),f=j/3,h=j%3,k=t%K,a=(h+1)%3,b=(h+2)%3;
            sum+=M(f,a,k)*V(k,b)-M(f,b,k)*V(k,a);}"""
    elif role == big:
        extent = 1
        coords = (
            "f=c/(3*K),a=(c/K)%3,k=c%K" if big == 0 else "k=c/(3*F),a=(c/F)%3,f=c%F"
        )
        calculation = f"I {coords},p=(a+1)%3,m=(a+2)%3;sum=V(k,p)*G(f,m)-V(k,m)*G(f,p);"
    else:
        extent = free
        coords = "k=c/3,b=c%3" if big == 0 else "b=c/K,k=c%K"
        calculation = f"""I {coords},p=(b+1)%3,m=(b+2)%3;
            for(I f=lane;f<F;f+=lanes) sum+=M(f,m,k)*G(f,p)-M(f,p,k)*G(f,m);"""
    lanes = 32 if extent >= 32 else 1
    scale = (1 if big == 0 else -1) / (nperm * math.sqrt(2 * inner))
    code = f"""
    using I=long long; using T={scalar_type(dtype)};
    {permutations_code(rank)}
    extern "C" __global__ void run(const T* x,const T* y,T* out,I count){{
        constexpr I K={inner},F={free},S={nperm},lanes={lanes};
        I id=I(blockIdx.x)*blockDim.x+threadIdx.x,lane=id%lanes;
        for(I i=id/lanes;i<count;i+=I(gridDim.x)*blockDim.x/lanes){{
            I row=i/{width},c=i%{width};
            const T* px=x+({row_x});const T* py=y+({row_y});
            {" ".join(functions)}
            T sum=0;{calculation}
            if constexpr(lanes==32){{
                #pragma unroll
                for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
            }}
            if(!lane)out[i]=sum*T({scale:.17g});
        }}
    }}"""
    return code, lanes


@contract.register_fake
def fake(x, y, degrees, role):
    return x.new_empty(
        (*torch.broadcast_shapes(x.shape[:-1], y.shape[:-1]), 3 ** degrees[role])
    )


def setup_context(ctx, inputs, output):
    x, y, ctx.degrees, ctx.role = inputs
    ctx.save_for_backward(x, y)


def backward(ctx, gradient):
    x, y = ctx.saved_tensors
    dx, dy = adjoints(
        contract, x, y, gradient, ctx.degrees, ctx.role, ctx.needs_input_grad[:2]
    )
    return dx, dy, None, None


contract.register_autograd(backward, setup_context=setup_context)


@lru_cache(maxsize=128)
def orbit_source(degrees, dtype, stride_x, stride_y, row_x, row_y):
    l1, l2, rank = degrees
    k = 3 ** (min(l1, l2) - 1)
    right = 3 ** max(0, l2 - l1)
    width = 3**rank
    groups = math.comb(rank + 2, 2)
    lanes = 32 if width * k >= 6 * groups else 1
    return (
        f"""
    using I=long long; using T={scalar_type(dtype)};
    extern "C" __global__ void run(const T* x,const T* y,const I* order,const I* offsets,
                                   T* out,I count){{
        constexpr I K={k},B={right},W={width},C={groups},lanes={lanes};
        I id=I(blockIdx.x)*blockDim.x+threadIdx.x, lane=id%lanes;
        for(I i=id/lanes;i<count;i+=I(gridDim.x)*blockDim.x/lanes){{
            I row=i/C,c=i%C,begin=offsets[c],end=offsets[c+1];
            const T* px=x+({row_x}); const T* py=y+({row_y}); T sum=0;
            for(I t=lane;t<(end-begin)*K;t+=lanes){{
                I j=order[begin+t/K],u=j/(3*B),h=(j/B)%3,v=j%B,k=t%K;
                I p=(h+1)%3,m=(h+2)%3;
                sum+=px[((u*3+p)*K+k)*{stride_x}]*py[((k*3+m)*B+v)*{stride_y}]
                    -px[((u*3+m)*K+k)*{stride_x}]*py[((k*3+p)*B+v)*{stride_y}];
            }}
            if constexpr(lanes==32){{
                #pragma unroll
                for(int d=16;d;d/=2)sum+=__shfl_down_sync(0xffffffff,sum,d);
                sum=__shfl_sync(0xffffffff,sum,0);
            }}
            T value=sum*(T({(2 * k) ** -0.5:.17g})/T(end-begin));
            for(I t=begin+lane;t<end;t+=lanes)out[row*W+order[t]]=value;
        }}
    }}""",
        lanes,
    )


@torch.library.custom_op(
    "eqx::ictp_orbit_product", mutates_args=(), device_types="cuda"
)
def orbit_product(
    x: torch.Tensor,
    y: torch.Tensor,
    degrees: list[int],
    indices: torch.Tensor,
    order: torch.Tensor,
    offsets: torch.Tensor,
    counts: torch.Tensor,
) -> torch.Tensor:
    """Contract each permutation class once and write its full Cartesian output."""
    l1, l2, rank = degrees
    if rank < 2 or min(l1, l2) < 1 or rank != abs(l1 - l2) + 1:
        raise ValueError("Expected a symmetrization-only path.")
    validate_uniform((x, y, counts))
    width, n = 3**rank, math.comb(rank + 2, 2)
    if (
        x.shape[-1],
        y.shape[-1],
        indices.shape,
        order.shape,
        offsets.shape,
        counts.shape,
    ) != (3**l1, 3**l2, (width,), (width,), (n + 1,), (n,)):
        raise ValueError("Invalid Cartesian widths or permutation classes.")
    if any(t.dtype != torch.long for t in (indices, order, offsets)):
        raise ValueError("Permutation tables must use int64.")
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    out = x.new_empty((*shape, width))
    if out.numel():
        source, lanes = orbit_source(
            tuple(degrees),
            x.dtype,
            x.stride(-1),
            y.stride(-1),
            broadcast_offset(x, shape, 1),
            broadcast_offset(y, shape, 1),
        )
        count = out.numel() // width * n
        launch(
            source,
            (x, y, order.contiguous(), offsets.contiguous(), out),
            (count,),
            count * lanes,
        )
    return out


@orbit_product.register_fake
def orbit_fake(x, y, degrees, indices, order, offsets, counts):
    return x.new_empty(
        (*torch.broadcast_shapes(x.shape[:-1], y.shape[:-1]), 3 ** degrees[2])
    )


def orbit_context(ctx, inputs, output):
    x, y, ctx.degrees, *tables = inputs
    ctx.save_for_backward(x, y, *tables)


def orbit_backward(ctx, gradient):
    x, y, indices, order, offsets, counts = ctx.saved_tensors
    g = gradient.new_zeros((*gradient.shape[:-1], counts.numel())).index_add(
        -1, indices, gradient
    )

    def transpose(a, b, degrees, role):
        return packed_contract(a, b, degrees, role, indices, order, offsets, counts)

    dx, dy = adjoints(transpose, x, y, g, ctx.degrees, 2, ctx.needs_input_grad[:2])
    return dx, dy, None, None, None, None, None


orbit_product.register_autograd(orbit_backward, setup_context=orbit_context)

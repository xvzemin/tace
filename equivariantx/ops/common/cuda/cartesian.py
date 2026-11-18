"""Native Cartesian contractions before STF projection."""

import math

import torch

from ....utils.autograd import forward_ad_active
from ... import config
from ...precision import validate_uniform
from ..channel_product import angular_product
from ..torch import cartesian as torch_kernel
from ..torch.sparse_bilinear import adjoints
from .pointwise import broadcast_offset, launch, scalar_type


@torch.library.custom_op(
    "eqx::cartesian_channel_mix", mutates_args=(), device_types="cuda"
)
def channel_mix(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Multiply narrow batched matrices without copying broadcast operands."""
    validate_uniform((x, y))
    if min(x.ndim, y.ndim) < 2 or x.size(-1) != y.size(-2):
        raise ValueError("Channel matrices must have matching contracted dimensions.")
    shape = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
    m, k, n = x.size(-2), x.size(-1), y.size(-1)
    result = x.new_empty((*shape, m, n))
    if not result.numel():
        return result
    lanes = 32 if k >= 32 and x.stride(-1) == 1 else 1
    source = f"""
    using I=long long; using T={scalar_type(x.dtype)};
    extern "C" __global__ void run(const T* x,const T* y,T* out,I count) {{
        constexpr int lanes={lanes};
        I thread=I(blockIdx.x)*blockDim.x+threadIdx.x,lane=thread%lanes;
        for(I i=thread/lanes;i<count;i+=I(gridDim.x)*blockDim.x/lanes) {{
            I row=i/{m * n}, a=(i/{n})%{m}, b=i%{n};
            const T* left=x+({broadcast_offset(x, shape, 2)})+a*{x.stride(-2)};
            const T* right=y+({broadcast_offset(y, shape, 2)})+b*{y.stride(-1)};
            T sum=0;
            #pragma unroll 1
            for(I j=lane;j<{k};j+=lanes)
                sum+=left[j*{x.stride(-1)}]*right[j*{y.stride(-2)}];
            if constexpr(lanes==32) {{
                #pragma unroll
                for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
            }}
            if(!lane) out[i]=sum;
        }}
    }}
    """
    launch(source, (x, y, result), (result.numel(),), result.numel() * lanes)
    return result


@channel_mix.register_fake
def channel_mix_fake(x, y):
    return x.new_empty(
        (*torch.broadcast_shapes(x.shape[:-2], y.shape[:-2]), x.size(-2), y.size(-1))
    )


def mix_context(ctx, inputs, output):
    ctx.save_for_backward(*inputs)


def mix_backward(ctx, gradient):
    x, y = ctx.saved_tensors
    return (
        channel_mix(gradient, y.transpose(-1, -2)).sum_to_size(x.shape)
        if ctx.needs_input_grad[0]
        else None,
        channel_mix(x.transpose(-1, -2), gradient).sum_to_size(y.shape)
        if ctx.needs_input_grad[1]
        else None,
    )


channel_mix.register_autograd(mix_backward, setup_context=mix_context)


def dimensions(degrees):
    l1, l2, l3 = degrees
    k, odd = divmod(l1 + l2 - l3, 2)
    return 3 ** (l1 - k - odd), 3**k, 3 ** (l2 - k - odd), odd


@torch.library.custom_op(
    "eqx::cartesian_contract_packed", mutates_args=(), device_types="cuda"
)
def packed_contract(
    x: torch.Tensor,
    y: torch.Tensor,
    degrees: list[int],
    role: int,
    indices: torch.Tensor,
    permutation: torch.Tensor,
    offsets: torch.Tensor,
    norm: torch.Tensor,
) -> torch.Tensor:
    """Contract directly into normalized permutation classes, or transpose the map."""
    validate_uniform((x, y, norm))
    if len(degrees) != 3 or role not in (0, 1, 2):
        raise ValueError("Expected three ranks and an output leg in {0, 1, 2}.")
    left, inner, right, odd = dimensions(degrees)
    p = 1 + 2 * odd
    width = math.comb(degrees[2] + 2, 2)
    if (
        indices.shape != (left * p * right,)
        or permutation.shape != indices.shape
        or offsets.shape != (width + 1,)
        or norm.shape != (width,)
        or any(v.dtype != torch.long for v in (indices, permutation, offsets))
    ):
        raise ValueError("Invalid Cartesian permutation classes.")
    widths = [3 ** degrees[0], 3 ** degrees[1], width]
    output = widths.pop(role)
    if [x.size(-1), y.size(-1)] != widths:
        raise ValueError("Operand widths do not match the Cartesian ranks.")
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    result = x.new_empty((*shape, output))
    if not result.numel():
        return result
    indices, permutation, offsets, norm = (
        v.contiguous() for v in (indices, permutation, offsets, norm)
    )
    if role == 2:
        setup = "I begin=offsets[c], length=(offsets[c+1]-begin)*K;"
        step = "I j=permutation[begin+t/K],u=j/(P*B),v=j%B,h=(j/B)%P,k=t%K;"
        size = left * p * right * inner // width
    elif role == 0:
        setup = f"I u=c/(P*K),h=(c/K)%P,k=c%K,length={right};"
        step, size = "I v=t;", right
    else:
        setup = f"I k=c/(P*B),h=(c/B)%P,v=c%B,length={left};"
        step, size = "I u=t;", left
    addresses = ["(u*P+{h})*K+k", "(k*P+{h})*B+v", "(u*P+{h})*B+v"]
    addresses.pop(role)

    def term(hx, hy):
        a, b = addresses[0].format(h=hx), addresses[1].format(h=hy)
        right_value = (
            f"py[ids[{b}]*{y.stride(-1)}]/norm[ids[{b}]]"
            if role != 2
            else f"py[({b})*{y.stride(-1)}]"
        )
        return f"px[({a})*{x.stride(-1)}]*({right_value})"

    value = term("hp", "hm") + "-" + term("hm", "hp") if odd else term("0", "0")
    scale = (-1 if role == 1 and odd else 1) / math.sqrt(inner * (1 + odd))
    lanes = 32 if size >= 12 else 1
    code = f"""
    using I=long long; using T={scalar_type(x.dtype)};
    extern "C" __global__ void run(const T* x,const T* y,const I* ids,
        const I* permutation,const I* offsets,const T* norm,T* out,I count) {{
        constexpr I K={inner}, B={right}, P={p}, lanes={lanes};
        I thread=I(blockIdx.x)*blockDim.x+threadIdx.x,lane=thread%lanes;
        for(I i=thread/lanes;i<count;i+=I(gridDim.x)*blockDim.x/lanes) {{
            I row=i/{output},c=i%{output};
            const T* px=x+({broadcast_offset(x, shape, 1)});
            const T* py=y+({broadcast_offset(y, shape, 1)});
            {setup}
            T sum=0;
            for(I t=lane;t<length;t+=lanes) {{
                {step}
                I hp=(h+1)%3,hm=(h+2)%3;
                sum+={value};
            }}
            if constexpr(lanes==32) {{
                #pragma unroll
                for(int d=16;d;d/=2) sum+=__shfl_down_sync(0xffffffff,sum,d);
            }}
            if(!lane) out[i]=sum*T({scale:.17g}){"/norm[c]" if role == 2 else ""};
        }}
    }}"""
    launch(
        code,
        (x, y, indices, permutation, offsets, norm, result),
        (result.numel(),),
        result.numel() * lanes,
    )
    return result


@packed_contract.register_fake
def packed_fake(x, y, degrees, role, indices, permutation, offsets, norm):
    width = math.comb(degrees[2] + 2, 2) if role == 2 else 3 ** degrees[role]
    return x.new_empty((*torch.broadcast_shapes(x.shape[:-1], y.shape[:-1]), width))


def packed_context(ctx, inputs, output):
    x, y, ctx.degrees, ctx.role, *tables = inputs
    ctx.save_for_backward(x, y, *tables)


def packed_backward(ctx, gradient):
    x, y, *tables = ctx.saved_tensors

    def operation(a, b, degrees, role):
        return packed_contract(a, b, degrees, role, *tables)

    dx, dy = adjoints(
        operation, x, y, gradient, ctx.degrees, ctx.role, ctx.needs_input_grad[:2]
    )
    return dx, dy, None, None, None, None, None, None


packed_contract.register_autograd(packed_backward, setup_context=packed_context)


def pack_product(x, y, l1, l2, basis):
    return packed_contract(
        x,
        y,
        [l1, l2, basis.rank],
        2,
        basis.indices,
        basis.permutation,
        basis.offsets,
        basis.norm,
    )


@torch.library.custom_op(
    "eqx::cartesian_contract", mutates_args=(), device_types="cuda"
)
def contract(
    x: torch.Tensor, y: torch.Tensor, degrees: list[int], role: int
) -> torch.Tensor:
    """Contract the two other legs of a normalized delta/epsilon product."""
    if len(degrees) != 3 or role not in (0, 1, 2):
        raise ValueError("Expected three degrees and an output leg in {0, 1, 2}.")
    torch_kernel.coupling_size(*degrees)
    validate_uniform((x, y))
    widths = [3**l for l in degrees]
    output = widths.pop(role)
    if [x.size(-1), y.size(-1)] != widths:
        raise ValueError("Cartesian operand widths do not match their degrees.")
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    result = x.new_empty((*shape, output))
    if not result.numel():
        return result

    left, inner, right, odd = dimensions(degrees)
    # x[u,h,k], y[k,h,v], z[u,h,v]; h is the epsilon axis for odd products.
    addresses = ["(u*P+{h})*K+k", "(k*P+{h})*B+v", "(u*P+{h})*B+v"]
    addresses.pop(role)
    ax, ay = addresses

    def product(hx, hy):
        return (
            f"x[({ax.format(h=hx)})*{x.stride(-1)}]"
            f"*y[({ay.format(h=hy)})*{y.stride(-1)}]"
        )

    term = (
        product("hp", "hm") + " - " + product("hm", "hp") if odd else product("0", "0")
    )
    # y's adjoint reverses the order of the epsilon axes.
    scale = (-1 if odd and role == 1 else 1) / math.sqrt(inner * (1 + odd))
    reduce, extent, coordinates = (
        ("v", right, "u=c/(P*K),k=c%K,h=(c/K)%P"),
        ("u", left, "k=c/(P*B),v=c%B,h=(c/B)%P"),
        ("k", inner, "u=c/(P*B),v=c%B,h=(c/B)%P"),
    )[role]
    lanes = 32 if extent >= 32 else 1
    source = f"""
    using I=long long; using T={scalar_type(x.dtype)};
    extern "C" __global__ void run(const T* px,const T* py,T* out,I count) {{
        constexpr I P={3 if odd else 1},K={inner},B={right},lanes={lanes};
        I thread=I(blockIdx.x)*blockDim.x+threadIdx.x,lane=thread%lanes;
        for(I i=thread/lanes;i<count;i+=I(gridDim.x)*blockDim.x/lanes) {{
            I row=i/{output},c=i%{output},{coordinates};
            int hp=(h+1)%3,hm=(h+2)%3;
            const T* x=px+({broadcast_offset(x, shape, 1)});
            const T* y=py+({broadcast_offset(y, shape, 1)});
            T total=0;
            #pragma unroll 1
            for(I {reduce}=lane;{reduce}<{extent};{reduce}+=lanes) total+={term};
            if constexpr(lanes==32) {{
                #pragma unroll
                for(int d=16;d;d/=2) total+=__shfl_down_sync(0xffffffff,total,d);
            }}
            if(!lane) out[i]=total*T({scale:.17g});
        }}
    }}
    """
    launch(source, (x, y, result), (result.numel(),), result.numel() * lanes)
    return result


@contract.register_fake
def fake(x, y, degrees, role):
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1])
    return x.new_empty((*shape, 3 ** degrees[role]))


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


def cartesian_product(x, y, l1, l2, l3):
    """Use literal low-degree paths or indexed contractions at higher ranks."""
    degrees = (l1, l2, l3)
    if not torch_kernel.literal_coupling(degrees, config.CARTESIAN_LITERAL_TERMS):
        if torch_kernel.inline_reduction(x, degrees):
            return torch_kernel.reduced_product(x, y, *degrees)
        if not forward_ad_active():
            return contract(x, y, list(degrees), 2)
    if forward_ad_active():
        return torch_kernel.cartesian_product(x, y, *degrees)
    return angular_product(
        x, y, torch_kernel.normalized_coefficients(l1, l2, l3), 3**l3
    )

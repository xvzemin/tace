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


def dimensions(degrees):
    l1, l2, l3 = degrees
    k, odd = divmod(l1 + l2 - l3, 2)
    return 3 ** (l1 - k - odd), 3**k, 3 ** (l2 - k - odd), odd


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

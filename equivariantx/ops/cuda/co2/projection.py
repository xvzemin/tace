"""Native plane polynomials and broadcast tensor-axis projection."""

import torch

from ...common.cuda.pointwise import broadcast_offset, launch, scalar_type
from ...common.cuda.pointwise import evaluate as pointwise
from ...common.cuda.tensor_axis import evaluate as tensor_axis
from ...common.cuda.tensor_axis import inline
from ...precision import validate_uniform
from ...torch.co2 import projection as torch_kernel


@torch.library.custom_op("eqx::plane_projector", mutates_args=(), device_types="cuda")
def plane_operator(direction: torch.Tensor) -> torch.Tensor:
    """Form I - n n.T in one CUDA launch."""
    validate_uniform((direction,))
    if direction.ndim < 1 or direction.shape[-1] != 3:
        raise ValueError("Plane projectors require vectors of size three.")
    direction = direction.contiguous()
    result = direction.new_empty((*direction.shape[:-1], 3, 3))
    return pointwise(
        result,
        (direction,),
        """
        I row=i/9; int a=(i%9)/3,b=i%3;
        out[i]=T(a==b)-p0[row*3+a]*p0[row*3+b];
    """,
    )


@plane_operator.register_fake
def plane_fake(direction):
    return direction.new_empty((*direction.shape[:-1], 3, 3))


def plane_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[0])


def plane_backward(ctx, gradient):
    (direction,) = ctx.saved_tensors
    return -((gradient + gradient.transpose(-1, -2)) * direction.unsqueeze(-2)).sum(-1)


plane_operator.register_autograd(plane_backward, setup_context=plane_context)


def native_detrace(tensor, plane, rank, coefficient):
    """Apply rank-two or rank-three trace corrections with broadcast strides."""
    validate_uniform((tensor, plane))
    if tensor.shape[-1] != 3**rank or plane.shape[-2:] != (3, 3):
        raise ValueError("Detracing requires matching tensors and 3x3 planes.")
    shape = torch.broadcast_shapes(tensor.shape[:-1], plane.shape[:-2])
    result = tensor.new_empty((*shape, 3**rank))
    if not result.numel():
        return result
    if rank == 3:
        # A warp owns one cubic tensor. Its first 27 lanes share the three
        # traces across all outputs, including for nonsymmetric input.
        row_x = broadcast_offset(tensor, shape, 1)
        row_p = broadcast_offset(plane, shape, 2)
        sx, pa, pb = tensor.stride(-1), *plane.stride()[-2:]
        source = f"""
        using T={scalar_type(tensor.dtype)}; using I=long long;
        extern "C" __global__ void run(const T* x,const T* p,T* out,I rows) {{
            int t=threadIdx.x%32, a=(t/9)%3, b=(t/3)%3, c=t%3;
            for(I row=(I(blockIdx.x)*blockDim.x+threadIdx.x)/32;
                row<rows;row+=I(gridDim.x)*blockDim.x/32) {{
                const T* plane=p+({row_p});
                T v=t<27?x[({row_x})+t*{sx}]:T(0);
                T q=(plane[b*{pa}+c*{pb}]+plane[c*{pa}+b*{pb}])*T(0.5);
                T s=(v+__shfl_sync(0xffffffff,v,9*b+3*a+c)
                      +__shfl_sync(0xffffffff,v,9*b+3*c+a))*q;
                #pragma unroll
                for(int d=1;d<=8;d*=2) {{
                    T other=__shfl_down_sync(0xffffffff,s,d);
                    if(t%9+d<9) s+=other;
                }}
                T u=__shfl_sync(0xffffffff,s,9*a);
                T w=__shfl_sync(0xffffffff,s,9*b);
                T z=__shfl_sync(0xffffffff,s,9*c);
                T ab=__shfl_sync(0xffffffff,q,3*a+b),ac=__shfl_sync(0xffffffff,q,3*a+c);
                if(t<27) out[row*27+t]=v+T({coefficient:.17g}/9.0)*(q*u+ac*w+ab*z);
            }}
        }}"""
        rows = result.numel() // 27
        launch(source, (tensor, plane, result), (rows,), rows * 32)
        return result
    prepare = f"""
        I row=i/{3**rank},xo={broadcast_offset(tensor, shape, 1)},po={broadcast_offset(plane, shape, 2)};
        T p[9];
        for(int a=0;a<3;++a) for(int b=0;b<3;++b)
            p[a*3+b]=(p1[po+a*{plane.stride(-2)}+b*{plane.stride(-1)}]
                       +p1[po+b*{plane.stride(-2)}+a*{plane.stride(-1)}])*T(0.5);
    """
    body = f"""
        T trace=0;
        for(int k=0;k<9;++k) trace+=p0[xo+k*{tensor.stride(-1)}]*p[k];
        out[i]=p0[xo+(i%9)*{tensor.stride(-1)}]+T({coefficient:.17g})*trace*p[i%9];
    """
    return pointwise(result, (tensor, plane), prepare + body)


@torch.library.custom_op(
    "eqx::planar_detrace_two", mutates_args=(), device_types="cuda"
)
def detrace_two(
    tensor: torch.Tensor, plane: torch.Tensor, coefficient: float = -0.5
) -> torch.Tensor:
    """Fuse rank-two trace correction without a dense projection matrix."""
    return native_detrace(tensor, plane, 2, coefficient)


@torch.library.custom_op(
    "eqx::planar_detrace_three", mutates_args=(), device_types="cuda"
)
def detrace_three(
    tensor: torch.Tensor, plane: torch.Tensor, coefficient: float = -0.75
) -> torch.Tensor:
    """Fuse rank-three trace correction without dense basis contractions."""
    return native_detrace(tensor, plane, 3, coefficient)


@detrace_two.register_fake
@detrace_three.register_fake
def detrace_fake(tensor, plane, coefficient=-0.5):
    shape = torch.broadcast_shapes(tensor.shape[:-1], plane.shape[:-2])
    return tensor.new_empty((*shape, tensor.shape[-1]))


def detrace_context(ctx, inputs, output):
    tensor, plane, ctx.coefficient = inputs
    ctx.save_for_backward(tensor, plane)


detrace_two.register_autograd(
    torch_kernel.DetraceTwo.backward, setup_context=detrace_context
)


detrace_three.register_autograd(
    torch_kernel.DetraceThree.backward, setup_context=detrace_context
)


def plane_projector(direction):
    """Select fused plane formation or compiler-visible ordinary algebra."""
    return (
        torch_kernel.plane_projector(direction)
        if inline()
        else plane_operator(direction)
    )


def project(module, tensor, plane):
    """Apply native broadcast contractions along every Cartesian tensor index."""
    if inline() or module.rank == 0:
        return torch_kernel.project(module, tensor, plane)
    for axis in range(module.rank):
        tensor = tensor_axis(tensor, plane, 3**axis, 3 ** (module.rank - axis - 1))
    return tensor


def detrace(module, tensor, plane):
    """Apply the native low-rank polynomial or compact native trace sequence."""
    if inline() or module.rank < 2:
        return torch_kernel.detrace(module, tensor, plane)
    if module.rank == 2:
        return detrace_two(tensor, plane, module.coefficients[1])
    if module.rank == 3:
        return detrace_three(tensor, plane, module.coefficients[1])
    return torch_kernel.compact_detrace(module, tensor, plane, tensor_axis)

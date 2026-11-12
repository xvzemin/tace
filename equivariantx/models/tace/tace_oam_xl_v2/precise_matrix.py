"""Live FP32 matrix products with recursive TF32x3 derivatives.

This opt-in experimental primitive is used to prevent interpolation from
amplifying single-TF32 rounding of radial table samples. It does not cache
parameter values or change the default matrix policy of a model.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def precise_product(A,B,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                    A0:tl.constexpr,A1:tl.constexpr,B0:tl.constexpr,B1:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    acc=tl.full((BM,BN),0.,tl.float32)
    for block in range(tl.cdiv(K,BK*SPLIT)):
        kk=(block*SPLIT+tl.program_id(2))*BK+k
        a=tl.load(A+m[:,None]*A0+kk[None,:]*A1,(m[:,None]<M)&(kk[None,:]<K),0.)
        b=tl.load(B+kk[:,None]*B0+n[None,:]*B1,(kk[:,None]<K)&(n[None,:]<N),0.)
        acc=tl.dot(a,b,acc,input_precision='tf32x3')
    tl.store(O+tl.program_id(2)*M*N+m[:,None]*N+n[None,:],acc,(m[:,None]<M)&(n[None,:]<N))


@torch.library.custom_op('eqx_oam_xl::precise_mm',mutates_args=())
def precise_mm(a:torch.Tensor,b:torch.Tensor)->torch.Tensor:
    """Multiply live matrices with FP32-quality products and FP32 accumulation."""
    if a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[0]:
        raise ValueError('Expected two compatible matrices')
    if a.dtype==torch.float64 or a.device.type=='cpu':return a@b
    if a.dtype!=torch.float32 or b.dtype!=a.dtype:raise ValueError('Expected FP32 operands')
    m,k=a.shape;n=b.shape[1]
    out=a.new_empty((m,n))
    bm,bn,bk=32,64,32
    split=8 if m<=256 and k>1024 else 1
    partial=out if split==1 else a.new_empty((split,m,n))
    precise_product[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
        a,b,partial,m,n,k,*a.stride(),*b.stride(),bm,bn,bk,split,num_warps=4,num_stages=3)
    if split>1:
        from .matrix_adjoint import reduce_products
        reduce_products[(triton.cdiv(m*n,256),)](partial,out,m*n,split,256)
    return out


@precise_mm.register_fake
def fake_precise_mm(a,b):return a.new_empty((a.shape[0],b.shape[1]))


def precise_context(ctx,inputs,output):ctx.save_for_backward(*inputs)


def precise_backward(ctx,gradient):
    a,b=ctx.saved_tensors
    return (precise_mm(gradient,b.T) if ctx.needs_input_grad[0] else None,
            precise_mm(a.T,gradient) if ctx.needs_input_grad[1] else None)


precise_mm.register_autograd(precise_backward,setup_context=precise_context)

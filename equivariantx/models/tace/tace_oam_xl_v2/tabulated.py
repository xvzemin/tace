"""Grouped differentiable table interpolation, including mixed adjoints.

This operator evaluates live trainable tables. It neither freezes a network nor
asserts that interpolation is exact. Geometry plans and interpolation accuracy
are responsibilities of the caller. Products accumulate in FP32 using TF32x3;
the FP64 fallback is available for independent derivative checks.
"""
import os
import torch
import triton
import triton.language as tl


@triton.jit
def table_value(A,B,Y,PTR,TASKS,OUT,ROLE:tl.constexpr,K:tl.constexpr,C:tl.constexpr,
                E:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,
                SPLIT:tl.constexpr,PRECISION:tl.constexpr):
    task=tl.program_id(0)
    group=tl.load(TASKS+task*2)
    begin=tl.load(PTR+group)
    end=tl.load(PTR+group+1)
    row=begin+tl.load(TASKS+task*2+1)+tl.arange(0,BM)
    col=tl.program_id(1)*BN+tl.arange(0,BN)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0.,tl.float32)
    if ROLE==2:
        for offset in range(tl.cdiv(K,BK)):
            k=offset*BK+kk
            a=tl.load(A+row[:,None]*K+k[None,:],(row[:,None]<end)&(k[None,:]<K),0.)
            b=tl.load(B+group*K*C+k[:,None]*C+col[None,:],(k[:,None]<K)&(col[None,:]<C),0.)
            acc=tl.dot(a,b,acc,input_precision=PRECISION)
        tl.store(OUT+row[:,None]*C+col[None,:],acc,(row[:,None]<end)&(col[None,:]<C))
    else:
        for offset in range(tl.cdiv(C,BK*SPLIT)):
            k=(offset*SPLIT+tl.program_id(2))*BK+kk
            y=tl.load(Y+row[:,None]*C+k[None,:],(row[:,None]<end)&(k[None,:]<C),0.)
            b=tl.load(B+group*K*C+col[None,:]*C+k[:,None],(k[:,None]<C)&(col[None,:]<K),0.)
            acc=tl.dot(y,b,acc,input_precision=PRECISION)
        tl.store(OUT+tl.program_id(2)*E*K+row[:,None]*K+col[None,:],acc,(row[:,None]<end)&(col[None,:]<K))


@triton.jit
def table_parameter(A,Y,PTR,OUT,K:tl.constexpr,C:tl.constexpr,
                    BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,PRECISION:tl.constexpr):
    group=tl.program_id(0)
    m=tl.program_id(1)*BM+tl.arange(0,BM)
    n=tl.program_id(2)*BN+tl.arange(0,BN)
    begin,end=tl.load(PTR+group),tl.load(PTR+group+1)
    kk=tl.arange(0,BK)
    acc=tl.full((BM,BN),0.,tl.float32)
    for offset in range(tl.cdiv(end-begin,BK)):
        k=begin+offset*BK+kk
        a=tl.load(A+k[None,:]*K+m[:,None],(k[None,:]<end)&(m[:,None]<K),0.)
        y=tl.load(Y+k[:,None]*C+n[None,:],(k[:,None]<end)&(n[None,:]<C),0.)
        acc=tl.dot(a,y,acc,input_precision=PRECISION)
    tl.store(OUT+group*K*C+m[:,None]*C+n[None,:],acc,(m[:,None]<K)&(n[None,:]<C))


@torch.library.custom_op('eqx_oam_xl::table_contract',mutates_args=())
def table_contract(role:int,a:torch.Tensor,b:torch.Tensor,y:torch.Tensor,
                   ptr:torch.Tensor,tasks:torch.Tensor)->torch.Tensor:
    """Evaluate one output of ``sum_group a[e,k] * b[group,k,c] * y[e,c]``."""
    values=(a,b,y)
    if role not in (0,1,2):raise ValueError('Expected a bilinear role')
    out=torch.empty_like(values[role],memory_format=torch.contiguous_format)
    if not a.shape[0]:return out.zero_()
    if a.dtype==torch.float64 or a.device.type=='cpu':
        positions=ptr.tolist()
        for group,(begin,end) in enumerate(zip(positions,positions[1:])):
            if role==2:out[begin:end]=a[begin:end]@b[group]
            elif role==0:out[begin:end]=y[begin:end]@b[group].T
            else:out[group]=a[begin:end].T@y[begin:end]
        return out
    if a.dtype!=torch.float32 or any(not v.is_contiguous() for i,v in enumerate(values) if i!=role):
        raise ValueError('Native tables require contiguous FP32 operands')
    k,c=b.shape[1:]
    precision=os.environ.get('EQX_XL_TABLE_PRECISION','tf32x3')
    bn=int(os.environ.get('EQX_XL_TABLE_COLUMNS','64'))
    bk=int(os.environ.get('EQX_XL_TABLE_REDUCTION','32'))
    warps=int(os.environ.get('EQX_XL_TABLE_WARPS','4'))
    if role==1:
        table_parameter[(b.shape[0],triton.cdiv(k,32),triton.cdiv(c,bn))](
            a,y,ptr,out,k,c,32,bn,bk,precision,num_warps=warps,num_stages=3)
    else:
        columns=c if role==2 else k
        split=int(os.environ.get('EQX_XL_TABLE_SPLIT','1')) if role==0 else 1
        partial=out if split==1 else out.new_empty((split,*out.shape))
        table_value[(tasks.shape[0],triton.cdiv(columns,bn if role==2 else 32),split)](
            a,b,y,ptr,tasks,partial,role,k,c,a.shape[0],32,bn if role==2 else 32,bk,
            split,precision,num_warps=warps,num_stages=3)
        if split>1:
            from .matrix_adjoint import reduce_products
            reduce_products[(triton.cdiv(out.numel(),256),)](partial,out,out.numel(),split,256)
    return out


@table_contract.register_fake
def fake_table(role,a,b,y,ptr,tasks):
    return torch.empty_like((a,b,y)[role],memory_format=torch.contiguous_format)


def table_context(ctx,inputs,output):
    role,a,b,y,ptr,tasks=inputs
    ctx.role=role
    ctx.save_for_backward(a,b,y,ptr,tasks)


def table_backward(ctx,gradient):
    a,b,y,ptr,tasks=ctx.saved_tensors
    values=[a,b,y];values[ctx.role]=gradient.contiguous()
    gradients=[None,None,None]
    for role in range(3):
        if role!=ctx.role and ctx.needs_input_grad[role+1]:
            gradients[role]=table_contract(role,*values,ptr,tasks)
    return (None,*gradients,None,None)


table_contract.register_autograd(table_backward,setup_context=table_context)


def interpolate_table(weights,table,ptr,tasks):
    """Apply per-group cardinal weights to differentiable network samples."""
    # The inactive operand carries only its output shape; it is never read.
    shape=weights.new_empty(1).expand(weights.shape[0],table.shape[-1])
    return table_contract(2,weights.contiguous(),table.contiguous(),shape,ptr,tasks)


def table_tasks(ptr,block=32):
    """Make static launch tiles from CPU group lengths."""
    positions=ptr.tolist()
    tasks=[(group,start) for group,(begin,end) in enumerate(zip(positions,positions[1:]))
           for start in range(0,end-begin,block)]
    return torch.tensor(tasks,dtype=torch.int64).reshape(-1,2)

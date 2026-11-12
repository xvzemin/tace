"""Experimental two-slice INT8 emulation of a floating-point matrix product.

Inspired by EmuGEMM/Ozaki Scheme I (arXiv:2606.25453). Row and column scales
are recomputed from live values on every call. Two signed base-128 slices
produce three exact INT8 products with INT32 accumulation; reconstruction is
FP32. Omitted lower slice products make this a precision experiment, not an
exact replacement for arbitrary FP32 matrices or a selected model default.
"""
import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def maxima(X,P,R:tl.constexpr,K:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,
           SPLITS:tl.constexpr,BR:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    k=tl.program_id(1)*BK+tl.arange(0,BK)
    x=tl.load(X+r[:,None]*S0+k[None,:]*S1,(r[:,None]<R)&(k[None,:]<K),0.)
    amax=tl.max(tl.abs(x),1)
    tl.store(P+r*SPLITS+tl.program_id(1),amax,r<R)


@triton.jit
def scale_rows(P,S,R:tl.constexpr,SPLITS:tl.constexpr,BS:tl.constexpr):
    row=tl.program_id(0)
    k=tl.arange(0,BS)
    amax=tl.max(tl.load(P+row*SPLITS+k,k<SPLITS,0.),0)
    scale=tl.exp2(7.-tl.ceil(tl.log2(tl.maximum(amax,1.e-30))))
    tl.store(S+row,scale)


@triton.jit
def split_rows(X,S,O,R:tl.constexpr,K:tl.constexpr,S0:tl.constexpr,S1:tl.constexpr,
               BR:tl.constexpr,BK:tl.constexpr):
    r=tl.program_id(0)*BR+tl.arange(0,BR)
    k=tl.program_id(1)*BK+tl.arange(0,BK)
    mask=(r[:,None]<R)&(k[None,:]<K)
    x=tl.load(X+r[:,None]*S0+k[None,:]*S1,mask,0.)
    scale=tl.load(S+r,r<R,1.)
    value=x*scale[:,None]
    high=tl.minimum(tl.maximum(libdevice.nearbyint(value),-128.),127.)
    low=tl.minimum(tl.maximum(libdevice.nearbyint((value-high)*128.),-128.),127.)
    at=r[:,None]*K+k[None,:]
    tl.store(O+at,high.to(tl.int8),mask)
    tl.store(O+R*K+at,low.to(tl.int8),mask)


def split_matrix(value):
    """Encode matrix rows using fresh power-of-two scales and two INT8 slices."""
    if value.ndim!=2 or value.dtype!=torch.float32:raise ValueError('Expected an FP32 matrix')
    rows,k=value.shape
    splits=triton.cdiv(k,512)
    partial=value.new_empty((rows,splits));scale=value.new_empty(rows)
    out=torch.empty((2,rows,k),device=value.device,dtype=torch.int8)
    maxima[(triton.cdiv(rows,16),splits)](value,partial,rows,k,*value.stride(),splits,16,512)
    scale_rows[(rows,)](partial,scale,rows,splits,triton.next_power_of_2(splits))
    split_rows[(triton.cdiv(rows,16),triton.cdiv(k,256))](value,scale,out,rows,k,*value.stride(),16,256)
    return out,scale


@triton.jit
def sliced_product(A,B,SA,SB,O,M:tl.constexpr,N:tl.constexpr,K:tl.constexpr,
                   BM:tl.constexpr,BN:tl.constexpr,BK:tl.constexpr,SPLIT:tl.constexpr):
    m=tl.program_id(0)*BM+tl.arange(0,BM)
    n=tl.program_id(1)*BN+tl.arange(0,BN)
    k=tl.arange(0,BK)
    high=tl.full((BM,BN),0,tl.int32)
    low=tl.full((BM,BN),0,tl.int32)
    for block in range(tl.cdiv(K,BK*SPLIT)):
        kk=(block*SPLIT+tl.program_id(2))*BK+k
        ai=m[:,None]*K+kk[None,:];bi=n[None,:]*K+kk[:,None]
        am=(m[:,None]<M)&(kk[None,:]<K);bm=(n[None,:]<N)&(kk[:,None]<K)
        ah=tl.load(A+ai,am,0);al=tl.load(A+M*K+ai,am,0)
        bh=tl.load(B+bi,bm,0);bl=tl.load(B+N*K+bi,bm,0)
        high=tl.dot(ah,bh,high)
        low=tl.dot(ah,bl,low)
        low=tl.dot(al,bh,low)
    sa=tl.load(SA+m,m<M,1.);sb=tl.load(SB+n,n<N,1.)
    value=((high.to(tl.float32)+low.to(tl.float32)*(1./128.))/sa[:,None])/sb[None,:]
    tl.store(O+tl.program_id(2)*M*N+m[:,None]*N+n[None,:],value,(m[:,None]<M)&(n[None,:]<N))


def integer_mm(a,b,config=(32,64,64,4,1),slices=None):
    """Reconstruct a matrix product; timing must include live slicing costs."""
    m,k=a.shape;kb,n=b.shape
    if k!=kb or k>65535:raise ValueError('Unsupported reduction length or mismatched shapes')
    aa,bb=(split_matrix(a),split_matrix(b.T)) if slices is None else slices
    bm,bn,bk,warps,split=config
    out=a.new_empty((m,n));partial=out if split==1 else a.new_empty((split,m,n))
    sliced_product[(triton.cdiv(m,bm),triton.cdiv(n,bn),split)](
        aa[0],bb[0],aa[1],bb[1],partial,m,n,k,bm,bn,bk,split,num_warps=warps,num_stages=3)
    if split>1:
        from .matrix_adjoint import reduce_products
        reduce_products[(triton.cdiv(m*n,256),)](partial,out,m*n,split,256)
    return out

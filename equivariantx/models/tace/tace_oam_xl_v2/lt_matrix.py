"""Select a live TF32 GEMM algorithm for wide radial parameter reductions.

This opt-in post-autodiff operator retains FP32 tensors and accumulation.
Plans contain layouts/algorithm choices, never cached parameter values or
output buffers. Calibration runs before CUDA Graph capture.
"""
import ctypes as ct
from functools import lru_cache
import os
import json
from pathlib import Path
import torch


class Algorithm(ct.Structure):
    _fields_=[('data',ct.c_uint64*8)]


class Heuristic(ct.Structure):
    _fields_=[('algo',Algorithm),('workspace',ct.c_size_t),('status',ct.c_int),
              ('waves',ct.c_float),('reserved',ct.c_int*4)]


def checked(status,operation):
    if status:raise RuntimeError(f'{operation}: cuBLASLt status {status}')


@lru_cache(maxsize=1)
def library():
    directory=Path(torch.__file__).resolve().parent.parent/'nvidia/cu13/lib'
    lib=ct.CDLL(str(directory/'libcublasLt.so.13'))
    P,I,Q,S=ct.c_void_p,ct.c_int,ct.c_uint64,ct.c_size_t
    signatures={
        'cublasLtCreate':[ct.POINTER(P)],
        'cublasLtMatmulDescCreate':[ct.POINTER(P),I,I],
        'cublasLtMatmulDescSetAttribute':[P,I,P,S],
        'cublasLtMatrixLayoutCreate':[ct.POINTER(P),I,Q,Q,ct.c_int64],
        'cublasLtMatrixLayoutSetAttribute':[P,I,P,S],
        'cublasLtMatmulPreferenceCreate':[ct.POINTER(P)],
        'cublasLtMatmulPreferenceSetAttribute':[P,I,P,S],
        'cublasLtMatmulAlgoGetHeuristic':[P,P,P,P,P,P,P,I,ct.POINTER(Heuristic),ct.POINTER(I)],
        'cublasLtMatmul':[P,P,P,P,P,P,P,P,P,P,P,P,ct.POINTER(Algorithm),P,S,P],
        'cublasLtDestroy':[P], 'cublasLtMatmulDescDestroy':[P],
        'cublasLtMatrixLayoutDestroy':[P], 'cublasLtMatmulPreferenceDestroy':[P],
    }
    for name,types in signatures.items():
        getattr(lib,name).argtypes=types;getattr(lib,name).restype=I
    return lib


class MatrixPlan:
    def __init__(self,left,right,workspace_bytes=128*1024*1024,candidates=32,sm_count=0):
        if left.dtype != torch.float32 or right.dtype != torch.float32:
            raise TypeError("The calibrated TF32 plan requires FP32 operands")
        self.lib=lib=library();self.handle=ct.c_void_p();self.desc=ct.c_void_p()
        checked(lib.cublasLtCreate(ct.byref(self.handle)),'create')
        checked(lib.cublasLtMatmulDescCreate(ct.byref(self.desc),77,0),'desc')
        if sm_count:
            value=ct.c_int(sm_count)
            checked(lib.cublasLtMatmulDescSetAttribute(self.desc,15,ct.byref(value),ct.sizeof(value)),'SM target')
        self.layouts=[]
        def layout(tensor):
            if tensor.stride(1)==1:rows,cols=tensor.shape;leading=tensor.stride(0);trans=0
            elif tensor.stride(0)==1:cols,rows=tensor.shape;leading=tensor.stride(1);trans=1
            else:raise ValueError('Dense matrices or transposes required')
            pointer=ct.c_void_p()
            checked(lib.cublasLtMatrixLayoutCreate(ct.byref(pointer),0,
                rows,cols,leading),'layout')
            order=ct.c_int(1)
            checked(lib.cublasLtMatrixLayoutSetAttribute(pointer,1,ct.byref(order),ct.sizeof(order)),'order')
            self.layouts.append(pointer)
            return pointer,trans
        example=left.new_empty((left.shape[0],right.shape[1]),dtype=torch.float32)
        self.a,ta=layout(left);self.b,tb=layout(right);self.d,_=layout(example)
        for attribute,value in ((3,ta),(4,tb)):
            datum=ct.c_int(value)
            checked(lib.cublasLtMatmulDescSetAttribute(self.desc,attribute,ct.byref(datum),ct.sizeof(datum)),'transpose')
        self.pref=ct.c_void_p()
        checked(lib.cublasLtMatmulPreferenceCreate(ct.byref(self.pref)),'preference')
        capacity=ct.c_size_t(workspace_bytes)
        checked(lib.cublasLtMatmulPreferenceSetAttribute(self.pref,1,ct.byref(capacity),ct.sizeof(capacity)),'workspace size')
        self.workspace=torch.empty(workspace_bytes,device=left.device,dtype=torch.uint8)
        self.algorithms=(Heuristic*candidates)();count=ct.c_int()
        checked(lib.cublasLtMatmulAlgoGetHeuristic(self.handle,self.desc,self.a,self.b,self.d,self.d,
            self.pref,candidates,self.algorithms,ct.byref(count)),'heuristics')
        self.count=count.value
        self.alpha,self.beta=ct.c_float(1.),ct.c_float(0.)
    def run(self,left,right,index,out=None):
        if out is None:out=left.new_empty((left.shape[0],right.shape[1]),dtype=torch.float32)
        algorithm=self.algorithms[index]
        checked(self.lib.cublasLtMatmul(self.handle,self.desc,ct.byref(self.alpha),
            left.data_ptr(),self.a,right.data_ptr(),self.b,ct.byref(self.beta),
            out.data_ptr(),self.d,out.data_ptr(),self.d,ct.byref(algorithm.algo),
            self.workspace.data_ptr(),self.workspace.numel(),torch.cuda.current_stream().cuda_stream),'matmul')
        return out
    def close(self):
        for pointer in self.layouts:checked(self.lib.cublasLtMatrixLayoutDestroy(pointer),'destroy layout')
        checked(self.lib.cublasLtMatmulPreferenceDestroy(self.pref),'destroy preference')
        checked(self.lib.cublasLtMatmulDescDestroy(self.desc),'destroy desc')
        checked(self.lib.cublasLtDestroy(self.handle),'destroy handle')



PLANS = {}


def eligible(left, right):
    """Limit calibration to the measured long-reduction radial shape family."""
    return (left.dtype == right.dtype == torch.float32 and left.ndim == right.ndim == 2
            and left.shape[0] == 128 and right.shape[1] == 5376
            and left.shape[1] == right.shape[0] and left.shape[1] >= 8192
            and left.stride() == (1,128) and right.stride() == (5376,1)
            and left.data_ptr()%256 == 0 and right.data_ptr()%256 == 0
            and torch.backends.cuda.matmul.fp32_precision == 'tf32')


def calibrate(left, right):
    """Choose among valid algorithms, checking their current live outputs."""
    target=int(os.environ.get('EQX_XL_LT_SM_TARGET','0'))
    requested=int(os.environ.get('EQX_XL_LT_ALGORITHM','-1'))
    plan = MatrixPlan(left,right,workspace_bytes=32*1024*1024,candidates=8,sm_count=target)
    expected = torch.mm(left,right)
    output = torch.empty_like(expected)
    def timing(call):
        call()
        start,end = torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(8):call()
        end.record();end.synchronize()
        return start.elapsed_time(end)/8
    baseline = timing(lambda:torch.mm(left,right,out=output))
    choices = []
    for index in range(plan.count):
        if plan.algorithms[index].status:continue
        try:
            actual=plan.run(left,right,index,output)
            error=((actual.double()-expected.double()).norm()/expected.double().norm().clamp_min(1e-30)).item()
            if not error <= 1e-4:continue
            elapsed=timing(lambda:plan.run(left,right,index,output))
            choices.append((elapsed,index,error))
        except RuntimeError:
            continue
    plan.choice=min(choices)[1] if choices and min(choices)[0] < baseline*.97 else -1
    if requested>=0:
        if requested not in {index for _,index,_ in choices}:
            raise RuntimeError('The requested overlapped GEMM algorithm failed live validation')
        plan.choice=requested
    plan.measurements=dict(reference_ms=baseline,candidates=choices,selected=plan.choice,
                           sm_target=target,requested_algorithm=requested)
    return plan


def selected_mm(left, right):
    """Reuse only algorithm plans; every call reads current operand values."""
    from .precision import require_tensors
    require_tensors((left, right))
    if os.environ.get('EQX_XL_LT_RADIAL','0') != '1' or not eligible(left,right):
        return torch.mm(left,right)
    stream=torch.cuda.current_stream(left.device)
    # Separate workspaces for independent streams; repeated calls on a stream
    # are ordered. Including stream avoids concurrent scratchpad reuse.
    key=(left.device.index,stream.cuda_stream,tuple(left.shape),tuple(right.shape),
         int(os.environ.get('EQX_XL_LT_SM_TARGET','0')),int(os.environ.get('EQX_XL_LT_ALGORITHM','-1')))
    plan=PLANS.get(key)
    if plan is None:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('Warm up the live radial GEMM plan on this stream before capture')
        plan=calibrate(left,right);PLANS[key]=plan
        manifest=os.environ.get('EQX_XL_LT_MANIFEST')
        if manifest:
            Path(manifest).write_text(json.dumps([
                dict(key=list(key),**value.measurements) for key,value in PLANS.items()],indent=2)+'\n')
    return torch.mm(left,right) if plan.choice<0 else plan.run(left,right,plan.choice)


@torch.library.custom_op('eqx_oam_xl::lt_radial_mm',mutates_args=(),device_types='cuda')
def lt_radial_mm(left:torch.Tensor,right:torch.Tensor)->torch.Tensor:
    """Execute an already differentiated matrix product under its TF32 policy."""
    return selected_mm(left,right)


@lt_radial_mm.register_fake
def fake_lt_radial_mm(left,right):
    return left.new_empty((left.shape[0],right.shape[1]))


def select_radial_matrices(graph):
    """Expose only the measured static output dimensions to the custom plan."""
    from .parallel import hint
    from .program import refresh_inputs
    count=0
    for node in graph.nodes:
        if node.target != torch.ops.aten.mm.default or node.kwargs:continue
        a,b=(item.meta.get('val') for item in node.args)
        if (isinstance(a,torch.Tensor) and isinstance(b,torch.Tensor)
                and a.dtype == b.dtype == torch.float32 and a.ndim == b.ndim == 2
                and hint(a.shape[0]) == 128 and hint(b.shape[1]) == 5376
                and hint(a.shape[1]) >= 8192):
            node.target=torch.ops.eqx_oam_xl.lt_radial_mm.default
            refresh_inputs(node);count+=1
    graph.lint()
    return count


def warm_capacity(edges, device):
    """Calibrate new matrix sizes without retaining an eager whole-model tape.

    Only streams that already executed an eligible radial product participate.
    Private seeded probe operands are discarded; training RNG is untouched.
    """
    if os.environ.get('EQX_XL_LT_RADIAL','0')!='1' or edges<8192:return
    from .parallel import matrix_stream
    parent=torch.cuda.current_stream(device)
    streams=(parent,matrix_stream(device,parent.cuda_stream))
    for stream in streams:
        previous=[key for key in PLANS if key[:2]==(device.index,stream.cuda_stream)]
        key=(device.index,stream.cuda_stream,(128,edges),(edges,5376),
             int(os.environ.get('EQX_XL_LT_SM_TARGET','0')),int(os.environ.get('EQX_XL_LT_ALGORITHM','-1')))
        if not previous or key in PLANS:continue
        stream.wait_stream(parent)
        with torch.cuda.stream(stream):
            generator=torch.Generator(device=device).manual_seed(829)
            left=torch.empty(edges,128,device=device).normal_(std=.05,generator=generator).T
            right=torch.empty(edges,5376,device=device).normal_(std=.05,generator=generator)
            selected_mm(left,right)
        parent.wait_stream(stream)

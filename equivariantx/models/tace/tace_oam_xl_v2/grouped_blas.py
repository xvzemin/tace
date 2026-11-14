"""Process-local grouped cuBLAS matrix products for explicit gradient graphs.

The GPU pointer table is refreshed on every invocation, including capture.
Only shapes, strides and compiled pointer-writing code are cached. All learned
values and outputs remain live tensors. This experimental backend is opt-in.
"""

from collections import defaultdict
import ctypes as ct
from functools import lru_cache
import torch

from equivariantx.ops.common.cuda.runtime import kernels, runtime
from .blas import library


@lru_cache(maxsize=1)
def grouped_library():
    lib = library()
    ptr = ct.c_void_p
    integer = ct.c_int
    lib.cublasSetStream_v2.argtypes = [ptr, ptr]
    lib.cublasGemmGroupedBatchedEx.argtypes = [
        ptr,
        ptr,
        ptr,
        ptr,
        ptr,
        ptr,
        ptr,
        ptr,
        integer,
        ptr,
        ptr,
        integer,
        ptr,
        ptr,
        ptr,
        integer,
        ptr,
        integer,
        ptr,
        integer,
    ]
    return lib


def layout(value):
    if value.ndim != 2:
        raise ValueError("Expected a matrix")
    if value.stride(1) == 1:
        return 0, value.stride(0)
    if value.stride(0) == 1:
        return 1, value.stride(1)
    raise ValueError("Grouped BLAS needs row-major or column-major dense strides")


@lru_cache(maxsize=32)
def pointer_kernel(count, device):
    arguments = ",".join(f"const float* p{i}" for i in range(count * 3))
    body = "".join(
        f"if(threadIdx.x=={i}) out[{i}]=(unsigned long long)p{i};"
        for i in range(count * 3)
    )
    code = (
        f'extern "C" __global__ void run({arguments},unsigned long long* out){{{body}}}'
    )
    return kernels([code], device)[code]


class GroupedMatrices:
    """Plan compatible independent FP32 products without storing input tensors."""

    def __init__(self, left, right):
        if not left or len(left) != len(right):
            raise ValueError("Expected nonempty paired operands")
        groups = defaultdict(list)
        self.signatures = []
        for i, (a, b) in enumerate(zip(left, right)):
            if a.dtype != torch.float32 or b.dtype != a.dtype or b.device != a.device:
                raise ValueError(
                    "This grouped backend currently requires FP32 on one device"
                )
            m, k = a.shape
            kb, n = b.shape
            if k != kb:
                raise ValueError("Matrix dimensions disagree")
            ta, lda = layout(a)
            tb, ldb = layout(b)
            signature = (tb, ta, n, m, k, ldb, lda, n)
            groups[signature].append(i)
            self.signatures.append(
                (tuple(a.shape), tuple(a.stride()), tuple(b.shape), tuple(b.stride()))
            )
        self.order = tuple(i for group in groups.values() for i in group)
        self.groups = len(groups)
        self.count = len(left)
        integer = lambda values: (ct.c_int * len(values))(*values)
        self.columns = [integer([key[i] for key in groups]) for i in range(8)]
        self.sizes = integer([len(group) for group in groups.values()])
        self.alpha = (ct.c_float * self.groups)(*[1.0] * self.groups)
        self.beta = (ct.c_float * self.groups)(*[0.0] * self.groups)
        self.device = left[0].device

    def __call__(self, left, right):
        if len(left) != self.count or len(right) != self.count:
            raise ValueError("Matrix count changed")
        for a, b, signature in zip(left, right, self.signatures):
            if (
                tuple(a.shape),
                tuple(a.stride()),
                tuple(b.shape),
                tuple(b.stride()),
            ) != signature:
                raise ValueError(
                    "Grouped matrix shape or stride changed; build a new plan"
                )
        out = [a.new_empty((a.shape[0], b.shape[1])) for a, b in zip(left, right)]
        pointers = torch.empty(self.count * 3, device=self.device, dtype=torch.int64)
        ordered = [values[i] for values in (right, left, out) for i in self.order]
        kernel = pointer_kernel(self.count, self.device)
        args = [x.data_ptr() for x in ordered] + [pointers.data_ptr()]
        current = torch.cuda.current_stream(self.device)
        runtime().launch([(kernel, args, 1, 1, 128, 0)], current.cuda_stream)
        lib = grouped_library()
        handle = torch.cuda.current_blas_handle()
        status = lib.cublasSetStream_v2(handle, current.cuda_stream)
        if status:
            raise RuntimeError(f"cuBLAS grouped stream: {status}")
        ta, tb, m, n, k, lda, ldb, ldc = self.columns
        status = lib.cublasGemmGroupedBatchedEx(
            handle,
            ta,
            tb,
            m,
            n,
            k,
            self.alpha,
            pointers.data_ptr(),
            0,
            lda,
            pointers.data_ptr() + self.count * 8,
            0,
            ldb,
            self.beta,
            pointers.data_ptr() + self.count * 16,
            0,
            ldc,
            self.groups,
            self.sizes,
            77,
        )
        if status:
            raise RuntimeError(f"cublasGemmGroupedBatchedEx: {status}")
        return out

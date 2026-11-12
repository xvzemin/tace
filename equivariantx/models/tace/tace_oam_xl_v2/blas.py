"""Process-local cuBLAS heuristics for concurrent kernel experiments."""
from contextlib import contextmanager
import ctypes
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def library():
    path=Path(torch.__file__).resolve().parents[1]/'nvidia/cu13/lib/libcublas.so.13'
    lib=ctypes.CDLL(str(path))
    lib.cublasSetSmCountTarget.argtypes=[ctypes.c_void_p,ctypes.c_int]
    lib.cublasGetSmCountTarget.argtypes=[ctypes.c_void_p,ctypes.POINTER(ctypes.c_int)]
    return lib


@contextmanager
def sm_target(count):
    """Temporarily select available-SM heuristics on the current stream handle."""
    lib=library();handle=torch.cuda.current_blas_handle();previous=ctypes.c_int()
    status=lib.cublasGetSmCountTarget(handle,ctypes.byref(previous))
    if status:raise RuntimeError(f'cublasGetSmCountTarget: {status}')
    status=lib.cublasSetSmCountTarget(handle,count)
    if status:raise RuntimeError(f'cublasSetSmCountTarget: {status}')
    try:yield
    finally:
        status=lib.cublasSetSmCountTarget(handle,previous.value)
        if status:raise RuntimeError(f'cuBLAS SM restoration: {status}')

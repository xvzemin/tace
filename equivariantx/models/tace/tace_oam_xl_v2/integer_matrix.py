"""Experimental two-slice INT8 emulation of a floating-point matrix product.

Inspired by EmuGEMM/Ozaki Scheme I (arXiv:2606.25453). Row and column scales
are recomputed from live values on every call. Two signed base-128 slices
produce three exact INT8 products with INT32 accumulation; reconstruction is
FP32. Omitted lower slice products make this a precision experiment, not an
exact replacement for arbitrary FP32 matrices or a selected model default.
"""

import torch


def split_matrix(value):
    """Encode matrix rows with power-of-two scales and two signed INT8 slices."""
    if value.ndim != 2 or value.dtype != torch.float32:
        raise ValueError("Expected an FP32 matrix.")
    maximum = value.abs().amax(-1).clamp_min(1e-30)
    scale = torch.exp2(7.0 - torch.ceil(torch.log2(maximum)))
    scaled = value * scale[:, None]
    high = scaled.round().clamp(-128, 127)
    low = ((scaled - high) * 128).round().clamp(-128, 127)
    return torch.stack((high, low)).to(torch.int8), scale


def integer_mm(a, b, slices=None):
    """Reconstruct the three integer products of a two-slice approximation."""
    if a.shape[1] != b.shape[0] or a.shape[1] > 65535:
        raise ValueError("Unsupported reduction length or mismatched shapes.")
    aa, bb = (split_matrix(a), split_matrix(b.T)) if slices is None else slices
    ah, al = aa[0].unbind(0)
    bh, bl = bb[0].unbind(0)

    def product(x, y):
        if x.is_cuda:
            return torch._int_mm(x.contiguous(), y.T.contiguous())
        return x.to(torch.int32) @ y.to(torch.int32).T

    high = product(ah, bh)
    low = product(ah, bl) + product(al, bh)
    return (high.to(a.dtype) + low.to(a.dtype) / 128) / aa[1][:, None] / bb[1][None, :]

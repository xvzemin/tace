"""Experimental Tensor Core sum of the two mixed linear weight adjoints.

The two products keep independent FP32 accumulators. Split-K partials are
reduced explicitly; no concatenated matrix or intermediate full gradient is
written. TF32 operands use nearest rounding, as in the production policy.
"""

import torch
import triton
import triton.language as tl

from .linear_kernel import rounded_tf32


@triton.jit
def paired_product(
    A,
    B,
    C,
    D,
    O,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    A0: tl.constexpr,
    A1: tl.constexpr,
    B0: tl.constexpr,
    B1: tl.constexpr,
    C0: tl.constexpr,
    C1: tl.constexpr,
    D0: tl.constexpr,
    D1: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    start = tl.program_id(2)
    k = tl.arange(0, BK)
    first = tl.full((BM, BN), 0.0, tl.float32)
    second = tl.full((BM, BN), 0.0, tl.float32)
    for block in range(tl.cdiv(K, BK * SPLIT)):
        kk = (block * SPLIT + start) * BK + k
        a = tl.load(
            A + m[:, None] * A0 + kk[None, :] * A1,
            (m[:, None] < M) & (kk[None, :] < K),
            0.0,
        )
        b = tl.load(
            B + kk[:, None] * B0 + n[None, :] * B1,
            (kk[:, None] < K) & (n[None, :] < N),
            0.0,
        )
        c = tl.load(
            C + m[:, None] * C0 + kk[None, :] * C1,
            (m[:, None] < M) & (kk[None, :] < K),
            0.0,
        )
        d = tl.load(
            D + kk[:, None] * D0 + n[None, :] * D1,
            (kk[:, None] < K) & (n[None, :] < N),
            0.0,
        )
        first = tl.dot(rounded_tf32(a), rounded_tf32(b), first, input_precision="tf32")
        second = tl.dot(
            rounded_tf32(c), rounded_tf32(d), second, input_precision="tf32"
        )
    tl.store(
        O + start * M * N + m[:, None] * N + n[None, :],
        first + second,
        (m[:, None] < M) & (n[None, :] < N),
    )


@triton.jit
def reduce_products(P, O, SIZE: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    parts = tl.arange(0, triton.next_power_of_2(SPLIT))
    values = tl.load(
        P + parts[:, None] * SIZE + index[None, :],
        (parts[:, None] < SPLIT) & (index[None, :] < SIZE),
        0.0,
    )
    tl.store(O + index, tl.sum(values, 0), index < SIZE)


def matrix_adjoint(a, b, c, d, config=(32, 64, 32, 4, 4)):
    """Return ``a @ b + c @ d`` for equal-shaped FP32 matrix pairs."""
    if any(x.dtype != torch.float32 or x.ndim != 2 for x in (a, b, c, d)):
        raise ValueError("Paired matrix adjoints currently require FP32 matrices")
    if a.shape != c.shape or b.shape != d.shape or a.shape[1] != b.shape[0]:
        raise ValueError("Paired matrix dimensions must agree")
    m, k = a.shape
    n = b.shape[1]
    bm, bn, bk, warps, split = config
    output = a.new_empty((m, n))
    partial = output if split == 1 else a.new_empty((split, m, n))
    paired_product[(triton.cdiv(m, bm), triton.cdiv(n, bn), split)](
        a,
        b,
        c,
        d,
        partial,
        m,
        n,
        k,
        *a.stride(),
        *b.stride(),
        *c.stride(),
        *d.stride(),
        bm,
        bn,
        bk,
        split,
        num_warps=warps,
        num_stages=3,
    )
    if split > 1:
        reduce_products[(triton.cdiv(m * n, 256),)](partial, output, m * n, split, 256)
    return output

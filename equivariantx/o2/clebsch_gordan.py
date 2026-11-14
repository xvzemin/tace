"""Real O(2) coupling maps shared by tensor products, gates and contractions."""

import math

import torch

from ..ops import resolve_backend
from .irreps import Irrep


def coupling_entries(ir1, ir2, ir_out):
    """Return analytic ``(input1, input2, output, coefficient)`` entries."""
    if not ir1.m or not ir2.m:
        scalar, tensor = (ir1, ir2) if not ir1.m else (ir2, ir1)
        odd = scalar.p == -1 and tensor.m > 0
        entries = []
        for k in range(tensor.dim):
            index = 1 - k if odd else k
            i, j = (0, index) if not ir1.m else (index, 0)
            entries.append((i, j, k, -1.0 if odd and k == 0 else 1.0))
        return tuple(entries)
    scale = math.sqrt(0.5)
    if ir_out.m == ir1.m + ir2.m:
        return ((0, 0, 0, scale), (1, 1, 0, -scale), (0, 1, 1, scale), (1, 0, 1, scale))
    real = ((0, 0, 0, scale), (1, 1, 0, scale))
    if ir_out.is_even_scalar():
        return real
    sign = 1 if ir1.m >= ir2.m else -1
    imaginary = (
        (1, 0, int(bool(ir_out.m)), sign * scale),
        (0, 1, int(bool(ir_out.m)), -sign * scale),
    )
    return real + imaginary if ir_out.m else imaginary


def coupling_type(ir1, ir2, ir_out):
    """Return representative irreps with the same real coupling matrix."""
    if not ir1.m or not ir2.m:
        return tuple(Irrep(int(bool(ir.m)), ir.p) for ir in (ir1, ir2, ir_out))
    if ir_out.m == ir1.m + ir2.m:
        return Irrep(1, 0), Irrep(1, 0), Irrep(2, 0)
    if not ir_out.m:
        return Irrep(1, 0), Irrep(1, 0), Irrep(0, ir_out.p)
    return (
        (Irrep(2, 0), Irrep(1, 0), Irrep(1, 0))
        if ir1.m > ir2.m
        else (Irrep(1, 0), Irrep(2, 0), Irrep(1, 0))
    )


def channel_product(input1, input2, ir1, ir2, ir_out):
    """Couple broadcast channel blocks with trailing angular dimensions."""
    return clebsch_gordan_product(
        input1.transpose(-1, -2),
        ir1,
        input2.transpose(-1, -2),
        ir2,
        ir_out,
        elementwise=True,
        backend="torch",
    ).transpose(-1, -2)


def quarter_turn(features: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Apply the signed basis permutation (a, b) -> (-b, a)."""
    return torch.stack((-features.select(dim, 1), features.select(dim, 0)), dim=dim)


def clebsch_gordan_product(
    input1, ir1, input2, ir2, ir_out, *, elementwise=False, backend=None
):
    """Evaluate one component-normalized real O(2) coupling map.

    Parameters
    ----------
    input1, input2 : torch.Tensor
        Features with trailing axes ``(ir.dim, channels)``.
    ir1, ir2, ir_out : Irrep
        Valid input and output representations.
    elementwise : bool, optional
        Retain one broadcast channel axis instead of both input axes.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution provider.

    Returns
    -------
    torch.Tensor
        Features with trailing ``(ir_out.dim, channels)`` or
        ``(ir_out.dim, channels1, channels2)`` axes.
    """
    backend = resolve_backend(backend, input1.device)
    return clebsch_gordan_products(
        input1, ir1, input2, ir2, (ir_out,), elementwise=elementwise, backend=backend
    )[0]


def clebsch_gordan_products(
    input1, ir1, input2, ir2, irreps_out, *, elementwise, backend=None
):
    """Evaluate output irreps while sharing their real products."""
    backend = resolve_backend(backend, input1.device)
    from ..ops.precision import validate_uniform

    validate_uniform((input1, input2))
    if backend == "cuda":
        if not input1.is_cuda:
            raise ValueError("backend='cuda' requires CUDA inputs.")
        from ..ops.cuda.o2.clebsch_gordan import evaluate
    elif backend == "torch":
        from ..ops.torch.o2.clebsch_gordan import evaluate
    else:
        raise ValueError("backend must be 'torch' or 'cuda'.")
    return evaluate(input1, ir1, input2, ir2, tuple(irreps_out), elementwise)

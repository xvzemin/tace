"""Real O(2) coupling maps shared by tensor products, gates and contractions."""

import math

import torch

from .irreps import Irrep


def quarter_turn(features: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Apply the signed basis permutation (a, b) -> (-b, a)."""
    return torch.stack((-features.select(dim, 1), features.select(dim, 0)), dim=dim)


def clebsch_gordan_product(
    input1: torch.Tensor,
    ir1: Irrep,
    input2: torch.Tensor,
    ir2: Irrep,
    ir_out: Irrep,
    *,
    elementwise: bool = False,
) -> torch.Tensor:
    """Evaluate one component-normalized real O(2) coupling map.

    Inputs have trailing axes ``(ir.dim, mul)``. Elementwise coupling retains
    one broadcast channel axis; otherwise both input channel axes are kept.
    The caller validates the coupling and applies path weights.
    """
    dim = -2 if elementwise else -3
    if not elementwise:
        input1 = input1.unsqueeze(-1)
        input2 = input2.unsqueeze(-2)
    if ir1.m == 0 and ir2.m == 0:
        return input1 * input2
    if ir1.m == 0:
        return input1 * (quarter_turn(input2, dim) if ir1.p == -1 else input2)
    if ir2.m == 0:
        return (quarter_turn(input1, dim) if ir2.p == -1 else input1) * input2

    real1, imag1 = input1.unbind(dim=dim)
    real2, imag2 = input2.unbind(dim=dim)
    scale = math.sqrt(0.5)
    if ir_out.m == ir1.m + ir2.m:
        real = real1 * real2 - imag1 * imag2
        imaginary = real1 * imag2 + imag1 * real2
        return torch.stack((real, imaginary), dim=dim) * scale

    real = real1 * real2 + imag1 * imag2
    imaginary = imag1 * real2 - real1 * imag2
    if ir2.m > ir1.m:
        imaginary = -imaginary
    if ir_out.m > 0:
        return torch.stack((real, imaginary), dim=dim) * scale
    if ir_out.is_even_scalar():
        return real.unsqueeze(dim) * scale
    return imaginary.unsqueeze(dim) * scale

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

    if ir_out.is_even_scalar():
        return (real1 * real2 + imag1 * imag2).unsqueeze(dim) * scale
    imaginary = imag1 * real2 - real1 * imag2
    if ir2.m > ir1.m:
        imaginary = -imaginary
    if ir_out.m > 0:
        real = real1 * real2 + imag1 * imag2
        return torch.stack((real, imaginary), dim=dim) * scale
    return imaginary.unsqueeze(dim) * scale


def clebsch_gordan_products(input1, ir1, input2, ir2, irreps_out, *, elementwise):
    """Evaluate multiple output irreps while sharing the four real products."""
    if len(irreps_out) == 1 or not ir1.m or not ir2.m:
        return tuple(
            clebsch_gordan_product(
                input1, ir1, input2, ir2, ir, elementwise=elementwise
            )
            for ir in irreps_out
        )
    dim = -2 if elementwise else -3
    if not elementwise:
        input1 = input1.unsqueeze(-1)
        input2 = input2.unsqueeze(-2)
    a, b = input1.unbind(dim)
    c, d = input2.unbind(dim)
    ac, bd, ad, bc = a * c, b * d, a * d, b * c
    scale = math.sqrt(0.5)
    result = []
    for ir in irreps_out:
        if ir.m == ir1.m + ir2.m:
            value = torch.stack((ac - bd, ad + bc), dim=dim)
        elif ir.is_even_scalar():
            value = (ac + bd).unsqueeze(dim)
        else:
            imaginary = bc - ad if ir1.m >= ir2.m else ad - bc
            value = (
                torch.stack((ac + bd, imaginary), dim=dim)
                if ir.m
                else imaginary.unsqueeze(dim)
            )
        result.append(value * scale)
    return tuple(result)

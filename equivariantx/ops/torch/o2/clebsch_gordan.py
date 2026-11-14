"""Ordinary real O(2) formulas and sparse operand-only coupling adjoints."""

import math
from functools import lru_cache

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from equivariantx.o2.clebsch_gordan import coupling_entries
from equivariantx.o2.irreps import Irrep

from ... import config
from ...common.sparse_coefficients import bilinear_coefficients
from ...common.torch import literal_bilinear


def quarter_turn(features: torch.Tensor, dim: int = -2) -> torch.Tensor:
    """Apply the signed basis permutation (a, b) -> (-b, a)."""
    return torch.stack((-features.select(dim, 1), features.select(dim, 0)), dim=dim)


def reference(
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


def shared_reference(input1, ir1, input2, ir2, irreps_out, *, elementwise):
    """Evaluate multiple output irreps while sharing the four real products."""
    if len(irreps_out) == 1 or not ir1.m or not ir2.m:
        return tuple(
            reference(input1, ir1, input2, ir2, ir, elementwise=elementwise)
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


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def short_program(left, right, outputs):
    ir1, ir2 = Irrep(*left), Irrep(*right)
    irreps_out = tuple(Irrep(*entry) for entry in outputs)
    entries, width = [], 0
    for ir in irreps_out:
        entries.extend(
            (i, j, width + k, c) for i, j, k, c in coupling_entries(ir1, ir2, ir)
        )
        width += ir.dim
    return (ir1.dim, ir2.dim, width), tuple(entries)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def coefficient_data(left, right, outputs):
    dimensions, entries = short_program(left, right, outputs)
    return bilinear_coefficients(dimensions, entries)


@lru_cache(maxsize=128)
def coefficient_tables(ir1, ir2, irreps_out, dtype, device):
    return tuple(
        torch.tensor(data, dtype=dtype if i % 3 == 2 else torch.long, device=device)
        for i, data in enumerate(
            coefficient_data(
                (ir1.m, ir1.p), (ir2.m, ir2.p), tuple((ir.m, ir.p) for ir in irreps_out)
            )
        )
    )


def sparse(input1, ir1, input2, ir2, irreps_out, elementwise, operation):
    """Share one sparse bilinear map across requested output irreps."""
    tracing = (
        torch.compiler.is_compiling()
        or is_fake(input1)
        or get_proxy_mode() is not None
        or torch._C._are_functorch_transforms_active()
    )
    tables = coefficient_tables.__wrapped__ if tracing else coefficient_tables
    tables = tables(ir1, ir2, tuple(irreps_out), input1.dtype, input1.device)
    x, y = input1.transpose(-1, -2), input2.transpose(-1, -2)
    if not elementwise:
        x, y = x.unsqueeze(-2), y.unsqueeze(-3)
    value = operation(x, y, tables)
    return tuple(
        part.movedim(-1, -2 if elementwise else -3)
        for part in value.split([ir.dim for ir in irreps_out], dim=-1)
    )


def evaluate(input1, ir1, input2, ir2, irreps_out, elementwise=False):
    if config.TORCH_REFERENCE:
        return tuple(
            reference(input1, ir1, input2, ir2, ir, elementwise=elementwise)
            for ir in irreps_out
        )
    compiling = torch.compiler.is_compiling() or get_proxy_mode() is not None
    if not config.TORCH_O2_COUPLING or not compiling or input1.is_cuda:
        return shared_reference(
            input1, ir1, input2, ir2, irreps_out, elementwise=elementwise
        )
    return compiled_coupling(
        input1,
        input2,
        (ir1.m, ir1.p),
        (ir2.m, ir2.p),
        tuple((ir.m, ir.p) for ir in irreps_out),
        elementwise,
        config.TORCH_O2_COUPLING_ADJOINT,
    )


@torch.compiler.allow_in_graph
def compiled_coupling(input1, input2, left, right, outputs, elementwise, adjoint):
    """Keep original channel layout when differentiating short compiled formulas."""
    left, right = tuple(int(v) for v in left), tuple(int(v) for v in right)
    outputs = tuple(tuple(int(v) for v in entry) for entry in outputs)
    irreps_out = tuple(Irrep(*entry) for entry in outputs)
    dimensions, entries = short_program(left, right, outputs)
    x, y = input1, input2
    dim = -2 if elementwise else -3
    if not elementwise:
        x, y = x.unsqueeze(-1), y.unsqueeze(-2)
    operation = literal_bilinear.evaluate if adjoint else literal_bilinear.primal
    value = operation(x, y, dimensions, entries, dim=dim)
    return value.split([ir.dim for ir in irreps_out], dim=dim)

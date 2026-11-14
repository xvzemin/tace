"""Exact Cartesian harmonics without redundant low-rank tensor powers."""

import math
from fractions import Fraction
from functools import lru_cache
from itertools import product

import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config


def degrees(module):
    """Return angular degrees in flattened output-entry order."""
    return tuple(ir.l for mul, ir in module.irreps_out for _ in range(mul))


def scale(degree, normalization):
    """Convert unit-norm harmonics to the requested normalization."""
    if normalization == "component":
        return math.sqrt(2 * degree + 1)
    if normalization == "integral":
        return math.sqrt((2 * degree + 1) / (4 * math.pi))
    return 1.0


@lru_cache(maxsize=None)
def monomial_powers(degree):
    """Cartesian exponent triples of a homogeneous degree, in fixed order."""
    return tuple(
        (a, b, degree - a - b)
        for a in range(degree, -1, -1)
        for b in range(degree - a, -1, -1)
    )


@lru_cache(maxsize=None)
def polynomial_coefficients(degree):
    """Exact rational STF coefficients, indexed by Cartesian exponent counts."""
    powers = monomial_powers(degree)
    denominator = math.prod(range(2 * degree - 1, 0, -2))
    rows = []
    for counts in powers:
        row = {}
        for pairs in product(*(range(count // 2 + 1) for count in counts)):
            paired = sum(pairs)
            coefficient = Fraction(
                (-1) ** paired * math.prod(range(2 * degree - 2 * paired - 1, 0, -2)),
                denominator,
            )
            for count, number in zip(counts, pairs):
                coefficient *= math.factorial(count) // (
                    math.factorial(count - 2 * number)
                    * 2**number
                    * math.factorial(number)
                )
            for radial in monomial_powers(paired):
                multiplicity = math.factorial(paired) // math.prod(
                    math.factorial(value) for value in radial
                )
                power = tuple(
                    count - 2 * number + 2 * value
                    for count, number, value in zip(counts, pairs, radial)
                )
                row[power] = row.get(power, Fraction(0)) + coefficient * multiplicity
        rows.append(
            tuple((powers.index(power), value) for power, value in row.items() if value)
        )
    return tuple(rows)


@lru_cache(maxsize=None)
def polynomial_matrix(degree, normalization):
    """Return a cached CPU FP64 monomial-to-Cartesian STF matrix."""
    powers = monomial_powers(degree)
    matrix = torch.zeros(len(powers), len(powers), dtype=torch.float64, device="cpu")
    normalization_scale = math.sqrt(math.comb(2 * degree, degree) / 2**degree)
    normalization_scale *= scale(degree, normalization)
    for column, row in enumerate(polynomial_coefficients(degree)):
        for index, value in row:
            matrix[index, column] = float(value) * normalization_scale
    indices = [
        powers.index(tuple(entry.count(axis) for axis in range(3)))
        for entry in product(range(3), repeat=degree)
    ]
    return matrix[:, indices].clone()


def polynomial(module, vectors):
    """Share coordinate powers and evaluate each retained high degree once."""
    if module.normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    orders = degrees(module)
    unique = set(orders)
    blocks = {
        degree: low_order(vectors, (degree,), module.normalization)
        for degree in unique
        if degree <= 2
    }
    if 3 in unique:
        values = o3.spherical_harmonics(3, vectors, False, module.normalization)
        blocks[3] = values @ module.basis_3.to(vectors).T
    coordinates = vectors.unbind(-1)
    powers = [[value * 0 + 1] for value in coordinates]
    for coordinate, axis in zip(coordinates, powers):
        for _ in range(module.lmax):
            axis.append(axis[-1] * coordinate)
    for degree in sorted(unique):
        if degree < 4:
            continue
        values = []
        for exponents in monomial_powers(degree):
            terms = [axis[n] for axis, n in zip(powers, exponents) if n]
            value = terms[0]
            for factor in terms[1:]:
                value = value * factor
            values.append(value)
        blocks[degree] = torch.stack(values, -1) @ getattr(
            module, f"polynomial_{degree}"
        ).to(vectors)
    outputs = [blocks[degree] for degree in orders]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)


def low_order(vectors, orders, normalization):
    """Evaluate ranks zero, one, and two by their exact STF formulas."""
    blocks = {}
    if 0 in orders:
        blocks[0] = torch.ones_like(vectors[..., :1])
    if 1 in orders:
        blocks[1] = vectors
    if 2 in orders:
        product = vectors.unsqueeze(-1) * vectors.unsqueeze(-2)
        trace = vectors.square().sum(-1) / 3
        diagonal = torch.diag_embed(trace[..., None].expand(*trace.shape, 3))
        blocks[2] = (product - diagonal).flatten(-2) * math.sqrt(1.5)
    outputs = [blocks[degree] * scale(degree, normalization) for degree in orders]
    if not outputs:
        return vectors[..., :0]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)


def spherical(module, vectors):
    """Evaluate compact harmonics and expand each irreducible block once."""
    orders = degrees(module)
    if not orders:
        return vectors[..., :0]
    values = o3.spherical_harmonics(
        list(orders), vectors, module.normalize, module.normalization
    )
    blocks = values.split([2 * degree + 1 for degree in orders], -1)
    outputs = [
        value @ getattr(module, f"basis_{degree}").to(vectors).T
        for degree, value in zip(orders, blocks)
    ]
    return outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)


def evaluate(module, vectors):
    """Select measured exact formulas while retaining the dense reference."""
    if config.TORCH_REFERENCE or not config.TORCH_CARTESIAN_HARMONICS:
        return reference(module, vectors)
    if module.lmax <= 2:
        if module.normalize:
            vectors = torch.nn.functional.normalize(vectors, dim=-1)
        return low_order(vectors, degrees(module), module.normalization)
    if module.lmax <= 6 and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        if (
            config.TORCH_CARTESIAN_HARMONIC_POLYNOMIAL
            and vectors.device.type == "cpu"
            and module.lmax >= 4
            and all(
                hasattr(module, f"polynomial_{degree}")
                for degree in set(degrees(module))
                if degree >= 4
            )
        ):
            return polynomial(module, vectors)
        return spherical(module, vectors)
    return reference(module, vectors)


def reference(module, vectors):
    """Ordinary full Cartesian powers and dense STF projection."""
    if module.normalize:
        vectors = torch.nn.functional.normalize(vectors, dim=-1)
    powers = [torch.ones_like(vectors[..., :1])]
    for _ in range(module.lmax):
        powers.append((powers[-1].unsqueeze(-1) * vectors.unsqueeze(-2)).flatten(-2))
    values = []
    for mul, ir in module.irreps_out:
        x = (
            powers[ir.l]
            @ getattr(module, f"basis_{ir.l}").to(vectors)
            @ getattr(module, f"scaled_basis_{ir.l}").to(vectors)
        )
        values.extend([x] * mul)
    return torch.cat(values, dim=-1) if values else vectors[..., :0]

"""Independent Cartesian coordinates and symmetric trace-free contractions."""

import math
from collections import defaultdict
from fractions import Fraction
from functools import lru_cache
from itertools import product

import torch

from .irreps import Irreps
from .symmetric import symmetric_powers


@lru_cache(maxsize=None)
def trace_free_components(counts):
    """Express a symmetric trace-free component through at most one z index."""
    x, y, z = counts
    pairs, z = divmod(z, 2)
    return tuple(
        ((x + 2 * k, y + 2 * (pairs - k), z), (-1) ** pairs * math.comb(pairs, k))
        for k in range(pairs + 1)
    )


@lru_cache(maxsize=None)
def trace_free_polynomial(counts):
    """Return exact detracing coefficients for one symmetric component."""
    degree = sum(counts)
    terms = defaultdict(Fraction)
    for pairs in product(*(range(n // 2 + 1) for n in counts)):
        k = sum(pairs)
        coefficient = Fraction(
            (-1) ** k, math.prod(range(2 * degree - 2 * k + 1, 2 * degree, 2))
        )
        for n, t in zip(counts, pairs):
            coefficient *= math.factorial(n) // (
                math.factorial(n - 2 * t) * 2**t * math.factorial(t)
            )
        for a, b, c in symmetric_powers(k):
            powers = tuple(
                n - 2 * t + 2 * p for n, t, p in zip(counts, pairs, (a, b, c))
            )
            terms[powers] += coefficient * math.comb(k, a) * math.comb(k - a, b)
    return tuple((powers, value) for powers, value in sorted(terms.items()) if value)


@lru_cache(maxsize=None)
def trace_free_coupling(l1, l2, l3):
    """Return exact delta/epsilon couplings in independent STF coordinates.

    Parameters
    ----------
    l1, l2, l3 : int
        Input and output degrees satisfying the triangle inequality.

    Returns
    -------
    tuple
        Nonzero ``(i, j, k, coefficient)`` entries. Coefficients are rational
        and exclude path normalization. Component order follows
        ``symmetric_powers(l)`` restricted to at most one z index.
    """
    if min(l1, l2, l3) < 0 or not abs(l1 - l2) <= l3 <= l1 + l2:
        raise ValueError("Degrees must satisfy the triangle inequality.")
    bases = [tuple(p for p in symmetric_powers(l) if p[2] < 2) for l in (l1, l2, l3)]
    lookup1, lookup2 = ({p: i for i, p in enumerate(basis)} for basis in bases[:2])
    k, odd = divmod(l1 + l2 - l3, 2)
    if not odd and k == min(l1, l2):
        # All free indices belong to one STF input. Their symmetry and zero
        # traces are inherited, so output projection is the identity.
        result = []
        for c, powers in enumerate(bases[2]):
            row = defaultdict(int)
            for gamma in symmetric_powers(k):
                count = math.comb(k, gamma[0]) * math.comb(k - gamma[0], gamma[1])
                a = tuple(x + y for x, y in zip(gamma, powers)) if l1 >= l2 else gamma
                b = tuple(x + y for x, y in zip(gamma, powers)) if l2 > l1 else gamma
                for (x, sx), (y, sy) in product(
                    trace_free_components(a), trace_free_components(b)
                ):
                    row[lookup1[x], lookup2[y]] += count * sx * sy
            result.extend(
                (a, b, c, Fraction(value))
                for (a, b), value in sorted(row.items())
                if value
            )
        return tuple(result)
    left, right = l1 - k - odd, l2 - k - odd
    raw = defaultdict(lambda: defaultdict(Fraction))
    epsilon = (
        tuple(
            (c, p, q, sign)
            for c, (a, b) in enumerate(((1, 2), (2, 0), (0, 1)))
            for p, q, sign in ((a, b, 1), (b, a, -1))
        )
        if odd
        else ((-1, -1, -1, 1),)
    )
    for alpha, beta in product(symmetric_powers(left), symmetric_powers(right)):
        multiplicity = (
            math.comb(left, alpha[0])
            * math.comb(left - alpha[0], alpha[1])
            * math.comb(right, beta[0])
            * math.comb(right - beta[0], beta[1])
        )
        for c, p, q, sign in epsilon:
            out = tuple(
                a + b + (axis == c) for axis, (a, b) in enumerate(zip(alpha, beta))
            )
            scale = Fraction(
                sign * multiplicity,
                math.comb(l3, out[0]) * math.comb(l3 - out[0], out[1]),
            )
            for gamma in symmetric_powers(k):
                count = math.comb(k, gamma[0]) * math.comb(k - gamma[0], gamma[1])
                a = tuple(
                    v + w + (axis == p) for axis, (v, w) in enumerate(zip(alpha, gamma))
                )
                b = tuple(
                    v + w + (axis == q) for axis, (v, w) in enumerate(zip(beta, gamma))
                )
                for (x, sx), (y, sy) in product(
                    trace_free_components(a), trace_free_components(b)
                ):
                    raw[out][lookup1[x], lookup2[y]] += scale * count * sx * sy
    result = []
    for c, powers in enumerate(bases[2]):
        row = defaultdict(Fraction)
        for p, scale in trace_free_polynomial(powers):
            for indices, value in raw[p].items():
                row[indices] += scale * value
        result.extend(
            (a, b, c, value) for (a, b), value in sorted(row.items()) if value
        )
    return tuple(result)


@lru_cache(maxsize=None)
def trace_free_basis(l):
    """Return selection indices and an STF reconstruction matrix in CPU float64."""
    if l < 0:
        raise ValueError("The tensor rank must be non-negative.")
    powers = tuple(p for p in symmetric_powers(l) if p[2] < 2)
    lookup = {p: i for i, p in enumerate(powers)}
    indices = []
    for counts in powers:
        index = 0
        for axis, count in enumerate(counts):
            for _ in range(count):
                index = 3 * index + axis
        indices.append(index)
    matrix = torch.zeros(2 * l + 1, 3**l, dtype=torch.float64, device="cpu")
    for j, axes in enumerate(product(range(3), repeat=l)):
        for p, value in trace_free_components(tuple(axes.count(a) for a in range(3))):
            matrix[lookup[p], j] = value
    return torch.tensor(indices, dtype=torch.long, device="cpu"), matrix


class TraceFreeBasis(torch.nn.Module):
    """Pack STF tensors into independent Cartesian components.

    Parameters
    ----------
    irreps : Irreps or str
        Cartesian representation labels and multiplicities.

    Notes
    -----
    Both layouts use ``mul_ir`` order. Each packed rank-l tensor has
    ``2*l+1`` components containing at most one z index. These coordinates
    are not orthonormal. Packing assumes STF inputs; it is not a projection.
    """

    def __init__(self, irreps):
        super().__init__()
        self.irreps = Irreps(irreps)
        self.dim = sum(mul * (2 * ir.l + 1) for mul, ir in self.irreps)
        self.identity = all(ir.l < 2 for _, ir in self.irreps)
        groups, indices, offset = [], [], 0
        for mul, ir in self.irreps:
            selection, _ = trace_free_basis(ir.l)
            indices.extend(
                offset + u * ir.dim + a for u in range(mul) for a in selection.tolist()
            )
            if groups and groups[-1][1] == ir.l:
                groups[-1] = (groups[-1][0] + mul, ir.l)
            else:
                groups.append((mul, ir.l))
            offset += mul * ir.dim
        self.groups = tuple(groups)
        self.widths = tuple(mul * (2 * l + 1) for mul, l in groups)
        self.register_buffer(
            "indices", torch.tensor(indices, dtype=torch.long), persistent=False
        )
        for l in sorted({ir.l for _, ir in self.irreps}):
            self.register_buffer(
                f"expansion_{l}",
                trace_free_basis(l)[1].to(torch.get_default_dtype()).clone(),
                persistent=False,
            )

    def pack(self, features):
        """Select independent components from STF features (..., irreps.dim)."""
        if features.shape[-1] != self.irreps.dim:
            raise ValueError("The feature dimension does not match irreps.")
        return features if self.identity else features.index_select(-1, self.indices)

    def unpack(self, features):
        """Reconstruct full Cartesian tensors from features (..., dim)."""
        if features.shape[-1] != self.dim:
            raise ValueError("The feature dimension does not match packed irreps.")
        if self.identity:
            return features
        return torch.cat(
            [
                (
                    block.reshape(*features.shape[:-1], mul, 2 * l + 1)
                    @ getattr(self, f"expansion_{l}")
                ).flatten(-2)
                for block, (mul, l) in zip(features.split(self.widths, -1), self.groups)
            ],
            dim=-1,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for l in {ir.l for _, ir in self.irreps}:
            name = f"expansion_{l}"
            self._buffers[name] = trace_free_basis(l)[1].to(self._buffers[name]).clone()
        return self

    def extra_repr(self):
        return str(self.irreps)

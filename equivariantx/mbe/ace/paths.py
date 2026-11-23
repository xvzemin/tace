"""Ordered coupling paths for channel-wise tensor products."""

import math
from collections import defaultdict
from typing import NamedTuple, Union

from e3nn import o3


class ContractionInstruction(NamedTuple):
    """Couple a previous-order block and an input block to one output irrep."""

    i_in1: int
    i_in2: int
    ir_out: o3.Irrep


class SymmetricProductPaths:
    """Remove identically zero paths in successive channel-wise self products.

    Parameters
    ----------
    irreps : o3.Irreps
        Representations of the original features. Each entry denotes an
        independent field, including repeated entries of the same irrep.
    """

    def __init__(self, irreps: o3.Irreps) -> None:
        self.irreps_in = o3.Irreps(irreps)
        self.offsets = []
        self.lowering = []
        self.polynomials = []
        for _, ir in self.irreps_in:
            offset = len(self.lowering)
            self.offsets.append(offset)
            self.lowering.extend(range(2 * ir.l, -1, -1))
            self.polynomials.append((((offset,), 1),))
        self._couplings = {}
        self._descendants = {}

    def _couple(self, polynomial, l1: int, j: int, l2: int, l3: int):
        key = (polynomial, l1, j, l3)
        if key in self._couplings:
            return self._couplings[key]

        # In this basis D x_s = (2l - s) x_{s+1}. Apply D by the product rule.
        k = l1 + l2 - l3
        descendants = self._descendants.setdefault(polynomial, [polynomial])
        while len(descendants) <= k:
            lowered = defaultdict(int)
            for monomial, coefficient in descendants[-1]:
                for i, variable in enumerate(monomial):
                    factor = self.lowering[variable]
                    if factor:
                        term = tuple(
                            sorted(monomial[:i] + (variable + 1,) + monomial[i + 1 :])
                        )
                        lowered[term] += coefficient * factor
            descendants.append(
                tuple(sorted((term, value) for term, value in lowered.items() if value))
            )

        # Highest-weight CG contraction, up to a nonzero path-wise factor:
        # sum_q (-1)^q binom(k,q) (2l1-q)!/(2l1-k)! (D^q P) y_{k-q}.
        result = defaultdict(int)
        denominator = math.factorial(2 * l1 - k)
        for q in range(k + 1):
            factor = (
                (-1) ** q
                * math.comb(k, q)
                * (math.factorial(2 * l1 - q) // denominator)
            )
            variable = self.offsets[j] + k - q
            for monomial, coefficient in descendants[q]:
                term = tuple(sorted(monomial + (variable,)))
                result[term] += factor * coefficient
        result = sorted((term, value) for term, value in result.items() if value)
        if result:
            # Normalize only the whole polynomial, not individual descendants.
            divisor = math.gcd(*(value for _, value in result))
            if result[0][1] < 0:
                divisor = -divisor
            result = [(term, value // divisor) for term, value in result]
        result = tuple(result)
        self._couplings[key] = result
        return result

    def filter(self, instructions, irreps_out: o3.Irreps, irreps_base: o3.Irreps):
        """Filter one correlation order and retain its polynomials for the next."""
        paths, irrep_list, polynomials = [], [], []
        for i, j, k, mode, trainable in instructions:
            if mode != "uuu" or trainable:
                raise ValueError("SymmetricProductPaths requires unweighted uuu paths")
            polynomial = self._couple(
                self.polynomials[i],
                self.irreps_in[i].ir.l,
                j,
                irreps_base[j].ir.l,
                irreps_out[k].ir.l,
            )
            if polynomial:
                paths.append((i, j, len(irrep_list), mode, trainable))
                irrep_list.append(irreps_out[k])
                polynomials.append(polynomial)
        self.irreps_in = o3.Irreps(irrep_list)
        self.polynomials = polynomials
        self._couplings.clear()
        self._descendants.clear()
        return paths, self.irreps_in


def satisfy(l1: int, l2: int, restriction: Union[str, None] = None) -> bool:
    if restriction is None:
        return True
    elif restriction == "<":
        return l1 < l2
    elif restriction == "<=":
        return l1 <= l2
    elif restriction == ">":
        return l1 > l2
    elif restriction == ">=":
        return l1 >= l2
    elif restriction == "==":
        return l1 == l2
    elif restriction == "!=":
        return l1 != l2
    else:
        raise ValueError(f"Unknown restriction: {restriction}")


def generate_paths(
    irreps_out: o3.Irreps,
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    *,
    l1l2: Union[str, None] = None,
    l2l3: Union[str, None] = None,
    l3l1: Union[str, None] = None,
    e3nn_mode="uvu",
    trainable: bool = False,
    identical_inputs: bool = False,
):

    e3nn_paths: list[tuple[int, int, int, str, bool]] = []
    e3nn_out_irreps: list[tuple[int, o3.Irrep]] = []

    if identical_inputs and irreps_in1 != irreps_in2:
        raise ValueError("identical_inputs requires matching input irreps")

    for _, (_, ir_out) in enumerate(irreps_out):
        for i, (mul, ir1) in enumerate(irreps_in1):
            for j, (_, ir2) in enumerate(irreps_in2):
                l1 = ir1.l
                l2 = ir2.l
                l3 = ir_out.l

                if (
                    ir_out in ir1 * ir2
                    and satisfy(l1, l2, l1l2)
                    and satisfy(l2, l3, l2l3)
                    and satisfy(l3, l1, l3l1)
                ):
                    if identical_inputs and i == j and (l1 + l2 - l3) % 2 == 1:
                        continue

                    k = len(e3nn_out_irreps)
                    e3nn_out_irreps.append((mul, ir_out))
                    e3nn_paths.append(
                        (i, j, k, e3nn_mode, e3nn_mode == "uvu" or trainable)
                    )

    return e3nn_paths, o3.Irreps(e3nn_out_irreps)


def contraction_paths(irreps_in, outputs, l1l2, selector):
    base = previous = o3.Irreps(irreps_in)
    orders = []
    for target in outputs:
        paths, out = generate_paths(
            o3.Irreps(target), previous, base, l1l2=l1l2, e3nn_mode="uuu"
        )
        if selector is not None:
            paths, out = selector.filter(paths, out, base)
        orders.append(
            tuple(ContractionInstruction(i, j, out[k].ir) for i, j, k, _, _ in paths)
        )
        previous = out
    return tuple(orders)


def symmetric_contraction_paths(irreps_in, outputs, *, l1l2=None):
    """Return ordered uuu paths which survive for repeated inputs.

    Parameters
    ----------
    irreps_in : o3.Irreps
        Input representations.
    outputs : sequence of o3.Irreps
        Allowed outputs for each order, starting at two.
    l1l2 : str, optional
        Relation between the two coupled angular degrees.
    """
    return contraction_paths(irreps_in, outputs, l1l2, SymmetricProductPaths(irreps_in))


def asymmetric_contraction_paths(irreps_in, outputs, *, l1l2=None):
    """Return all ordered uuu paths for independent inputs.

    Parameters
    ----------
    irreps_in : o3.Irreps
        Input representations.
    outputs : sequence of o3.Irreps
        Allowed outputs for each order, starting at two.
    l1l2 : str, optional
        Relation between the two coupled angular degrees.
    """
    return contraction_paths(irreps_in, outputs, l1l2, None)

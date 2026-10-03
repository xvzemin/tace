"""Spherical contractions of transverse irreducible tensors."""

import math
from collections import Counter
from functools import lru_cache

import torch
from e3nn import o3

from ...co2.spherical import coupling_polynomial, coupling_recurrence
from ...kernels.codegen import ScalarProgram, contraction_source
from ...o3 import so3_generators
from ..angular import generator_action


def coupling_source(
    l1, l2, l3, vector, features, normalization, cache, lines, recurrence=False
):
    """Emit a transverse restriction and lift without Cartesian intermediates."""
    from ..o3.harmonics import polynomial_coefficients

    delta = abs(l1 - l3)
    degree = min(l1, l3)
    spectral_scale = degree * (degree + 1) / max(1, degree**2)

    def bridge(features):
        bridge_key = "transverse_bridge", l1, l3, vector, features
        if bridge_key in cache:
            return cache[bridge_key]
        if l1 == l3:
            scale = (2 * l1 + 1) ** -0.5
            values = tuple(
                contraction_source(((scale, (value,)),), cache, lines)
                for value in features
            )
        else:
            cg = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")
            harmonics = tuple(
                contraction_source(
                    tuple(
                        (
                            c,
                            tuple(
                                vector[a]
                                for a, p in enumerate(powers)
                                for _ in range(p)
                            ),
                        )
                        for powers, c in polynomial
                    ),
                    cache,
                    lines,
                )
                for polynomial in polynomial_coefficients(delta, "norm")
            )
            entries = tuple(
                (a, b, c, float(cg[a, b, c])) for a, b, c in cg.nonzero().tolist()
            )
            values = tuple(
                contraction_source(
                    tuple(
                        (scale, (features[a], harmonics[b]))
                        for a, b, c, scale in entries
                        if c == out
                    ),
                    cache,
                    lines,
                )
                for out in range(2 * l3 + 1)
            )
        cache[bridge_key] = values
        return values

    values = features if l1 < l3 else bridge(features)
    if recurrence:
        values = tuple(
            contraction_source(((math.sqrt(2 * delta + 1), (x,)),), cache, lines)
            for x in values
        )
        previous = ("T(0)",) * len(values)
        coefficients = coupling_recurrence(l1, l3, l2)
        odd = (l2 - delta) % 2
        if odd:
            values = tuple(
                contraction_source(((1 / coefficients[0], (x,)),), cache, lines)
                for x in generator_action(
                    degree, vector, values, cache, lines, normalized=True
                )
            )
        for k in range(odd, l2 - delta, 2):
            acted = generator_action(
                degree,
                vector,
                generator_action(degree, vector, values, cache, lines, normalized=True),
                cache,
                lines,
                normalized=True,
            )
            scale = 1 / (coefficients[k] * coefficients[k + 1])
            diagonal = coefficients[k] ** 2 + (coefficients[k - 1] ** 2 if k else 0)
            following = tuple(
                contraction_source(
                    ((scale, (a,)), (scale * diagonal, (b,)))
                    + (
                        ((-scale * coefficients[k - 1] * coefficients[k - 2], (c,)),)
                        if k >= 2
                        else ()
                    ),
                    cache,
                    lines,
                )
                for a, b, c in zip(acted, values, previous)
            )
            previous, values = values, following
        scale = (
            1 / math.sqrt(2 * l2 + 1)
            if normalization == "norm"
            else 1 / math.sqrt(4 * math.pi)
            if normalization == "integral"
            else 1
        )
        values = tuple(
            contraction_source(((scale, (x,)),), cache, lines) for x in values
        )
        return bridge(values) if l1 < l3 else values
    coefficients = coupling_polynomial(l1, l2, l3, normalization)
    if (l1 + l2 + l3) % 2:
        values = generator_action(degree, vector, values, cache, lines, normalized=True)
    terms = [[(coefficients[0], (value,))] for value in values]
    previous = values
    for k, coefficient in enumerate(coefficients[1:], start=1):
        squared = generator_action(
            degree,
            vector,
            generator_action(degree, vector, values, cache, lines, normalized=True),
            cache,
            lines,
            normalized=True,
        )
        following = tuple(
            contraction_source(
                ((-2.0 * spectral_scale, (a,)), (-1.0, (b,)))
                if k == 1
                else ((-4.0 * spectral_scale, (a,)), (-2.0, (b,)), (-1.0, (c,))),
                cache,
                lines,
            )
            for a, b, c in zip(squared, values, previous)
        )
        previous, values = values, following
        for row, value in zip(terms, values):
            row.append((coefficient, (value,)))
    result = tuple(contraction_source(tuple(row), cache, lines) for row in terms)
    return bridge(result) if l1 < l3 else result


def coupling_derivative_source(
    l1,
    l2,
    l3,
    vector,
    features,
    normalization,
    cache,
    lines,
    directions,
    recurrence=False,
):
    """Differentiate the generator polynomial by the exact product rule.

    Each subset of ``directions`` stores one mixed directional derivative.
    Generator actions remain sparse at every order; derivatives do not switch
    to a full harmonic CG contraction.
    """
    if recurrence:
        nodes, outputs = coupling_program(l1, l2, l3, normalization, len(directions))
        return polynomial_source(
            nodes, outputs, (features, vector, *directions), cache, lines
        )
    from ..o3.harmonics import polynomial_coefficients

    degree, delta = min(l1, l3), abs(l1 - l3)
    coefficients = coupling_polynomial(l1, l2, l3, normalization)
    spectral_scale = degree * (degree + 1) / max(1, degree**2)
    size = 1 << len(directions)
    subsets = tuple(
        tuple(s for s in range(size) if s & mask == s) for mask in range(size)
    )

    def contract(terms):
        return contraction_source(
            tuple((c, f) for c, f in terms if c and "T(0)" not in f), cache, lines
        )

    def generator(values):
        result = []
        for mask, value in enumerate(values):
            terms = [
                generator_action(degree, vector, value, cache, lines, normalized=True)
            ]
            terms.extend(
                generator_action(
                    degree,
                    direction,
                    values[mask ^ (1 << k)],
                    cache,
                    lines,
                    normalized=True,
                )
                for k, direction in enumerate(directions)
                if mask & (1 << k)
            )
            result.append(
                tuple(contract(tuple((1.0, (x,)) for x in row)) for row in zip(*terms))
            )
        return tuple(result)

    def bridge(values):
        if l1 == l3:
            scale = (2 * l1 + 1) ** -0.5
            return tuple(
                tuple(contract(((scale, (x,)),)) for x in row) for row in values
            )
        harmonics = []
        for polynomial in polynomial_coefficients(delta, "norm"):
            terms = [[] for _ in range(size)]
            for powers, coefficient in polynomial:
                jet = ("T(1)",) + ("T(0)",) * (size - 1)
                for a, power in enumerate(powers):
                    for _ in range(power):
                        jet = tuple(
                            contract(
                                ((1.0, (jet[mask], vector[a])),)
                                + tuple(
                                    (1.0, (jet[mask ^ (1 << k)], direction[a]))
                                    for k, direction in enumerate(directions)
                                    if mask & (1 << k)
                                )
                            )
                            for mask in range(size)
                        )
                for row, value in zip(terms, jet):
                    row.append((coefficient, (value,)))
            harmonics.append(tuple(contract(row) for row in terms))
        cg = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")
        entries = tuple(
            (a, b, c, float(cg[a, b, c])) for a, b, c in cg.nonzero().tolist()
        )
        return tuple(
            tuple(
                contract(
                    tuple(
                        (coefficient, (values[s][a], harmonics[b][mask ^ s]))
                        for a, b, c, coefficient in entries
                        if c == out
                        for s in subsets[mask]
                    )
                )
                for out in range(2 * l3 + 1)
            )
            for mask in range(size)
        )

    values = (features,) + (("T(0)",) * len(features),) * (size - 1)
    if l1 >= l3:
        values = bridge(values)
    if (l1 + l2 + l3) % 2:
        values = generator(values)
    terms = [[[(coefficients[0], (x,))] for x in row] for row in values]
    previous = values
    for k, coefficient in enumerate(coefficients[1:], start=1):
        squared = generator(generator(values))
        following = tuple(
            tuple(
                contract(
                    ((-2.0 * spectral_scale, (a,)), (-1.0, (b,)))
                    if k == 1
                    else ((-4.0 * spectral_scale, (a,)), (-2.0, (b,)), (-1.0, (c,)))
                )
                for a, b, c in zip(row_a, row_b, row_c)
            )
            for row_a, row_b, row_c in zip(squared, values, previous)
        )
        previous, values = values, following
        for rows, row in zip(terms, values):
            for entries, value in zip(rows, row):
                entries.append((coefficient, (value,)))
    result = tuple(tuple(contract(entries) for entries in row) for row in terms)
    return (bridge(result) if l1 < l3 else result)[-1]


@lru_cache(maxsize=256)
def coupling_program(l1, l2, l3, normalization, rank):
    """Construct a generator polynomial and its directional derivatives.

    Parameters
    ----------
    l1, l2, l3 : int
        Input, harmonic and output degrees.
    normalization : {"component", "norm", "integral"}
        Spherical-harmonic normalization.
    rank : int
        Number of mixed directional derivatives before transposition.
    """
    from ..o3.harmonics import polynomial_coefficients

    program = ScalarProgram()
    zero = program.constant(0)
    features = tuple(program.add("input", 0, i) for i in range(2 * l1 + 1))
    vector = tuple(program.add("input", 1, i) for i in range(3))
    degree, delta = min(l1, l3), abs(l1 - l3)
    matrix = so3_generators(degree) / math.sqrt(max(1, degree * (degree + 1)))
    entries = tuple(
        (a, i, j, float(matrix[a, i, j])) for a, i, j in matrix.nonzero().tolist()
    )

    def contract(terms):
        result = zero
        for coefficient, factors in terms:
            value = program.constant(coefficient)
            for factor in factors:
                value = program.add("mul", value, factor)
            result = program.add("add", result, value)
        return result

    def generator(values):
        return tuple(
            contract(
                (scale, (vector[a], values[j]))
                for a, i, j, scale in entries
                if i == out
            )
            for out in range(2 * degree + 1)
        )

    def bridge(values):
        if l1 == l3:
            return tuple(contract((((2 * l1 + 1) ** -0.5, (x,)),)) for x in values)
        harmonics = tuple(
            contract(
                (
                    scale,
                    tuple(vector[a] for a, p in enumerate(powers) for _ in range(p)),
                )
                for powers, scale in polynomial
            )
            for polynomial in polynomial_coefficients(delta, "norm")
        )
        cg = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")
        terms = tuple(
            (a, b, c, float(cg[a, b, c])) for a, b, c in cg.nonzero().tolist()
        )
        return tuple(
            contract(
                (scale, (values[a], harmonics[b]))
                for a, b, c, scale in terms
                if c == out
            )
            for out in range(2 * l3 + 1)
        )

    values = features if l1 < l3 else bridge(features)
    values = tuple(contract(((math.sqrt(2 * delta + 1), (x,)),)) for x in values)
    previous = (zero,) * len(values)
    coefficients = coupling_recurrence(l1, l3, l2)
    odd = (l2 - delta) % 2
    if odd:
        values = tuple(
            contract(((1 / coefficients[0], (x,)),)) for x in generator(values)
        )
    for k in range(odd, l2 - delta, 2):
        acted = generator(generator(values))
        scale = 1 / (coefficients[k] * coefficients[k + 1])
        diagonal = coefficients[k] ** 2 + (coefficients[k - 1] ** 2 if k else 0)
        following = tuple(
            contract(
                ((scale, (a,)), (scale * diagonal, (b,)))
                + (
                    ((-scale * coefficients[k - 1] * coefficients[k - 2], (c,)),)
                    if k >= 2
                    else ()
                )
            )
            for a, b, c in zip(acted, values, previous)
        )
        previous, values = values, following
    scale = (
        1 / math.sqrt(2 * l2 + 1)
        if normalization == "norm"
        else 1 / math.sqrt(4 * math.pi)
        if normalization == "integral"
        else 1
    )
    output = tuple(contract(((scale, (x,)),)) for x in values)
    if l1 < l3:
        output = bridge(output)

    for k in range(rank):
        count = len(program.nodes)
        tangent = {i: program.add("input", k + 2, a) for a, i in enumerate(vector)}
        for i, (op, *args) in enumerate(program.nodes[:count]):
            if op == "add":
                a, b = args
                tangent[i] = program.add(
                    "add", tangent.get(a, zero), tangent.get(b, zero)
                )
            elif op == "mul":
                a, b = args
                tangent[i] = program.add(
                    "add",
                    program.add("mul", tangent.get(a, zero), b),
                    program.add("mul", a, tangent.get(b, zero)),
                )
        output = tuple(tangent.get(i, zero) for i in output)
    return tuple(program.nodes), output


def coupling_adjoint_source(requests, cache, lines, feature_adjoint=False):
    """Transpose shared path polynomials before reducing their adjoints.

    Each request supplies the degrees, direction, feature and cotangent
    expressions, normalization, other derivative directions and path weight.
    Differentiate the common source features or direction. All other operands
    are held fixed, including operands that alias an active input at runtime.
    """
    program = ScalarProgram()
    zero = program.constant(0)
    vector = requests[0][3]
    active_values = requests[0][4] if feature_adjoint else vector
    active_group = 0 if feature_adjoint else 1
    active = tuple(program.add("input", 0, i) for i in range(len(active_values)))
    external, indices = [], {}

    def input(value):
        if value == "T(0)":
            return zero
        if value not in indices:
            indices[value] = program.add("input", 1, len(external))
            external.append(value)
        return indices[value]

    loss = zero
    for (
        l1,
        l2,
        l3,
        _,
        features,
        cotangent,
        normalization,
        directions,
        weight,
    ) in requests:
        nodes, outputs = coupling_program(l1, l2, l3, normalization, len(directions))
        inputs = features, vector, *directions
        needed = set(outputs)
        for i in range(len(nodes) - 1, -1, -1):
            if i in needed and nodes[i][0] not in ("input", "constant"):
                needed.update(nodes[i][1:])
        mapped = {}
        for i, (op, *args) in enumerate(nodes):
            if i not in needed:
                continue
            if op == "input":
                mapped[i] = (
                    active[args[1]]
                    if args[0] == active_group
                    else input(inputs[args[0]][args[1]])
                )
            elif op == "constant":
                mapped[i] = program.constant(args[0])
            else:
                mapped[i] = program.add(op, *(mapped[j] for j in args))
        for i, seed in zip(outputs, cotangent):
            seed = program.add("mul", input(weight), input(seed))
            loss = program.add("add", loss, program.add("mul", seed, mapped[i]))
    outputs, _ = program.transpose(((loss,),), (len(active), len(external)), (0,))
    return polynomial_source(
        program.nodes, outputs[0], (active_values, external, ("T(1)",)), cache, lines
    )


def polynomial_source(nodes, outputs, inputs, cache, lines):
    """Emit the reachable expressions of a scalar polynomial program."""
    needed = set(outputs)
    uses = Counter(outputs)
    for i in range(len(nodes) - 1, -1, -1):
        if i in needed and nodes[i][0] not in ("input", "constant"):
            needed.update(nodes[i][1:])
            uses.update(nodes[i][1:])
    values = {}
    for i, (op, *args) in enumerate(nodes):
        if i not in needed:
            continue
        if op == "input":
            terms = ((1.0, (inputs[args[0]][args[1]],)),)
        elif op == "constant":
            terms = ((args[0], ()),)
        else:
            a, b = (values[j] for j in args)
            if op == "add":
                terms = a + b
            elif len(a) == len(b) == 1:
                terms = ((a[0][0] * b[0][0], a[0][1] + b[0][1]),)
            else:
                terms = (
                    (1.0, tuple(contraction_source(t, cache, lines) for t in (a, b))),
                )
        terms = tuple((c, f) for c, f in terms if c and "T(0)" not in f)
        if (uses[i] > 1 and op not in ("input", "constant")) or i in outputs:
            terms = ((1.0, (contraction_source(terms, cache, lines),)),)
        values[i] = terms
    return tuple(values[i][0][1][0] for i in outputs)

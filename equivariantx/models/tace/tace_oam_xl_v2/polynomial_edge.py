"""Exact polynomial factoring across weighted CG terms and mixed adjoints."""

from collections import defaultdict

from equivariantx.ops.common.cuda.codegen import contraction_source


def edge_polynomials(paths, program, dimensions, shared):
    """Emit complete output expressions with common factors shared across paths.

    The input contract is a direct weighted ``uvu`` convolution with angular
    multiplicity one. All feature, radial and adjoint operands stay live.
    Coefficients are the same constants stored in the native CG metadata.
    """
    cache, lines = {}, []
    groups = (defaultdict(list), defaultdict(list), defaultdict(list))

    def load(mapping, role, column):
        pointer = mapping[role]
        row = {0: "source_node", 1: "edge", 3: "edge", 4: "target_node"}[role]
        if role in (1, 3) and shared[pointer]:
            row = "0"
        channel = "0" if role == 3 else "u"
        key = ("load", pointer, row, column, channel)
        if key not in cache:
            name = f"v{len(cache)}"
            lines.append(
                f"const T {name}=active ? p{pointer}[{row}*{dimensions[pointer]}+{column}+{channel}] : T(0);"
            )
            cache[key] = name
        return cache[key]

    for mapping, _, pairs in program:
        for path in paths:
            start, attr, end, channels, mul2, di, dy, do, offset, factor, cg = path
            if mul2 != 1 or offset < 0:
                raise ValueError(
                    "Polynomial edges require weighted angular multiplicity one"
                )
            for role, slot in pairs:
                if role not in (0, 1, 3, 4):
                    raise ValueError("Polynomial edges require direct operands")
                for i, j, k, coefficient in cg:
                    columns = {
                        0: start + i * channels,
                        1: offset,
                        3: attr + j,
                        4: end + k * channels,
                    }
                    factors = tuple(
                        load(mapping, r, columns[r]) for r in (0, 1, 3, 4) if r != role
                    )
                    pointer = mapping[role]
                    if role in (0, 4):
                        group, key = (
                            groups[0],
                            (slot, role, dimensions[pointer], columns[role]),
                        )
                    elif role == 3:
                        group, key = (
                            groups[1],
                            (slot, dimensions[pointer], columns[role], shared[pointer]),
                        )
                    else:
                        group, key = (
                            groups[2],
                            (slot, dimensions[pointer], offset, 1, 0, shared[pointer]),
                        )
                    group[key].append((coefficient * factor, factors))
    result = []
    for group in groups:
        output = {}
        for key, terms in group.items():
            value = contraction_source(terms, cache, lines)
            output[key] = [("T(1)", value)]
        result.append(output)
    return cache, lines, *result

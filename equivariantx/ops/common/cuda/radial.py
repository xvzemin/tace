"""Bounded radial projections shared by convolution kernels and their adjoints."""

import torch


def project(
    weight_numel,
    source,
    target,
    calls,
    contract,
    *,
    complete,
    output_role,
    chunk_size=16384,
    workspace_bytes=512 << 20,
):
    """Fuse a mixed-adjoint program with bounded radial workspaces."""
    # Preserve the original broadcasting layout even for a one-edge tail.
    shared = tuple(calls[0][-3][i].size(0) == 1 for i in (1, *range(3, output_role)))
    groups = {}
    factors = {}
    for *prefix, outputs, values, results, weighted in calls:
        destinations = dict(zip(outputs, results))
        key = (
            id(values[1]),
            id(values[2]),
            id(destinations.get(1)),
            id(destinations.get(2)),
        )
        groups.setdefault(key, []).append(
            (*prefix, outputs, values, destinations, weighted)
        )
        if any(i not in (1, 2) for i in outputs):
            factors[key[:2]] = values[1:3]
    radial = calls[0][-3][1]
    adjoints, aliases = {}, {}
    for key, terms in groups.items():
        if 1 not in terms[0][-2] and 2 not in terms[0][-2]:
            continue
        # The projected weight adjoint depends only on angular operands.
        # Different radial/projection tangents can reuse this contraction.
        signature = (
            terms[0][-3][1].size(0) == 1,
            tuple(
                (
                    tuple(prefix),
                    tuple(
                        id(value) for i, value in enumerate(values) if i not in (1, 2)
                    ),
                    weighted,
                )
                for *prefix, _, values, _, weighted in terms
            ),
        )
        aliases[key] = adjoints.setdefault(signature, key)
    count = len(factors) + len(adjoints)
    chunk = max(
        1,
        workspace_bytes // (count * weight_numel * radial.element_size()),
    )
    if chunk >= 32:
        chunk = chunk // 32 * 32
    chunk = min(chunk_size, chunk, source.numel())
    workspaces = {
        key: radial.new_empty(1 if r.size(0) == 1 else chunk, weight_numel)
        for key, (r, _) in factors.items()
    }
    gradients = {
        key: radial.new_empty(
            1 if terms[0][-3][1].size(0) == 1 else chunk, weight_numel
        )
        for key in adjoints.values()
        for terms in (groups[key],)
    }
    # Each active uvu path owns a disjoint range of weight columns. A single
    # non-broadcast adjoint can therefore write its workspace without reading
    # or clearing it first. Retain accumulation for shared or mixed adjoints,
    # and for instructions with inactive weight columns.
    initialize = tuple(
        key
        for key in gradients
        if complete and len(groups[key]) == 1 and groups[key][0][-3][1].size(0) != 1
    )
    # Weight-only adjoints never read this operand. Retain its shape without
    # allocating a second edge-by-path workspace.
    unused = next(iter(workspaces.values()), None)
    if unused is None:
        unused = radial.new_empty(1).expand(chunk, weight_numel)
    empty = radial.new_empty(0, weight_numel)
    for key, (r, projection) in factors.items():
        if r.size(0) == 1:
            torch.mm(r, projection, out=workspaces[key])
    for key, gradient in gradients.items():
        if groups[key][0][-3][1].size(0) == 1:
            gradient.zero_()
    for start in range(0, source.numel(), chunk):
        stop = min(start + chunk, source.numel())
        views = {}

        def edge_view(value):
            if id(value) not in views:
                views[id(value)] = value if value.size(0) == 1 else value[start:stop]
            return views[id(value)]

        weights = {}
        for key, (r, projection) in factors.items():
            is_shared = r.size(0) == 1
            r = edge_view(r)
            weights[key] = workspaces[key][: r.size(0)]
            if not is_shared:
                torch.mm(r, projection, out=weights[key])
        direct = []
        weight_gradients = {
            key: gradients[original][
                : 1 if groups[key][0][-3][1].size(0) == 1 else stop - start
            ]
            for key, original in aliases.items()
        }
        for key in gradients:
            if groups[key][0][-3][1].size(0) != 1 and key not in initialize:
                weight_gradients[key].zero_()
        for key, terms in groups.items():
            r, projection = terms[0][-3][1:3]
            rows = 1 if r.size(0) == 1 else stop - start
            w = weights.get(key[:2], unused[:rows])
            for *prefix, outputs, values, destinations, weighted in terms:
                edge_roles = tuple(i for i in range(3, len(values)) if i != output_role)
                result = {
                    i: edge_view(value) if i in edge_roles else value
                    for i, value in destinations.items()
                    if i not in (1, 2)
                }
                if key in gradients:
                    result[1] = weight_gradients[key]
                if not result:
                    continue
                direct.append(
                    (
                        *prefix,
                        tuple(result),
                        tuple(
                            w
                            if i == 1
                            else empty
                            if i == 2
                            else edge_view(value)
                            if i in edge_roles
                            else value
                            for i, value in enumerate(values)
                        ),
                        tuple(result.values()),
                        weighted,
                    )
                )
        contract(
            source[start:stop],
            target[start:stop],
            direct,
            shared,
            tuple(weight_gradients[key] for key in initialize),
        )
        for key, gradient in weight_gradients.items():
            r, projection = groups[key][0][-3][1:3]
            if r.size(0) == 1:
                continue
            destinations = groups[key][0][-2]
            if 1 in destinations:
                edge_view(destinations[1]).addmm_(gradient, projection.T)
            if 2 in destinations:
                destinations[2].addmm_(edge_view(r).T, gradient)
    # Shared weights have one projected cotangent, summed over every chunk.
    # Apply its projection transpose only once.
    for key, original in aliases.items():
        gradient = gradients[original]
        r, projection = groups[key][0][-3][1:3]
        if r.size(0) != 1:
            continue
        destinations = groups[key][0][-2]
        if 1 in destinations:
            destinations[1].addmm_(gradient, projection.T)
        if 2 in destinations:
            destinations[2].addmm_(r.T, gradient)

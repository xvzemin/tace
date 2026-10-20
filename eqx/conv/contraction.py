"""Transpose multilinear convolution programs without truncating derivatives."""


def adjoint_program(program, operands, grad_outputs, needs_grad, has_unweighted):
    """Transpose requested outputs, retaining the dependencies of each path."""
    values = list(operands)
    cotangents = {}
    for slot, value in enumerate(grad_outputs):
        if value is not None:
            cotangents[slot] = len(values)
            values.append(value)
    terms = {}
    destinations = {}
    for mapping, weighted_only, pairs in program:
        for output, slot in pairs:
            if slot not in cotangents:
                continue
            if output == 2 and not operands[mapping[2]].numel():
                continue
            replacement = list(mapping)
            replacement[output] = cotangents[slot]
            # Roles one and two are radial features and their projection.
            key = (
                tuple(replacement),
                has_unweighted and (weighted_only or output in (1, 2)),
            )
            for role, index in enumerate(mapping):
                if role != output and needs_grad[index]:
                    destination = destinations.setdefault(index, len(destinations))
                    terms.setdefault(key, []).append((role, destination))
    program = tuple(
        (mapping, weighted, tuple(pairs))
        for (mapping, weighted), pairs in terms.items()
    )
    return program, values, destinations

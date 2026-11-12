"""Sparse O(3) contractions and their ordinary Torch transposes."""

import math
from functools import lru_cache

import torch

from ....utils.harmonics import polynomial_coefficients
from ....utils.metadata import parse_metadata
from ....utils.outputs import contraction_outputs


@lru_cache(maxsize=128)
def harmonic_coefficients(dimension, normalization):
    """Return the exact polynomial used by the native geometric contraction."""
    if isinstance(normalization, tuple):
        from ....co3.symmetric import symmetric_powers
        from ....conv.co3.polynomials import component_powers, harmonic_polynomial

        kind, normalization = normalization
        if kind == "cartesian_symmetric":
            degree = (math.isqrt(8 * dimension + 1) - 3) // 2
            powers = symmetric_powers(degree)
        elif kind == "cartesian":
            degree, width = 0, dimension
            while width > 1:
                degree, width = degree + 1, width // 3
            powers = tuple(component_powers(i, degree) for i in range(dimension))
        else:
            raise NotImplementedError(f"Torch O3 geometry does not support {kind!r}.")
        return tuple(harmonic_polynomial(power, normalization) for power in powers)
    return polynomial_coefficients((dimension - 1) // 2, normalization)


def harmonic_values(coefficients, vector, directions):
    """Evaluate mixed directional polynomial derivatives without AD nesting."""
    rank = len(directions)
    zero = vector[..., 0] * 0
    outputs = []
    for polynomial in coefficients:
        value = zero
        for powers, coefficient in polynomial:
            if rank > sum(powers):
                continue
            jet = [zero + coefficient] + [zero] * ((1 << rank) - 1)
            for axis, power in enumerate(powers):
                for _ in range(power):
                    jet = [
                        term * vector[..., axis]
                        + sum(
                            jet[mask ^ (1 << k)] * direction[..., axis]
                            for k, direction in enumerate(directions)
                            if mask & (1 << k)
                        )
                        for mask, term in enumerate(jet)
                    ]
            value = value + jet[-1]
        outputs.append(value)
    return torch.stack(outputs, dim=-1)


def contraction(metadata, program, source, target, operands, *, batch_paths=False):
    """Evaluate sparse paths, including projected radial and geometric roles.

    Parameters
    ----------
    metadata, program : str
        Native path coefficients and transposed operand mappings.
    source, target : torch.Tensor or None
        Source and target owners; both may be omitted for edge-local execution.
    operands : sequence of torch.Tensor
        Features, radial values, projection, attributes, output placeholder,
        and optional directions. FP32 or FP64; TF32 follows Torch settings.
    batch_paths : bool, optional
        Native scheduling hint. Does not alter the Torch expression.
    """
    description = parse_metadata(metadata)
    if len(description) > 3:
        geometry = description[3]
        if isinstance(geometry, tuple) and geometry[0] == "transverse":
            raise NotImplementedError(
                "Transverse O3 lowering requires its generator polynomial."
            )
        if len(description) > 5 and description[5]:
            raise NotImplementedError("Angular O3 lowering requires Lie derivatives.")
    if (source is None) != (target is None):
        raise ValueError("Both graph indices must be supplied or omitted.")
    outputs = [
        torch.zeros_like(value) for value in contraction_outputs(program, operands)
    ]
    for mapping, weighted, pairs in parse_metadata(program):
        values = tuple(operands[index] for index in mapping)
        for role, slot in pairs:
            outputs[slot] = outputs[slot] + pairing(
                description, source, target, values, role, weighted
            )
    return outputs


def pairing(description, source, target, values, role, weighted_only):
    """Contract all occupied factors except the requested destination."""
    x, radial, projection, attributes, output, *geometry = values
    paths, _, _, *options = description
    geometric = bool(options)
    projected = projection.numel() != 0
    if geometric and role == 5:
        raise ValueError("Base-direction adjoints use an appended direction role.")
    edges = x.shape[0] if source is None else source.numel()
    result = torch.zeros_like(values[role])

    def gather(value, begin, width, index):
        value = value[:, begin : begin + width]
        return value if index is None else value.index_select(0, index)

    for (
        start,
        attr,
        end,
        channels,
        multiplicity,
        dx,
        dy,
        dz,
        weight,
        scale,
        sparse,
    ) in paths:
        if (weight < 0 and (weighted_only or role in (1, 2))) or (
            role == 2 and not projected
        ):
            continue
        coefficients = [[[0.0] * dz for _ in range(dy)] for _ in range(dx)]
        for a, b, c, coefficient in sparse:
            coefficients[a][b][c] += scale * coefficient
        tensors = [
            None
            if role == 0
            else gather(x, start, dx * channels, source).reshape(edges, dx, channels),
            None,
            None,
            None,
            None
            if role == 4
            else gather(output, end, dz * channels, target).reshape(
                edges, dz, channels
            ),
        ]
        tokens = ["eau", "eh", "huv", "ev" if geometric else "ebv", "ecu"]
        if role != 3:
            shape = (multiplicity,) if geometric else (dy, multiplicity)
            tensors[3] = (
                attributes[:, attr : attr + math.prod(shape)]
                .reshape(-1, *shape)
                .expand(edges, *shape)
            )
        width = channels * multiplicity
        if weight >= 0:
            if projected:
                if role != 1:
                    tensors[1] = radial.expand(edges, -1)
                if role != 2:
                    tensors[2] = projection[:, weight : weight + width].reshape(
                        projection.shape[0], channels, multiplicity
                    )
            else:
                tokens[1] = "euv"
                if role != 1:
                    tensors[1] = (
                        radial[:, weight : weight + width]
                        .reshape(-1, channels, multiplicity)
                        .expand(edges, -1, -1)
                    )
        active = [
            (token, value) for token, value in zip(tokens, tensors) if value is not None
        ]
        if weight < 0 and role == 3:
            active.append(("v", x.new_ones(multiplicity)))
        output_token = tokens[role] if role < 5 else "ek"
        if geometric:
            offset = options[3] if len(options) > 3 else 0
            vector = geometry[0][:, offset : offset + 3].expand(edges, -1)
            directions = [value[:, :3].expand(edges, -1) for value in geometry[1:]]
            if role >= 6:
                vector = vector[:, None, :].expand(-1, 3, -1)
                directions = [
                    torch.eye(3, dtype=x.dtype, device=x.device)[None]
                    if k == role - 6
                    else value[:, None, :]
                    for k, value in enumerate(directions)
                ]
            harmonics = harmonic_values(
                harmonic_coefficients(dy, options[0]), vector, directions
            )
            active.append(("ekb" if role >= 6 else "eb", harmonics))
        term = torch.einsum(
            ",".join([token for token, _ in active] + ["abc"]) + "->" + output_token,
            *[value for _, value in active],
            x.new_tensor(coefficients),
        )
        if role in (0, 4):
            index = source if role == 0 else target
            begin, size = (start, dx * channels) if role == 0 else (end, dz * channels)
            term = term.reshape(edges, size)
            if index is not None:
                term = term.new_zeros((result.shape[0], size)).index_add(0, index, term)
        elif role == 2:
            begin, size = weight, width
            term = term.reshape(projection.shape[0], width)
        else:
            if role == 3:
                begin, size = attr, (1 if geometric else dy) * multiplicity
            elif role >= 6:
                begin, size = 0, 3
            elif projected:
                begin, size = 0, radial.shape[1]
            else:
                begin, size = weight, width
            term = term.reshape(edges, size)
            if result.shape[0] == 1:
                term = term.sum(0, keepdim=True)
        result = result + torch.nn.functional.pad(
            term, (begin, result.shape[1] - begin - size)
        )
    return result + sum(value[..., :0].sum() for value in values)

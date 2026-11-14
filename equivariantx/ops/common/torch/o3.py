"""Sparse O(3) contractions and their ordinary Torch transposes."""

import math
from functools import lru_cache

import torch

from ....utils.harmonics import polynomial_coefficients
from ....utils.metadata import parse_metadata
from ... import config
from .convolution import contract_program
from .einsum import einsum


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


@lru_cache(maxsize=64)
@torch.compiler.assume_constant_result
def generator_data(degree):
    """Return immutable real rotation generators outside tracing contexts."""
    from torch.utils._python_dispatch import _disable_current_modes

    from ....o3.rotation import so3_generators

    with _disable_current_modes(), torch._C._DisableFuncTorch():
        return so3_generators(degree).tolist()


def angular_values(dimension, normalization, vector, directions):
    """Apply ordered Lie derivatives, with the first direction acting last."""
    value = harmonic_values(harmonic_coefficients(dimension, normalization), vector, ())
    generator = vector.new_tensor(generator_data((dimension - 1) // 2))
    for direction in reversed(directions):
        value = torch.einsum("...a,aij,...j->...i", direction, generator, value)
    return value


@lru_cache(maxsize=256)
@torch.compiler.assume_constant_result
def coupling_data(l1, l2, l3, normalization, method):
    """Return immutable CG, generator and transverse-polynomial constants."""
    from e3nn import o3
    from torch.utils._python_dispatch import _disable_current_modes

    from ....co2.spherical import coupling_polynomial, coupling_recurrence
    from ....o3.rotation import so3_generators

    # Constants must not retain fake tensors or functional-transform wrappers.
    with _disable_current_modes(), torch._C._DisableFuncTorch():
        degree, delta = min(l1, l3), abs(l1 - l3)
        cg = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")
        generator = so3_generators(degree) / math.sqrt(max(1, degree * (degree + 1)))
        coefficients = (
            coupling_recurrence(l1, l3, l2)
            if method == "recurrence"
            else coupling_polynomial(l1, l2, l3, normalization)
        )
        return cg.tolist(), generator.tolist(), coefficients


def transverse_values(l1, l2, l3, normalization, method, vector, directions):
    """Evaluate transverse matrices and exact mixed Cartesian derivatives.

    The generator polynomial is evaluated at the supplied vector, including
    away from the unit sphere. Derivative subsets obey the product rule.
    """
    cg, generator, coefficients = coupling_data(l1, l2, l3, normalization, method)
    degree, delta = min(l1, l3), abs(l1 - l3)
    size = 1 << len(directions)
    generator = vector.new_tensor(generator)
    matrices = [
        torch.einsum("...a,aij->...ij", v, generator) for v in (vector, *directions)
    ]
    zero = vector[..., :1, None] * 0
    identity = torch.eye(2 * degree + 1, dtype=vector.dtype, device=vector.device)
    if l1 == l3:
        bridge = [identity / math.sqrt(2 * l1 + 1) + zero]
        bridge += [bridge[0] * 0] * (size - 1)
    else:
        cg = vector.new_tensor(cg)
        bridge = [
            torch.einsum(
                "...b,abc->...ca",
                harmonic_values(
                    harmonic_coefficients(2 * delta + 1, "norm"),
                    vector,
                    tuple(v for k, v in enumerate(directions) if mask & (1 << k)),
                ),
                cg,
            )
            for mask in range(size)
        ]
    values = [identity + zero] + [identity * zero] * (size - 1) if l1 < l3 else bridge

    def generator_action(values):
        return [
            matrices[0] @ value
            + sum(
                matrices[k + 1] @ values[mask ^ (1 << k)]
                for k in range(len(directions))
                if mask & (1 << k)
            )
            for mask, value in enumerate(values)
        ]

    odd = (l2 - delta) % 2
    if method == "recurrence":
        values = [math.sqrt(2 * delta + 1) * value for value in values]
        if odd:
            values = [value / coefficients[0] for value in generator_action(values)]
        previous = [value * 0 for value in values]
        for k in range(odd, l2 - delta, 2):
            acted = generator_action(generator_action(values))
            diagonal = coefficients[k] ** 2 + (coefficients[k - 1] ** 2 if k else 0)
            previous, values = (
                values,
                [
                    (
                        a
                        + diagonal * b
                        - (
                            coefficients[k - 1] * coefficients[k - 2] * c
                            if k >= 2
                            else 0
                        )
                    )
                    / (coefficients[k] * coefficients[k + 1])
                    for a, b, c in zip(acted, values, previous)
                ],
            )
        scale = (
            1 / math.sqrt(2 * l2 + 1)
            if normalization == "norm"
            else 1 / math.sqrt(4 * math.pi)
            if normalization == "integral"
            else 1
        )
        result = [scale * value for value in values]
    else:
        if odd:
            values = generator_action(values)
        previous = values
        result = [coefficients[0] * value for value in values]
        spectral_scale = degree * (degree + 1) / max(1, degree**2)
        for k, coefficient in enumerate(coefficients[1:], start=1):
            acted = generator_action(generator_action(values))
            following = [
                -2 * spectral_scale * a - b
                if k == 1
                else -4 * spectral_scale * a - 2 * b - c
                for a, b, c in zip(acted, values, previous)
            ]
            previous, values = values, following
            result = [a + coefficient * b for a, b in zip(result, values)]
    value = (
        sum(bridge[mask] @ result[(size - 1) ^ mask] for mask in range(size))
        if l1 < l3
        else result[-1]
    )
    return value.transpose(-1, -2)


def contraction_metadata(metadata):
    """Resolve baseline transverse paths to their homogeneous CG polynomial."""
    description = parse_metadata(metadata)
    if len(description) > 3:
        geometry = description[3]
        if isinstance(geometry, tuple) and geometry[0] == "transverse":
            if len(geometry) < 3 or geometry[2] == "baseline":
                description = (*description[:3], geometry[1], *description[4:])
            elif geometry[2] not in ("generator", "recurrence"):
                raise NotImplementedError(f"Unknown transverse method {geometry[2]!r}.")
    return description


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
    if (source is None) != (target is None):
        raise ValueError("Both graph indices must be supplied or omitted.")
    return contract_program(
        pairing,
        contraction_metadata(metadata),
        program,
        source,
        target,
        operands,
        optimized=not config.TORCH_REFERENCE,
    )


def reference(metadata, program, source, target, operands, *, batch_paths=False):
    """Evaluate independent paths without combining their radial projections."""
    return contract_program(
        pairing,
        contraction_metadata(metadata),
        program,
        source,
        target,
        operands,
        optimized=False,
    )


def pairing(description, source, target, values, role, weighted_only, *, reuse=False):
    """Contract all occupied factors except the requested destination."""
    x, radial, projection, attributes, output, *geometry = values
    paths, _, _, *options = description
    geometric = bool(options)
    projected = projection.numel() != 0
    batched_projection = projection.ndim == 3
    if geometric and role == 5:
        raise ValueError("Base-direction adjoints use an appended direction role.")
    edges = x.shape[0] if source is None else source.numel()
    result = torch.zeros_like(values[role])
    gathered, geometries = {}, {}

    def gather(value, begin, width, index, owner):
        key = owner, begin, width
        if reuse and key in gathered:
            return gathered[key]
        value = value[:, begin : begin + width]
        value = value if index is None else value.index_select(0, index)
        if reuse:
            gathered[key] = value
        return value

    def geometry_value(dx, dy, dz):
        normalization = options[0]
        transverse = (
            isinstance(normalization, tuple) and normalization[0] == "transverse"
        )
        key = (dx, dy, dz) if transverse else dy
        if reuse and key in geometries:
            return geometries[key]
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
        if transverse:
            value = transverse_values(
                (dx - 1) // 2,
                (dy - 1) // 2,
                (dz - 1) // 2,
                normalization[1],
                normalization[2],
                vector,
                directions,
            )
        elif len(options) > 2 and options[2]:
            value = angular_values(dy, normalization, vector, directions)
        else:
            value = harmonic_values(
                harmonic_coefficients(dy, normalization), vector, directions
            )
        if reuse:
            geometries[key] = value
        return value

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
            else gather(x, start, dx * channels, source, 0).reshape(
                edges, dx, channels
            ),
            None,
            None,
            None,
            None
            if role == 4
            else gather(output, end, dz * channels, target, 4).reshape(
                edges, dz, channels
            ),
        ]
        tokens = ["eau", "eh", "huv", "ev" if geometric else "ebv", "ecu"]
        if batched_projection:
            tokens[2] = "ehuv"
        if role != 3:
            shape = (multiplicity,) if geometric else (dy, multiplicity)
            tensors[3] = (
                attributes[:, attr : attr + math.prod(shape)]
                .reshape(attributes.shape[0], *shape)
                .expand(edges, *shape)
            )
        width = channels * multiplicity
        if weight >= 0:
            if projected:
                if role != 1:
                    tensors[1] = radial.expand(edges, -1)
                if role != 2:
                    tensors[2] = projection[..., weight : weight + width].reshape(
                        *projection.shape[:-1], channels, multiplicity
                    )
            else:
                tokens[1] = "euv"
                if role != 1:
                    tensors[1] = (
                        radial[:, weight : weight + width]
                        .reshape(radial.shape[0], channels, multiplicity)
                        .expand(edges, -1, -1)
                    )
        active = [
            (token, value) for token, value in zip(tokens, tensors) if value is not None
        ]
        if weight < 0 and role == 3:
            active.append(("v", x.new_ones(multiplicity)))
        output_token = tokens[role] if role < 5 else "ek"
        if geometric:
            normalization = options[0]
            value = geometry_value(dx, dy, dz)
            if isinstance(normalization, tuple) and normalization[0] == "transverse":
                active.append(("ekac" if role >= 6 else "eac", scale * value))
            else:
                active.append(("ekb" if role >= 6 else "eb", value))
                active.append(("abc", x.new_tensor(coefficients)))
        else:
            active.append(("abc", x.new_tensor(coefficients)))
        term = einsum(
            ",".join(token for token, _ in active) + "->" + output_token,
            *[value for _, value in active],
        )
        if (
            geometric
            and isinstance(options[0], tuple)
            and options[0][0] == "transverse"
        ):
            # Eager and fake TensorIterator can choose different strides after
            # broadcasted matrix products; export needs an explicit layout.
            term = term.clone(memory_format=torch.contiguous_format)
        if role in (0, 4):
            index = source if role == 0 else target
            begin, size = (start, dx * channels) if role == 0 else (end, dz * channels)
            term = term.reshape(edges, size)
            if index is not None:
                term = term.new_zeros((result.shape[0], size)).index_add(0, index, term)
        elif role == 2:
            begin, size = weight, width
            term = term.reshape(*projection.shape[:-1], width)
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
            term, (begin, result.shape[-1] - begin - size)
        )
    return result + sum(value[..., :0].sum() for value in values)

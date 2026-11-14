"""Tensor-product expressions shared by radial execution policies."""

from collections import defaultdict
from math import prod

import torch

from ...utils.harmonics import harmonic_program
from ...utils.metadata import parse_metadata, static_dimensions
from ...utils.program import Program
from .. import config
from .torch.edge import evaluate_recomputed, evaluate_torch
from ..precision import validate_precision


def radial_input(program, inputs, radial, network=None):
    """Append partitioned radial inputs and optional network parameters."""
    from .radial import radial_parts, radial_program

    values = []
    for tensor, storage in radial_parts(radial):
        values.append(
            program.input(len(inputs), storage, static_dimensions(tensor, (-1,))[0])
        )
        inputs.append(tensor)
    value = program.concatenate(values)
    return value if network is None else radial_program(network, inputs, program, value)


def tensor_product_program(
    program, metadata, features, attributes, weights, output_dim
):
    """Lower sparse ``uvu`` paths to matrix products and multilinear terms."""
    paths = parse_metadata(metadata)[0]
    blocks = defaultdict(list)
    for (
        start,
        attr,
        end,
        channels,
        multiplicity,
        dx,
        dy,
        dz,
        offset,
        scale,
        cg,
    ) in paths:
        x = program.slice(features, start, dx * channels)
        y = program.slice(attributes, attr, dy * multiplicity)
        if offset >= 0:
            w = program.slice(weights, offset, channels * multiplicity)
            w = program.transpose(w, channels, multiplicity)
        else:
            w = program.constant(multiplicity * channels, 1)
        y = program.matmul(y, w, dy, multiplicity, channels)
        polynomial = (
            channels,
            (dx, dy, dz),
            tuple(((a, b, c), scale * value) for a, b, c, value in cg),
        )
        value = program.product(
            (x, y, program.constant(dz * channels, 0)), polynomial, 2
        )
        blocks[end, dz * channels].append(value)
    values, position = [], 0
    for (start, width), terms in sorted(blocks.items()):
        if start < position:
            raise ValueError("Tensor-product output sections must not overlap.")
        if start > position:
            values.append(program.constant(start - position, 0))
        values.append(program.sum(terms, width))
        position = start + width
    if position < output_dim:
        values.append(program.constant(output_dim - position, 0))
    return program.concatenate(values)


def aligned_product_program(
    program, metadata, features, wigner, weights, amplitudes, output_dim
):
    """Lower aligned paths while sharing input and output frame rotations."""
    paths, _, _, couplings = parse_metadata(metadata)
    blocks, rotations = defaultdict(list), {}
    for (mode, path), cg in zip(paths, couplings):
        start, end, ci, co, di, do, wi, wo, offset, harmonic = path
        x = program.slice(features, start, di * ci)
        rotation = program.slice(wigner, wi, di * di)
        x = program.matmul(rotation, x, di, di, ci)
        polynomial = (ci, (di, do), tuple(((i, j), value) for i, j, value in cg))
        value = program.product((x, program.constant(do * ci, 0)), polynomial, 1)
        if offset >= 0:
            width = ci if mode == "uvu" else ci * co
            weight = program.slice(weights, offset, width)
            if mode == "uvu":
                value = program.binary(
                    "mul", value, program.gather(weight, tuple(range(ci)) * do)
                )
            else:
                value = program.matmul(value, weight, do, ci, co)
        elif mode == "uvw":
            value = program.matmul(value, program.constant(ci * co, 1), do, ci, co)
        amplitude = program.slice(amplitudes, harmonic, 1)
        value = program.binary(
            "mul", value, program.gather(amplitude, (0,) * (do * co))
        )
        blocks[end, do * co].append(value)
        rotations[end] = wo, do, co
    values, position = [], 0
    for (start, width), terms in sorted(blocks.items()):
        if start > position:
            values.append(program.constant(start - position, 0))
        wo, do, co = rotations[start]
        rotation = program.transpose(program.slice(wigner, wo, do * do), do, do)
        value = program.matmul(rotation, program.sum(terms, width), do, do, co)
        values.append(value)
        position = start + width
    if position < output_dim:
        values.append(program.constant(output_dim - position, 0))
    return program.concatenate(values)


def transverse_product_program(
    program,
    metadata,
    layout,
    features,
    direction,
    weights,
    amplitudes,
    output_dim,
    *,
    native=True,
):
    """Mix channels after exact angular coupling, optionally inside CUDA tiles."""
    paths, _, _, geometry = parse_metadata(metadata)
    blocks = defaultdict(list)
    empty, unit = program.constant(0, 0), program.constant(1, 1)
    degree = max((max(path[5:8]) // 2 for path in paths), default=0)
    polynomial = (not native or degree <= config.TRANSVERSE_POLYNOMIAL_DEGREE) and (
        isinstance(geometry, str) or len(geometry) == 2
    )
    normalization = geometry if isinstance(geometry, str) else geometry[1]
    for (mode, co), path in zip(layout, paths):
        start, harmonic, end, ci, _, di, dh, do, offset, scale, cg = path
        x = program.slice(features, start, di * ci)
        if polynomial:
            harmonics = harmonic_program(
                program, direction, (dh - 1) // 2, normalization
            )
            harmonics = program.gather(
                harmonics, tuple(a for a in range(dh) for _ in range(ci))
            )
            angular = (ci, (di, dh, do), tuple(((a, b, c), v) for a, b, c, v in cg))
            value = program.product(
                (x, harmonics, program.constant(do * ci, 0)), angular, 2
            )
        else:
            angular = repr(
                (((0, 0, 0, ci, 1, di, dh, do, -1, 1.0, cg),), 0, True, geometry)
            )
            value = program.add(
                "convolution",
                do * ci,
                (x, empty, empty, unit, program.constant(do * ci, 0), direction),
                ("o3", angular, 4, False, (0, 0), 0),
            )
        if offset >= 0:
            width = ci if mode == "uvu" else ci * co
            weight = program.slice(weights, offset, width)
            value = (
                program.binary(
                    "mul", value, program.gather(weight, tuple(range(ci)) * do)
                )
                if mode == "uvu"
                else program.matmul(value, weight, do, ci, co)
            )
        elif mode == "uvw":
            value = program.matmul(value, program.constant(ci * co, 1), do, ci, co)
        amplitude = program.scale(program.slice(amplitudes, harmonic, 1), scale)
        value = program.binary(
            "mul", value, program.gather(amplitude, (0,) * (do * co))
        )
        blocks[end, do * co].append(value)
    values, position = [], 0
    for (start, width), terms in sorted(blocks.items()):
        if start > position:
            values.append(program.constant(start - position, 0))
        values.append(program.sum(terms, width))
        position = start + width
    if position < output_dim:
        values.append(program.constant(output_dim - position, 0))
    return program.concatenate(values)


def convolve(
    metadata,
    features,
    attributes,
    radial,
    projection,
    edge_index,
    num_nodes,
    output_dim,
    *,
    network=None,
    recompute=False,
    amplitudes=None,
    transverse=None,
    backend="torch",
):
    """Evaluate tensor-product expressions with Torch or tiled CUDA adjoints.

    ``network`` includes hidden radial layers in the same expression program.
    ``recompute`` saves operands rather than intermediate MLP activations.
    Matrix adjoints reduce edges before forming shared parameter gradients.
    """
    if backend not in ("torch", "cuda"):
        raise ValueError("backend must be torch or cuda.")
    indexed = edge_index is not None
    input_dim = static_dimensions(features, (-1,))[0]
    attr_dim = static_dimensions(attributes, (-1,))[0]
    projection_shape = static_dimensions(projection)
    source, target = (
        (edge_index[0], edge_index[1])
        if indexed
        else (torch.arange(features.shape[0], device=features.device),) * 2
    )
    program = Program()
    inputs = [features, attributes, projection.reshape(1, -1)]
    x = program.input(0, "source" if indexed else "edge", input_dim)
    attrs = program.input(1, "shared" if attributes.shape[0] == 1 else "edge", attr_dim)
    matrix = program.input(2, "shared", prod(projection_shape))
    radial = radial_input(program, inputs, radial, network)
    if projection.numel():
        width = program.size(radial)
        bias = width + 1 == projection.shape[0]
        if width + int(bias) != projection.shape[0]:
            raise ValueError("Radial network output does not match projection input.")
        if (
            bias
            and config.RADIAL_AFFINE_MIN_WIDTH > 0
            and width >= config.RADIAL_AFFINE_MIN_WIDTH
            and width % 8 == 0
        ):
            columns = projection_shape[1]
            value = program.matmul(
                radial, program.slice(matrix, 0, width * columns), 1, width, columns
            )
            radial = program.binary(
                "add", value, program.slice(matrix, width * columns, columns)
            )
        else:
            if bias:
                radial = program.concatenate((radial, program.constant(1, 1)))
            radial = program.matmul(radial, matrix, 1, *projection_shape)
    elif program.size(radial) != projection.shape[1]:
        raise ValueError("Radial network output does not match path weights.")
    if amplitudes is None:
        result = tensor_product_program(program, metadata, x, attrs, radial, output_dim)
    else:
        amplitude = program.input(
            len(inputs),
            "shared" if amplitudes.shape[0] == 1 else "edge",
            static_dimensions(amplitudes, (-1,))[0],
        )
        inputs.append(amplitudes)
        result = (
            aligned_product_program(
                program, metadata, x, attrs, radial, amplitude, output_dim
            )
            if transverse is None
            else transverse_product_program(
                program,
                metadata,
                transverse,
                x,
                attrs,
                radial,
                amplitude,
                output_dim,
                native=backend == "cuda",
            )
        )
    validate_precision(inputs)
    expressions = repr(
        (tuple(program.nodes), ((result, 0, "target" if indexed else "edge"),))
    )
    if backend == "cuda":
        from .edge import evaluate

        return evaluate(expressions, inputs, source, target, num_nodes)[0]
    if recompute or config.TORCH_RECOMPUTE:
        return evaluate_recomputed(expressions, inputs, source, target, num_nodes)[0]
    return evaluate_torch(
        expressions, inputs, source, target, num_nodes, optimized=True
    )[0]

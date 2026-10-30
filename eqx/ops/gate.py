"""Fused O(2) scalar activations, basis permutations and gating."""

import math

import torch

from ..o2.layout import irrep_layout


def gate_program(gate, transverse=False, *, cartesian=False, blocks=False):
    """Build the shared gate expression for standalone and convolution kernels."""
    from .program import Program, activate

    program = Program()
    inputs = (
        tuple(
            (ir.dim, mul, section.start)
            for (mul, ir), section in zip(gate.irreps_in, gate.irreps_in.slices())
        )
        if cartesian
        else irrep_layout(gate.irreps_in, transverse)
    )
    if blocks:
        values = [
            program.input(i, "edge", dim * mul)
            for i, (dim, mul, _) in enumerate(inputs)
        ]
    else:
        features = program.input(0, "edge", sum(dim * mul for dim, mul, _ in inputs))
        values = [
            program.slice(features, start, dim * mul) for dim, mul, start in inputs
        ]
    direction = (
        program.input(len(values) if blocks else 1, "edge", 3) if transverse else None
    )
    scalars, gates = [], []
    for locations, activation, outputs in (
        (gate._scalar_locations, gate.act_scalars, scalars),
        (gate._gate_locations, gate.act_gates, gates),
    ):
        for i, act in zip(locations, activation.acts):
            outputs.append(activate(program, values[i], act))
    outputs = list(scalars)
    for path in gate._paths:
        index = gate._gated_locations[path.i_gated]
        dim, _, _ = inputs[index]
        features = values[index]
        width = gate.irreps_gated[path.i_gated].mul
        scalar = program.slice(gates[path.i_gate], path.gate_start, path.mul)
        if cartesian:
            odd = path.ir_gate.is_odd_scalar() and path.ir_gated.m > 0
            indices = tuple(
                (c + path.gated_start) * dim + ((a + dim // 2) % dim if odd else a)
                for c in range(path.mul)
                for a in range(dim)
            )
            value = program.gather(features, indices)
            if odd:
                signs = program.concatenate(
                    [program.constant(dim // 2, -1), program.constant(dim // 2, 1)]
                )
                value = program.binary(
                    "mul", value, program.gather(signs, tuple(range(dim)) * path.mul)
                )
            outputs.append(
                program.binary(
                    "mul",
                    value,
                    program.gather(
                        scalar, tuple(c for c in range(path.mul) for _ in range(dim))
                    ),
                )
            )
            continue
        if transverse and path.ir_gate.is_odd_scalar() and path.ir_gated.m > 0:
            from ..o3.rotation import so3_generators

            value = program.concatenate(
                [
                    program.slice(features, a * width + path.gated_start, path.mul)
                    for a in range(dim)
                ]
            )
            vector = program.gather(
                direction, tuple(a for a in range(3) for _ in range(path.mul))
            )
            generator = so3_generators(path.ir_gated.m) / path.ir_gated.m
            paths = tuple(
                ((i, a, j), float(generator[a, j, i]))
                for a, j, i in generator.nonzero().tolist()
            )
            value = program.product(
                (value, vector, program.constant(dim * path.mul, 0)),
                (path.mul, (dim, 3, dim), paths),
                2,
            )
            outputs.append(
                program.binary(
                    "mul", value, program.gather(scalar, tuple(range(path.mul)) * dim)
                )
            )
            continue
        components = []
        for a in range(dim):
            odd = path.ir_gate.is_odd_scalar() and path.ir_gated.m > 0
            b = 1 - a if odd else a
            value = program.slice(features, b * width + path.gated_start, path.mul)
            value = program.scale(value, -1 if odd and a == 0 else 1)
            components.append(program.binary("mul", value, scalar))
        outputs.append(program.concatenate(components))
    if blocks:
        return repr(
            (
                tuple(program.nodes),
                tuple((value, i, "edge") for i, value in enumerate(outputs)),
            )
        )
    output = program.concatenate(outputs)
    return repr((tuple(program.nodes), ((output, 0, "edge"),)))


def gate(features, metadata, output_dim):
    """Evaluate a gate without materializing activated or expanded gate arrays."""
    from .edge import evaluate

    if not features.is_cuda or features.dtype not in (torch.float32, torch.float64):
        raise TypeError("Gate kernels require CUDA float32 or float64 features.")
    batch = math.prod(features.shape[:-1])
    index = torch.zeros(batch, dtype=torch.int64, device=features.device)
    output = evaluate(
        metadata, [features.reshape(batch, features.shape[-1])], index, index, batch
    )[0]
    return output.reshape(*features.shape[:-1], output_dim)

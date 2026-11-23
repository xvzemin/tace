"""Fixed nonlinear ACE expressions with externally owned parameters."""

from functools import lru_cache

import torch

from ...co3 import Projector
from ...mbe.ace.generalized import spherical_rows
from ...mbe.ace.ictd_symmetric_contraction import coupling_entries
from ...ops.common.edge import evaluate
from ...utils.metadata import numeric_rows
from ...utils.program import Program, activate


@lru_cache(maxsize=None)
def projector_matrix(degree):
    projector = Projector([(1, (degree, (-1) ** degree))], backend="torch").double()
    return projector(torch.eye(3**degree, dtype=torch.float64, device="cpu"))


def angular_layout(program, value, irreps, inverse=False):
    blocks, offset = [], 0
    for mul, ir in irreps:
        block = program.slice(value, offset, mul * ir.dim)
        rows, columns = (ir.dim, mul) if inverse else (mul, ir.dim)
        blocks.append(program.transpose(block, rows, columns))
        offset += mul * ir.dim
    return program.concatenate(blocks)


def product_metadata(product):
    """Return channel-wise coupling paths in angular-major layout."""
    irreps = product.irreps_in1, product.irreps_in2, product.irreps_out
    widths = {mul for entries in irreps for mul, _ in entries}
    if len(widths) != 1 or any(
        path.connection_mode != "uuu" or path.has_weight
        for path in product.instructions
    ):
        raise ValueError("Nonlinear ACE requires unweighted, equal-channel uuu paths.")
    width = widths.pop()
    dimensions = tuple(sum(ir.dim for _, ir in entries) for entries in irreps)
    offsets = [
        tuple(section.start // width for section in entries.slices())
        for entries in irreps
    ]
    cartesian = hasattr(product, "path_factors")
    paths = []
    for index, path in enumerate(product.instructions):
        indices = path.i_in1, path.i_in2, path.i_out
        if cartesian:
            entries = coupling_entries(product, index)
        else:
            if path.path_weight == 0:
                continue
            rows = spherical_rows(*(irreps[k][i].ir.l for k, i in enumerate(indices)))
            entries = ((a, b, c, value * path.path_weight) for a, b, c, value in rows)
        starts = tuple(offsets[k][i] for k, i in enumerate(indices))
        paths.extend(
            ((starts[0] + a, starts[1] + b, starts[2] + c), coefficient)
            for a, b, c, coefficient in entries
        )
    return width, dimensions, tuple(paths)


def tensor_product(program, product, left, right):
    irreps = product.irreps_in1, product.irreps_in2, product.irreps_out
    metadata = product_metadata(product)
    width, dimensions, _ = metadata
    values = (
        angular_layout(program, left, irreps[0]),
        angular_layout(program, right, irreps[1]),
        program.constant(width * dimensions[2], 0),
    )
    result = program.product(values, metadata, 2)
    if (
        hasattr(product, "path_factors")
        and product.project
        and product.projection_indices
    ):
        paths = []
        for index, (_, ir) in enumerate(irreps[2]):
            start = irreps[2].slices()[index].start // width
            if index in product.projection_indices:
                matrix = projector_matrix(ir.l)
                paths.extend(
                    ((start + a, start + b), float(matrix[a, b]))
                    for a, b in matrix.nonzero().tolist()
                )
            else:
                paths.extend(((start + a, start + a), 1.0) for a in range(ir.dim))
        result = program.product(
            (result, program.constant(width * dimensions[2], 0)),
            (width, (dimensions[2], dimensions[2]), tuple(paths)),
            1,
        )
    return angular_layout(program, result, irreps[2], inverse=True)


def build_product(product):
    """Build a spherical CGTP in its native ``mul_ir`` layout."""
    width, _, paths = product_metadata(product)
    irreps = product.irreps_in1, product.irreps_in2, product.irreps_out
    layouts = [
        tuple(
            (section.start + angular, ir.dim)
            for (_, ir), section in zip(entries, entries.slices())
            for angular in range(ir.dim)
        )
        for entries in irreps
    ]
    rows = tuple(
        (
            tuple(layouts[role][index][0] for role, index in enumerate(indices)),
            coefficient,
            width,
            tuple(layouts[role][index][1] for role, index in enumerate(indices)),
        )
        for indices, coefficient in paths
    )
    return numeric_rows(0, tuple(entries.dim for entries in irreps), rows)


def product_program(module, num_inputs):
    program, bindings = Program(), []

    def parameter(path, name, width):
        binding = path, name
        if binding not in bindings:
            bindings.append(binding)
        return program.input(num_inputs + bindings.index(binding), "shared", width)

    def linear(value, path):
        layer = module.get_submodule(path)
        operation = layer.linear
        weight = parameter(path, "weight", operation.weight_numel)
        grouped = [[] for _ in operation.irreps_out]
        offset = 0
        for instruction in operation.instructions:
            if instruction.i_in < 0:
                raise ValueError(
                    "Nonlinear ACE linear biases must be externally owned."
                )
            ci, _ = operation.irreps_in[instruction.i_in]
            co, _ = operation.irreps_out[instruction.i_out]
            section = operation.irreps_in.slices()[instruction.i_in]
            grouped[instruction.i_out].append(
                (
                    program.slice(value, section.start, section.stop - section.start),
                    program.scale(
                        program.slice(weight, offset, ci * co), instruction.path_weight
                    ),
                    ci,
                )
            )
            offset += ci * co
        bias = (
            parameter(path, "bias", layer.bias.numel())
            if layer.bias is not None
            else None
        )
        blocks, bias_offset = [], 0
        for (co, ir), terms in zip(operation.irreps_out, grouped):
            if terms:
                ci = sum(term[2] for term in terms)
                inputs = program.transpose(
                    program.concatenate([term[0] for term in terms]), ci, ir.dim
                )
                matrix = program.concatenate([term[1] for term in terms])
                block = program.transpose(
                    program.matmul(inputs, matrix, ir.dim, ci, co), ir.dim, co
                )
            else:
                block = program.constant(co * ir.dim, 0)
            if bias is not None and ir.is_scalar():
                block = program.binary(
                    "add", block, program.slice(bias, bias_offset, co)
                )
                bias_offset += co
            blocks.append(block)
        return program.concatenate(blocks)

    return program, bindings, linear, parameter


def generalized_product(program, block, operands, channels, dimensions):
    args = [*operands, program.constant(block.width * channels, 0)]
    return program.product(
        args, (channels, (*dimensions, block.width), block.angular_rows()), block.arity
    )


def build(module):
    """Build one expression for the factor projections, gates and product maps."""
    program, bindings, linear, parameter = product_program(module, 2)
    features = program.input(0, "edge", module.irreps_in.dim)
    embedding = program.input(1, "edge", module.num_hidden_channel)
    projected = program.gather(
        linear(features, "linear_up"), module.factor_index.tolist()
    )
    scalar = program.slice(projected, 0, module.num_hidden_channel)
    gate_input = program.concatenate((scalar, embedding))
    gate = module.gate[0]
    gate_weight = parameter("gate.0", "weight", gate.in_features * gate.out_features)
    gate_weight = program.scale(
        program.transpose(gate_weight, gate.out_features, gate.in_features), gate.alpha
    )
    gates = program.matmul(
        gate_input, gate_weight, 1, gate.in_features, gate.out_features
    )
    if gate.bias is not None:
        gates = program.binary(
            "add", gates, parameter("gate.0", "bias", gate.out_features)
        )
    gates = activate(program, gates, module.gate[1])
    factors = [
        program.slice(
            projected,
            module.num_hidden_channel + i * module.factor_dim,
            module.factor_dim,
        )
        for i in range(module.num_inputs)
    ]
    corr, contributions = factors[0], []
    if module.use_generalized_cg:
        op = module.generalized
        packed = [angular_layout(program, x, module.irreps_hidden) for x in factors]
        previous = [packed[0]]
    for degree, (start, stop) in enumerate(module.gate_slices):
        if degree and module.use_generalized_cg:
            block = op.blocks[degree - 1]
            dims = (op.angular_dims[block.start],) + (op.angular_dims[0],) * (
                degree - block.start
            )
            args = [previous[block.start]] + [
                packed[j if module.asymmetric else 0]
                for j in range(block.start + 1, degree + 1)
            ]
            value = generalized_product(program, block, args, op.channels, dims)
            previous.append(value)
            corr = angular_layout(
                program, value, op.irreps_orders[degree], inverse=True
            )
        elif degree:
            corr = tensor_product(
                program,
                module.aces[degree - 1].tp,
                corr,
                factors[degree if module.asymmetric else 0],
            )
        gate = program.gather(
            program.slice(gates, start, stop - start),
            getattr(module, f"gate_index_{degree + 1}").tolist(),
        )
        contributions.append(
            linear(program.binary("mul", corr, gate), f"coefs.{degree}")
        )
    output = linear(program.sum(contributions, module.irreps_coefs_out.dim), "linear")
    paths = tuple((*path.split("."), name) for path, name in bindings)
    return repr((tuple(program.nodes), ((output, 0, "edge"),))), paths


def build_path_gates(module):
    """Build the asymmetric product with one invariant coefficient per path."""
    program, bindings, linear, _ = product_program(module, 1)
    c, op = module.num_hidden_channel, module.generalized
    features = program.input(0, "edge", module.irreps_in.dim)
    projected = linear(features, "linear_up")
    order = module.factor_index.tolist()
    factors = [
        program.gather(projected, order[k : k + module.irreps_hidden.dim])
        for k in range(0, len(order), module.irreps_hidden.dim)
    ]
    gates = activate(
        program,
        program.slice(projected, module.gate_start, module.weight_numel),
        module.activation,
    )
    gate = program.gather(
        gates, [p for p in range(module.weight_numel) for _ in range(c)]
    )
    packed = [factors[0]]
    for degree, block in enumerate(op.blocks, 1):
        args = [packed[block.start], *factors[block.start + 1 : degree + 1]]
        dims = [op.angular_dims[block.start]] + [module.angular_dim] * (
            degree - block.start
        )
        if op.gated and degree == len(op.blocks):
            args.append(
                program.slice(
                    gate, (module.weight_numel - op.gate_count) * c, op.gate_count * c
                )
            )
            dims.append(op.gate_count)
        packed.append(generalized_product(program, block, args, c, dims))
    out_dim = module.irreps_coefs_out.dim // c
    rows = [[] for _ in packed]
    last = []
    base = 0
    for j, (dim, blocks, scale) in enumerate(module.groups):
        for stage, offset, paths, start, _ in blocks:
            if op.gated and stage == len(op.blocks):
                continue
            rows[stage].extend(
                ((offset + p * dim + m, start + p, base + m), scale)
                for p in range(paths)
                for m in range(dim)
            )
        if op.gated:
            offset, tiles = op.gate_slices[j]
            last.extend(
                ((offset + t * dim + m, base + m), scale)
                for t in range(tiles)
                for m in range(dim)
            )
        base += dim
    zero = program.constant(out_dim * c, 0)
    terms = [
        program.product(
            (value, gate, zero),
            (c, (op.angular_dims[i], module.weight_numel, out_dim), tuple(entries)),
            2,
        )
        for i, (value, entries) in enumerate(zip(packed, rows))
        if entries
    ]
    if last:
        terms.append(
            program.product(
                (packed[-1], zero), (c, (op.blocks[-1].width, out_dim), tuple(last)), 1
            )
        )
    output = linear(
        angular_layout(
            program,
            program.sum(terms, out_dim * c),
            module.irreps_coefs_out,
            inverse=True,
        ),
        "linear",
    )
    return repr((tuple(program.nodes), ((output, 0, "edge"),))), tuple(
        (path, name) for path, name in bindings
    )


def run(module, features, embedding=None):
    """Execute the cached expression without registering duplicate parameters."""
    inputs = [features] if embedding is None else [features, embedding]
    for path in module.eqx_bindings:
        value = module
        for name in path:
            value = getattr(value, name)
        if isinstance(value, torch.nn.ParameterList):
            value = torch.cat([part.reshape(-1) for part in value])
        inputs.append(value.reshape(1, -1))
    nodes = features.shape[0]
    indices = torch.arange(nodes, device=features.device, dtype=torch.long)
    return evaluate(module.eqx_metadata, inputs, indices, indices, nodes)[0]

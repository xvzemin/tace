"""Fixed nonlinear ACE expressions with externally owned parameters."""

from functools import lru_cache

import torch
from e3nn import o3

from ...co3 import Projector
from ...mbe.ace.ictd_symmetric_contraction import coupling_entries
from ...ops.common.edge import evaluate
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
            coefficients = (
                o3.wigner_3j(
                    *(entries[i].ir.l for entries, i in zip(irreps, indices)),
                    dtype=torch.float64,
                    device="cpu",
                )
                * path.path_weight
            )
            entries = [
                (*row, float(coefficients[tuple(row)]))
                for row in coefficients.nonzero().tolist()
            ]
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
    return repr((0, tuple(entries.dim for entries in irreps), rows))


def build(module):
    """Build one expression for all projections, gates and recursive products."""
    program, bindings = Program(), []
    features = program.input(0, "edge", module.irreps_in.dim)
    embedding = program.input(1, "edge", module.num_hidden_channel)

    def parameter(path, name, width):
        binding = path, name
        if binding not in bindings:
            bindings.append(binding)
        return program.input(2 + bindings.index(binding), "shared", width)

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
    for degree, (start, stop) in enumerate(module.gate_slices):
        if degree:
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
    return repr((tuple(program.nodes), ((output, 0, "edge"),))), tuple(bindings)


def run(module, features, embedding):
    """Execute the cached expression without registering duplicate parameters."""
    inputs = [features, embedding]
    for path, name in module.eqx_bindings:
        value = getattr(module.get_submodule(path), name)
        if isinstance(value, torch.nn.ParameterList):
            value = torch.cat([part.reshape(-1) for part in value])
        inputs.append(value.reshape(1, -1))
    nodes = features.shape[0]
    indices = torch.arange(nodes, device=features.device, dtype=torch.long)
    return evaluate(module.eqx_metadata, inputs, indices, indices, nodes)[0]

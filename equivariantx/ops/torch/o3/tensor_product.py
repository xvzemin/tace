"""Factorized spherical products in independent and compact coordinates."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.program import Program
from ... import config
from ...common.torch.literal_bilinear import evaluate as literal_product
from ...common.torch.sparse_bilinear import evaluate as sparse_product


def reference(module, x, y, weight):
    """Original quadrature or matrix contraction with ordinary autograd."""
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
    x, y, weight = (
        value.expand(shape + (value.shape[-1],)) for value in (x, y, weight)
    )
    result = x.new_zeros(shape + (module.output_dim,))
    if not module.routes:
        return result + (x.sum(-1) + y.sum(-1) + weight.sum(-1)).unsqueeze(-1) * 0
    inputs = []
    for value, entries in zip((x, y), module.inputs):
        transformed = []
        for name, mul, dim in entries:
            features = value.index_select(-1, getattr(module, f"{name}_index"))
            factors = weight.index_select(-1, getattr(module, f"{name}_weight_index"))
            features = (features * factors).reshape(shape + (mul, dim))
            features = features * getattr(module, f"{name}_scale")
            transformed.append(
                module.transform(features, getattr(module, f"{name}_basis"))
            )
        inputs.append(transformed)
    mul = module.num_channels
    for i, j, name, dim in module.routes:
        value = module.product(inputs[0][i], inputs[1][j], name)
        factors = weight.index_select(-1, getattr(module, f"{name}_weight_index"))
        value = value * factors.reshape(shape + (mul, dim))
        value = value * getattr(module, f"{name}_scale")
        result = result.index_add(
            -1, getattr(module, f"{name}_index"), value.reshape(shape + (mul * dim,))
        )
    return result


def compact(module, x, y, weight, angular, *, native=False):
    """Reuse weighted inputs and choose compact or structured angular routes."""
    shape = torch.broadcast_shapes(x.shape[:-1], y.shape[:-1], weight.shape[:-1])
    result = x.new_zeros(shape + (module.output_dim,))
    if not module.routes:
        return result + (x.sum(-1) + y.sum(-1) + weight.sum(-1)).unsqueeze(-1) * 0
    compact_routes = getattr(module, "compact_routes", {})
    selected = {
        name
        for _, _, name, _ in module.routes
        if name in compact_routes
        and (
            x.is_cuda
            and (native or len(module.program_entries[name]) <= 256)
            or (
                compact_routes[name][1]
                and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
            )
        )
    }
    if not selected:
        return reference(module, x, y, weight)
    inputs, transformed = [], []
    for side, (value, entries) in enumerate(zip((x, y), module.inputs)):
        raw, basis = [], {}
        required = {
            (i, j)[side] for i, j, name, _ in module.routes if name not in selected
        }
        for index, (name, mul, dim) in enumerate(entries):
            features = value.index_select(-1, getattr(module, f"{name}_index"))
            factors = weight.index_select(-1, getattr(module, f"{name}_weight_index"))
            features = features.reshape(value.shape[:-1] + (mul, dim))
            factors = factors.reshape(weight.shape[:-1] + (mul, dim))
            features = features * factors
            raw.append(features)
            if index in required:
                basis[index] = module.transform(
                    features * getattr(module, f"{name}_scale"),
                    getattr(module, f"{name}_basis"),
                )
        inputs.append(raw)
        transformed.append(basis)
    mul = module.num_channels
    for i, j, name, dim in module.routes:
        if name in selected:
            value = angular(module, inputs[0][i], inputs[1][j], name)
        else:
            value = module.product(transformed[0][i], transformed[1][j], name)
            value = value * getattr(module, f"{name}_scale")
        factors = weight.index_select(-1, getattr(module, f"{name}_weight_index"))
        value = value * factors.reshape(weight.shape[:-1] + (mul, dim))
        result = result.index_add(
            -1, getattr(module, f"{name}_index"), value.reshape(shape + (mul * dim,))
        )
    return result


def angular(module, x, y, name):
    dimensions, entries = module.compact_routes[name]
    if not x.is_cuda and entries:
        return literal_product(x, y, dimensions, entries)
    tables = tuple(getattr(module, f"{name}_sparse_{k}") for k in range(9))
    return sparse_product(x, y, tables)


def evaluate(module, x, y, weight):
    if config.TORCH_REFERENCE or not config.TORCH_FACTORIZED_PRODUCT:
        return reference(module, x, y, weight)
    if getattr(module, "method", None) == "quadrature" and not (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        return reference(module, x, y, weight)
    return compact(module, x, y, weight, angular)


def convolution_program(module):
    """Encode compact factors once for shared tiled radial execution."""
    if not module.routes or len(getattr(module, "compact_routes", {})) != len(
        module.routes
    ):
        return None
    program = Program()
    features = program.input(0, "source", module.input_dim1)
    attributes = program.input(1, "edge", module.input_dim2)
    weights = program.input(
        2, "shared" if module.shared_weights else "edge", module.weight_numel
    )
    inputs = []
    channels = module.num_channels
    for value, groups in zip((features, attributes), module.inputs):
        values = []
        for name, mul, dim in groups:
            index = module.constants[f"{name}_index"].tolist()
            wi = module.constants[f"{name}_weight_index"].tolist()
            weighted = program.binary(
                "mul", program.gather(value, index), program.gather(weights, wi)
            )
            weighted = program.transpose(weighted, mul, dim)
            if mul == 1 and channels != 1:
                weighted = program.gather(
                    weighted, tuple(k for k in range(dim) for _ in range(channels))
                )
            values.append(weighted)
        inputs.append(values)
    outputs = []
    for i, j, name, dim in module.routes:
        dimensions = module.compact_routes[name][0]
        polynomial = (
            channels,
            dimensions,
            tuple(
                ((a, b, c), coefficient)
                for a, b, c, coefficient in module.program_entries[name]
            ),
        )
        value = program.product(
            (inputs[0][i], inputs[1][j], program.constant(channels * dim, 0)),
            polynomial,
            2,
        )
        value = program.transpose(value, dim, channels)
        value = program.binary(
            "mul",
            value,
            program.gather(weights, module.constants[f"{name}_weight_index"].tolist()),
        )
        outputs.append(
            program.scatter(
                value, module.constants[f"{name}_index"].tolist(), module.output_dim
            )
        )
    result = program.sum(outputs, module.output_dim)
    return repr((tuple(program.nodes), ((result, 0, "target"),)))


def convolution_reference(
    module, features, attributes, edge_index, weight, num_nodes, radial, network
):
    """Materialize messages and radial factors with ordinary Torch operations."""
    from ...common.radial import materialize, radial_reference

    if network is not None:
        weight = radial_reference(network, materialize(radial, edge_index))
    weight = module.tensor_product.weight if weight is None else weight
    messages = reference(
        module.tensor_product, features[edge_index[0]], attributes, weight
    )
    return messages.new_zeros((num_nodes, module.output_dim)).index_add(
        0, edge_index[1], messages
    )


def convolve(
    module,
    features,
    attributes,
    edge_index,
    weight,
    num_nodes,
    radial,
    network,
    *,
    backend="torch",
):
    """Share radial GEMMs and recursively recompute requested edge adjoints."""
    from ...common.convolution import radial_input
    from ...common.radial import radial_features, split_radial
    from ...common.torch.edge import evaluate_recomputed, evaluate_torch

    captured = torch.compiler.is_compiling() or get_proxy_mode() is not None
    eager_torch = backend == "torch" and not captured
    if (
        config.TORCH_REFERENCE
        or getattr(module, "program_metadata", None) is None
        or eager_torch
        and module.radial_mode == "last"
    ):
        return convolution_reference(
            module, features, attributes, edge_index, weight, num_nodes, radial, network
        )
    if (
        captured
        and module.radial_mode == "full"
        and network is not None
        and (backend == "torch" or config.EDGE_PROGRAM_INLINE)
    ):
        # Save radial operands; let joint compilation reuse angular intermediates.
        program, inputs = Program(), []
        value = radial_input(program, inputs, radial, network)
        if program.size(value) != module.weight_numel:
            raise ValueError("Radial network output must equal weight_numel.")
        metadata = repr((tuple(program.nodes), ((value, 0, "edge"),)))
        source, target = edge_index.unbind(0)
        weights = evaluate_recomputed(metadata, inputs, source, target, num_nodes)[0]
        return evaluate_torch(
            module.program_metadata,
            (features, attributes, weights),
            source,
            target,
            num_nodes,
            optimized=True,
        )[0]
    if network is None:
        weight = module.tensor_product.weight if weight is None else weight
        inputs = [features, attributes, weight.reshape(-1, module.weight_numel)]
        metadata = module.program_metadata
        if weight.ndim == 1 and not module.tensor_product.shared_weights:
            program = Program()
            x = program.input(0, "source", module.input_dim)
            y = program.input(1, "edge", module.edge_dim)
            w = program.input(2, "shared", module.weight_numel)
            result = program.compose(metadata, (x, y, w))[0]
            metadata = repr((tuple(program.nodes), ((result, 0, "target"),)))
    else:
        if module.radial_mode == "last":
            hidden, network = split_radial(network)
            radial = radial_features(hidden, radial, edge_index)
        program = Program()
        inputs = [features, attributes]
        x = program.input(0, "source", module.input_dim)
        y = program.input(1, "edge", module.edge_dim)
        w = radial_input(program, inputs, radial, network)
        if program.size(w) != module.weight_numel:
            raise ValueError("Radial network output must equal weight_numel.")
        result = program.compose(module.program_metadata, (x, y, w))[0]
        metadata = repr((tuple(program.nodes), ((result, 0, "target"),)))
    if backend == "cuda":
        from ...common.edge import evaluate

        return evaluate(metadata, inputs, *edge_index.unbind(0), num_nodes)[0]
    if module.radial_mode == "full" or config.TORCH_RECOMPUTE:
        return evaluate_recomputed(metadata, inputs, *edge_index.unbind(0), num_nodes)[
            0
        ]
    return evaluate_torch(
        metadata, inputs, *edge_index.unbind(0), num_nodes, optimized=True
    )[0]

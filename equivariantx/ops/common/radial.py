"""Radial networks evaluated inside tiled convolution programs."""

import math

import torch

from ...utils.metadata import parse_metadata, static_dimensions
from ...utils.program import Program, activate
from .. import config
from ..precision import validate_precision


def radial_program(network, inputs, program, value):
    """Append a radial network to an edge expression graph.

    Parameters
    ----------
    network : torch.nn.Module
        Sequential scalar network. Linear layers, smooth activations, layer
        normalization and RMS normalization are supported.
    inputs : list of torch.Tensor
        Program inputs. Network parameters are appended without copying them.
    program : Program
        Destination expression graph.
    value : int
        Input expression index.
    """

    def parameter(tensor):
        slot = len(inputs)
        inputs.append(tensor.reshape(1, -1))
        return program.input(slot, "shared", math.prod(static_dimensions(tensor)))

    for layer in radial_layers(network):
        if isinstance(layer, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            width = program.size(value)
            if tuple(layer.normalized_shape) != (width,):
                raise ValueError("Radial normalization must act on the channel axis.")
            eps = layer.eps
            if eps is None:
                eps = torch.finfo(inputs[0].dtype).eps
            value = program.add(
                "normalize",
                width,
                (value,),
                (eps, isinstance(layer, torch.nn.LayerNorm)),
            )
            if layer.weight is not None:
                value = program.binary("mul", value, parameter(layer.weight))
            if getattr(layer, "bias", None) is not None:
                value = program.binary("add", value, parameter(layer.bias))
        elif hasattr(layer, "weight") or hasattr(layer, "weights"):
            if isinstance(layer, torch.nn.Linear):
                weight = layer.weight.T
            elif hasattr(layer, "get_weight"):
                weight = layer.get_weight()
            elif hasattr(layer, "weights"):
                weight = layer.weights
            else:
                weight = layer.weight
                if hasattr(layer, "alpha"):
                    weight = weight * layer.alpha
            if hasattr(layer, "h_in"):
                scale = layer.h_in * layer.var_in
                if layer.act is None:
                    scale /= layer.var_out
                weight = weight / math.sqrt(scale)
            if weight.ndim != 2 or weight.shape[0] != program.size(value):
                raise ValueError("Radial linear weight has an incompatible shape.")
            value = program.matmul(
                value, parameter(weight), 1, *static_dimensions(weight)
            )
            if getattr(layer, "bias", None) is not None:
                value = program.binary("add", value, parameter(layer.bias))
            if hasattr(layer, "h_in") and layer.act is not None:
                value = program.scale(
                    activate(program, value, layer.act), math.sqrt(layer.var_out)
                )
        else:
            value = activate(program, value, layer)
    return value


def radial_features(network, radial, edge_index, projection=None):
    """Evaluate hidden layers with higher-order normalization derivatives."""
    if config.TORCH_REFERENCE:
        value = radial_reference(network, materialize(radial, edge_index))
    else:
        from .convolution import radial_input
        from .torch.edge import evaluate_torch

        program, inputs = Program(), []
        root = radial_input(program, inputs, radial, network)
        validate_precision(inputs)
        if not isinstance(radial, tuple) and radial.shape[0] == 1:
            source = target = edge_index.new_zeros(1)
        else:
            source, target = edge_index
        metadata = repr((tuple(program.nodes), ((root, 0, "edge"),)))
        value = evaluate_torch(metadata, inputs, source, target, 0, optimized=True)[0]
    if projection is not None and value.shape[-1] + 1 == projection.shape[0]:
        value = torch.cat((value, torch.ones_like(value[:, :1])), -1)
    return value


def radial_layers(network):
    """Flatten supported sequential containers while retaining their parameters."""
    layers = network.mlp if hasattr(network, "mlp") else network
    if isinstance(layers, torch.nn.Module) and not isinstance(
        layers, (torch.nn.Sequential, torch.nn.ModuleList)
    ):
        return (layers,)
    result = []
    for layer in layers:
        if isinstance(layer, (torch.nn.Sequential, torch.nn.ModuleList)):
            result.extend(radial_layers(layer))
        else:
            result.append(layer)
    return tuple(result)


def split_radial(network):
    """Separate hidden layers from the last projection and its pointwise tail."""
    layers = radial_layers(network)
    for index in range(len(layers) - 1, -1, -1):
        layer = layers[index]
        if not isinstance(layer, (torch.nn.LayerNorm, torch.nn.RMSNorm)) and (
            hasattr(layer, "weight") or hasattr(layer, "weights")
        ):
            return layers[:index], layers[index:]
    return layers, ()


def cast_radial(radial, dtype):
    """Convert radial inputs to the network dtype without materializing gathers."""
    if isinstance(radial, tuple):
        return tuple((value.to(dtype=dtype), storage) for value, storage in radial)
    return radial.to(dtype=dtype)


def align_projection(radial, projection):
    """Align CUDA FP32 matrix strides, including an appended affine bias."""
    if projection.is_cuda and projection.dtype == torch.float32 and projection.numel():
        extra = -projection.shape[0] % 8
        if extra:
            radial = torch.nn.functional.pad(radial, (0, extra))
            projection = torch.nn.functional.pad(projection, (0, 0, 0, extra))
    return radial, projection


def compose_radial(metadata, inputs, slot, network):
    """Replace an edge input by a tiled radial network evaluation."""
    nodes, outputs = parse_metadata(metadata)
    inputs = list(inputs)
    program = Program()
    radial = inputs[slot]
    parts = []
    for i, (feature, storage) in enumerate(radial_parts(radial)):
        index = slot if i == 0 else len(inputs)
        if i == 0:
            inputs[slot] = feature
        else:
            inputs.append(feature)
        parts.append(
            program.input(index, storage, static_dimensions(feature, (-1,))[0])
        )
    value = program.concatenate(parts)
    value = radial_program(network, inputs, program, value)
    translated = {}
    for i, (op, size, args, data) in enumerate(nodes):
        if op == "input" and data[0] == slot:
            if size != program.size(value):
                raise ValueError(
                    "Radial network output does not match convolution weights."
                )
            translated[i] = value
        else:
            translated[i] = program.add(
                op, size, tuple(translated[j] for j in args), data
            )
    outputs = tuple((translated[root], output, kind) for root, output, kind in outputs)
    return repr((tuple(program.nodes), outputs)), inputs


def convolve(kind, metadata, operands, edge_index, network):
    """Evaluate a full radial network and tensor product in bounded edge tiles."""
    from .edge import evaluate

    output_role = 4 if kind == "o3" else 6
    output = operands[output_role]
    inputs = []
    program = Program()
    values = []
    for role, tensor in enumerate(operands):
        if role == 1:
            parts = []
            for feature, storage in radial_parts(tensor):
                parts.append(
                    program.input(
                        len(inputs), storage, static_dimensions(feature, (-1,))[0]
                    )
                )
                inputs.append(feature)
            values.append(program.concatenate(parts))
            continue
        if role == output_role:
            values.append(program.constant(static_dimensions(tensor, (-1,))[0], 0))
            continue
        storage = (
            "source"
            if role == 0
            else "shared"
            if role == 2 or tensor.shape[0] == 1
            else "edge"
        )
        # Program widths are model constants, unlike dynamic node/edge counts.
        width = (
            math.prod(static_dimensions(tensor))
            if storage == "shared"
            else static_dimensions(tensor, (-1,))[0]
        )
        values.append(program.input(len(inputs), storage, width))
        inputs.append(tensor.reshape(1, -1) if storage == "shared" else tensor)
    values[1] = radial_program(network, inputs, program, values[1])
    projection_shape = static_dimensions(operands[2])
    width = projection_shape[0] or projection_shape[1]
    if program.size(values[1]) + 1 == width:
        values[1] = program.concatenate((values[1], program.constant(1, 1)))
    if program.size(values[1]) != width:
        raise ValueError("Radial network output does not match projection input.")
    root = program.add(
        "convolution",
        static_dimensions(output, (-1,))[0],
        values,
        (
            kind,
            metadata,
            output_role,
            False,
            projection_shape,
            0,
        ),
    )
    return evaluate(
        repr((tuple(program.nodes), ((root, 0, "target"),))),
        inputs,
        edge_index[0],
        edge_index[1],
        output.shape[0],
    )[0]


def materialize(radial, edge_index):
    """Gather partitioned radial inputs for the PyTorch reference."""
    if not isinstance(radial, tuple):
        return radial
    parts = []
    for value, kind in radial_parts(radial):
        if kind in ("edge", "shared"):
            if value.shape[0] == 1:
                value = value.expand(edge_index.shape[1], -1)
            elif kind == "shared":
                raise ValueError("Shared radial inputs must contain exactly one row.")
            elif value.shape[0] != edge_index.shape[1]:
                raise ValueError("Radial edge inputs must match the number of edges.")
        elif kind in ("source", "target"):
            value = value[edge_index[0 if kind == "source" else 1]]
        else:
            raise ValueError("Radial storage must be edge, source, target or shared.")
        parts.append(value)
    return torch.cat(parts, dim=-1)


def radial_parts(radial):
    """Validate radial partitions and normalize broadcast rows to shared storage."""
    parts = radial if isinstance(radial, tuple) else ((radial, "edge"),)
    if not parts:
        raise ValueError("Radial partitions must not be empty.")
    result = []
    for value, kind in parts:
        if kind not in ("edge", "source", "target", "shared"):
            raise ValueError("Radial storage must be edge, source, target or shared.")
        if value.ndim != 2:
            raise ValueError("Radial inputs must have shape (rows, channels).")
        if kind == "shared" and value.shape[0] != 1:
            raise ValueError("Shared radial inputs must contain exactly one row.")
        if kind == "edge" and value.shape[0] == 1:
            kind = "shared"
        result.append((value, kind))
    return tuple(result)


def radial_reference(network, value):
    """Evaluate independent Torch layers with differentiable normalization."""
    validate_precision((value,))
    for layer in radial_layers(network):
        validate_precision((tuple(layer.parameters()), tuple(layer.buffers())))
        if isinstance(layer, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
            if tuple(layer.normalized_shape) != (value.shape[-1],):
                raise ValueError("Radial normalization must act on the channel axis.")
            if isinstance(layer, torch.nn.LayerNorm):
                value = value - value.mean(-1, keepdim=True)
            eps = layer.eps
            if eps is None:
                eps = torch.finfo(value.dtype).eps
            value = value * (value.square().mean(-1, keepdim=True) + eps).rsqrt()
            if layer.weight is not None:
                value = value * layer.weight
            if getattr(layer, "bias", None) is not None:
                value = value + layer.bias
        else:
            value = layer(value)
    return value

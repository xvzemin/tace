"""Translate radial networks into tensor-expression programs."""

import math

import torch

from ..ops.program import activate


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
        return program.input(slot, "shared", int(tensor.numel()))

    layers = network.mlp if hasattr(network, "mlp") else network
    for layer in layers:
        if isinstance(layer, torch.nn.Sequential):
            value = radial_program(layer, inputs, program, value)
        elif isinstance(layer, (torch.nn.LayerNorm, torch.nn.RMSNorm)):
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
                value, parameter(weight), 1, *(int(dim) for dim in weight.shape)
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

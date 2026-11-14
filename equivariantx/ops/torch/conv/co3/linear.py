"""Cartesian layout, geometry and compact coefficient execution."""

import torch

from .....utils.layout import permute
from .... import config


def reference(module, features, weight, bias):
    """Expand Cartesian entries and contract individual paths with ordinary AD."""
    shape = torch.broadcast_shapes(
        features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
    )
    expanded = features.index_select(-1, module.input_expansion)
    zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
    outputs = []
    for i, ((mul, ir), group) in enumerate(zip(module.cartesian_out, module.groups)):
        value = None
        for ins, section in group:
            if ins.i_in < 0 or not all(ins.path_shape):
                continue
            x = expanded[..., module.slices_in[ins.i_in]].reshape(
                *features.shape[:-1], ins.path_shape[0], ir.dim
            )
            w = weight[..., section].reshape(*weight.shape[:-1], *ins.path_shape)
            part = torch.einsum("...ui,...uv->...vi", x, w) * ins.path_weight
            value = part if value is None else value + part
        if value is None:
            dim = ir.spherical_dim if module.output_basis == "spherical" else ir.dim
            value = features.new_zeros((*shape, mul, dim)) + zero
        else:
            matrix = getattr(module, f"reference_basis_{ir.l}")
            value = value @ matrix
            if module.output_basis == "cartesian":
                value = value @ matrix.T
        for ins, section in group:
            if ins.i_in < 0:
                value = value + bias[..., section].unsqueeze(-1)
        outputs.append(value.expand(*shape, *value.shape[-2:]).flatten(-2))
    return (
        torch.cat(outputs, -1) + zero
        if outputs
        else features[..., :0].expand(*shape, 0) + zero
    )


def basis_first_available(module, features, weight, index):
    """Use a smaller shared channel GEMM for wide partial rank-three tensors."""
    dest, dim, mul_in, mul_out = module.compact_groups[index]
    return (
        config.TORCH_COMPACT_LINEAR_BASIS_FIRST
        and features.device.type == "cpu"
        and features.ndim == 2
        and weight.ndim == 1
        and module.output_basis == "spherical"
        and module.input_identity
        and module.weight_identity[index]
        and module.uniform_scales[index]
        and module.cartesian_out[dest].ir.l == 3
        and dim == 18
        and mul_in == mul_out
        and mul_in in (128, 256, 512, 1024)
    )


def evaluate(module, features, weight, bias):
    """Evaluate validated Cartesian operands."""
    if config.TORCH_REFERENCE:
        return reference(module, features, weight, bias)
    shape = torch.broadcast_shapes(
        features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
    )
    zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
    if not module.input_identity:
        features = (
            permute(features, module.input_index, module.input_inverse)
            if module.input_permutation
            else features.index_select(-1, module.input_index)
        )
    outputs = [None] * len(module.cartesian_out)
    for i, (dest, dim, mul_in, mul_out) in enumerate(module.compact_groups):
        section = module.input_sections[i]
        x = features[..., section]
        x = x.reshape(*features.shape[:-1], dim, mul_in)
        w = weight[..., module.weight_sections[i]]
        if module.weight_rows[i]:
            rows = (
                module.weight_sections[i].stop - module.weight_sections[i].start
            ) // mul_out
            w = w.reshape(*weight.shape[:-1], rows, mul_out)
        if not module.weight_identity[i]:
            w = w.index_select(
                -2 if module.weight_rows[i] else -1,
                getattr(module, f"weight_index_{i}"),
            )
        w = w.reshape(*weight.shape[:-1], mul_in, mul_out)
        if not module.uniform_scales[i]:
            w = w * getattr(module, f"weight_scale_{i}")
        basis = getattr(module, f"basis_{i}")
        if basis_first_available(module, features, weight, i):
            value = ((basis.T @ x) @ w).transpose(-1, -2)
        else:
            value = (x @ w).transpose(-1, -2) @ basis
        outputs[dest] = value if outputs[dest] is None else outputs[dest] + value
    for i, ((mul, ir), group) in enumerate(zip(module.cartesian_out, module.groups)):
        if outputs[i] is None:
            dim = ir.spherical_dim if module.output_basis == "spherical" else ir.dim
            outputs[i] = features.new_zeros((*shape, mul, dim)) + zero
        for ins, section in group:
            if ins.i_in < 0:
                outputs[i] = outputs[i] + bias[..., section].unsqueeze(-1)
        outputs[i] = outputs[i].expand(*shape, *outputs[i].shape[-2:])
    return (
        torch.cat([value.flatten(-2) for value in outputs], -1) + zero
        if outputs
        else features[..., :0].expand(*shape, 0) + zero
    )

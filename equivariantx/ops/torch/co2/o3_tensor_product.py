"""Cartesian reference paths and compact projected angular contractions."""

import math

import torch

from ....co2.restriction import transverse_quarter_turn
from ... import config
from ..co3 import basis as basis_kernel
from . import restriction as restriction_kernel


def use_reference(module, features):
    return config.TORCH_REFERENCE and (
        getattr(module, "backend", "torch") == "torch" or not features.is_cuda
    )


def apply_basis(module, features, ordinary):
    """Apply ordinary basis maps without dispatching a reference's child backend."""
    if not ordinary or isinstance(module, torch.nn.Identity):
        return module(features)
    if hasattr(module, "inverse"):
        return basis_kernel.change_basis_reference(module, features)
    values = [
        features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
        for (mul, ir), section in zip(module.irreps_in, module.slices)
    ]
    values = basis_kernel.project_reference(module, values)
    return (
        torch.cat([value.flatten(-2) for value in values], -1)
        if values
        else features[..., :0]
    )


def unprojected(module, features, vectors, weight, *, ordinary=False):
    """Retain the original unprojected transverse Cartesian construction."""
    direction = (
        vectors / torch.linalg.vector_norm(vectors, dim=-1, keepdim=True)
    ).unsqueeze(-2)
    shape = torch.broadcast_shapes(
        features.shape[:-1], vectors.shape[:-1], weight.shape[:-1]
    )
    zero = features[..., :0].sum() + vectors[..., :0].sum() + weight[..., :0].sum()
    outputs = [
        features.new_zeros((*shape, mul, dim)) + zero
        for mul, dim in module.output_shapes
    ]
    inputs = {}
    for index, mul, l, dim, section in module.input_paths:
        x = features[..., section].reshape(*features.shape[:-1], mul, dim)
        restriction = module.restrictions[str(l)]
        inputs[index] = (
            restriction_kernel.restrict_reference(restriction, x, direction)
            if ordinary
            else restriction(x, direction)
        )
    offset = 0
    for ins, (l1, l2, l3), coefficients in zip(
        module.instructions, module.paths, module.coefficients
    ):
        value = None
        for m, coefficient in enumerate(coefficients):
            if value is not None:
                value = (direction.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
            if coefficient == 0.0:
                continue
            local = inputs[ins.i_in1][m]
            if (l1 + l2 + l3) % 2:
                local = (
                    restriction_kernel.quarter_turn_reference(local, direction)
                    if ordinary
                    else transverse_quarter_turn(
                        local, direction, backend=getattr(module, "backend", "torch")
                    )
                )
            term = local * (coefficient * ins.path_weight)
            value = term if value is None else value + term
        if value is not None:
            for _ in range(l3 - len(coefficients) + 1):
                value = (direction.unsqueeze(-1) * value.unsqueeze(-2)).flatten(-2)
        if value is None:
            value = (
                features.new_zeros(
                    (*shape, module.input_multiplicities[ins.i_in1], 3**l3)
                )
                + zero
            )
        if ins.has_weight:
            size = math.prod(ins.path_shape)
            w = weight[..., offset : offset + size].reshape(
                *weight.shape[:-1], *ins.path_shape
            )
            offset += size
            value = (
                torch.einsum("...ud,...uw->...wd", value, w.squeeze(-2))
                if ins.connection_mode == "uvw"
                else value * w
            )
        outputs[ins.i_out] = outputs[ins.i_out] + value
    return (
        torch.cat([v.flatten(-2) for v in outputs], -1)
        if outputs
        else features.new_zeros((*shape, 0)) + zero
    )


def reference(module, features, vectors, weight):
    """Evaluate unprojected Cartesian products using ordinary Torch operators."""
    return unprojected(module, features, vectors, weight, ordinary=True)


def contract(module, features, vectors, weight):
    return unprojected(
        module, features, vectors, weight, ordinary=use_reference(module, features)
    )


def compact(module, features, vectors, weight):
    """Reuse compact spherical inputs and angle maps across weighted paths."""
    shape = torch.broadcast_shapes(
        features.shape[:-1], vectors.shape[:-1], weight.shape[:-1]
    )
    zero = features[..., :0].sum() + vectors[..., :0].sum() + weight[..., :0].sum()
    inputs = {
        index: features[..., section].reshape(*features.shape[:-1], mul, dim)
        for index, mul, dim, section in module.spherical_input_paths
    }
    outputs = [None] * len(module.spherical_output_shapes)
    direction = vectors.unsqueeze(-2)
    offset = 0
    angular = {}
    for ins, degrees in zip(module.instructions, module.paths):
        key = (ins.i_in1, degrees)
        if key not in angular:
            angular[key] = module.couplings["_".join(map(str, degrees))](
                inputs[ins.i_in1], direction
            )
        value = angular[key]
        if ins.has_weight:
            size = math.prod(ins.path_shape)
            w = weight[..., offset : offset + size].reshape(
                *weight.shape[:-1], *ins.path_shape
            )
            offset += size
            value = (
                torch.einsum("...ud,...uw->...wd", value, w.squeeze(-2))
                if ins.connection_mode == "uvw"
                else value * w
            )
        value = value * ins.path_weight
        previous = outputs[ins.i_out]
        outputs[ins.i_out] = value if previous is None else previous + value
    outputs = [
        features.new_zeros((*shape, mul, dim))
        if value is None
        else value.expand(*shape, mul, dim)
        for value, (mul, dim) in zip(outputs, module.spherical_output_shapes)
    ]
    return (
        torch.cat([value.flatten(-2) for value in outputs], -1)
        if outputs
        else features.new_zeros((*shape, 0))
    ) + zero


def use_compact(module, features):
    return (
        not use_reference(module, features)
        and config.TORCH_TRANSVERSE_PRODUCT
        and hasattr(module, "couplings")
        and getattr(module, "compact_product", True)
        and (module.project or module.output_basis == "spherical")
    )


def forward(module, features, vectors, weight):
    if use_compact(module, features):
        return module.compact_projection(
            compact(module, module.compact_embedding(features), vectors, weight)
        )
    ordinary = use_reference(module, features)
    return apply_basis(
        module.projection,
        unprojected(
            module,
            apply_basis(module.embedding, features, ordinary),
            vectors,
            weight,
            ordinary=ordinary,
        ),
        ordinary,
    )


def scatter(module, features, vectors, edge_index, weight, num_nodes):
    source, target = edge_index
    if use_compact(module, features):
        features = module.compact_embedding(features)
        message = compact(module, features[source], vectors, weight)
        nodes = message.new_zeros((num_nodes, message.shape[-1])).index_add(
            0, target, message
        )
        return module.compact_projection(nodes)
    ordinary = use_reference(module, features)
    features = apply_basis(module.embedding, features, ordinary)
    message = unprojected(module, features[source], vectors, weight, ordinary=ordinary)
    nodes = message.new_zeros((num_nodes, module.output_dim)).index_add(
        0, target, message
    )
    return apply_basis(module.projection, nodes, ordinary)

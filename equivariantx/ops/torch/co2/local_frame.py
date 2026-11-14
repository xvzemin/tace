"""Cartesian frame restriction and synthesis with rectangular tensor maps."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....co2.tensor_product import quarter_turn
from ... import config
from ...common.torch import tensor_axis
from . import restriction as restriction_kernel


def matrix_enabled(module, features, contract, rank, channels):
    """Share bounded direct-frame maps in the default compiled CPU schedule."""
    return (
        config.TORCH_CARTESIAN_FRAME_MATRIX
        and not config.TORCH_REFERENCE
        and features.device.type == "cpu"
        and module.method == "direct"
        and contract is tensor_axis.primal
        and 2 <= rank <= 4
        and config.CARTESIAN_FRAME_MATRIX_MIN_CHANNELS <= channels <= 1024
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )


def local_matrix(module, rotation, rank, orders):
    """Map Cartesian indices to retained planar orders once per node."""
    direction = rotation[..., 1, :]
    axes = (
        torch.stack((rotation[..., 2, :], rotation[..., 0, :]), -2)
        .transpose(-1, -2)
        .unsqueeze(-3)
    )
    values = []
    for m in range(orders):
        basis = getattr(module, f"basis_{rank}_{m}").to(rotation)
        value = tensor_axis.transform(basis, axes, m, tensor_axis.primal)
        for _ in range(rank - m):
            value = (value.unsqueeze(-1) * direction[..., None, None, :]).flatten(-2)
        values.append(value)
    return torch.cat(values, -2)


def global_matrix(module, rotation, rank):
    """Rotate the synthesis rows once for all channels at each node."""
    return tensor_axis.transform(
        getattr(module, f"synthesis_{rank}").to(rotation).T,
        rotation.transpose(-1, -2).unsqueeze(-3),
        rank,
        tensor_axis.primal,
    )


def to_local(module, features, rotation, contract=None):
    """Contract Cartesian indices without global edge-frame expansion."""
    default_contract = contract is None
    contract = tensor_axis.select() if contract is None else contract
    rotation = rotation.to(features)
    direction = rotation[..., 1, :].unsqueeze(-2)
    axes = torch.stack((rotation[..., 2, :], rotation[..., 0, :]), -2).unsqueeze(-3)
    shape = torch.broadcast_shapes(features.shape[:-1], rotation.shape[:-2])
    outputs = [[] for _ in module.irreps_out]
    matrices = {}
    for section, mul, l, odd, paths in module.entries:
        tensor = (
            features[..., section]
            .reshape(*features.shape[:-1], mul, 3**l)
            .expand(*shape, mul, 3**l)
        )
        matrix = default_contract and matrix_enabled(module, features, contract, l, mul)
        if matrix:
            if l not in matrices:
                matrices[l] = local_matrix(module, rotation, l, len(paths))
            values = (tensor @ matrices[l].transpose(-1, -2)).split(
                tuple(2**m for m in range(len(paths))), -1
            )
        elif module.method == "projection":
            restriction = module.restrictions[str(l)]
            values = (
                restriction_kernel.restrict_reference(restriction, tensor, direction)
                if default_contract and config.TORCH_REFERENCE
                else restriction(tensor, direction)
            )
        elif module.method == "rotation":
            tensor = tensor_axis.transform(tensor, rotation.unsqueeze(-3), l, contract)
        else:
            values = [tensor]
            for rank in range(l, 0, -1):
                values.append(
                    contract(values[-1], direction.unsqueeze(-2), 3 ** (rank - 1), 1)
                )
            values.reverse()
        for m, (index, _) in enumerate(paths):
            if matrix:
                value = values[m]
            elif module.method == "rotation":
                value = tensor.index_select(-1, getattr(module, f"indices_{l}_{m}"))
            else:
                value = tensor_axis.transform(values[m], axes, m, contract)
            if not matrix:
                value = value @ getattr(module, f"basis_{l}_{m}").to(features).T
            if odd and m and module.basis_change:
                value = quarter_turn(value)
            outputs[index].append(value)
    return (
        torch.cat([torch.cat(parts, -2).flatten(-2) for parts in outputs], -1)
        if outputs
        else features[..., :0]
    )


def to_global(module, features, rotation, contract=None):
    """Contract Cartesian indices without global edge-frame expansion."""
    default_contract = contract is None
    contract = tensor_axis.select() if contract is None else contract
    rotation = rotation.to(features)
    direction = rotation[..., 1, :].unsqueeze(-2)
    axes = (
        torch.stack((rotation[..., 2, :], rotation[..., 0, :]), -2)
        .transpose(-1, -2)
        .unsqueeze(-3)
    )
    local = [
        features[..., section].reshape(*features.shape[:-1], mul, ir.dim)
        for (mul, ir), section in zip(module.irreps_out, module.irreps_out.slices())
    ]
    outputs = []
    matrices = {}
    shape = torch.broadcast_shapes(features.shape[:-1], rotation.shape[:-2])
    for _, mul, l, odd, paths in module.entries:
        values = []
        orders = l + 1 if module.method == "projection" else len(paths)
        for m in range(orders):
            if m >= len(paths):
                values.append(features.new_zeros((*shape, mul, 3**m)))
                continue
            index, section = paths[m]
            value = local[index][..., section, :]
            if odd and m and module.basis_change:
                value = -quarter_turn(value)
            if module.method == "projection":
                value = value @ getattr(module, f"basis_{l}_{m}").to(features)
                value = tensor_axis.transform(value, axes, m, contract)
            values.append(value)
        if default_contract and matrix_enabled(module, features, contract, l, mul):
            if l not in matrices:
                matrices[l] = global_matrix(module, rotation, l)
            value = (torch.cat(values, -1) @ matrices[l]).expand(*shape, mul, 3**l)
        elif module.method == "projection":
            restriction = module.restrictions[str(l)]
            value = (
                restriction_kernel.inverse_reference(
                    restriction, values, direction, True
                )
                if default_contract and config.TORCH_REFERENCE
                else restriction.inverse(values, direction)
            )
        else:
            value = (
                torch.cat(values, -1) @ getattr(module, f"synthesis_{l}").to(features).T
            )
            value = value.expand(*shape, mul, 3**l)
            value = tensor_axis.transform(
                value, rotation.transpose(-1, -2).unsqueeze(-3), l, contract
            )
        outputs.append(value.flatten(-2))
    return torch.cat(outputs, -1) if outputs else features[..., :0]


def rotate(module, features, rotation, contract=None):
    """Rotate Cartesian blocks using the same rectangular-axis primitive."""
    contract = tensor_axis.select() if contract is None else contract
    rotation = rotation.to(features)
    shape = torch.broadcast_shapes(features.shape[:-1], rotation.shape[:-2])
    values = [
        tensor_axis.transform(
            features[..., section]
            .reshape(*features.shape[:-1], mul, 3**rank)
            .expand(*shape, mul, 3**rank),
            rotation.unsqueeze(-3),
            rank,
            contract,
        ).flatten(-2)
        for section, mul, rank, _, _ in module.entries
    ]
    return torch.cat(values, -1) if values else features[..., :0]


def transform_tensor(tensor, matrix, rank):
    """Apply the selected Torch axis primitive to each Cartesian index."""
    return tensor_axis.transform(tensor, matrix, rank, tensor_axis.select())

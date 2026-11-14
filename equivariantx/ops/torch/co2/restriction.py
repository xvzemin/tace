"""Coordinate-free restriction using shared tensor and symmetric axes."""

import torch

from ... import config
from ...common.torch import tensor_axis
from ..co3 import symmetric as sym
from .projection import plane_projector


def transverse_reference(module, tensor, direction):
    if module.m == 0:
        shape = torch.broadcast_shapes(tensor.shape[:-1], direction.shape[:-1])
        return tensor.expand(*shape, 1)
    plane = module.identity - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    if module.m == 1:
        return torch.einsum("...ij,...j->...i", plane, tensor)
    if module.m == 2:
        value = plane @ tensor.unflatten(-1, (3, 3)) @ plane
        trace = value.diagonal(dim1=-2, dim2=-1).sum(-1)
        return (value - 0.5 * trace[..., None, None] * plane).flatten(-2)
    basis = module.bases[0]
    traces = [sym.transform_compact(basis, sym.pack_factorized(basis, tensor), plane)]
    entries = torch.cat((plane[..., 0, :], plane[..., 1, 1:], plane[..., 2, 2:]), -1)
    for basis in module.bases[:-1]:
        traces.append(sym.trace_reference(basis, traces[-1]))
    value = traces[-1] * module.coefficients[-1]
    for k in range(len(traces) - 2, -1, -1):
        value = module.coefficients[k] * traces[k] + sym.multiply_reference(
            module.bases[k], value, entries
        )
    return sym.unpack_factorized(module.bases[0], value)


def restrict_reference(module, tensor, direction):
    if module.l == 0:
        shape = torch.broadcast_shapes(tensor.shape[:-1], direction.shape[:-1])
        return (tensor.expand(*shape, 1),)
    if module.l == 1:
        scalar = (tensor * direction).sum(-1, keepdim=True)
        return scalar, tensor - scalar * direction
    plane = module.identity - direction.unsqueeze(-1) * direction.unsqueeze(-2)
    if module.l == 2:
        tensor = tensor.unflatten(-1, (3, 3))
        vector = torch.einsum("...ij,...j->...i", tensor, direction)
        scalar = (vector * direction).sum(-1, keepdim=True)
        transverse = plane @ tensor @ plane + 0.5 * scalar.unsqueeze(-1) * plane
        return (
            module.scales[0] * scalar,
            module.scales[1] * (vector - scalar * direction),
            transverse.flatten(-2),
        )
    contractions = [sym.pack_factorized(module.bases[-1], tensor)]
    for m in range(module.l, 0, -1):
        contractions.append(
            sym.contract_reference(module.bases[m], contractions[-1], direction)
        )
    contractions.reverse()
    transverse = [
        sym.transform_compact(basis, x, plane)
        for basis, x in zip(module.bases, contractions)
    ]
    entries = torch.cat((plane[..., 0, :], plane[..., 1, 1:], plane[..., 2, 2:]), -1)
    values = []
    for m, (basis, scale, coefficients) in enumerate(
        zip(module.bases, module.scales, module.coefficients)
    ):
        # Tr^k(U_lm) = (-1)^k U_l,m-2k for a three-dimensional STF input.
        # Nested symmetric products avoid repeated trace and permutation sums.
        k = len(coefficients) - 1
        value = coefficients[k] * transverse[m - 2 * k]
        for k in range(k - 1, -1, -1):
            value = coefficients[k] * transverse[m - 2 * k] + sym.multiply_reference(
                module.bases[m - 2 * k], value, entries
            )
        values.append(sym.unpack_factorized(basis, value * scale))
    return tuple(values)


def inverse_reference(module, tensors, direction, project):
    result = None
    for m, (tensor, scale) in enumerate(zip(tensors, module.scales)):
        if tensor.shape[-1] != 3**m:
            raise ValueError("A transverse tensor has an incorrect trailing size.")
        value = scale * tensor
        result = (
            value
            if result is None
            else (direction.unsqueeze(-1) * result.unsqueeze(-2)).flatten(-2) + value
        )
    if module.l == 0:
        shape = torch.broadcast_shapes(result.shape[:-1], direction.shape[:-1])
        result = result.expand(*shape, 1)
    return (result @ module.basis) @ module.basis.T if project else result


def transform_symmetric(basis, tensor, matrix, basis_ops):
    """Keep compact CPU matrix products inside the joint derivative graph."""
    if basis_ops is sym and not tensor.is_cuda:
        return sym.transform_compact(basis, tensor, matrix)
    return basis_ops.transform(basis, tensor, matrix)


def unpack_symmetric(basis, tensor, basis_ops):
    """Avoid wide CPU scatter adjoints by expanding high ranks in stages."""
    if basis_ops is sym and not tensor.is_cuda and basis.rank >= 6 and sym.compiling():
        return sym.unpack_factorized(basis, tensor)
    return basis_ops.unpack(basis, tensor)


def transverse_program(module, tensor, direction, axes, basis_ops):
    if module.m == 0:
        shape = torch.broadcast_shapes(tensor.shape[:-1], direction.shape[:-1])
        return tensor.expand(*shape, 1)
    plane = plane_projector(direction)
    if module.m == 1:
        return axes(tensor, plane, 1, 1)
    if module.m == 2:
        value = axes(axes(tensor, plane, 1, 3), plane, 3, 1)
        trace = (
            value.unflatten(-1, (3, 3)).diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True)
        )
        return value - 0.5 * trace * plane.flatten(-2)
    basis = module.bases[0]
    traces = [
        transform_symmetric(basis, basis_ops.pack(basis, tensor), plane, basis_ops)
    ]
    entries = torch.cat((plane[..., 0, :], plane[..., 1, 1:], plane[..., 2, 2:]), -1)
    for basis in module.bases[:-1]:
        traces.append(basis_ops.trace(basis, traces[-1]))
    value = traces[-1] * module.coefficients[-1]
    for k in range(len(traces) - 2, -1, -1):
        value = module.coefficients[k] * traces[k] + basis_ops.multiply(
            module.bases[k], value, entries
        )
    return unpack_symmetric(module.bases[0], value, basis_ops)


def restrict_program(module, tensor, direction, axes, basis_ops):
    if module.l < 2:
        return restrict_reference(module, tensor, direction)
    plane = plane_projector(direction)
    if module.l == 2:
        vector = axes(tensor, direction.unsqueeze(-2), 3, 1)
        scalar = (vector * direction).sum(-1, keepdim=True)
        transverse = axes(
            axes(tensor, plane, 1, 3), plane, 3, 1
        ) + 0.5 * scalar * plane.flatten(-2)
        return (
            module.scales[0] * scalar,
            module.scales[1] * (vector - scalar * direction),
            transverse,
        )
    contractions = [basis_ops.pack(module.bases[-1], tensor)]
    for m in range(module.l, 0, -1):
        contractions.append(
            basis_ops.contract(module.bases[m], contractions[-1], direction)
        )
    contractions.reverse()
    transverse = [
        transform_symmetric(basis, x, plane, basis_ops)
        for basis, x in zip(module.bases, contractions)
    ]
    entries = torch.cat((plane[..., 0, :], plane[..., 1, 1:], plane[..., 2, 2:]), -1)
    values = []
    for m, (basis, scale, coefficients) in enumerate(
        zip(module.bases, module.scales, module.coefficients)
    ):
        k = len(coefficients) - 1
        value = coefficients[k] * transverse[m - 2 * k]
        for k in range(k - 1, -1, -1):
            value = coefficients[k] * transverse[m - 2 * k] + basis_ops.multiply(
                module.bases[m - 2 * k], value, entries
            )
        values.append(unpack_symmetric(basis, value * scale, basis_ops))
    return tuple(values)


def inverse_program(module, tensors, direction, project, axes):
    result = None
    for m, (tensor, scale) in enumerate(zip(tensors, module.scales)):
        if tensor.shape[-1] != 3**m:
            raise ValueError("A transverse tensor has an incorrect trailing size.")
        value = scale * tensor
        result = (
            value
            if result is None
            else axes(result, direction.unsqueeze(-1), 1, 3 ** (m - 1)) + value
        )
    if module.l == 0:
        shape = torch.broadcast_shapes(result.shape[:-1], direction.shape[:-1])
        result = result.expand(*shape, 1)
    return (result @ module.basis) @ module.basis.T if project else result


def transverse(module, tensor, direction):
    if (
        config.TORCH_REFERENCE
        or not config.TORCH_RESTRICTION
        or (module.m <= 2 and not sym.compiling())
    ):
        return transverse_reference(module, tensor, direction)
    return transverse_program(module, tensor, direction, tensor_axis.select(), sym)


def restrict(module, tensor, direction):
    if (
        config.TORCH_REFERENCE
        or not config.TORCH_RESTRICTION
        or (module.l <= 2 and not sym.compiling())
    ):
        return restrict_reference(module, tensor, direction)
    return restrict_program(module, tensor, direction, tensor_axis.select(), sym)


def inverse(module, tensors, direction, project):
    return inverse_reference(module, tensors, direction, project)


def quarter_turn_reference(tensor, direction):
    value = tensor.reshape(*tensor.shape[:-1], 3, tensor.shape[-1] // 3)
    value = torch.linalg.cross(direction.unsqueeze(-2), value.transpose(-1, -2), dim=-1)
    return value.transpose(-1, -2).flatten(-2)


def quarter_turn(tensor, direction):
    return quarter_turn_reference(tensor, direction)

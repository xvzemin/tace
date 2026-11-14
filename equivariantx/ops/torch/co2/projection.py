"""Plane projection and detracing with independent ordinary Torch references."""

import torch

from ... import config
from ...common.torch import tensor_axis


def plane_projector(direction):
    """Form I - n n.T without changing the direction normalization."""
    return torch.eye(
        3, dtype=direction.dtype, device=direction.device
    ) - direction.unsqueeze(-1) * direction.unsqueeze(-2)


def project_reference(module, tensor, plane):
    if module.rank == 0:
        shape = torch.broadcast_shapes(tensor.shape[:-1], plane.shape[:-2])
        return tensor.expand(*shape, 1)
    for axis in range(module.rank):
        tensor = torch.einsum(
            "...ij,...ajb->...aib",
            plane,
            tensor.unflatten(-1, (3**axis, 3, 3 ** (module.rank - axis - 1))),
        ).flatten(-3)
    return tensor


def detrace_reference(module, tensor, plane):
    if module.rank < 2:
        shape = torch.broadcast_shapes(tensor.shape[:-1], plane.shape[:-2])
        return tensor.expand(*shape, module.dim)
    value = (tensor @ module.basis).unsqueeze(-1)
    contractions, traces = [], []
    for k in range(len(module.coefficients) - 1):
        contraction = torch.einsum(
            "...ab,abij->...ij", plane, getattr(module, f"trace_{k}")
        )
        value = contraction @ value
        contractions.append(contraction)
        traces.append(value)
    correction = module.coefficients[-1] * traces[-1]
    for k in range(len(contractions) - 1, 0, -1):
        correction = (
            contractions[k].transpose(-1, -2) @ correction
            + module.coefficients[k] * traces[k - 1]
        )
    correction = contractions[0].transpose(-1, -2) @ correction
    return tensor + (module.basis @ correction).squeeze(-1)


def matrix(module, plane):
    """Construct the explicit dense map only when requested."""
    identity = module.identity
    if not identity.numel():
        identity = torch.eye(module.dim, dtype=identity.dtype, device=identity.device)
    if module.rank < 2:
        return identity.expand(*plane.shape[:-2], module.dim, module.dim)
    trace = module.trace_identity
    correction = torch.zeros_like(module.trace_identity)
    for k, coefficient in enumerate(module.coefficients[1:]):
        contraction = torch.einsum(
            "...ab,abij->...ij", plane, getattr(module, f"trace_{k}")
        )
        trace = contraction @ trace
        correction = correction + coefficient * (trace.transpose(-1, -2) @ trace)
    return identity + module.basis @ correction @ module.basis.T


def project(module, tensor, plane):
    """Use short explicit tensor axes or the independent einsum reference."""
    if config.TORCH_REFERENCE or not config.TORCH_TENSOR_AXIS or module.rank == 0:
        return project_reference(module, tensor, plane)
    function = tensor_axis.select()
    for axis in range(module.rank):
        tensor = function(tensor, plane, 3**axis, 3 ** (module.rank - axis - 1))
    return tensor


def symmetric_plane(plane):
    """Retain the symmetric matrix action used by the detracing basis."""
    return ((plane + plane.transpose(-1, -2)) * 0.5).flatten(-2)


def detrace_two(tensor, plane, coefficient=-0.5):
    """Apply the rank-two trace correction without constructing dense matrices."""
    p = symmetric_plane(plane)
    return tensor + coefficient * (tensor * p).sum(-1, keepdim=True) * p


def trace_three(tensor, plane):
    """Contract each pair of a cubic tensor with a symmetric plane matrix."""
    value = tensor.unflatten(-1, (3, 3, 3))
    plane = (plane + plane.transpose(-1, -2)) * 0.5
    return (
        (value * plane.unsqueeze(-1)).sum((-3, -2))
        + (value * plane.unsqueeze(-2)).sum((-3, -1))
        + (value * plane.unsqueeze(-3)).sum((-2, -1))
    ) / 3


def insert_three(vector, plane):
    """Insert and average a symmetric plane over all three pairs."""
    plane = (plane + plane.transpose(-1, -2)) * 0.5
    return (
        plane.unsqueeze(-1) * vector[..., None, None, :]
        + plane.unsqueeze(-2) * vector[..., None, :, None]
        + plane.unsqueeze(-3) * vector[..., :, None, None]
    ).flatten(-3) / 3


def matrix_three(tensor, vector):
    """Differentiate the cubic trace with respect to the symmetric plane."""
    value = tensor.unflatten(-1, (3, 3, 3))
    matrix = (
        (value * vector[..., None, None, :]).sum(-1)
        + (value * vector[..., None, :, None]).sum(-2)
        + (value * vector[..., :, None, None]).sum(-3)
    ) / 3
    return (matrix + matrix.transpose(-1, -2)) * 0.5


def detrace_three(tensor, plane, coefficient=-0.75):
    """Apply the exact cubic trace correction without dense basis contractions."""
    return tensor + coefficient * insert_three(trace_three(tensor, plane), plane)


@torch.compiler.allow_in_graph
class DetraceThree(torch.autograd.Function):
    """Differentiate the cubic polynomial using its self-adjoint trace map."""

    generate_vmap_rule = True

    @staticmethod
    def forward(tensor, plane, coefficient):
        return detrace_three(tensor, plane, coefficient)

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, plane, ctx.coefficient = inputs
        ctx.save_for_backward(tensor, plane)
        ctx.save_for_forward(tensor, plane)

    @staticmethod
    def backward(ctx, gradient):
        from ....utils.autograd import torch_gradient_mask

        tensor, plane = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[:2])
        dx = (
            detrace_three(gradient, plane, ctx.coefficient).sum_to_size(tensor.shape)
            if needed[0]
            else None
        )
        dp = None
        if needed[1]:
            dp = ctx.coefficient * (
                matrix_three(gradient, trace_three(tensor, plane))
                + matrix_three(tensor, trace_three(gradient, plane))
            )
            dp = dp.sum_to_size(plane.shape)
        return dx, dp, None

    @staticmethod
    def jvp(ctx, dx, dp, unused):
        tensor, plane = ctx.saved_tensors
        dx = torch.zeros_like(tensor) if dx is None else dx
        dp = torch.zeros_like(plane) if dp is None else dp
        return detrace_three(dx, plane, ctx.coefficient) + ctx.coefficient * (
            insert_three(trace_three(tensor, plane), dp)
            + insert_three(trace_three(tensor, dp), plane)
        )


@torch.compiler.allow_in_graph
class DetraceTwo(torch.autograd.Function):
    """Retain tensor and plane with exact recursive polynomial derivatives."""

    generate_vmap_rule = True

    @staticmethod
    def forward(tensor, plane, coefficient):
        return detrace_two(tensor, plane, coefficient)

    @staticmethod
    def setup_context(ctx, inputs, output):
        tensor, plane, ctx.coefficient = inputs
        ctx.save_for_backward(tensor, plane)
        ctx.save_for_forward(tensor, plane)

    @staticmethod
    def backward(ctx, gradient):
        from ....utils.autograd import torch_gradient_mask

        tensor, plane = ctx.saved_tensors
        needed = torch_gradient_mask(ctx, ctx.needs_input_grad[:2])
        p = symmetric_plane(plane)
        a = (tensor * p).sum(-1, keepdim=True)
        b = (gradient * p).sum(-1, keepdim=True)
        dx = (
            (gradient + ctx.coefficient * b * p).sum_to_size(tensor.shape)
            if needed[0]
            else None
        )
        dp = None
        if needed[1]:
            dp = (ctx.coefficient * (b * tensor + a * gradient)).unflatten(-1, (3, 3))
            dp = ((dp + dp.transpose(-1, -2)) * 0.5).sum_to_size(plane.shape)
        return dx, dp, None

    @staticmethod
    def jvp(ctx, dx, dp, unused):
        tensor, plane = ctx.saved_tensors
        dx = torch.zeros_like(tensor) if dx is None else dx
        dp = torch.zeros_like(plane) if dp is None else dp
        p, q = symmetric_plane(plane), symmetric_plane(dp)
        a = (tensor * p).sum(-1, keepdim=True)
        da = (dx * p + tensor * q).sum(-1, keepdim=True)
        return dx + ctx.coefficient * (da * p + a * q)


def detrace(module, tensor, plane):
    """Select the exact low-rank polynomial or general compact trace sequence."""
    if config.TORCH_REFERENCE or not config.TORCH_PLANAR_DETRACING or module.rank < 2:
        return detrace_reference(module, tensor, plane)
    if module.rank == 2:
        return detrace_two(tensor, plane, module.coefficients[1])
    if module.rank == 3:
        if config.TORCH_PLANAR_CUBIC:
            return detrace_three(tensor, plane, module.coefficients[1])
        if tensor.is_cuda and tensor.dtype == torch.float64:
            return detrace_reference(module, tensor, plane)
    return compact_detrace(module, tensor, plane, tensor_axis.primal)


def compact_detrace(module, tensor, plane, contract):
    """Keep trace vectors compact and use one shared GEMM for expansion."""
    value = tensor @ module.basis
    contractions, traces = [], []
    for k in range(len(module.coefficients) - 1):
        contraction = torch.einsum(
            "...ab,abij->...ij", plane, getattr(module, f"trace_{k}")
        )
        value = contract(value, contraction, 1, 1)
        contractions.append(contraction)
        traces.append(value)
    correction = module.coefficients[-1] * traces[-1]
    for k in range(len(contractions) - 1, 0, -1):
        correction = (
            contract(correction, contractions[k].transpose(-1, -2), 1, 1)
            + module.coefficients[k] * traces[k - 1]
        )
    correction = contract(correction, contractions[0].transpose(-1, -2), 1, 1)
    return tensor + correction @ module.basis.T

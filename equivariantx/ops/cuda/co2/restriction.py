"""Native coordinate-free restriction built from reusable tensor axes."""

import torch

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda.tensor_axis import evaluate as axes
from ...torch.co2 import restriction as torch_kernel
from ..co3 import symmetric


def inline():
    return forward_ad_active() or (
        config.RESTRICTION_INLINE and torch_kernel.sym.compiling()
    )


def transverse(module, tensor, direction):
    if inline():
        return torch_kernel.transverse(module, tensor, direction)
    return torch_kernel.transverse_program(module, tensor, direction, axes, symmetric)


def restrict(module, tensor, direction):
    if inline():
        return torch_kernel.restrict(module, tensor, direction)
    return torch_kernel.restrict_program(module, tensor, direction, axes, symmetric)


def inverse(module, tensors, direction, project):
    if inline():
        return torch_kernel.inverse(module, tensors, direction, project)
    return torch_kernel.inverse_program(module, tensors, direction, project, axes)


def quarter_turn(tensor, direction):
    if inline():
        return torch_kernel.quarter_turn(tensor, direction)
    x, y, z = direction.unbind(-1)
    zero = torch.zeros_like(x)
    matrix = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).unflatten(
        -1, (3, 3)
    )
    return axes(tensor, matrix, 1, tensor.shape[-1] // 3)

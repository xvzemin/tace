"""O(2) Fourier transforms and projected grid functions."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda.grid import evaluate
from ...torch.o2 import o2grid as torch_kernel
from .grid_projection import project, transform


def inline():
    return forward_ad_active() or (
        config.O2_GRID_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    )


def band_transform(module):
    return (
        transform
        if module.num_sheets * module.resolution <= config.O2_GRID_MAX_SAMPLES
        else torch_kernel.band_map
    )


def to_grid(module, features, features2=None):
    if inline():
        return torch_kernel.to_grid(module, features, features2)
    if torch_kernel.use_bands(module):
        return torch_kernel.band_synthesis(
            module, features, features2, band_transform(module)
        )
    return torch_kernel.to_grid_program(module, features, features2, evaluate)


def from_grid(module, features, split=False):
    if inline():
        return torch_kernel.from_grid(module, features, split)
    if torch_kernel.use_bands(module):
        return torch_kernel.band_analysis(
            module, features, split, band_transform(module)
        )
    return torch_kernel.from_grid_program(module, features, split, evaluate)


def supported(module):
    return (
        torch_kernel.use_bands(module)
        and not inline()
        and module.num_sheets * module.resolution <= config.O2_GRID_MAX_SAMPLES
        and module.num_sheets * (2 * module.mmax + 1) <= config.O2_GRID_MAX_BANDS
    )


def projected(module, features, kind, coeff=None, bound=1.0):
    args = [torch_kernel.band_input(module, x) for x in features]
    result = project(
        module.synthesis_bands,
        module.analysis_bands,
        coeff,
        args,
        [0] * len(args),
        1,
        kind,
        0,
        bound,
    )
    return torch_kernel.band_output(module, result)


def activation(module, act):
    from ....nn.activation import PolynomialActivation

    if isinstance(act, torch.nn.Module):
        hooks = torch.nn.modules.module
        if (
            "forward" in act.__dict__
            or getattr(act, "inplace", False)
            or act._forward_hooks
            or act._forward_pre_hooks
            or act._backward_hooks
            or act._backward_pre_hooks
            or hooks._global_forward_hooks
            or hooks._global_forward_pre_hooks
            or hooks._global_backward_hooks
            or hooks._global_backward_pre_hooks
        ):
            return None
    if type(act) is PolynomialActivation:
        if (
            act.coefficients.requires_grad
            or act.coefficients.dtype != module.grid.dtype
            or act.coefficients.device != module.grid.device
            or act.degree > 32
        ):
            return None
        return "polynomial", act.coefficients, act.bound
    if act is torch.nn.functional.gelu or (
        type(act) is torch.nn.GELU and act.approximate == "none"
    ):
        return "gelu", None, 1.0
    if act is torch.nn.functional.softplus or (
        type(act) is torch.nn.Softplus and act.beta == 1 and act.threshold == 20
    ):
        return "softplus", None, 1.0
    if act is torch.relu:
        return "relu", None, 1.0
    for cls, fn, name in (
        (torch.nn.SiLU, torch.nn.functional.silu, "silu"),
        (torch.nn.ReLU, torch.nn.functional.relu, "relu"),
        (torch.nn.Tanh, torch.tanh, "tanh"),
        (torch.nn.Sigmoid, torch.sigmoid, "sigmoid"),
    ):
        if type(act) is cls or act is fn:
            return name, None, 1.0
    if act is torch.square or act is torch.sin:
        return ("square" if act is torch.square else "sin"), None, 1.0
    return None


def activate(module, x, act):
    if supported(module):
        args = activation(module, act)
        if args is not None:
            return projected(module, (x,), *args)
    return from_grid(module, torch_kernel.sample_activation(to_grid(module, x), act))


def product(module, x, y):
    if supported(module) and x.shape == y.shape:
        if x is y:
            return projected(module, (x,), "square")
        return projected(module, (x, y), "product")
    u = to_grid(module, x)
    return from_grid(module, u.square() if x is y else u * to_grid(module, y))

"""Local CUDA selection for TACE species-indexed linear maps."""

import torch

from equivariantx import co3, o3
from tace.models.linear import e3nnElementLinear, e3nnMoEElementLinear
from tace.utils.env import acceleration_enabled

from .matching import custom_forward


def use_element_kernel(module):
    module.use_eqx_linear = True
    return module


def use_torch_element_kernel(module):
    module.use_eqx_linear = False
    return module


def element_regions(rep, *, backend):
    """Only change a previously selected map if it was selected by this planner."""
    for path, module in rep.named_modules():
        if type(module) not in (e3nnElementLinear, e3nnMoEElementLinear):
            continue
        old = getattr(module, "_foundation_regional_provider", None)
        if old is None:
            continue
        if old[0] == backend:
            yield path, None, old[1]
            continue
        if acceleration_enabled("eqx", kernel="linear") or custom_forward(
            module, recursive=True
        ):
            raise ValueError(
                f"Cannot retarget {path}: its provider is globally enabled or has customized forward/hooks. Use the original host model."
            )
        lower = use_torch_element_kernel if backend == "torch" else use_element_kernel
        yield path, lower, f"species-indexed linear using {backend}"
    for path in element_maps(rep, backend=backend):
        if (
            getattr(rep.get_submodule(path), "_foundation_regional_provider", None)
            is None
        ):
            yield (
                path,
                use_element_kernel,
                "species-indexed linear on the host's native layout",
            )


def element_maps(rep, *, backend):
    """Yield species maps outside the product coefficient schedule."""
    if backend != "cuda" or acceleration_enabled("eqx", kernel="linear"):
        return
    for path, module in rep.named_modules():
        if type(module) not in (e3nnElementLinear, e3nnMoEElementLinear):
            continue
        if path.startswith("products.") and ".coefs." in path:
            continue
        if (
            module.use_eqx_linear
            or module.eqx_linear is None
            or module.torch_linear is None
        ):
            continue
        if type(module.eqx_linear) is not type(module.torch_linear) or type(
            module.eqx_linear
        ) not in (
            o3.ElementLinear,
            o3.MoEElementLinear,
            co3.ElementLinear,
            co3.MoEElementLinear,
        ):
            continue
        if next(module.parameters()).dtype not in (torch.float32, torch.float64):
            continue
        if custom_forward(module.torch_linear, recursive=True):
            continue
        yield path

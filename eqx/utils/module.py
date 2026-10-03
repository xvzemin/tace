"""Module conversion and construction helpers."""

from contextlib import contextmanager
from copy import deepcopy

import torch


@contextmanager
def default_dtype(dtype):
    """Set the default dtype within a context and restore it on exit.

    Parameters
    ----------
    dtype : torch.dtype
        Floating-point dtype used inside the context.
    """
    previous = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        yield
    finally:
        torch.set_default_dtype(previous)


def copy_model(model):
    """Copy a module while sharing stateless compiled functions.

    Parameters
    ----------
    model : torch.nn.Module
        Module to copy.

    Returns
    -------
    torch.nn.Module
        Independent copy of parameters, buffers, and submodules.
    """
    functions = {
        id(value): value
        for module in model.modules()
        for value in vars(module).values()
        if isinstance(value, torch.jit.ScriptFunction)
    }
    return deepcopy(model, functions)


def convert_modules(model, factory, *, inplace=False):
    """Replace selected modules while preserving shared submodules.

    Parameters
    ----------
    model : torch.nn.Module
        Root module to convert.
    factory : callable
        Return a replacement for a module, or None to visit its children.
        Children of replacement modules are not visited.
    inplace : bool, optional
        Modify the supplied model. Otherwise, copy it before conversion.

    Returns
    -------
    torch.nn.Module
        Converted root module.
    """
    if not inplace:
        model = copy_model(model)
    replacements = {}

    def convert(module):
        if id(module) in replacements:
            return replacements[id(module)]
        result = factory(module)
        replacements[id(module)] = module if result is None else result
        if result is None:
            for name, child in module._modules.items():
                if child is not None:
                    module._modules[name] = convert(child)
            return module
        return result

    return convert(model)

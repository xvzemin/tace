"""TACE model conversion, parameter ownership and local-owner ML-IAP export."""

import copy
from pathlib import Path

import torch


def _model_copy(model, memo=None):
    from e3nn import o3

    from .matching import customized_codegen

    if customized_codegen(model):
        raise ValueError(
            "Cannot copy an e3nn generated module with customized execution."
        )
    memo = {} if memo is None else memo
    for child in model.modules():
        if type(child) is o3.SphericalHarmonics:
            evaluator = getattr(child, "sph_func", None)
            if isinstance(evaluator, torch.jit.ScriptFunction):
                # The scripted harmonic evaluator is code, not model state.
                memo[id(evaluator)] = evaluator
    persistent = set(model.state_dict())
    result = copy.deepcopy(model, memo)
    for path, child in result.named_modules():
        for name in child._buffers.keys():
            if f"{path}.{name}".lstrip(".") not in persistent:
                child._non_persistent_buffers_set.add(name)
    return result


def shared_parameter_copy(model):
    """Copy a module while sharing its parameters.

    Examples
    --------
    >>> model = torch.nn.Linear(3, 2)
    >>> adapter = shared_parameter_copy(model)
    >>> adapter is model, adapter.weight is model.weight
    (False, True)
    """
    return _model_copy(model, {id(p): p for p in model.parameters()})


def frozen_export_copy(model):
    """Own one frozen export snapshot, independent of any caller or optimizer.

    Compiled variants may share *these* frozen parameters, never the live model's.
    Deepcopy retains tied parameters while isolating their storage and all buffers.
    """
    return _model_copy(model).eval().requires_grad_(False)


def export_local_lammps(
    frozen, output, *, loader_class, aoti_package=None, receiver_model=None
):
    """Export a single-rank FP32 energy and pair-force model.

    Parameters
    ----------
    frozen : torch.nn.Module
        Frozen CUDA model for the compiled branch.
    output : str or Path
        Destination of the ML-IAP loader.
    loader_class : type
        Local-owner ML-IAP calculator.
    aoti_package : str or Path, optional
        Destination of the compiled force graph.
    receiver_model : torch.nn.Module, optional
        Model for receiver-tiled large-system evaluation.
    """
    from tace.models.compile import export_lammps_aotinductor

    output = Path(output)
    package = (
        Path(aoti_package) if aoti_package is not None else output.with_suffix(".pt2")
    )
    if package.suffix != ".pt2":
        package = Path(str(package) + ".pt2")
    if package.resolve() == output.resolve():
        raise ValueError("The ML-IAP loader and AOTI package need distinct paths.")
    if (
        frozen.get_model_dtype() != torch.float32
        or next(frozen.parameters()).device.type != "cuda"
    ):
        raise ValueError("Single-rank foundation export requires FP32 on CUDA.")
    frozen.eval().requires_grad_(False)
    package = export_lammps_aotinductor(frozen, package, local_owners=True)
    loader = loader_class(
        package,
        frozen.get_atomic_numbers(),
        frozen.get_cutoff(),
        frozen.get_model_dtype(),
    )
    loader.graph_max_atoms = 1024
    if receiver_model is not None:
        loader.attach_frozen_receiver(receiver_model)
    torch.save(loader, output, pickle_protocol=5)
    return output

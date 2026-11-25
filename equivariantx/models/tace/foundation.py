"""Local-owner ML-IAP exports for spherical TACE foundation models."""

import copy
from pathlib import Path

import torch


def shared_parameter_copy(model):
    """Copy a module while sharing its parameters.

    Examples
    --------
    >>> model = torch.nn.Linear(3, 2)
    >>> adapter = shared_parameter_copy(model)
    >>> adapter is model, adapter.weight is model.weight
    (False, True)
    """
    persistent = set(model.state_dict())
    result = copy.deepcopy(model, {id(p): p for p in model.parameters()})
    for path, child in result.named_modules():
        for name in child._buffers:
            if f"{path}.{name}".lstrip(".") not in persistent:
                child._non_persistent_buffers_set.add(name)
    return result


def representation_architecture(rep) -> str | None:
    """Identify a released interaction and product schedule."""
    from tace.models.spherical.inter import O3CgtpInteraction
    from tace.models.spherical.prod import CgtpACE
    from tace.models.spherical.tece_oam_rra import BilinearMoEACE, UvSO2Interaction

    interactions = getattr(rep, "interactions", ())
    products = getattr(rep, "products", ())
    if not interactions or len(products) != len(interactions):
        return None
    local = tuple(isinstance(inter, UvSO2Interaction) for inter in interactions)
    if any(
        not is_local and type(inter) is not O3CgtpInteraction
        for inter, is_local in zip(interactions, local, strict=True)
    ):
        return None
    channels = rep.num_channel
    if (
        channels == 64
        and local == (False,) * 5
        and all(isinstance(p, CgtpACE) for p in products)
    ):
        return "l"
    if all(isinstance(p, BilinearMoEACE) for p in products):
        if channels == 128 and local == (False,) * 5:
            return "7m"
        if channels == 64 and local == (False, True) * 4:
            return "rra"
    return None


def foundation_architecture(model) -> str:
    """Return the specialized execution schedule for a TACE adapter."""
    name = representation_architecture(
        getattr(model.readout_fn, "representation", None)
    )
    if name is None:
        raise ValueError(
            "No specialized ML-IAP backend for this architecture; use mliap or aoti."
        )
    return name


def optimize_foundation(model):
    """Return a differentiable execution schedule sharing the original parameters.

    Parameters
    ----------
    model : torch.nn.Module
        Loaded OAM or OMat24 L, 7M, or TECE-RRA-1.0 model.

    Examples
    --------
    >>> model = optimize_foundation(model).train()  # doctest: +SKIP
    """
    arch = foundation_architecture(model)
    if arch == "7m":
        from .tace_oam_7m import optimize_oam_7m

        return optimize_oam_7m(model)
    if arch == "rra":
        from .tece_oam_rra.model import optimize_rra

        return optimize_rra(model)
    from .tace_oam_l import optimize_oam_l

    return optimize_oam_l(model)


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

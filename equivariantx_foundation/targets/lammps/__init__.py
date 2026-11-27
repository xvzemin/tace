"""LAMMPS deployment helpers."""

from ...registry import require_cuda


def export_distributed(model, output):
    """Export an eager GPU ML-IAP loader with TACE's original MPI halo exchange.

    Parameters
    ----------
    model : torch.nn.Module
        Eager TACE model. Each ML-IAP rank selects its CUDA device at execution.
    output : str or pathlib.Path
        Destination of the ML-IAP loader.

    Examples
    --------
    >>> path = export_distributed(model, "potential.pt")  # doctest: +SKIP
    """
    from .distributed import export_distributed as save

    return save(model, output)


def export(model, output, *, capacity=True, aoti_package=None):
    """Export a single-rank FP32 ML-IAP force model.

    Parameters
    ----------
    model : torch.nn.Module
        A CUDA TACE-OAM-L, TACE-OAM-7M or TECE-OAM-RRA model.
    output : str or pathlib.Path
        Destination of the ML-IAP loader.
    capacity : bool, optional
        Enable receiver tiling for large systems.
    aoti_package : str or pathlib.Path, optional
        Destination of the compiled force graph.

    Examples
    --------
    >>> from equivariantx_foundation.targets.lammps import export
    >>> path = export(model, "potential.pt")  # doctest: +SKIP
    """
    from importlib import import_module

    from equivariantx_foundation.models.tace.planner import plan

    require_cuda(model)
    schedule = plan(model, task="lammps", backend="cuda").schedule
    export_lammps = import_module(
        f"equivariantx_foundation.models.tace.{schedule}.lammps"
    ).export_lammps
    return export_lammps(model, output, capacity=capacity, aoti_package=aoti_package)

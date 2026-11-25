"""Interaction layout for TECE-OAM-RRA."""

from .layout import LocalSplit


def export_lammps(model, output, *, capacity, aoti_package=None):
    """Export a TECE-RRA local-owner ML-IAP package on one CUDA device."""
    from .lammps import export_lammps as export

    return export(model, output, capacity=capacity, aoti_package=aoti_package)


__all__ = ["LocalSplit", "export_lammps"]

"""Interaction layout for TECE-OAM-RRA."""

TASKS = {"inference", "training", "training_buckets", "lammps"}


def matches(model, *, task):
    """Check the alternating CGTP and rotary-attention architecture."""
    from ..matching import spherical_release_representation

    rep = spherical_release_representation(model)
    if rep is None or (rep.num_channel, rep.num_layers) != (64, 8):
        return False
    from tace.models.spherical.representation import NodeRepresentation

    from .model import validate_rra
    from .radial import RRARepresentation

    if type(rep) not in (NodeRepresentation, RRARepresentation):
        return False
    validate_rra(model)
    return True


def optimize(model, **options):
    from .model import optimize_rra

    return optimize_rra(model, **options)


def __getattr__(name):
    if name == "LocalSplit":
        from .layout import LocalSplit

        return LocalSplit
    raise AttributeError(name)


def export_lammps(model, output, *, capacity, aoti_package=None):
    """Export a TECE-RRA local-owner ML-IAP package on one CUDA device."""
    from .lammps import export_lammps as export

    return export(model, output, capacity=capacity, aoti_package=aoti_package)


__all__ = ["LocalSplit", "export_lammps"]

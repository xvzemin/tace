"""Whole-model operators for the released TACE-7M architecture."""

__all__ = ["optimize_oam_7m"]

TASKS = {"inference", "training", "training_buckets", "lammps"}


def matches(model, *, task):
    """Check the five-layer CGTP and bilinear MoE architecture."""
    from ..matching import spherical_release_representation

    rep = spherical_release_representation(model)
    if rep is None or (rep.num_channel, rep.num_layers) != (128, 5):
        return False
    from tace.models.spherical.representation import NodeRepresentation

    from .model import SevenRepresentation, validate_representation

    if type(rep) not in (NodeRepresentation, SevenRepresentation):
        return False
    validate_representation(rep)
    return True


def optimize(model, **options):
    from .model import optimize_oam_7m

    return optimize_oam_7m(model, **options)


def __getattr__(name):
    if name == "optimize_oam_7m":
        from .model import optimize_oam_7m

        return optimize_oam_7m
    raise AttributeError(name)

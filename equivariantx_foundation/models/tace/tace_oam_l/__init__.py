"""Differentiable whole-model execution schedule for TACE-OAM-L."""

__all__ = ["optimize_oam_l"]

TASKS = {"inference", "training", "training_buckets", "lammps"}


def matches(model, *, task):
    """Check the five-layer quadratic CGTP architecture."""
    from ..matching import (
        canonical_component_harmonics,
        spherical_release_representation,
    )

    rep = spherical_release_representation(model)
    if rep is None or (rep.num_channel, rep.num_layers) != (64, 5):
        return False
    from tace.models.spherical.representation import NodeRepresentation

    from .model import _OAMLRepresentation, _validate

    if not canonical_component_harmonics(getattr(rep, "o3_angular_basis", None), 5):
        raise ValueError("OAM-L requires canonical component harmonics l=0..5.")
    if type(rep) not in (NodeRepresentation, _OAMLRepresentation):
        return False
    _validate(rep)
    if task == "lammps":
        from .tangent import supports_radial_embedding

        if not supports_radial_embedding(rep):
            raise ValueError(
                "The ML-IAP tangent kernel requires the eight-mode j0 and scaled SiLU embedding."
            )
    return True


def optimize(model, **options):
    from .model import optimize_oam_l

    return optimize_oam_l(model, **options)


def __getattr__(name):
    if name == "optimize_oam_l":
        from .model import optimize_oam_l

        return optimize_oam_l
    raise AttributeError(name)

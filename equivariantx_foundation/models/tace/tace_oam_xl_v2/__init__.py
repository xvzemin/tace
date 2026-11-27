"""Architecture-specialized E/F/S training for TACE-OAM-XL-v2."""

TASKS = {"joint_training"}


def matches(model, *, task):
    """Check a nonlinear correlation-two architecture for joint E/F/S training."""
    from tace.models.spherical.representation import NodeRepresentation

    from ..matching import spherical_release_representation

    if type(spherical_release_representation(model)) is not NodeRepresentation:
        return False
    from .model import validate

    validate(model)
    return True


def optimize(model, **options):
    from .model import TACEOAMXLV2

    return TACEOAMXLV2.from_tace(model, **options)


def __getattr__(name):
    if name == "TACEOAMXLV2":
        from .model import TACEOAMXLV2

        return TACEOAMXLV2
    if name == "register_runtime_ops":
        from .registration import register_runtime_ops

        return register_runtime_ops
    if name == "GradientWorkspace":
        from .training import GradientWorkspace

        return GradientWorkspace
    raise AttributeError(name)


__all__ = ["TACEOAMXLV2", "GradientWorkspace", "register_runtime_ops"]

"""Register TACE-OAM-7M operators for compiled graphs."""

from importlib import import_module

for module in ("product", "radial_batch", "reciprocal", "tangent"):
    import_module(f"..models.tace.tace_oam_7m.{module}", __package__)

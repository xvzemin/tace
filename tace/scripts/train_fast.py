"""Train using a Foundation plan with the ordinary TACE Hydra configuration."""

import os
from collections.abc import Mapping

from tace.utils.cfg import training_cli


def launch(cfg):
    """Select a general plan, or explicitly request the captured OAM-v2 schedule."""
    from equivariantx_foundation.registry import require_cuda

    schedule = cfg.get("fast_schedule", "foundation")
    if schedule not in ("foundation", "oam_v2"):
        raise ValueError("fast_schedule must be 'foundation' or 'oam_v2'")
    require_cuda()
    if schedule == "oam_v2":
        from tace.scripts import train_oamv2

        # The legacy schedule configures FP32/TF32, kernels and matrix-weight
        # layout before importing EQX or the training/model modules.
        train_oamv2.configure_environment()
        return train_oamv2.launch(cfg)
    from tace.scripts.train import train_from_config

    replay = any(
        isinstance(cb, Mapping) and cb.get("replay_bucket") is not None
        for cb in cfg.get("callbacks", {}).values()
    )
    # The native constructor implicitly selects EquiTorch for higher-order
    # products. Leave those products available to the requested replay plan.
    default_eqt = (
        replay
        and "TACE_USE_EQT" not in os.environ
        and "TACE_USE_EQT" not in cfg.get("misc", {}).get("env", {})
    )
    if default_eqt:
        os.environ["TACE_USE_EQT"] = "0"
    try:
        return train_from_config(cfg, fast=True)
    finally:
        if default_eqt:
            os.environ.pop("TACE_USE_EQT", None)


def main():
    previous = None

    @training_cli
    def job(cfg):
        nonlocal previous
        schedule = cfg.get("fast_schedule", "foundation")
        if previous is not None and schedule != previous:
            raise ValueError(
                "Run different fast schedules in separate processes; OAM-v2 fixes process-wide precision and compiler settings."
            )
        previous = schedule
        return launch(cfg)

    job()


if __name__ == "__main__":
    main()

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Train using a Foundation plan with the ordinary TACE Hydra configuration."""

import logging

from tace.utils.cfg import training_cli


def select_schedule(cfg):
    """Resolve the training schedule before constructing the model or its optimizer."""
    schedule = cfg.get("fast_schedule", "auto")
    if schedule == "auto":
        from tace.scripts.train_oamv2 import matches_config

        return "oam_v2" if matches_config(cfg) else "foundation"
    if schedule not in ("foundation", "oam_v2"):
        raise ValueError("fast_schedule must be 'auto', 'foundation' or 'oam_v2'")
    return schedule


def launch(cfg, *, schedule=None):
    """Run the selected Foundation training schedule."""
    from equivariantx_foundation.registry import require_cuda

    if schedule is None:
        schedule = select_schedule(cfg)
    require_cuda()
    logging.getLogger(__name__).info("Fast training schedule: %s", schedule)
    if schedule == "oam_v2":
        from tace.scripts import train_oamv2

        # The legacy schedule configures FP32/TF32, kernels and matrix-weight
        # layout before importing EQX or the training/model modules.
        train_oamv2.configure_environment(
            preserve_existing=cfg.get("fast_schedule", "auto") == "auto"
        )
        return train_oamv2.launch(cfg)
    from tace.scripts.train import train_from_config

    return train_from_config(cfg, fast=True)


def main():
    previous = None

    @training_cli
    def job(cfg):
        nonlocal previous
        schedule = select_schedule(cfg)
        if previous is not None and schedule != previous:
            raise ValueError(
                "Run different fast schedules in separate processes; OAM-v2 fixes process-wide precision and compiler settings."
            )
        previous = schedule
        return launch(cfg, schedule=schedule)

    job()


if __name__ == "__main__":
    main()

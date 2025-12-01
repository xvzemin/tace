################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict


import yaml
from importlib.resources import files


from tace import utils


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v
    return d


def update_cfg(cfg: Dict):
    # === load default cfg ===
    config_yaml = files(utils) / "config.yaml"

    with config_yaml.open("r") as f:
        default_cfg = yaml.safe_load(f)

    # update
    ignore_default_config = cfg.get("ignore_default_config", False)

    if not ignore_default_config:
        logging.info("Useing user's config and default config")
        cfg = deep_update(default_cfg, cfg)
    else:
        logging.info("Useing only user's config")

    return cfg

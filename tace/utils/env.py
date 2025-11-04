################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import os
from typing import Dict

def set_env(cfg: Dict):
    env = cfg.get("misc", {}).get("env", {})
    for k, v in env.items():
        os.environ[k] = v

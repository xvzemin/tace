"""Native recoupling with AOT-visible sparse angular contractions."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config


def evaluate(module, values):
    inline = config.WIGNER6J_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    )
    return module.schedule.evaluate(values, "torch" if inline else "cuda")

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import packaging


import torch

_BOOL = {
    0: False,
    1: True,
}

DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}


_TORCH_VERSION = packaging.version.parse(torch.__version__)
_TORCH_GE_2_9 = _TORCH_VERSION >= packaging.version.parse("2.9")
_GLOBAL_STATE_INITIALIZED = False
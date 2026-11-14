"""Native smooth alignment charts visible to joint AOT compilation."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.cuda import scalar
from ...torch.o2.rotation_matrix import evaluate as torch_evaluate
from ...torch.o2.rotation_matrix import reference


def evaluate(vector, axis, eps=1e-7, quaternion=False):
    """Fuse the selected chart and each requested transpose into native kernels."""
    if torch.compiler.is_compiling() and eps != 1e-7:
        return reference(vector, axis, eps, quaternion)
    if config.ALIGNMENT_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        return torch_evaluate(vector, axis, eps, quaternion)
    value = scalar.evaluate(axis, eps, quaternion, (), vector.reshape(-1, 3))[0]
    return value.reshape(*vector.shape[:-1], *((4,) if quaternion else (3, 3)))

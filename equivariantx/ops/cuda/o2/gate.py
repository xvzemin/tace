"""Fused gate expressions in the module's representation layout."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.gate import gate
from ...torch.o2.gate import evaluate as torch_gate


def evaluate(module, features):
    """Apply the native expression or its compiler-visible Torch lowering."""
    if config.GATE_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    ):
        return torch_gate(module, features)
    return gate(features, module.cuda_program, module.irreps_out.dim)

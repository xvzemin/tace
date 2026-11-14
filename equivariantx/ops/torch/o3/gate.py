"""Spherical gate evaluation using ordinary Torch operations."""

from e3nn.nn import Gate
import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.gate import torch_gate


def reference(module, features):
    """Evaluate the independent e3nn gate reference."""
    return Gate.forward(module, features)


def evaluate(module, features):
    """Expose supported gate expressions while retaining the ordinary reference."""
    if (
        not config.TORCH_REFERENCE
        and config.GATE_INLINE
        and module.metadata is not None
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return torch_gate(features, module.metadata, module.output_dim)
    return reference(module, features)

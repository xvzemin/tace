"""Scalar activations and gate application in the declared storage layout."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.gate import torch_gate


def activate(module, features):
    """Evaluate the module's declared paths."""
    return module.activation(features)


def reference(module, features):
    """Evaluate independent scalar activations and tensor-product gates."""
    scalars = module.act_scalars(features.index_select(-1, module.indices_scalars))
    if not module.irreps_gates.dim:
        return scalars
    gates = module.act_gates(features.index_select(-1, module.indices_gates))
    gated = module.mul(features.index_select(-1, module.indices_gated), gates)
    return torch.cat((scalars, gated), dim=-1)


def evaluate(module, features):
    """Fuse static channel broadcasts and their adjoints in compiled graphs."""
    if (
        not config.TORCH_REFERENCE
        and config.GATE_INLINE
        and module.metadata is not None
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return torch_gate(features, module.metadata, module.irreps_out.dim)
    return reference(module, features)

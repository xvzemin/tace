################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch


# only act, in TACE, tensors are not gated
ACT = {
    None: torch.nn.Identity,
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "leaky_relu": torch.nn.LeakyReLU,
    "prelu": torch.nn.PReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    "gelu": torch.nn.GELU,
    "silu": torch.nn.SiLU,  # Swish
    "mish": torch.nn.Mish,
    "softplus": torch.nn.Softplus,
    "softsign": torch.nn.Softsign,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "hardtanh": torch.nn.Hardtanh,
    "hardswish": torch.nn.Hardswish,
    "hardsigmoid": torch.nn.Hardsigmoid,
    "tanhshrink": torch.nn.Tanhshrink,
}
"""Mapping from activation function name to torch.nn modules.

You can specify the key to use these activation functions in TACE.

- None: ``torch.nn.Identity``
- identity: ``torch.nn.Identity``
- relu: ``torch.nn.ReLU``
- leaky_relu: ``torch.nn.LeakyReLU``
- prelu: ``torch.nn.PReLU``
- elu: ``torch.nn.ELU``
- selu: ``torch.nn.SELU``
- gelu: ``torch.nn.GELU``
- silu: ``torch.nn.SiLU`` (Swish)
- mish: ``torch.nn.Mish``
- softplus: ``torch.nn.Softplus``
- softsign: ``torch.nn.Softsign``
- tanh: ``torch.nn.Tanh``
- sigmoid: ``torch.nn.Sigmoid``
- hardtanh: ``torch.nn.Hardtanh``
- hardswish: ``torch.nn.Hardswish``
- hardsigmoid: ``torch.nn.Hardsigmoid``
- tanhshrink: ``torch.nn.Tanhshrink``
"""
GATE=ACT

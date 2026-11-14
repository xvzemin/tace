"""Native shared channel maps in the CO2 representation layout."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda.channel_linear import evaluate
from ...common.torch.channel_linear import supported
from ...torch.co2 import linear as torch_kernel


def linear(module, features, weight, bias):
    """Fuse shared path contractions, scaling, output layout and scalar bias."""
    if (
        hasattr(module, "channel_metadata")
        and supported(features, weight, bias, module.layout)
        and (not forward_ad_active())
    ):
        output = evaluate(module.channel_metadata, features, weight, bias)
        return module.projection(output)
    return torch_kernel.linear(module, features, weight, bias)


def uu_linear(module, features, weight):
    """Fuse copy paths without mixing channels or retaining expanded features."""
    if (
        config.UU_LINEAR_INLINE
        and (module.path_mode == "sum" or module.num_channel >= 1024)
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return torch_kernel.uu_linear(module, features, weight)

    from ...common.cuda.uu_linear import evaluate as channelwise
    from ...common.torch.uu_linear import supported as supported_channelwise

    if (
        hasattr(module, "uu_metadata")
        and supported_channelwise(features, weight)
        and (not forward_ad_active())
    ):
        return channelwise(module.uu_metadata, features, weight)
    return torch_kernel.uu_linear(module, features, weight)

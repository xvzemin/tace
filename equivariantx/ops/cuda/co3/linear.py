"""Native shared channel maps in the CO3 representation layout."""

from ....utils.autograd import forward_ad_active
from ...common.cuda.channel_linear import evaluate
from ...common.torch.channel_linear import supported
from ...torch.co3 import linear as torch_kernel


def linear(module, features, weight, bias):
    """Fuse shared path contractions, scaling, output layout and scalar bias."""
    if (
        hasattr(module, "channel_metadata")
        and supported(features, weight, bias, module.channel_layout)
        and (not forward_ad_active())
    ):
        output = evaluate(module.channel_metadata, features, weight, bias)
        return module.projection(output)
    return torch_kernel.linear(module, features, weight, bias)

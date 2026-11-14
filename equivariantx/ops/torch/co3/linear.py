"""Channel maps in the module's representation layout."""

from ... import config
from ...common.torch import channel_linear
from ...common.torch.linear import cartesian_linear


def linear(module, features, weight, bias):
    """Evaluate the declared Linear paths and output layout."""
    if (
        not config.TORCH_REFERENCE
        and config.TORCH_CHANNEL_LINEAR
        and hasattr(module, "channel_metadata")
        and channel_linear.supported(features, weight, bias, module.channel_layout)
    ):
        output = channel_linear.evaluate(
            module.channel_metadata, features, weight, bias
        )
        return module.projection(output)

    output, zero = cartesian_linear(
        features,
        weight,
        bias,
        module.output_shapes,
        module.groups,
        module.slices_in,
        module.group_sections,
    )
    return module.projection(output) + zero

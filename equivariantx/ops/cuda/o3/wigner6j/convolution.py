"""Node coupling followed by fused indexed recoupled convolution."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.autograd import forward_ad_active
from .... import config


def recoupled(module, values, edge_index, num_nodes):
    """Couple nodes once and contract edge paths directly into destination nodes."""
    x1, x2, x3, weight1, weight2 = values
    features = module.couple_inputs(x1, x3, backend="cuda")
    weight = module.combine_weights(weight1, weight2, backend="cuda")
    if weight.ndim == 1:
        weight = weight.unsqueeze(0)
    output = module.conv132(
        features,
        x2.index_select(-1, module.input2_index),
        weight,
        weight.new_empty((0, module.tp132.weight_numel)),
        edge_index,
        num_nodes,
    )
    return output.index_select(-1, module.output_index)


def evaluate(module, values, edge_index, num_nodes):
    inline = config.WIGNER6J_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    )
    if (
        not inline
        and not forward_ad_active()
        and module.schedule.use_recoupling(values, edge_index[0])
    ):
        return recoupled(module, values, edge_index, num_nodes)
    message = module.schedule.evaluate(
        values, "torch" if inline else "cuda", edge_index[0]
    )
    return message.new_zeros((num_nodes, module.output_dim)).index_add(
        0, edge_index[1], message
    )

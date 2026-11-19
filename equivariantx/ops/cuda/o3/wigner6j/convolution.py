"""Node coupling followed by fused indexed recoupled convolution."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from .....utils.autograd import forward_ad_active
from .... import config


def recoupled(module, values, edge_index, num_nodes, edge_pairs=None):
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
        edge_pairs=edge_pairs,
    )
    return output.index_select(-1, module.output_index)


def use_pair_contraction(module, values, pairs):
    """Fuse pair-indexed recoupled weights with the directed node reduction."""
    if forward_ad_active() or (
        config.WIGNER6J_INLINE
        and (torch.compiler.is_compiling() or get_proxy_mode() is not None)
    ):
        return False
    schedule = module.schedule
    choice = config.WIGNER6J_RECOUPLE
    choice = schedule.recouple if choice is None else choice
    if choice is False:
        return False
    nodes, edges = values[0].shape[0], values[1].shape[0]
    pairs_size = pairs.representatives.numel()
    if not statically_known_true(
        pairs_size * module.tp132.weight_numel * values[0].element_size()
        <= config.RADIAL_PROJECT_MAX_BYTES
    ):
        return False
    # The native reduction avoids two edge messages and combines the two
    # weights once per bond. Bound the angular overhead of changing schedules.
    return choice is True or (
        schedule.recoupled_cost <= 2 * schedule.standard_cost
        and statically_known_true(4 * nodes <= edges)
        and statically_known_true(3 * pairs_size <= 2 * edges)
    )


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

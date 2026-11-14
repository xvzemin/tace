"""Native node coupling and indexed recoupled message reduction."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config


def evaluate(module, values, edge_index, num_nodes):
    inline = config.WIGNER6J_INLINE and (
        torch.compiler.is_compiling() or get_proxy_mode() is not None
    )
    message = module.schedule.evaluate(
        values, "torch" if inline else "cuda", edge_index[0]
    )
    return message.new_zeros((num_nodes, module.output_dim)).index_add(
        0, edge_index[1], message
    )

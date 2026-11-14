"""Node-level angular coupling with shared edge channel projections."""

from .... import config


def recoupled(module, values, edge_index, num_nodes):
    x1, x2, x3, weight1, weight2 = values
    weight = module.combine_weights(weight1, weight2)
    message = module.tp132(
        module.tp13(x1, x3)[edge_index[0]],
        x2,
        weight.unsqueeze(0) if weight.ndim == 1 else weight,
    )
    return message.new_zeros((num_nodes, module.output_dim)).index_add(
        0, edge_index[1], message
    )


def evaluate(module, values, edge_index, num_nodes):
    if config.TORCH_REFERENCE:
        x1, x2, x3, weight1, weight2 = values
        return module.forward_reference(
            x1, x2, x3, edge_index, weight1, weight2, num_nodes
        )
    if not config.TORCH_WIGNER6J or (
        config.WIGNER6J_RECOUPLE is None
        and not values[0].is_cuda
        and module.schedule.scalar_edge is not None
        and all(weight.ndim == 1 or weight.shape[0] == 1 for weight in values[3:])
        and module.schedule.use_recoupling(values, edge_index[0])
    ):
        return recoupled(module, values, edge_index, num_nodes)
    message = module.schedule.evaluate(values, "torch", edge_index[0])
    return message.new_zeros((num_nodes, module.output_dim)).index_add(
        0, edge_index[1], message
    )

"""Node coupling followed by fused indexed recoupled convolution."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode
from torch.fx.experimental.symbolic_shapes import statically_known_true

from .....utils.autograd import forward_ad_active
from .... import config

# Per-tile intermediate (not a bound on parameters, outputs or all AD saves).
STANDARD_TILE_BYTES = 64 << 20


def standard_expression(module):
    from .....utils.metadata import compact_metadata
    from .....utils.program import Program

    program = Program()
    widths = (*module.input_dims, module.tp1.weight_numel, module.tp2.weight_numel)
    x, attrs, spin, w1, w2 = (
        program.input(i, kind, width)
        for i, (kind, width) in enumerate(
            zip(("source", "edge", "edge", "edge", "edge"), widths)
        )
    )
    empty = program.constant(0, 0)

    def couple(layer, features, attributes, weights):
        return program.add(
            "convolution",
            layer.output_dim,
            (
                features,
                weights,
                empty,
                attributes,
                program.constant(layer.output_dim, 0),
            ),
            ("o3", layer.kernel_metadata, 4, False, (0, 0), 0),
        )

    mid = couple(module.standard12, x, attrs, w1)
    result = couple(module.standard123, mid, spin, w2)
    return compact_metadata(repr((tuple(program.nodes), ((result, 0, "target"),))))


def standard_dynamic(module, values, edge_index, num_nodes):
    from ....common.edge import edge_program

    x, attrs, spin, w1, w2 = values
    src, dst = edge_index[0], edge_index[1]
    (out,) = edge_program(
        "cuda",
        module.standard_metadata,
        num_nodes,
        src,
        dst,
        x.index_select(-1, module.input1_index),
        attrs,
        spin.index_select(0, src),
        w1,
        w2,
        inline=False,
    )
    return out.index_select(-1, module.output_index)


def use_standard_contractions(module, values):
    if not hasattr(module, "standard12") or any(
        v.dtype != values[0].dtype or v.device != values[0].device for v in values[1:]
    ):
        return False  # Includes pre-existing serialized operators.
    _, attrs, _, weight1, weight2 = values
    edges = attrs.shape[0]
    # forward has validated that a weight table has either one or E rows.
    # Strict export can prove non-broadcast rows before it proves equal symbols.
    return statically_known_true(edges > 0) and all(
        w.ndim == 2
        and (
            statically_known_true(w.shape[0] > 1)
            or statically_known_true(w.shape[0] == edges)
        )
        for w in (weight1, weight2)
    )


def standard(module, values, edge_index, num_nodes, tile):
    x1, x2, x3, w1, w2 = values
    # Both attribute inputs contain only one copy per irrep. Their host
    # mul_ir and CUDA ir_mul layouts coincide. Keep the intermediate ir_mul.
    x1 = x1.index_select(-1, module.input1_index)
    nil1 = w1.new_empty((0, w1.shape[1]))
    nil2 = w2.new_empty((0, w2.shape[1]))

    def contract(src, dst, attrs, first, second):
        rows = torch.arange(src.shape[0], device=src.device)
        mid = module.standard12(x1, attrs, first, nil1, (src, rows), src.shape[0])
        return module.standard123(mid, x3[src], second, nil2, (rows, dst), num_nodes)

    src, dst = edge_index.unbind(0)
    if statically_known_true(src.shape[0] <= tile):
        out = contract(src, dst, x2, w1, w2)
    else:
        out = None
        for start in range(0, src.shape[0], tile):
            k = slice(start, start + tile)
            part = contract(src[k], dst[k], x2[k], w1[k], w2[k])
            out = part if out is None else out + part
    return out.index_select(-1, module.output_index)


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
    if not inline and not forward_ad_active():
        if module.schedule.use_recoupling(values, edge_index[0]):
            return recoupled(module, values, edge_index, num_nodes)
        if (
            edge_index.dtype == torch.int64
            and edge_index.device == values[0].device
            and use_standard_contractions(module, values)
        ):
            width = max(module.standard12.output_dim, module.input_dims[2])
            tile = max(1, STANDARD_TILE_BYTES // (width * values[0].element_size()))
            # Do not specialize a dynamic graph by its number of tiles.
            if statically_known_true(values[1].shape[0] <= tile) or not (
                torch.compiler.is_compiling() or get_proxy_mode() is not None
            ):
                return standard(module, values, edge_index, num_nodes, tile)
            return standard_dynamic(module, values, edge_index, num_nodes)
    message = module.schedule.evaluate(
        values, "torch" if inline else "cuda", edge_index[0]
    )
    return message.new_zeros((num_nodes, module.output_dim)).index_add(
        0, edge_index[1], message
    )

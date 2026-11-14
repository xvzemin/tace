"""Local Torch channel contractions and compiler-visible angular expressions."""

import torch

from .... import config, get_kernel
from ....common.torch.linear import linear as linear_blocks
from ....common.uv_o2 import prepare, program_inputs
from . import transverse as transverse_kernel


def evaluate(
    module,
    features,
    edge_features,
    conv_weights,
    wigner,
    wigner_inv,
    cutoff,
    parameters,
    edge_index,
    vectors,
    radial_network,
):
    """Execute ordinary Torch or the differentiated expression schedule."""
    edge_features, conv_weights, radial_network, expanded = prepare(
        module,
        features,
        edge_features,
        conv_weights,
        edge_index,
        vectors,
        radial_network,
    )
    if expanded:
        from equivariantx.ops.common.torch.edge import evaluate_torch

        metadata, inputs = program_inputs(
            module,
            features,
            edge_features,
            conv_weights,
            wigner,
            wigner_inv,
            cutoff,
            parameters,
            *edge_index,
            vectors,
            radial_network,
            True,
        )
        return evaluate_torch(
            metadata, inputs, *edge_index, features.shape[0], optimized=True
        )[0]
    execute = get_kernel("edge_program", "torch")
    if module.radial_mode == "full" and not config.TORCH_REFERENCE:
        from equivariantx.ops.common.torch.edge import evaluate_recomputed

        execute = evaluate_recomputed
    return staged(
        module,
        features,
        edge_features,
        conv_weights,
        wigner,
        wigner_inv,
        cutoff,
        parameters,
        edge_index,
        vectors,
        radial_network,
        execute,
    )


def staged(
    module,
    features,
    edge_features,
    conv_weights,
    wigner,
    wigner_inv,
    cutoff,
    parameters,
    edge_index,
    vectors,
    radial_network,
    execute,
):
    """Evaluate channel maps and gates with ordinary Torch autograd."""
    source, target = edge_index
    transverse = vectors is not None
    if transverse:
        if any(
            (
                f is None and description is not None
                for f, description in zip(module.transverse_frames, module.frames)
            )
        ):
            raise ValueError("Transverse evaluation requires canonical O(2) frames.")
        direction = vectors / vectors.norm(dim=-1, keepdim=True)
        edges = source.numel()
        edge_ids = torch.arange(edges, device=source.device)
        node_frame, edge_frame, _ = module.transverse_frames

        def project(frame, features, source, target, count, inverse=False):
            if module.backend == "torch" and config.TORCH_REFERENCE:
                return transverse_kernel.evaluate(
                    frame, features, direction, source, target, count, inverse
                )
            return frame(features, direction, source, target, count, inverse=inverse)

        projected_source = project(node_frame, features, source, edge_ids, edges)
        projected_target = project(node_frame, features, target, edge_ids, edges)
        projected_edge = (
            project(edge_frame, edge_features, edge_ids, edge_ids, edges)
            if edge_frame is not None
            else edge_features
        )
        local_inputs = [
            projected_source,
            projected_target,
            projected_edge,
            conv_weights,
        ]
        local_dim = global_dim = 0
        packed = False
        linears = module.transverse_linears
    else:
        if wigner is None:
            raise ValueError("Supply Wigner matrices or vectors.")
        packed = wigner.ndim == 2
        local_dim, global_dim = (wigner.shape[1], 1) if packed else wigner.shape[1:]
        if packed:
            wigner_inv = wigner
        local_inputs = [features, edge_features, conv_weights, wigner]
        linears = module.linears
    metadata = module.prepare_program(local_dim, global_dim, packed, transverse)
    if radial_network is not None:
        from equivariantx.ops.common.radial import compose_radial

        metadata, local_inputs = compose_radial(
            metadata, local_inputs, 3 if transverse else 2, radial_network
        )
    local = execute(metadata, local_inputs, source, target, features.shape[0])
    values = [
        value.reshape(source.shape[0], dim, mul)
        for value, (dim, mul) in zip(
            local[0].split([dim * mul for dim, mul in linears[0][0]], -1), linears[0][0]
        )
    ]
    gates = module.transverse_gates if transverse else module.gates
    for layer, description in enumerate(linears):
        values = linear_blocks(
            values,
            *parameters[2 * layer : 2 * layer + 2],
            description,
            (source.shape[0],),
        )
        if layer < len(gates):
            values = execute(
                gates[layer],
                [value.flatten(-2) for value in values]
                + ([direction] if transverse else []),
                source,
                target,
                features.shape[0],
            )
            values = [
                value.reshape(source.shape[0], dim, mul)
                for value, (dim, mul) in zip(values, linears[layer + 1][0])
            ]
    message = torch.cat([value.flatten(-2) for value in values], -1)
    del values, local, local_inputs
    output = execute(
        module.output_program(local_dim, global_dim, packed, transverse),
        [
            message,
            message.new_empty((source.numel(), 0)) if transverse else wigner_inv,
            cutoff,
        ],
        source,
        target,
        features.shape[0],
    )[0]
    if transverse:
        output = project(
            module.transverse_frames[2],
            output,
            edge_ids,
            target,
            features.shape[0],
            inverse=True,
        )
    return output

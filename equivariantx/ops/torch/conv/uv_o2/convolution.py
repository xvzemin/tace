"""Local Torch channel contractions and compiler-visible angular expressions."""

import torch
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .... import config, get_kernel
from ....common.torch.linear import linear as linear_blocks
from ....common.torch.linear import linear_flat as linear
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
    radial_attention,
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
    if expanded and not module.attention:
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
        radial_attention,
        edge_index,
        vectors,
        radial_network,
        execute,
    )


def prepare(
    module, features, edge_features, conv_weights, edge_index, vectors, radial_network
):
    """Prepare the radial partition and select explicit polynomial expansion."""
    source, target = edge_index
    transverse = vectors is not None
    expanded = (
        transverse
        and module.radial_mode == "last"
        and (not module.attention)
        and (not config.TORCH_REFERENCE)
        and (module.degree <= config.TRANSVERSE_POLYNOMIAL_DEGREE)
        and (
            torch.compiler.is_compiling()
            or get_proxy_mode() is not None
            or torch._C._are_functorch_transforms_active()
        )
    )
    if radial_network is not None:
        from equivariantx.ops.common.radial import radial_features, split_radial

        if config.TORCH_REFERENCE:
            conv_weights = radial_features(radial_network, conv_weights, edge_index)
            radial_network = None
        elif module.radial_mode == "last":
            hidden, projection = split_radial(radial_network)
            if hidden or not projection:
                conv_weights = radial_features(hidden, conv_weights, edge_index)
            radial_network = projection or None
    if edge_features is None:
        edge_features = features.new_empty((source.shape[0], 0))
    return (edge_features, conv_weights, radial_network, expanded)


def program_inputs(
    module,
    features,
    edge_features,
    conv_weights,
    wigner,
    wigner_inv,
    cutoff,
    parameters,
    source,
    target,
    vectors,
    radial_network,
    expanded,
):
    """Construct the local convolution program and its tensor operands."""
    transverse = vectors is not None
    if transverse:
        if any(
            (
                f is None and description is not None
                for f, description in zip(module.transverse_frames, module.frames)
            )
        ):
            raise ValueError("Transverse evaluation requires canonical O(2) frames.")
        from equivariantx.ops.common.wigner import alignment_cuda

        geometry = (
            vectors / vectors.norm(dim=-1, keepdim=True)
            if expanded
            else alignment_cuda(repr("normalize"), [vectors])[0]
        )
        inverse = features.new_empty((source.numel(), 0))
        local_dim = global_dim = 0
        packed = False
    else:
        if wigner is None:
            raise ValueError("Supply Wigner matrices or vectors.")
        packed = wigner.ndim == 2
        local_dim, global_dim = (wigner.shape[1], 1) if packed else wigner.shape[1:]
        geometry, inverse = (wigner, wigner if packed else wigner_inv)
    metadata = module.convolution_program(
        local_dim, global_dim, packed, transverse, native=not expanded
    )
    inputs = [features, edge_features, conv_weights, geometry, inverse, cutoff]
    inputs.extend((value.reshape(1, -1) for value in parameters[:4]))
    if radial_network is not None:
        from equivariantx.ops.common.radial import compose_radial

        metadata, inputs = compose_radial(metadata, inputs, 2, radial_network)
    return (metadata, inputs)


def staged(
    module,
    features,
    edge_features,
    conv_weights,
    wigner,
    wigner_inv,
    cutoff,
    parameters,
    radial_attention,
    edge_index,
    vectors,
    radial_network,
    execute,
    native=False,
):
    """Evaluate channel GEMMs, gates, optional attention, and aggregation."""
    from equivariantx.conv.attention import graph_softmax

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
        if native:
            from equivariantx.ops.common.wigner import alignment_cuda

            direction = alignment_cuda(repr("normalize"), [vectors])[0]
        else:
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
    message = linear_blocks(values, *parameters[:2], linears[0], (source.shape[0],))
    message = execute(
        module.transverse_gate_metadata if transverse else module.gate_metadata,
        [value.flatten(-2) for value in message] + ([direction] if transverse else []),
        source,
        target,
        features.shape[0],
    )
    message = linear_blocks(
        [
            value.reshape(source.shape[0], dim, mul)
            for value, (dim, mul) in zip(message, linears[1][0])
        ],
        *parameters[2:4],
        linears[1],
        (source.shape[0],),
    )
    message = torch.cat([value.flatten(-2) for value in message], -1)
    scale = cutoff
    if module.attention:
        if radial_attention is None:
            raise ValueError("Attention requires radial scale and shift.")
        query = linear(local[2], *parameters[4:6], linears[2])
        key = linear(local[1], *parameters[6:8], linears[3])
        score = execute(
            module.transverse_score_metadata if transverse else module.score_metadata,
            [query, key, radial_attention],
            source,
            target,
            features.shape[0],
        )[0]
        scale = (
            graph_softmax(
                score,
                target,
                features.shape[0],
                cutoff,
                module.eps,
                backend=module.backend,
            )
            * cutoff
        )
    output = execute(
        module.output_program(local_dim, global_dim, packed, transverse),
        [
            message,
            message.new_empty((source.numel(), 0)) if transverse else wigner_inv,
            scale,
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

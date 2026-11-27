"""Local Torch channel contractions and compiler-visible angular expressions."""

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.autograd import forward_ad_active
from .... import config, get_kernel
from ....common.edge import EdgeProgram
from ....common.torch.linear import linear as linear_blocks
from ....common.uv_o2 import prepare, program_inputs, rotation_layout
from ...nn.activation import scalar_blocks, scalar_values
from . import transverse as transverse_kernel


def gate_layout(module):
    """Describe the scalar maps and the circular blocks each of them gates."""
    groups = []
    for locations, acts in (
        (module._scalar_locations, module.act_scalars),
        (module._gate_locations, module.act_gates),
    ):
        blocks = [
            scalar_blocks((size,), (act,)) for size, act in zip(acts._sizes, acts.acts)
        ]
        if any(not block for block in blocks):
            return None
        groups.append(tuple(zip(locations, blocks)))
    paths = tuple(
        (
            module._gated_locations[path.i_gated],
            path.gated_start,
            path.i_gate,
            path.gate_start,
            path.mul,
            path.ir_gate.is_odd_scalar() and path.ir_gated.m > 0,
        )
        for path in module._paths
    )
    return (*groups, paths) if groups[0] or paths else None


def gate_blocks(values, layout):
    """Apply scalar gates to blocked circular features with native broadcasts."""
    scalar_maps, gate_maps, paths = layout
    outputs = [scalar_values(values[i], block) for i, block in scalar_maps]
    gates = [scalar_values(values[i], block) for i, block in gate_maps]
    for i, start, g, offset, mul, odd in paths:
        value = values[i].narrow(-1, start, mul)
        if odd:
            value = torch.stack((-value[..., 1, :], value[..., 0, :]), -2)
        outputs.append(value * gates[g].narrow(-1, offset, mul))
    return outputs


class CircularGate(EdgeProgram):
    """Broadcast the primal; replay the bounded expression for its derivatives."""

    @staticmethod
    def forward(plan, inline, metadata, nodes, source, target, *inputs):
        layout, shapes = plan
        values = [x.reshape(x.shape[0], d, c) for x, (d, c) in zip(inputs, shapes)]
        return tuple(value.flatten(-2) for value in gate_blocks(values, layout))

    @staticmethod
    def setup_context(ctx, inputs, output):
        EdgeProgram.setup_context(ctx, ("torch", *inputs[1:]), output)


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
    target_features=None,
):
    """Execute ordinary Torch or the differentiated expression schedule."""
    receivers = (
        features.shape[0] if target_features is None else target_features.shape[0]
    )
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
            target_features=target_features,
        )
        return evaluate_torch(metadata, inputs, *edge_index, receivers, optimized=True)[
            0
        ]
    execute = get_kernel("edge_program", "torch")
    if module.radial_mode == "full" and not config.TORCH_REFERENCE:
        from equivariantx.ops.common.torch.edge import evaluate_recomputed

        execute = evaluate_recomputed
        if (
            vectors is None
            and features.is_cuda
            and config.TORCH_EDGE_TILE_BYTES
            and not torch.compiler.is_compiling()
            and not is_fake(features)
            and get_proxy_mode() is None
            and not torch._C._are_functorch_transforms_active()
            and edge_index.shape[1]
            * max(features.shape[1], module.weight_numel)
            * features.element_size()
            > config.TORCH_EDGE_TILE_BYTES
        ):
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
                target_features=target_features,
            )
            return execute(metadata, inputs, *edge_index, receivers)[0]
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
        target_features,
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
    target_features,
):
    """Evaluate channel maps and gates with ordinary Torch autograd."""
    source, target = edge_index
    receivers = (
        features.shape[0] if target_features is None else target_features.shape[0]
    )
    target_slot = 0
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
        projected_target = project(
            node_frame,
            features if target_features is None else target_features,
            target,
            edge_ids,
            edges,
        )
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
        local_dim, global_dim, packed = rotation_layout(wigner)
        if packed:
            wigner_inv = wigner
        local_inputs = [features, edge_features, conv_weights, wigner]
        if target_features is not None:
            target_slot = len(local_inputs)
            local_inputs.append(target_features)
        linears = module.linears
    separate = not config.TORCH_REFERENCE and not transverse
    input_blocks = separate and bool(linears[0][0])
    output_blocks = separate and bool(linears[-1][1])
    metadata = module.prepare_program(
        local_dim, global_dim, packed, transverse, input_blocks, target_slot
    )
    if radial_network is not None:
        from equivariantx.ops.common.radial import compose_radial

        metadata, local_inputs = compose_radial(
            metadata,
            local_inputs,
            3 if transverse else 2,
            radial_network,
            num_targets=None if target_features is None else receivers,
        )
    local = execute(metadata, local_inputs, source, target, receivers)
    pieces = (
        local
        if input_blocks
        else local[0].split([dim * mul for dim, mul in linears[0][0]], -1)
    )
    values = [
        value.reshape(source.shape[0], dim, mul)
        for value, (dim, mul) in zip(pieces, linears[0][0])
    ]
    del local, pieces, local_inputs
    gates = module.transverse_gates if transverse else module.gates
    for layer, description in enumerate(linears):
        values = linear_blocks(
            values,
            *parameters[2 * layer : 2 * layer + 2],
            description,
            (source.shape[0],),
        )
        if layer < len(gates):
            layout = module.gate_layouts[layer]
            if (
                separate
                and packed
                and features.is_cuda
                and module.radial_mode == "full"
                and layout is not None
                and not torch.compiler.is_compiling()
                and get_proxy_mode() is None
                and not forward_ad_active()
            ):
                if not torch.is_grad_enabled():
                    values = gate_blocks(values, layout)
                    continue
                values = CircularGate.apply(
                    (layout, linears[layer][1]),
                    True,
                    gates[layer],
                    receivers,
                    source,
                    target,
                    *(value.flatten(-2) for value in values),
                )
            else:
                values = execute(
                    gates[layer],
                    [value.flatten(-2) for value in values]
                    + ([direction] if transverse else []),
                    source,
                    target,
                    receivers,
                )
            values = [
                value.reshape(source.shape[0], dim, mul)
                for value, (dim, mul) in zip(values, linears[layer + 1][0])
            ]
    messages = [value.flatten(-2) for value in values]
    if not output_blocks:
        messages = [torch.cat(messages, -1)]
    del values
    outputs = execute(
        module.output_program(local_dim, global_dim, packed, transverse, output_blocks),
        [
            *messages,
            features.new_empty((source.numel(), 0)) if transverse else wigner_inv,
            cutoff,
        ],
        source,
        target,
        receivers,
    )
    output = outputs[0] if len(outputs) == 1 else torch.cat(outputs, -1)
    if transverse:
        output = project(
            module.transverse_frames[2],
            output,
            edge_ids,
            target,
            receivers,
            inverse=True,
        )
    return output

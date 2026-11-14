"""Native edge scheduling with exact recursively generated mixed adjoints."""

from .precision import require_unscaled, require_tensors

from functools import lru_cache
import os

import torch

from equivariantx.ops.common.contraction import adjoint_program
from equivariantx.utils.autograd import gradient_mask
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs
from .edge_codegen import edge_groups, edge_source
from .packed_edge import pack_edge_metadata, pack_edge_source, packed_description

EDGE_PATH_TILE = int(os.environ.get("EQX_XL_EDGE_PATH_TILE", "0"))
SKIP_EDGE_ZERO = os.environ.get("EQX_XL_SKIP_EDGE_ZERO", "0") == "1"
PACKED_EDGES = os.environ.get("EQX_XL_PACKED_EDGES", "0") == "1"
CSR_NODE_WARPS = int(os.environ.get("EQX_XL_CSR_NODE_WARPS", "0"))
RECEIVER_PIPELINE = os.environ.get("EQX_XL_RECEIVER_PIPELINE", "off")
COTANGENT_RESIDENT = os.environ.get("EQX_XL_COTANGENT_RESIDENT", "0") == "1"


@lru_cache(maxsize=256)
def complete_weight_columns(metadata, width):
    """Require disjoint paths that assign the entire radial result row."""
    offset = 0
    for path in sorted(parse_metadata(metadata)[0], key=lambda item: item[8]):
        if path[8] != offset:
            return False
        offset += path[3] * path[4]
    return offset == width


@lru_cache(maxsize=256)
def edge_plan(
    metadata,
    program,
    dimensions,
    shared,
    dtype,
    device,
    node_warps=0,
    edge_block=False,
    radial_indexed=False,
    radial_scale=0,
):
    """Compile static path groups, retaining weights as runtime arguments."""
    require_unscaled(radial_scale, "radial_scale")
    from equivariantx.ops.common.cuda.runtime import kernels

    paths = parse_metadata(metadata)[0]
    outputs = 1 + max(slot for _, _, pairs in program for _, slot in pairs)
    groups = edge_groups(paths, program, path_tile=EDGE_PATH_TILE)
    codes = [
        edge_source(
            paths,
            terms,
            dimensions,
            shared,
            dtype,
            outputs,
            owner,
            accumulate=EDGE_PATH_TILE > 0,
            node_warps=node_warps,
            edge_block=edge_block,
        )
        for paths, terms, owner in groups
    ]
    packing = packed_description(metadata)
    if packing is not None:
        codes = [
            pack_edge_source(code, packing, terms)
            for code, (_, terms, _) in zip(codes, groups)
        ]
    if radial_indexed:
        from .compressed import compressed_source

        codes = [
            compressed_source(code, terms) for code, (_, terms, _) in zip(codes, groups)
        ]
    from .cache_policy import source_policy

    codes = [
        source_policy(code, terms, dtype) for code, (_, terms, _) in zip(codes, groups)
    ]
    compiled = kernels(codes, device)
    result = []
    for code, (paths, _, owner) in zip(codes, groups):
        kernel = compiled[code]
        threads = max(
            (64, 128, 256),
            key=lambda n: (n * kernel.active_blocks(n, 0), -abs(n - 128)),
        )
        if owner >= 0 and node_warps:
            threads = node_warps * 32
        elif owner < 0 and edge_block:
            threads = (paths[0][3] + 31) // 32 * 32
        elif not node_warps:
            requested = int(os.environ.get("EQX_XL_CG_THREADS", "0"))
            if requested:
                if requested not in (64, 128, 256) or not kernel.active_blocks(
                    requested, 0
                ):
                    raise ValueError(
                        "Requested CG thread count exceeds kernel resource limits"
                    )
                threads = requested
        result.append((kernel, paths[0][3], owner, threads))
    return tuple(result)


def run_edges(
    metadata,
    program,
    source,
    target,
    operands,
    *,
    node_warps=0,
    edge_block=False,
    radial_map=None,
    radial_scale=0,
):
    """Execute owner-separated native kernels for the OAM-XL contract."""
    require_unscaled(radial_scale, "radial_scale")
    require_tensors(operands)
    from .graph import prepare_graph
    from equivariantx.ops.common.cuda.runtime import runtime

    meta, terms = parse_metadata(metadata), parse_metadata(program)
    if node_warps not in (0, 1, 2, 4, 8):
        raise ValueError("Cooperative node rows support one, two, four or eight warps")
    edge_block = edge_block and max(p[3] for p in meta[0]) <= 256
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    destinations = {}
    for mapping, _, pairs in terms:
        for role, slot in pairs:
            destinations.setdefault(slot, set()).add(role)
    supported = (
        (len(meta) == 3 or packed_description(metadata) is not None)
        and not meta[2]
        and source is not None
        and all(p[4] == 1 and p[8] >= 0 for p in meta[0])
        and all(not operands[mapping[2]].numel() for mapping, _, _ in terms)
        and roles <= {0, 1, 3, 4}
        and all(len(owners) == 1 for owners in destinations.values())
        and len({p[3] for p in meta[0]}) == 1
    )
    if not supported:
        if radial_scale:
            raise ValueError(
                "Scaled storage requires the native independent-path contract"
            )
        from .cg_convolution import contraction

        return contraction(
            metadata, program, source, target, operands, batch_paths=True
        )
    if (
        os.environ.get("EQX_XL_SOURCE_VJP", "0") == "1"
        and roles == {0, 1, 3}
        and len(meta[0]) > 5
        and meta[0][0][3] == 128
        and radial_map is None
        and all(operands[m[1]].shape[0] != 1 for m, _, _ in terms)
        and all(
            complete_weight_columns(metadata, operands[m[1]].shape[1])
            for m, _, _ in terms
        )
    ):
        from .source_vjp import source_vjp_edges

        return source_vjp_edges(
            metadata,
            program,
            source,
            target,
            operands,
            grouping="angular",
            full_row=True,
            radial_scale=radial_scale,
        )
    results = o3_outputs(metadata, program, source, target, operands)
    for slot, value in enumerate(results):
        # Every edge weight column is assigned by exactly one path group.
        # Keep all other initialization unchanged, including zero-edge graphs.
        complete_weights = (
            radial_map is None
            and SKIP_EDGE_ZERO
            and source.numel() > 0
            and destinations[slot] == {1}
            and all(
                operands[mapping[1]].shape[0] != 1
                for mapping, _, pairs in terms
                if (1, slot) in pairs
            )
            and complete_weight_columns(metadata, value.shape[1])
        )
        if not complete_weights:
            value.zero_()
    if not source.numel():
        return results
    read = {
        index
        for mapping, _, pairs in terms
        for output, _ in pairs
        for role, index in enumerate(mapping)
        if role != output
    }
    operands = [
        value.contiguous() if i in read else value for i, value in enumerate(operands)
    ]
    source, target = source.contiguous(), target.contiguous()
    plan = edge_plan(
        metadata,
        terms,
        tuple(x.size(1) for x in operands),
        tuple(x.size(0) == 1 for x in operands),
        "float" if operands[0].dtype == torch.float32 else "double",
        operands[0].device,
        node_warps,
        edge_block,
        radial_map is not None,
        radial_scale,
    )
    mapping = terms[0][0]
    counts = {role: operands[mapping[role]].shape[0] for role in (0, 4)}
    orders = {
        owner: prepare_graph(source, target, int(owner == 4), counts[owner])
        for owner in {owner for _, _, owner, _ in plan}
        if owner >= 0
    }
    pointers = [x.data_ptr() for x in (*operands, *results)]
    extra = [counts[0]] if packed_description(metadata) is not None else []
    if radial_map is not None:
        extra.append(radial_map.data_ptr())
    launches = []
    for kernel, width, owner, threads in plan:
        order, offsets = orders[owner] if owner >= 0 else (None, None)
        rows = counts[owner] if owner >= 0 else source.numel()
        arguments = [
            *pointers,
            source.data_ptr(),
            target.data_ptr(),
            0 if order is None else order.data_ptr(),
            0 if offsets is None else offsets.data_ptr(),
            rows,
            *extra,
        ]
        launches.append(
            (
                kernel,
                arguments,
                rows
                if node_warps and owner >= 0 or edge_block and owner < 0
                else (rows + threads // 32 - 1) // (threads // 32),
                1 if edge_block and owner < 0 else (width + 31) // 32,
                threads,
                0,
            )
        )
    runtime().launch(
        launches, torch.cuda.current_stream(operands[0].device).cuda_stream
    )
    return results


def scheduled_edges(metadata, program, source, target, operands):
    """Choose full CSR rows only for the measured force-adjoint program.

    The complete CSR candidate remains separately benchmarkable. Its forward and
    mixed reverse were slower at train-2 sizes, so those phases use the original
    shared-path schedule, including all parameter derivatives.
    """
    roles = {role for _, _, pairs in parse_metadata(program) for role, _ in pairs}
    mixed_owner = os.environ.get("EQX_XL_MIXED_OWNER", "off") != "off"
    if (
        (COTANGENT_RESIDENT or mixed_owner)
        and roles == {0, 1}
        and (source.numel() >= 1024 or mixed_owner)
    ):
        paths = parse_metadata(metadata)[0]
        if len(paths) > 5 and all(
            p[3] == 128 and p[4] == 1 and p[8] >= 0 for p in paths
        ):
            from .cotangent import cotangent_edges

            return cotangent_edges(metadata, program, source, target, operands)
    if RECEIVER_PIPELINE != "off" and roles == {4}:
        from .receiver import receiver_edges

        paths = parse_metadata(metadata)[0]
        if paths and all(p[3] == 128 and p[4] == 1 and p[8] >= 0 for p in paths):
            asynchronous = RECEIVER_PIPELINE == "async" or (
                RECEIVER_PIPELINE == "auto" and len(parse_metadata(program)) == 1
            )
            return receiver_edges(
                metadata, program, source, target, operands, asynchronous
            )
    if (
        source.numel() >= 1024 or os.environ.get("EQX_XL_SOURCE_VJP", "0") == "1"
    ) and roles in ({0, 1, 3}, {1, 3}):
        return run_edges(
            metadata,
            program,
            source,
            target,
            operands,
            node_warps=CSR_NODE_WARPS,
            edge_block=bool(CSR_NODE_WARPS)
            or os.environ.get("EQX_XL_EDGE_ANGULAR_BLOCK", "0") == "1",
        )
    from .cg_convolution import contraction

    return contraction(metadata, program, source, target, operands, batch_paths=True)


@torch.library.custom_op("eqx_oam_xl::edge", mutates_args=(), device_types="cuda")
def edge_contract(
    metadata: str,
    program: str,
    source: torch.Tensor,
    target: torch.Tensor,
    operands: list[torch.Tensor],
) -> list[torch.Tensor]:
    """Evaluate sparse edge programs and their transposes on native CSR rows."""
    if len(parse_metadata(metadata)) != 3 and packed_description(metadata) is None:
        raise ValueError("OAM-XL native edges require explicit angular features")
    return scheduled_edges(metadata, program, source, target, operands)


edge_contract.register_fake(o3_outputs)


def edge_context(ctx, inputs, output):
    metadata, program, source, target, operands = inputs
    ctx.kernel_metadata, ctx.program = metadata, parse_metadata(program)
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(source, target, *operands)


def edge_backward(ctx, grad_outputs):
    source, target, *operands = ctx.saved_tensors
    program, values, destinations = adjoint_program(
        ctx.program,
        operands,
        grad_outputs,
        gradient_mask(operands, ctx.needs_input_grad[4]),
        parse_metadata(ctx.kernel_metadata)[2],
    )
    gradients = [None] * len(operands)
    if program:
        outputs = edge_contract(
            ctx.kernel_metadata, repr(program), source, target, values
        )
        for index, slot in destinations.items():
            gradients[index] = outputs[slot]
    return None, None, None, None, gradients


edge_contract.register_autograd(edge_backward, setup_context=edge_context)


def native_message(
    rejector, nodes, attrs, weights, edge_index, packed=False, radial_map=None
):
    """Use the original CG metadata and parameter-independent angular layout."""
    convolution = rejector.eqx_tp
    metadata = (
        pack_edge_metadata(convolution.kernel_metadata)
        if packed
        else convolution.kernel_metadata
    )
    if packed:
        layout = [
            (channels * paths, degree)
            for _, channels, degree, paths in packed_description(metadata)
        ]
        if layout != [(mul, ir.dim) for mul, ir in rejector.irreps_out.simplify()]:
            raise ValueError(
                "Packed convolution groups do not match the following irreps"
            )
    features = rejector.reshape_in(nodes)
    operands = [
        features,
        weights,
        weights.new_empty((0, weights.shape[1])),
        attrs,
        features.new_empty(1).expand(nodes.shape[0], convolution.output_dim),
    ]
    operation = edge_contract
    if radial_map is not None:
        from .compressed import compressed_edge

        operation = compressed_edge
    message = operation(
        metadata,
        repr((((0, 1, 2, 3, 4), False, ((4, 0),)),)),
        edge_index[0],
        edge_index[1],
        operands,
        *(() if radial_map is None else (radial_map,)),
    )[0]
    return message if packed else rejector.reshape_out(message)

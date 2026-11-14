"""Receiver-local mixed adjoints with live wide cotangents in registers.

The sparse polynomial is unchanged. CSR segments bound work while reusing
receiver cotangents. Source gradients use atomics; each radial output column
has one writer. Parameter updates invalidate no state because plans contain
only metadata and generated code, never model values.
"""

from .precision import require_unscaled, require_tensors
from collections import defaultdict, deque
from functools import lru_cache
import os
import re

import torch

from .cg_codegen import convolution_source
from .graph import prepare_graph
from equivariantx.ops.common.cuda.runtime import kernels, runtime
from equivariantx.utils.metadata import parse_metadata
from equivariantx.utils.outputs import o3_outputs
from .packed_edge import packed_description, pack_edge_source


@lru_cache(maxsize=256)
def cotangent_plan(
    metadata, program, dimensions, shared, dtype, device, radial_scale=0
):
    """Compile complete mixed expressions, splitting paths for register use."""
    require_unscaled(radial_scale, "radial_scale")
    paths = parse_metadata(metadata)[0]
    outputs = 1 + max(slot for _, _, pairs in program for _, slot in pairs)
    weights = {slot for _, _, pairs in program for role, slot in pairs if role == 1}
    groups = defaultdict(list)
    for path in paths:
        groups[path[0]].append(path)
    path_limit = int(os.environ.get("EQX_XL_COTANGENT_PATH_LIMIT", "0"))
    pending = deque(
        tuple(group[start : start + (path_limit or len(group))])
        for group in groups.values()
        for start in range(0, len(group), path_limit or len(group))
    )
    result = []
    while pending:
        group = pending.popleft()
        code = convolution_source(
            group,
            program,
            dimensions,
            shared,
            dtype,
            1,
            outputs,
            initialize=weights,
            hoist_owner=True,
            shared_owner=int(os.environ.get("EQX_XL_SHARED_COTANGENT", "0")),
        )
        packing = packed_description(metadata)
        if packing is not None:
            code = pack_edge_source(code, packing, program)
        kernel = kernels([code], device)[code]
        stored = re.search(r"EQX_OWNER_VALUES=(\d+)", code)
        per_thread = int(stored[1]) * (4 if dtype == "float" else 8) if stored else 0
        choices = [
            n
            for n in (64, 128, 256)
            if n * per_thread <= 49152 and kernel.active_blocks(n, n * per_thread)
        ]
        if (kernel.local_bytes or kernel.registers > 255 or not choices) and len(
            group
        ) > 1:
            half = len(group) // 2
            pending.extend((group[:half], group[half:]))
            continue
        if not choices:
            raise ValueError("Receiver cache exceeds the native shared-memory limit")
        threads = max(
            choices,
            key=lambda n: (n * kernel.active_blocks(n, n * per_thread), -abs(n - 128)),
        )
        requested = int(os.environ.get("EQX_XL_CG_THREADS", "0"))
        if requested:
            if requested not in choices:
                raise ValueError(
                    "Requested CG thread count exceeds kernel resource limits"
                )
            threads = requested
        result.append((kernel, group[0][3], threads, threads * per_thread))
    return tuple(result)


def cotangent_edges(
    metadata, program, source, target, operands, row_size=16, *, radial_scale=0
):
    """Evaluate the complete source/radial mixed reverse program.

    Parameters
    ----------
    metadata, program : str
        Explicit weighted UVU paths and recursively differentiated terms.
    source, target : torch.Tensor
        Current int64 connectivity, including arbitrary order and duplicates.
    operands : list of torch.Tensor
        Live FP32 or FP64 inputs. No parameter or cotangent is cached.
    row_size : int
        Maximum number of adjacent CSR edges processed by one warp.

    Returns
    -------
    list of torch.Tensor
        Node and radial gradients in their declared output slots.
    """
    require_unscaled(radial_scale, "radial_scale")
    require_tensors(operands)
    paths, _, weighted, *_ = parse_metadata(metadata)
    terms = parse_metadata(program)
    roles = {role for _, _, pairs in terms for role, _ in pairs}
    destinations = defaultdict(set)
    for _, _, pairs in terms:
        for role, slot in pairs:
            destinations[slot].add(role)
    offsets = sorted((p[8], p[3]) for p in paths)
    complete = (
        bool(offsets)
        and offsets[0][0] == 0
        and all(
            offset + width == following
            for (offset, width), (following, _) in zip(offsets, offsets[1:])
        )
    )
    weight_width = operands[terms[0][0][1]].shape[1]
    complete = complete and offsets[-1][0] + offsets[-1][1] == weight_width
    if (
        roles != {0, 1}
        or weighted
        or row_size < 1
        or not complete
        or any(p[3] != 128 or p[4] != 1 for p in paths)
        or any(owners not in ({0}, {1}) for owners in destinations.values())
        or any(
            operands[m[1]].shape[0] == 1 or operands[m[2]].numel() for m, _, _ in terms
        )
    ):
        raise ValueError(
            "Cotangent scheduling requires complete nonshared 128-channel UVU mixed adjoints"
        )
    traversal = os.environ.get("EQX_XL_MIXED_OWNER", "off")
    if traversal not in ("off", "source", "target"):
        raise ValueError(f"Unknown mixed-gradient traversal: {traversal}")
    if traversal != "off":
        from .source_vjp import source_vjp_edges

        return source_vjp_edges(
            metadata,
            program,
            source,
            target,
            operands,
            grouping="input",
            full_row=True,
            radial_scale=radial_scale,
            owner=int(traversal == "target"),
        )
    results = o3_outputs(metadata, program, source, target, operands)
    for slot, result in enumerate(results):
        if destinations[slot] != {1} or source.numel() == 0:
            result.zero_()
    if not source.numel():
        return results
    required = {
        i
        for m, _, pairs in terms
        for output, _ in pairs
        for role, i in enumerate(m)
        if role != output
    }
    operands = [
        value.contiguous() if i in required else value
        for i, value in enumerate(operands)
    ]
    source, target = source.contiguous(), target.contiguous()
    dims = tuple(x.shape[1] for x in operands)
    shared = tuple(x.shape[0] == 1 for x in operands)
    dtype = "float" if operands[0].dtype == torch.float32 else "double"
    plan = cotangent_plan(
        metadata, terms, dims, shared, dtype, operands[0].device, radial_scale
    )
    nodes = operands[terms[0][0][4]].shape[0]
    order, _ = prepare_graph(source, target, 1, nodes)
    tasks = (source.numel() + row_size - 1) // row_size
    arguments = [x.data_ptr() for x in (*operands, *results)]
    arguments += [
        source.data_ptr(),
        target.data_ptr(),
        order.data_ptr(),
        source.numel(),
        tasks,
        row_size,
    ]
    if packed_description(metadata) is not None:
        arguments.append(nodes)
    runtime().launch(
        [
            (
                kernel,
                arguments,
                (tasks + threads // 32 - 1) // (threads // 32),
                (width + 31) // 32,
                threads,
                shared_bytes,
            )
            for kernel, width, threads, shared_bytes in plan
        ],
        torch.cuda.current_stream(operands[0].device).cuda_stream,
    )
    return results

"""Bounded path scheduling on the shared NVRTC runtime."""

from collections import defaultdict, deque
from functools import lru_cache

import torch

from ...kernels.cuda import kernels, runtime
from ..graph import prepare_graph
from .codegen import convolution_source
from .convolution import parse_metadata


@lru_cache(maxsize=256)
def execution_plan(metadata, program, dimensions, shared, projected, dtype, device):
    """Partition by multiplicity and compiled register usage, not degree."""
    paths, _, _ = parse_metadata(metadata)
    phases = []
    for mapping, weighted_only, pairs in program:
        # Projection gradients reduce an edge tile before writing shared
        # parameters. Other adjoints share angular products within an edge.
        for projection_gradient in (False, True):
            active = tuple(
                (role, slot)
                for role, slot in pairs
                if (role == 2) == projection_gradient
                and (role != 2 or projected[mapping[2]])
            )
            if not active:
                continue
            groups = defaultdict(list)
            for path in paths:
                if path[8] < 0 and (
                    weighted_only or all(role in (1, 2) for role, _ in active)
                ):
                    continue
                groups[path[3]].append(path)
            for entries in groups.values():
                # Keep register lifetimes bounded before asking the compiler
                # for the actual occupancy and spill count.
                tiles, tile, cost = [], [], 0
                for path in sorted(entries, key=lambda p: (p[0], p[1], p[2])):
                    extra = path[5] + path[6] + path[7] + len(path[-1]) // 8
                    if tile and cost + extra > 64:
                        tiles.append(tuple(tile))
                        tile, cost = [], 0
                    tile.append(path)
                    cost += extra
                if tile:
                    tiles.append(tuple(tile))
                pending = deque(tiles)
                while pending:
                    tile = pending.popleft()
                    roles = tuple(role for role, _ in active)
                    owner = 1 if set(roles) == {4} else 0 if set(roles) == {0} else -1
                    code, edges_per_block = convolution_source(
                        tile,
                        roles,
                        tuple(dimensions[i] for i in mapping),
                        tuple(shared[i] for i in mapping),
                        projected[mapping[2]],
                        dtype,
                        owner,
                    )
                    kernel = kernels((code,), device)[code]
                    if len(tile) > 1 and (kernel.local_bytes or kernel.registers > 192):
                        half = len(tile) // 2
                        pending.extend((tile[:half], tile[half:]))
                        continue
                    phases.append(
                        (
                            kernel,
                            mapping,
                            tuple(slot for _, slot in active),
                            tile[0][3],
                            edges_per_block,
                            owner,
                        )
                    )
    return tuple(phases)


def contract(metadata, program, source, target, operands, results):
    """Launch contractions without retaining per-edge angular intermediates."""
    device, dtype = operands[0].device, operands[0].dtype
    if dtype not in (torch.float32, torch.float64):
        raise TypeError("The O3 CUDA convolution requires float32 or float64.")
    if any(value.device != device or value.dtype != dtype for value in operands):
        raise ValueError("Convolution operands must share a device and dtype.")
    if source.dtype != torch.int64 or target.dtype != torch.int64:
        raise TypeError("Edge indices must have dtype int64.")
    if source.device != device or target.device != device:
        raise ValueError("Edge indices and features must be on the same device.")
    operands = tuple(value.contiguous() for value in operands)
    source, target = source.contiguous(), target.contiguous()
    phases = execution_plan(
        metadata,
        program,
        tuple(value.size(1) for value in operands),
        tuple(value.size(0) == 1 for value in operands),
        tuple(value.numel() != 0 for value in operands),
        "float" if dtype == torch.float32 else "double",
        device,
    )
    orders = {
        owner: prepare_graph(source, target, owner)
        for owner in {phase[-1] for phase in phases}
        if owner >= 0
    }
    launches = []
    for kernel, mapping, slots, width, edges_per_block, owner in phases:
        arguments = [operands[i].data_ptr() for i in mapping]
        arguments += [results[i].data_ptr() for i in slots]
        arguments += [
            source.data_ptr(),
            target.data_ptr(),
            orders[owner].data_ptr() if owner >= 0 else 0,
            source.numel(),
        ]
        launches.append(
            (
                kernel,
                arguments,
                (source.numel() + edges_per_block - 1) // edges_per_block,
                (width + 31) // 32,
                128,
                0,
            )
        )
    with torch.cuda.device(device):
        runtime().launch(launches, torch.cuda.current_stream(device).cuda_stream)

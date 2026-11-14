"""Owner-specific CUDA programs for the OAM-XL trilinear edge operator.

Static CG coefficients are generated code; every learned value is a runtime
operand. Node programs reduce a complete CSR row, including isolated nodes.
Edge programs share angular contractions between radial and angular adjoints.
"""

from collections import defaultdict
import os

PATH_GROUPING = os.environ.get("EQX_XL_CG_GROUPING", "native")
GROUPING_PHASE = os.environ.get("EQX_XL_CG_GROUPING_PHASE", "all")

from equivariantx.ops.common.cuda.codegen import HEADER
from .cg_codegen import angular_source


def edge_groups(paths, program, path_tile=0):
    """Partition disjoint output columns by their reduction owner."""
    roles = {role for _, _, pairs in program for role, _ in pairs}
    groups = []
    for owner in (0, 4, -1):
        selected = roles & ({1, 3} if owner == -1 else {owner})
        if not selected:
            continue
        terms = tuple(
            (mapping, weighted, tuple(p for p in pairs if p[0] in selected))
            for mapping, weighted, pairs in program
            if any(role in selected for role, _ in pairs)
        )
        by_column = defaultdict(list)
        for path in paths:
            # An output block belongs to exactly one kernel. For edge
            # derivatives, grouping by angular degree shares its reduction.
            key = path[0 if owner == 0 else 2 if owner == 4 else 1]
            if (
                owner == -1
                and PATH_GROUPING != "native"
                and (GROUPING_PHASE == "all" or roles == {0, 1})
            ):
                key = {
                    "output_degree": (path[7],),
                    "output_input": (path[7], path[0]),
                    "output_angular": (path[7], path[1]),
                }[PATH_GROUPING]
            by_column[key].append(path)
        for group in by_column.values():
            tile = path_tile or len(group)
            groups.extend(
                (tuple(group[start : start + tile]), terms, owner)
                for start in range(0, len(group), tile)
            )
    return tuple(groups)


def edge_source(
    paths,
    program,
    dimensions,
    shared,
    dtype,
    outputs,
    owner,
    accumulate=False,
    node_warps=0,
    edge_block=False,
):
    """Emit one complete node row or one edge/channel tile.

    The supported contract is weighted ``uvu``, angular multiplicity one, and
    explicit path weights. Projection and all neural-network operations remain
    outside this first native scheduling stage.
    """
    channels = paths[0][3]
    arguments = [f"const T* p{i}" for i in range(len(dimensions))]
    arguments += [f"T* g{i}" for i in range(outputs)]
    arguments += [
        "const int64_t* source",
        "const int64_t* target",
        "const int64_t* order",
        "const int64_t* pointers",
        "int64_t rows",
    ]
    lines = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ", ".join(arguments) + ") {",
        "const int lane = threadIdx.x % 32;",
        (
            "const int u = threadIdx.x;"
            if owner < 0 and edge_block
            else "const int u = int(blockIdx.y) * 32 + lane;"
        ),
        (
            "const bool active = true;"
            if channels == 128 and os.environ.get("EQX_XL_LAUNCH_CONTRACT", "0") == "1"
            else f"const bool active = u < {channels};"
        ),
        (
            "const int64_t row = blockIdx.x;"
            if node_warps and owner >= 0 or edge_block and owner < 0
            else "const int64_t row = int64_t(blockIdx.x) * (blockDim.x / 32) + threadIdx.x / 32;"
        ),
        "if (row >= rows) return;",
    ]
    cache, body, accumulators = {}, [], {}
    for mapping, _, pairs in program:
        needed = {role for role, _ in pairs}
        for path in paths:
            angular = angular_source(
                path, mapping, dimensions, shared, needed, cache, body
            )
            start, attr, end, _, _, dim1, dim2, dimout, offset, factor, _ = path
            weight = f"T({factor:.17g})"
            if needed - {1}:
                pointer = mapping[1]
                key = "radial", pointer, offset, factor
                if key not in cache:
                    name = f"v{len(cache)}"
                    edge = "0" if shared[pointer] else "edge"
                    body.append(
                        f"const T {name} = active ? T({factor:.17g}) * "
                        f"p{pointer}[{edge} * {dimensions[pointer]} + {offset} + u] : T(0);"
                    )
                    cache[key] = name
                weight = cache[key]
            for role, slot in pairs:
                pointer = mapping[role]
                width = dimensions[pointer]
                if role in (0, 4):
                    begin, count = (start, dim1) if role == 0 else (end, dimout)
                    for m in range(count):
                        key = slot, width, begin + m * channels, role, False
                        accumulators.setdefault(key, []).append(
                            (weight, angular[0, role, m])
                        )
                elif role == 3:
                    for m in range(dim2):
                        key = slot, width, attr + m, role, shared[pointer]
                        accumulators.setdefault(key, []).append(
                            (weight, angular[0, role, m])
                        )
                elif role == 1:
                    key = slot, width, offset, role, shared[pointer]
                    accumulators.setdefault(key, []).append(
                        (f"T({factor:.17g})", angular[0, role, 0])
                    )
    if owner >= 0:
        lines.extend(f"T sum{i} = 0;" for i in range(len(accumulators)))
        lines += [
            (
                "for (int64_t k = pointers[row] + threadIdx.x/32; k < pointers[row + 1]; k += blockDim.x/32) {"
                if node_warps
                else "for (int64_t k = pointers[row]; k < pointers[row + 1]; ++k) {"
            ),
            "const int64_t edge = order[k];",
        ]
    else:
        lines += ["const int64_t edge = row;"]
    lines += [
        "const int64_t source_node = source[edge];",
        "const int64_t target_node = target[edge];",
    ]
    body_start = len(lines)
    lines.extend(body)
    angular_slots = {
        key: index for index, key in enumerate(k for k in accumulators if k[3] == 3)
    }
    if owner < 0 and edge_block and angular_slots:
        lines.append(
            f"__shared__ T angular_sums[{(channels + 31) // 32}][{len(angular_slots)}];"
        )
    sink_prefix = lines[body_start + len(body) :]
    sinks = []
    for i, (key, terms) in enumerate(accumulators.items()):
        slot, width, column, role, is_shared = key
        sinks.append((len(lines), role))
        if owner < 0:
            lines.append(f"T sum{i} = 0;")
        for weight, value in terms:
            lines.append(f"sum{i} = fma({weight}, {value}, sum{i});")
        if owner < 0:
            address = f"g{slot} + {'0' if is_shared else 'edge'} * {width} + {column}"
            if role == 3:
                lines.append(f"sum{i} = warp_sum(sum{i});")
                if edge_block:
                    lines.append(
                        f"if (lane == 0) angular_sums[threadIdx.x/32][{angular_slots[key]}] = sum{i};"
                    )
                else:
                    lines.append(f"if (lane == 0) atomicAdd({address}, sum{i});")
            elif is_shared:
                lines.append(f"if (active) atomicAdd({address} + u, sum{i});")
            else:
                lines.append(f"if (active) *({address} + u) = sum{i};")
    mode = os.environ.get("EQX_XL_STREAM_ANGULAR_OUTPUTS", "off")
    if mode == "auto":
        from .angular_schedule import schedule_mode

        mode = schedule_mode(mode, program)
    if mode != "off" and sinks:
        from .angular_schedule import stream_body

        ends = [start for start, _ in sinks[1:]] + [len(lines)]
        chunks = [(lines[start:end], role) for (start, role), end in zip(sinks, ends)]
        lines = lines[:body_start] + sink_prefix + stream_body(body, chunks, mode)
    if owner < 0 and edge_block and angular_slots:
        lines.append("__syncthreads();")
        for key, index in angular_slots.items():
            slot, width, column, _, is_shared = key
            lines += [
                f"if (threadIdx.x == {index}) {{ T total=0;",
                f"for (int warp=0;warp<{(channels + 31) // 32};++warp) total+=angular_sums[warp][{index}];",
            ]
            address = f"g{slot} + {'0' if is_shared else 'row'}*{width}+{column}"
            if is_shared or accumulate:
                lines.append(f"atomicAdd({address},total);}}")
            else:
                lines.append(f"*({address})=total;}}")
    if owner >= 0:
        lines.append("}")
        if node_warps:
            lines.append(
                f"__shared__ T node_sums[{len(accumulators)}][{node_warps * 32}];"
            )
            for i in range(len(accumulators)):
                lines.append(f"node_sums[{i}][threadIdx.x]=sum{i};")
            lines.append("__syncthreads();")
            lines.append("if (threadIdx.x < 32) {")
            for i in range(len(accumulators)):
                lines += [
                    f"sum{i}=0;",
                    f"for (int warp=0;warp<{node_warps};++warp) sum{i}+=node_sums[{i}][warp*32+lane];",
                ]
        for i, (slot, width, column, _, _) in enumerate(accumulators):
            if accumulate:
                lines.append(
                    f"if (active) atomicAdd(g{slot}+row*{width}+{column}+u,sum{i});"
                )
            else:
                lines.append(
                    f"if (active) g{slot}[row * {width} + {column} + u] = sum{i};"
                )
        if node_warps:
            lines.append("}")
    lines.append("}")
    return "\n".join(lines)

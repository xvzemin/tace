"""CUDA contractions with sparse angular tensors and vector adjoints."""

from functools import lru_cache

from ...kernels.codegen import HEADER
from .convolution import kernel_plan
from .geometry import angular_coefficients


@lru_cache(maxsize=256)
def direction_source(metadata, indices, calls, layouts, dtype, owner=-1):
    """Generate a shared rotation tile for mixed direction derivatives."""
    plan = kernel_plan(metadata)
    paths = [plan.path_data[i][1] for i in indices]
    mul = paths[0][2]
    wide = mul > 32
    barrier = "__syncthreads();" if wide else "__syncwarp();"
    entries = {}
    matrices = {}
    vectors = {}
    matrix_size = 0
    for rank, outputs, operands, results, weighted in calls:
        for i, path in zip(indices, paths):
            if weighted and path[8] < 0:
                continue
            cg = angular_coefficients(metadata, rank)[i]
            entries[rank, i] = tuple(
                (*row, float(cg[tuple(row)])) for row in cg.nonzero().tolist()
            )
            if not entries[rank, i]:
                continue
            if i not in plan.scalar_paths:
                for p, offset, dim in (
                    (operands[3], path[6], path[4]),
                    (operands[4], path[7], path[5]),
                ):
                    if (p, offset, dim) not in matrices:
                        matrices[p, offset, dim] = matrix_size
                        matrix_size += dim * dim
            if rank and (operands[3], 1, 3) not in matrices:
                matrices[operands[3], 1, 3] = matrix_size
                matrix_size += 9
            for axis in range(rank):
                if any(output != 7 + axis for output in outputs):
                    key = operands[7 + axis], operands[3]
                    vectors.setdefault(key, 3 * len(vectors))

    pointers = len(layouts)
    destinations = {p for _, _, _, results, _ in calls for p in results}
    result_roles = {
        p: {
            role
            for _, outputs, _, results, _ in calls
            for role, result in zip(outputs, results)
            if result == p
        }
        for p in destinations
    }
    arguments = [
        f"{'T' if i in destinations else 'const T'}* __restrict__ p{i}"
        for i in range(pointers)
    ] + [
        "const long long* source",
        "const long long* target",
        "const long long* order",
        "long long edges",
        "long long task_count",
        "long long row_size",
    ]
    lines = [
        HEADER.replace("SCALAR", dtype),
        'extern "C" __global__ void run(' + ",".join(arguments) + ") {",
        "const int lane = threadIdx.x & 31;",
        "const long long item = blockIdx.x;"
        if wide
        else "const long long item = (long long)blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;",
        "if (item >= task_count) return;",
        "const int channel = blockIdx.y * blockDim.x + threadIdx.x;"
        if wide
        else "const int channel = blockIdx.y * 32 + lane;",
        f"const bool active = channel < {mul};",
    ]
    emit = lines.append

    def address(p, row, column):
        shared, stride, inner = layouts[p]
        return f"{('0' if shared else row)} * {stride} + ({column}) * {inner}"

    def load(p, row, column):
        return f"p{p}[{address(p, row, column)}]"

    def matrix(p, offset, dim, a, b):
        return f"matrix[{matrices[p, offset, dim] + a * dim + b}]"

    storage_size = matrix_size + 3 * len(vectors)
    if storage_size:
        emit("extern __shared__ T storage[];")
        emit(
            "T* matrix = storage;"
            if wide
            else f"T* matrix = storage + (threadIdx.x / 32) * {storage_size};"
        )
    owned_inputs, owned_gradients = {}, {}
    if owner >= 0:
        index = "source" if owner == 0 else "target"
        emit("const long long stop = min((item + 1) * row_size, edges);")
        emit("long long end = item * row_size;")
        emit("while (end < stop) {")
        emit("const long long begin = end;")
        emit(f"const long long node = {index}[order[begin]];")
        emit(f"do {{ ++end; }} while (end < stop && {index}[order[end]] == node);")
        emit(
            f"const bool exclusive = (begin == 0 || {index}[order[begin - 1]] != node) "
            f"&& (end == edges || {index}[order[end]] != node);"
        )
        declarations = len(lines)
        emit("for (long long position = begin; position < end; ++position) {")
        emit("const long long edge = order[position];")
    else:
        emit("const long long edge = item;")
    emit("const long long src = source[edge], dst = target[edge];")
    if storage_size:
        for (p, offset, dim), start in matrices.items():
            thread, width = ("threadIdx.x", "blockDim.x") if wide else ("lane", "32")
            emit(
                f"for (int k = {thread}; k < {dim * dim}; k += {width}) matrix[{start} + k] = {load(p, 'edge', f'{offset} + k')};"
            )
        emit(barrier)
        # Direction cotangents are shared by all feature channels. Rotate
        # each once per edge tile, not once per channel.
        for (p, frame), start in vectors.items():
            thread = "threadIdx.x" if wide else "lane"
            emit(f"if ({thread} < 3) {{ T value = 0;")
            for b in range(3):
                emit(
                    f"value = fma(matrix[{matrices[frame, 1, 3]} + 3 * {thread} + {b}], {load(p, 'edge', str(b))}, value);"
                )
            emit(f"matrix[{matrix_size + start} + {thread}] = value; }}")
        if vectors:
            emit(barrier)

    rotated = {}
    gradients = {}

    def local(p, row, start, dim, channels, frame, offset, selected):
        key = p, row, start, dim, channels, frame, offset
        name, available = rotated.setdefault(key, (f"v{len(rotated)}_", set()))
        owned = owner >= 0 and row == ("src" if owner == 0 else "dst")
        if owned:
            node_key = p, start, dim, channels
            node_name = owned_inputs.setdefault(node_key, f"node{len(owned_inputs)}_")
        for a in sorted(set(selected) - available):
            available.add(a)
            if channels == 0:
                emit(
                    f"const T {name}{a} = matrix[{matrix_size + vectors[p, frame] + a}];"
                )
            elif offset < 0:
                value = (
                    f"{node_name}{a}"
                    if owned
                    else f"active ? {load(p, row, f'{start + a * channels} + channel')} : T(0)"
                )
                emit(f"T {name}{a} = {value};")
            else:
                emit(f"T {name}{a} = 0;")
                for b in range(dim):
                    column = f"{start + b * channels} + channel"
                    value = (
                        f"{node_name}{b}"
                        if owned
                        else f"(active ? {load(p, row, column)} : T(0))"
                    )
                    emit(
                        f"{name}{a} = fma({matrix(frame, offset, dim, a, b)}, {value}, {name}{a});"
                    )
        return name

    def gradient(p, row, start, dim, channels, frame, offset, selected):
        key = p, row, start, dim, channels, frame, offset
        name, available = gradients.setdefault(key, (f"d{len(gradients)}_", set()))
        for a in sorted(set(selected) - available):
            available.add(a)
            emit(f"T {name}{a} = 0;")
        return name

    for i, path in zip(indices, paths):
        start, end, _, _, dim, dim_out, di, do, weight, harmonic = path
        if i in plan.scalar_paths:
            di = do = -1
        for rank, outputs, operands, results, weighted in calls:
            if weighted and weight < 0:
                continue
            cg = entries.get((rank, i), ())
            if not cg:
                continue
            for output, result in zip(outputs, results):
                # Nonempty projection adjoints are handled by the GEMM schedule.
                if output == 2:
                    continue
                if output in (1, 2) and weight < 0:
                    continue
                rows = sorted({row[0] for row in cg})
                columns = sorted({row[1] for row in cg})
                lx = (
                    gradient(result, "src", start, dim, mul, operands[3], di, rows)
                    if output == 0
                    else local(
                        operands[0], "src", start, dim, mul, operands[3], di, rows
                    )
                )
                ly = (
                    gradient(result, "dst", end, dim_out, mul, operands[4], do, columns)
                    if output == 6
                    else local(
                        operands[6], "dst", end, dim_out, mul, operands[4], do, columns
                    )
                )
                lv = []
                for axis in range(rank):
                    selected = {row[2 + axis] for row in cg}
                    if output == 7 + axis:
                        lv.append(
                            gradient(result, "edge", 0, 3, 0, operands[3], 1, selected)
                        )
                    else:
                        lv.append(
                            local(
                                operands[7 + axis],
                                "edge",
                                0,
                                3,
                                0,
                                operands[3],
                                1,
                                selected,
                            )
                        )
                factors = []
                if output != 1 and weight >= 0:
                    factors.append(
                        f"(active ? {load(operands[1], 'edge', f'{weight} + channel')} : T(0))"
                    )
                if output != 5:
                    factors.append(load(operands[5], "edge", str(harmonic)))
                factor = " * ".join(factors) or "T(1)"
                emit("{")
                emit(f"const T factor = {factor};")
                # Apply radial factors after the sparse angular contraction.
                values = {}
                for a, b, *tail in cg:
                    axes, coefficient = tail[:-1], tail[-1]
                    product = []
                    if output != 0:
                        product.append(f"{lx}{a}")
                    if output != 6:
                        product.append(f"{ly}{b}")
                    product.extend(
                        f"{lv[k]}{axis}"
                        for k, axis in enumerate(axes)
                        if output != 7 + k
                    )
                    destination = (
                        a
                        if output == 0
                        else b
                        if output == 6
                        else axes[output - 7]
                        if output >= 7
                        else 0
                    )
                    values.setdefault(destination, []).append(
                        (coefficient, " * ".join(product) or "T(1)")
                    )
                for destination, terms in values.items():
                    emit("{ T value = 0;")
                    for coefficient, product in terms:
                        emit(f"value = fma(T({coefficient:.17g}), ({product}), value);")
                    if output == 0:
                        name = f"{lx}{destination}"
                    elif output == 6:
                        name = f"{ly}{destination}"
                    elif output >= 7:
                        name = f"{lv[output - 7]}{destination}"
                    else:
                        column = f"{weight} + channel" if output == 1 else str(harmonic)
                        if output == 1 and not layouts[result][0]:
                            emit(
                                f"if (active) {load(result, 'edge', column)} += factor * value;"
                            )
                        else:
                            if output == 5:
                                emit("value = warp_sum(factor * value);")
                                emit(
                                    f"if (lane == 0) atomicAdd(p{result} + {address(result, 'edge', column)}, value);"
                                )
                            else:
                                emit(
                                    f"if (active) atomicAdd(p{result} + {address(result, 'edge', column)}, factor * value);"
                                )
                        emit("}")
                        continue
                    emit(f"{name} = fma(factor, value, {name}); }}")
                emit("}")

    for (p, row, start, dim, channels, frame, offset), (
        name,
        selected,
    ) in gradients.items():
        if channels == 0:
            # Reduce the local vector before rotating it back. Only lane zero
            # performs the inverse rotation and writes the edge cotangent.
            for a in sorted(selected):
                emit(f"{name}{a} = warp_sum({name}{a});")
            emit("if (lane == 0) {")
        for b in range(dim):
            emit("{ T value = 0;")
            for a in sorted(selected):
                if offset < 0:
                    if a == b:
                        emit(f"value = {name}{a};")
                else:
                    emit(
                        f"value = fma({matrix(frame, offset, dim, a, b)}, {name}{a}, value);"
                    )
            column = (
                str(start + b) if channels == 0 else f"{start + b * channels} + channel"
            )
            if channels == 0:
                if not wide and not layouts[p][0]:
                    emit(f"if (gridDim.y == 1) {load(p, row, column)} += value; else")
                emit(f"atomicAdd(p{p} + {address(p, row, column)}, value);")
            elif owner >= 0 and row == ("src" if owner == 0 else "dst"):
                key = p, start, dim, channels
                total = owned_gradients.setdefault(key, f"total{len(owned_gradients)}_")
                emit(f"{total}{b} += value;")
            else:
                emit(f"if (active) atomicAdd(p{p} + {address(p, row, column)}, value);")
            emit("}")
        if channels == 0:
            emit("}")
    if owner >= 0:
        if storage_size:
            emit(barrier)
        emit("}")
        initializers = []
        for (p, start, dim, channels), name in owned_inputs.items():
            for b in range(dim):
                column = f"{start + b * channels} + channel"
                initializers.append(
                    f"const T {name}{b} = active && begin < end ? {load(p, 'node', column)} : T(0);"
                )
        for (p, start, dim, channels), name in owned_gradients.items():
            private = result_roles[p] == {0 if owner == 0 else 6}
            for b in range(dim):
                initializers.append(f"T {name}{b} = 0;")
                column = f"{start + b * channels} + channel"
                emit(
                    f"if (active && begin < end) {{ if (exclusive && {str(private).lower()}) {load(p, 'node', column)} += {name}{b}; "
                    f"else atomicAdd(p{p} + {address(p, 'node', column)}, {name}{b}); }}"
                )
        lines[declarations:declarations] = initializers
        emit("}")
    emit("}")
    return "\n".join(lines), mul, storage_size * (4 if dtype == "float" else 8)

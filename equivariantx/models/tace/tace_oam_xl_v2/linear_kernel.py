"""Grouped Tensor Core programs for values and mixed linear adjoints."""

import triton
import triton.language as tl


@triton.jit
def rounded_tf32(value):
    # tl.dot(tf32) may truncate FP32 inputs. Match the nearest rounding of
    # the production GEMM path before issuing Tensor Core instructions.
    bits = tl.inline_asm_elementwise(
        "cvt.rn.tf32.f32 $0, $1;",
        constraints="=r,f",
        args=[value],
        dtype=tl.int32,
        is_pure=True,
        pack=1,
    )
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def node_program(
    P,
    O,
    rows,
    GROUPS: tl.constexpr,
    INPUT_LAYOUT: tl.constexpr,
    OUTPUT_LAYOUT: tl.constexpr,
    INPUT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    ROLE: tl.constexpr,
    PRECISION: tl.constexpr,
    BM: tl.constexpr = 64,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    task = tl.program_id(0)
    first = 0
    for group in tl.static_range(len(GROUPS)):
        # A group is one disjoint irrep matrix and one output, with all
        # bilinear terms contributing to it evaluated in the same tile.
        path: tl.constexpr
        path = GROUPS[group][0]
        slot: tl.constexpr
        slot = GROUPS[group][1]
        terms: tl.constexpr
        terms = GROUPS[group][2]
        ix: tl.constexpr
        ix = path[0]
        ci: tl.constexpr
        ci = path[1]
        iy: tl.constexpr
        iy = path[2]
        co: tl.constexpr
        co = path[3]
        degree: tl.constexpr
        degree = path[4]
        scale: tl.constexpr
        scale = path[5]
        width: tl.constexpr
        width = ci if ROLE == 0 else co
        reduction: tl.constexpr
        reduction = co if ROLE == 0 else ci
        columns: tl.constexpr
        columns = (width + BN - 1) // BN
        tiles = tl.cdiv(rows * degree, BM) * columns
        if task >= first and task < first + tiles:
            tile = task - first
            m = (tile // columns) * BM + tl.arange(0, BM)
            n = (tile % columns) * BN + tl.arange(0, BN)
            k = tl.arange(0, BK)
            node, angular = m // degree, m % degree
            value = tl.full((BM, BN), 0, tl.float32)
            for term in tl.static_range(len(terms)):
                mapping: tl.constexpr
                mapping = terms[term]
                x: tl.constexpr
                x = mapping[0]
                w: tl.constexpr
                w = mapping[1]
                y: tl.constexpr
                y = mapping[2]
                for begin in range((reduction + BK - 1) // BK):
                    kk = begin * BK + k
                    if ROLE == 2:
                        address = node[:, None] * INPUT_WIDTH + ix
                        if INPUT_LAYOUT == "ir_mul":
                            address += angular[:, None] * ci + kk[None, :]
                        else:
                            address += kk[None, :] * degree + angular[:, None]
                        a = tl.load(
                            P[x] + address,
                            (node[:, None] < rows) & (kk[None, :] < ci),
                            0,
                        )
                        b = tl.load(
                            P[w] + kk[:, None] * co + n[None, :],
                            (kk[:, None] < ci) & (n[None, :] < co),
                            0,
                        )
                    else:
                        address = node[:, None] * OUTPUT_WIDTH + iy
                        if OUTPUT_LAYOUT == "ir_mul":
                            address += angular[:, None] * co + kk[None, :]
                        else:
                            address += kk[None, :] * degree + angular[:, None]
                        a = tl.load(
                            P[y] + address,
                            (node[:, None] < rows) & (kk[None, :] < co),
                            0,
                        )
                        b = tl.load(
                            P[w] + n[None, :] * co + kk[:, None],
                            (n[None, :] < ci) & (kk[:, None] < co),
                            0,
                        )
                        # Match the original autograd order before TF32 rounding.
                        a *= scale
                    if PRECISION == "tf32":
                        a, b = rounded_tf32(a), rounded_tf32(b)
                    value = tl.dot(a, b, value, input_precision=PRECISION)
            if ROLE == 2:
                value *= scale
            if ROLE == 2:
                address = node[:, None] * OUTPUT_WIDTH + iy
                if OUTPUT_LAYOUT == "ir_mul":
                    address += angular[:, None] * co + n[None, :]
                else:
                    address += n[None, :] * degree + angular[:, None]
            else:
                address = node[:, None] * INPUT_WIDTH + ix
                if INPUT_LAYOUT == "ir_mul":
                    address += angular[:, None] * ci + n[None, :]
                else:
                    address += n[None, :] * degree + angular[:, None]
            tl.store(
                O[slot] + address, value, (node[:, None] < rows) & (n[None, :] < width)
            )
        first += tiles


@triton.jit
def weight_program(
    P,
    Partial,
    rows,
    GROUPS: tl.constexpr,
    INPUT_LAYOUT: tl.constexpr,
    OUTPUT_LAYOUT: tl.constexpr,
    INPUT_WIDTH: tl.constexpr,
    OUTPUT_WIDTH: tl.constexpr,
    PRECISION: tl.constexpr,
    SPLIT: tl.constexpr = 8,
    BM: tl.constexpr = 64,
    BN: tl.constexpr = 64,
    BK: tl.constexpr = 32,
):
    task, split = tl.program_id(0), tl.program_id(1)
    first: tl.constexpr
    first = 0
    for group in tl.static_range(len(GROUPS)):
        path: tl.constexpr
        path = GROUPS[group][0]
        terms: tl.constexpr
        terms = GROUPS[group][2]
        ix: tl.constexpr
        ix = path[0]
        ci: tl.constexpr
        ci = path[1]
        iy: tl.constexpr
        iy = path[2]
        co: tl.constexpr
        co = path[3]
        degree: tl.constexpr
        degree = path[4]
        scale: tl.constexpr
        scale = path[5]
        columns: tl.constexpr
        columns = (co + BN - 1) // BN
        tiles: tl.constexpr
        tiles = ((ci + BM - 1) // BM) * columns
        if task >= first and task < first + tiles:
            tile = task - first
            m = (tile // columns) * BM + tl.arange(0, BM)
            n = (tile % columns) * BN + tl.arange(0, BN)
            k = tl.arange(0, BK)
            value = tl.full((BM, BN), 0, tl.float32)
            for term in tl.static_range(len(terms)):
                mapping: tl.constexpr
                mapping = terms[term]
                x: tl.constexpr
                x = mapping[0]
                w: tl.constexpr
                w = mapping[1]
                y: tl.constexpr
                y = mapping[2]
                for begin in range(split, tl.cdiv(rows * degree, BK), SPLIT):
                    kk = begin * BK + k
                    node, angular = kk // degree, kk % degree
                    ax = node[None, :] * INPUT_WIDTH + ix
                    if INPUT_LAYOUT == "ir_mul":
                        ax += angular[None, :] * ci + m[:, None]
                    else:
                        ax += m[:, None] * degree + angular[None, :]
                    ay = node[:, None] * OUTPUT_WIDTH + iy
                    if OUTPUT_LAYOUT == "ir_mul":
                        ay += angular[:, None] * co + n[None, :]
                    else:
                        ay += n[None, :] * degree + angular[:, None]
                    a = tl.load(
                        P[x] + ax, (m[:, None] < ci) & (node[None, :] < rows), 0
                    )
                    b = tl.load(
                        P[y] + ay, (node[:, None] < rows) & (n[None, :] < co), 0
                    )
                    b *= scale
                    if PRECISION == "tf32":
                        a, b = rounded_tf32(a), rounded_tf32(b)
                    value = tl.dot(a, b, value, input_precision=PRECISION)
            address = split * ci * co + m[:, None] * co + n[None, :]
            tl.store(
                Partial[group] + address,
                value,
                (m[:, None] < ci) & (n[None, :] < co),
            )
        first += tiles


@triton.jit
def weight_reduce(
    Partial, O, GROUPS: tl.constexpr, SPLIT: tl.constexpr = 8, BLOCK: tl.constexpr = 256
):
    task = tl.program_id(0)
    first: tl.constexpr
    first = 0
    for group in tl.static_range(len(GROUPS)):
        path: tl.constexpr
        path = GROUPS[group][0]
        slot: tl.constexpr
        slot = GROUPS[group][1]
        width: tl.constexpr
        width = path[1] * path[3]
        tiles: tl.constexpr
        tiles = (width + BLOCK - 1) // BLOCK
        if task >= first and task < first + tiles:
            column = (task - first) * BLOCK + tl.arange(0, BLOCK)
            split = tl.arange(0, SPLIT)
            values = tl.load(
                Partial[group] + split[:, None] * width + column[None, :],
                column[None, :] < width,
                0,
            )
            tl.store(O[slot] + column, tl.sum(values, axis=0), column < width)
        first += tiles

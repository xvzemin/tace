################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""CUDA contractions for the explicitly selected Triton backend."""

import weakref
from collections import OrderedDict

import torch

try:
    import triton
    import triton.language as tl
except ImportError as error:
    raise ImportError(
        "The Triton convolution backend requires Triton. "
        "Install the optional dependency with pip install './tace/eqx[triton]'."
    ) from error

# Older Triton versions require at least 16 rows and columns for dot products.
SMALL_DOT = tl.constexpr(tuple(map(int, triton.__version__.split(".")[:2])) >= (3, 5))
PROJECTION_CHUNK_SIZE = 16384
PROJECTION_MIN_NUMEL = 4096
CHANNEL_TILE = 32
EDGE_TILE = 16
TILE_BUDGET = 2048
_EDGE_ORDERS = OrderedDict()


def edge_order(indices, chunk_size):
    """Cache bounded source/receiver traversal without retaining the input graph."""
    base = indices._base if indices._base is not None else indices
    version = None if base.is_inference() else base._version
    key = (
        id(base),
        version,
        torch.cuda.current_stream(indices.device).cuda_stream
        if indices.is_cuda
        else None,
        indices.storage_offset(),
        indices.numel(),
        chunk_size,
    )
    cached = _EDGE_ORDERS.get(key)
    if version is not None and cached is not None and cached[0]() is base:
        _EDGE_ORDERS.move_to_end(key)
        return cached[1]
    padding = (-indices.numel()) % chunk_size
    rows = (
        torch.cat(
            (indices, indices.new_full((padding,), torch.iinfo(indices.dtype).max))
        )
        if padding
        else indices
    )
    order = rows.view(-1, chunk_size).argsort(dim=1, stable=True).flatten()
    if version is not None:
        _EDGE_ORDERS[key] = (weakref.ref(base), order)
    while len(_EDGE_ORDERS) > 8:
        _EDGE_ORDERS.popitem(last=False)
    return order


@triton.jit
def _rotate(matrix, features):
    size_e: tl.constexpr = features.shape[0]
    size_d: tl.constexpr = features.shape[1]
    size_c: tl.constexpr = features.shape[2]
    if features.dtype == tl.float32 and size_d >= 16 and size_c >= 16:
        return tl.dot(matrix, features, input_precision="ieee")
    result = tl.full((size_e, size_d, size_c), 0, features.dtype)
    for k in tl.static_range(size_d):
        column = tl.gather(
            matrix, tl.full((size_e, size_d, 1), k, tl.int32), 2
        ).reshape(size_e, size_d)
        row = tl.gather(features, tl.full((size_e, 1, size_c), k, tl.int32), 1).reshape(
            size_e, size_c
        )
        result += column[:, :, None] * row[:, None, :]
    return result


@triton.jit
def _outer(left, right):
    size_e: tl.constexpr = left.shape[0]
    size_d: tl.constexpr = left.shape[1]
    size_c: tl.constexpr = left.shape[2]
    if left.dtype == tl.float32 and size_c >= 16 and (size_d >= 16 or SMALL_DOT):
        return tl.dot(left, tl.permute(right, (0, 2, 1)), input_precision="ieee")
    if size_d <= 8:
        return tl.sum(left[:, :, None, :] * right[:, None, :, :], 3)
    result = tl.full((size_e, size_d, size_d), 0, left.dtype)
    for k in tl.static_range(size_c):
        a = tl.gather(left, tl.full((size_e, size_d, 1), k, tl.int32), 2)
        b = tl.gather(right, tl.full((size_e, size_d, 1), k, tl.int32), 2)
        result += a * tl.permute(b, (0, 2, 1))
    return result


@triton.jit
def _segment_add(head_left, value_left, head_right, value_right):
    return head_left | head_right, tl.where(
        head_right, value_right, value_left + value_right
    )


@triton.jit
def _scatter(pointer, value, nodes, mask):
    # Combine consecutive contributions within the edge tile. Only segment
    # boundaries use atomics, including rows that cross tile boundaries.
    edges: tl.constexpr = value.shape[0]
    nodes = tl.where(tl.sum(tl.sum(mask.to(tl.int32), 2), 1) > 0, nodes, -1)
    previous = tl.gather(nodes, tl.maximum(tl.arange(0, edges) - 1, 0), 0)
    heads = (nodes != previous) | (tl.arange(0, edges) == 0)
    heads = tl.broadcast_to(heads[:, None, None], value.shape)
    _, total = tl.associative_scan((heads, value), 0, _segment_add)
    next_node = tl.gather(nodes, tl.minimum(tl.arange(0, edges) + 1, edges - 1), 0)
    last = (nodes != next_node) | (tl.arange(0, edges) == edges - 1)
    tl.atomic_add(pointer, total, mask & last[:, None, None], sem="relaxed")


@triton.jit
def _channelwise_path(
    R,
    DO,
    S,
    Y,
    GR,
    GDO,
    GS,
    GY,
    local_x,
    local_grad,
    dst,
    u,
    mask_e,
    er,
    eo,
    es,
    R_DIM: tl.constexpr,
    Y_DIM: tl.constexpr,
    D_DIM: tl.constexpr,
    S_DIM: tl.constexpr,
    R_SHARED: tl.constexpr,
    OUTPUTS: tl.constexpr,
    path: tl.constexpr,
    B_E: tl.constexpr,
    B_C: tl.constexpr,
    B_I: tl.constexpr,
    mul: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    WRITE_X: tl.constexpr = (OUTPUTS & 1) != 0
    WRITE_R: tl.constexpr = (OUTPUTS & 2) != 0
    WRITE_DI: tl.constexpr = (OUTPUTS & 8) != 0
    WRITE_DO: tl.constexpr = (OUTPUTS & 16) != 0
    WRITE_S: tl.constexpr = (OUTPUTS & 32) != 0
    WRITE_Y: tl.constexpr = (OUTPUTS & 64) != 0
    dtype = local_x.dtype
    a = tl.arange(0, B_I)
    end: tl.constexpr = path[1]
    dim_out: tl.constexpr = path[5]
    dend: tl.constexpr = path[7]
    weight: tl.constexpr = path[8]
    harmonic: tl.constexpr = path[9]
    b = tl.arange(0, triton.next_power_of_2(dim_out))
    mask_y = (
        mask_e[:, None, None] & (b[None, :, None] < dim_out) & (u[None, None, :] < mul)
    )
    mask_do = (
        mask_e[:, None, None]
        & (b[None, :, None] < dim_out)
        & (b[None, None, :] < dim_out)
    )
    if WRITE_X or WRITE_R or WRITE_DI or WRITE_DO or WRITE_Y:
        amplitude = tl.load(S + es * S_DIM + harmonic, mask_e, 0)
    if WRITE_X or WRITE_R or WRITE_DI or WRITE_S or WRITE_Y:
        dout = tl.load(
            DO
            + eo[:, None, None] * D_DIM
            + dend
            + b[None, :, None] * dim_out
            + b[None, None, :],
            mask_do,
            0,
        )
    if WRITE_X or WRITE_R or WRITE_DI or WRITE_DO or WRITE_S:
        y = tl.load(
            Y
            + dst[:, None, None] * Y_DIM
            + end
            + b[None, :, None] * mul
            + u[None, None, :],
            mask_y,
            0,
        )
    if WRITE_R or WRITE_DO or WRITE_S or WRITE_Y:
        index = tl.full((b.shape[0],), 0, tl.int32)
        coefficient = tl.full((b.shape[0],), 0, dtype)
        for cg_index in tl.static_range((len(path) - 10) // 3):
            m = path[10 + 3 * cg_index]
            n = path[11 + 3 * cg_index]
            c = tl.full((), path[12 + 3 * cg_index], dtype)
            index = tl.where(b == n, m, index)
            coefficient = tl.where(b == n, c, coefficient)
        cx = (
            tl.gather(
                local_x,
                index[None, :, None] + tl.zeros((B_E, b.shape[0], B_C), tl.int32),
                1,
            )
            * coefficient[None, :, None]
        )
    if WRITE_X or WRITE_R or WRITE_DI or WRITE_S:
        local_y = _rotate(dout, y)
    if WRITE_X or WRITE_DI or WRITE_DO or WRITE_S or WRITE_Y:
        if weight >= 0:
            w = tl.load(
                R + er[:, None] * R_DIM + weight + u[None, :],
                mask_e[:, None] & (u[None, :] < mul),
                0,
            )
        else:
            w = tl.full((B_E, B_C), 1, dtype)
    if WRITE_Y or WRITE_DO:
        local = cx * w[:, None, :] * amplitude[:, None, None]
        if WRITE_Y:
            value = _rotate(tl.permute(dout, (0, 2, 1)), local)
            _scatter(
                GY
                + dst[:, None, None] * Y_DIM
                + end
                + b[None, :, None] * mul
                + u[None, None, :],
                value,
                dst,
                mask_y,
            )
        if WRITE_DO:
            value = _outer(local, y)
            tl.atomic_add(
                GDO
                + eo[:, None, None] * D_DIM
                + dend
                + b[None, :, None] * dim_out
                + b[None, None, :],
                value,
                mask_do,
                sem="relaxed",
            )
    if WRITE_X or WRITE_DI:
        index = tl.full((B_I,), 0, tl.int32)
        coefficient = tl.full((B_I,), 0, dtype)
        for cg_index in tl.static_range((len(path) - 10) // 3):
            m = path[10 + 3 * cg_index]
            n = path[11 + 3 * cg_index]
            c = tl.full((), path[12 + 3 * cg_index], dtype)
            index = tl.where(a == m, n, index)
            coefficient = tl.where(a == m, c, coefficient)
        cy = (
            tl.gather(
                local_y, index[None, :, None] + tl.zeros((B_E, B_I, B_C), tl.int32), 1
            )
            * coefficient[None, :, None]
        )
        local_grad += cy * w[:, None, :] * amplitude[:, None, None]
    if WRITE_R or WRITE_S:
        q = tl.sum(cx * local_y, 1)
        if WRITE_S:
            tl.atomic_add(
                GS + es * S_DIM + harmonic, tl.sum(q * w, 1), mask_e, sem="relaxed"
            )
        if WRITE_R and weight >= 0:
            pointer = GR + er[:, None] * R_DIM + weight + u[None, :]
            value = q * amplitude[:, None]
            mask_r = mask_e[:, None] & (u[None, :] < mul)
            if R_SHARED or ACCUMULATE:
                tl.atomic_add(pointer, value, mask_r, sem="relaxed")
            else:
                tl.store(pointer, value, mask_r)
    return local_grad


@triton.jit(do_not_specialize=["E_DIM"])
def _channelwise(
    X,
    R,
    W,
    DI,
    DO,
    S,
    Y,
    GX,
    GR,
    GW,
    GDI,
    GDO,
    GS,
    GY,
    Source,
    Target,
    Order,
    E_DIM,
    X_DIM: tl.constexpr,
    Y_DIM: tl.constexpr,
    R_DIM: tl.constexpr,
    D_DIM: tl.constexpr,
    S_DIM: tl.constexpr,
    R_SHARED: tl.constexpr,
    DI_SHARED: tl.constexpr,
    DO_SHARED: tl.constexpr,
    S_SHARED: tl.constexpr,
    OUTPUTS: tl.constexpr,
    PATHS: tl.constexpr,
    WEIGHTED_ONLY: tl.constexpr,
    B_E: tl.constexpr,
    B_C: tl.constexpr,
    ACCUMULATE: tl.constexpr,
    ORDERED: tl.constexpr,
):
    WRITE_X: tl.constexpr = (OUTPUTS & 1) != 0
    WRITE_R: tl.constexpr = (OUTPUTS & 2) != 0
    WRITE_DI: tl.constexpr = (OUTPUTS & 8) != 0
    WRITE_DO: tl.constexpr = (OUTPUTS & 16) != 0
    WRITE_S: tl.constexpr = (OUTPUTS & 32) != 0
    WRITE_Y: tl.constexpr = (OUTPUTS & 64) != 0
    edge = tl.program_id(0).to(tl.int64) * B_E + tl.arange(0, B_E)
    mask_e = edge < E_DIM
    if ORDERED:
        edge = tl.load(Order + edge, mask_e, 0).to(tl.int64)
    src = tl.load(Source + edge, mask_e, 0).to(tl.int64)
    dst = tl.load(Target + edge, mask_e, 0).to(tl.int64)
    start: tl.constexpr = PATHS[0][0]
    mul: tl.constexpr = PATHS[0][2]
    dim: tl.constexpr = PATHS[0][4]
    dstart: tl.constexpr = PATHS[0][6]
    B_I: tl.constexpr = triton.next_power_of_2(dim)
    a = tl.arange(0, B_I)
    u = tl.program_id(1) * B_C + tl.arange(0, B_C)
    er = tl.full((B_E,), 0, tl.int64) if R_SHARED else edge
    ei = tl.full((B_E,), 0, tl.int64) if DI_SHARED else edge
    eo = tl.full((B_E,), 0, tl.int64) if DO_SHARED else edge
    es = tl.full((B_E,), 0, tl.int64) if S_SHARED else edge
    mask_x = mask_e[:, None, None] & (a[None, :, None] < dim) & (u[None, None, :] < mul)
    mask_di = (
        mask_e[:, None, None] & (a[None, :, None] < dim) & (a[None, None, :] < dim)
    )
    dtype = X.dtype.element_ty
    if WRITE_X or WRITE_R or WRITE_DO or WRITE_S or WRITE_Y:
        din = tl.load(
            DI
            + ei[:, None, None] * D_DIM
            + dstart
            + a[None, :, None] * dim
            + a[None, None, :],
            mask_di,
            0,
        )
    if WRITE_R or WRITE_DI or WRITE_DO or WRITE_S or WRITE_Y:
        x = tl.load(
            X
            + src[:, None, None] * X_DIM
            + start
            + a[None, :, None] * mul
            + u[None, None, :],
            mask_x,
            0,
        )
    if WRITE_R or WRITE_DO or WRITE_S or WRITE_Y:
        local_x = _rotate(din, x)
    else:
        local_x = tl.full((B_E, B_I, B_C), 0, dtype)
    local_grad = tl.full((B_E, B_I, B_C), 0, dtype)

    for path_index in tl.static_range(len(PATHS)):
        if not WEIGHTED_ONLY or PATHS[path_index][8] >= 0:
            local_grad = _channelwise_path(
                R,
                DO,
                S,
                Y,
                GR,
                GDO,
                GS,
                GY,
                local_x,
                local_grad,
                dst,
                u,
                mask_e,
                er,
                eo,
                es,
                R_DIM,
                Y_DIM,
                D_DIM,
                S_DIM,
                R_SHARED,
                OUTPUTS,
                PATHS[path_index],
                B_E,
                B_C,
                B_I,
                mul,
                ACCUMULATE,
            )

    if WRITE_X:
        value = _rotate(tl.permute(din, (0, 2, 1)), local_grad)
        _scatter(
            GX
            + src[:, None, None] * X_DIM
            + start
            + a[None, :, None] * mul
            + u[None, None, :],
            value,
            src,
            mask_x,
        )
    if WRITE_DI:
        value = _outer(local_grad, x)
        tl.atomic_add(
            GDI
            + ei[:, None, None] * D_DIM
            + dstart
            + a[None, :, None] * dim
            + a[None, None, :],
            value,
            mask_di,
            sem="relaxed",
        )


@triton.jit(do_not_specialize=["E_DIM"])
def _kernel(
    X,
    R,
    W,
    DI,
    DO,
    S,
    Y,
    GX,
    GR,
    GW,
    GDI,
    GDO,
    GS,
    GY,
    Source,
    Target,
    Paths,
    Tiles,
    TileOffsets,
    Indices,
    Coefficients,
    E_DIM,
    X_DIM: tl.constexpr,
    Y_DIM: tl.constexpr,
    W_DIM: tl.constexpr,
    R_DIM: tl.constexpr,
    D_DIM: tl.constexpr,
    S_DIM: tl.constexpr,
    R_SHARED: tl.constexpr,
    DI_SHARED: tl.constexpr,
    DO_SHARED: tl.constexpr,
    S_SHARED: tl.constexpr,
    PROJECTED: tl.constexpr,
    UVU: tl.constexpr,
    OUTPUTS: tl.constexpr,
    B_E: tl.constexpr,
    B_D: tl.constexpr,
    B_C: tl.constexpr,
    B_K: tl.constexpr,
    CG_WIDTH: tl.constexpr,
    WEIGHTED_ONLY: tl.constexpr,
    ACCUMULATE: tl.constexpr,
):
    WRITE_X: tl.constexpr = (OUTPUTS & 1) != 0
    WRITE_R: tl.constexpr = (OUTPUTS & 2) != 0
    WRITE_W: tl.constexpr = (OUTPUTS & 4) != 0
    WRITE_DI: tl.constexpr = (OUTPUTS & 8) != 0
    WRITE_DO: tl.constexpr = (OUTPUTS & 16) != 0
    WRITE_S: tl.constexpr = (OUTPUTS & 32) != 0
    WRITE_Y: tl.constexpr = (OUTPUTS & 64) != 0
    USE_DOT: tl.constexpr = UVU and B_E >= 16 and B_C >= 16
    edge = tl.program_id(0).to(tl.int64) * B_E + tl.arange(0, B_E)
    mask_e = edge < E_DIM
    group = tl.program_id(1)
    begin = tl.load(TileOffsets + group)
    stop = tl.load(TileOffsets + group + 1)
    path = tl.load(Tiles + begin * 3)
    u0 = tl.load(Tiles + begin * 3 + 1)
    src = tl.load(Source + edge, mask_e, 0).to(tl.int64)
    dst = tl.load(Target + edge, mask_e, 0).to(tl.int64)
    start = tl.load(Paths + path * 10)
    mul = tl.load(Paths + path * 10 + 2)
    dim = tl.load(Paths + path * 10 + 4)
    dstart = tl.load(Paths + path * 10 + 6)
    a = tl.arange(0, B_D)
    u = u0 + tl.arange(0, B_C)
    er = tl.full((B_E,), 0, tl.int64) if R_SHARED else edge
    ei = tl.full((B_E,), 0, tl.int64) if DI_SHARED else edge
    eo = tl.full((B_E,), 0, tl.int64) if DO_SHARED else edge
    es = tl.full((B_E,), 0, tl.int64) if S_SHARED else edge
    dtype = X.dtype.element_ty
    if WRITE_X or WRITE_R or WRITE_W or WRITE_DO or WRITE_S or WRITE_Y:
        din = tl.load(
            DI
            + ei[:, None, None] * D_DIM
            + dstart
            + a[None, :, None] * dim
            + a[None, None, :],
            mask_e[:, None, None] & (a[None, :, None] < dim) & (a[None, None, :] < dim),
            0,
        )
    if WRITE_R or WRITE_W or WRITE_DI or WRITE_DO or WRITE_S or WRITE_Y:
        x = tl.load(
            X
            + src[:, None, None] * X_DIM
            + start
            + a[None, :, None] * mul
            + u[None, None, :],
            mask_e[:, None, None] & (a[None, :, None] < dim) & (u[None, None, :] < mul),
            0,
        )
    if WRITE_R or WRITE_W or WRITE_DO or WRITE_S or WRITE_Y:
        local_x = _rotate(din, x)
    if WRITE_X or WRITE_DI:
        local_grad = tl.full((B_E, B_D, B_C), 0, dtype)

    # Consume all paths sharing this input tile before discarding its rotation.
    for tile in range(begin, stop):
        path = tl.load(Tiles + tile * 3)
        v0 = tl.load(Tiles + tile * 3 + 2)
        end = tl.load(Paths + path * 10 + 1)
        mul_out = tl.load(Paths + path * 10 + 3)
        dim_out = tl.load(Paths + path * 10 + 5)
        dend = tl.load(Paths + path * 10 + 7)
        weight = tl.load(Paths + path * 10 + 8)
        harmonic = tl.load(Paths + path * 10 + 9)
        v = v0 + tl.arange(0, B_C)
        if WRITE_X or WRITE_R or WRITE_W or WRITE_DI or WRITE_DO or WRITE_Y:
            amplitude = tl.load(S + es * S_DIM + harmonic, mask_e, 0)
        if WRITE_X or WRITE_R or WRITE_W or WRITE_DI or WRITE_S or WRITE_Y:
            dout = tl.load(
                DO
                + eo[:, None, None] * D_DIM
                + dend
                + a[None, :, None] * dim_out
                + a[None, None, :],
                mask_e[:, None, None]
                & (a[None, :, None] < dim_out)
                & (a[None, None, :] < dim_out),
                0,
            )
        if WRITE_X or WRITE_R or WRITE_W or WRITE_DI or WRITE_DO or WRITE_S:
            y = tl.load(
                Y
                + dst[:, None, None] * Y_DIM
                + end
                + a[None, :, None] * mul_out
                + v[None, None, :],
                mask_e[:, None, None]
                & (a[None, :, None] < dim_out)
                & (v[None, None, :] < mul_out),
                0,
            )

        if WRITE_R or WRITE_W or WRITE_DO or WRITE_S or WRITE_Y:
            index = tl.load(Indices + path * 2 * CG_WIDTH + a, a < CG_WIDTH, 0)
            coefficient = tl.load(
                Coefficients + path * 2 * CG_WIDTH + a, a < CG_WIDTH, 0
            ).to(dtype)
            if WEIGHTED_ONLY:
                coefficient = tl.where(weight >= 0, coefficient, 0)
            cx = (
                tl.gather(
                    local_x,
                    index[None, :, None] + tl.zeros((B_E, B_D, B_C), tl.int32),
                    1,
                )
                * coefficient[None, :, None]
            )
        if WRITE_X or WRITE_R or WRITE_W or WRITE_DI or WRITE_S:
            local_y = _rotate(dout, y)

        if UVU:
            wi = weight + u
            mask_w = u < mul
        else:
            wi = weight + u[:, None] * mul_out + v[None, :]
            mask_w = (u[:, None] < mul) & (v[None, :] < mul_out)

        if WRITE_X or WRITE_DI or WRITE_DO or WRITE_S or WRITE_Y:
            if UVU:
                w = tl.full((B_E, B_C), 1, dtype)
            else:
                w = tl.full((B_E, B_C, B_C), 1, dtype)
            if weight >= 0:
                if PROJECTED:
                    w = w * 0
                    for k0 in range(tl.cdiv(R_DIM, B_K)):
                        k = k0 * B_K + tl.arange(0, B_K)
                        radial = tl.load(
                            R + er[:, None] * R_DIM + k[None, :],
                            mask_e[:, None] & (k[None, :] < R_DIM),
                            0,
                        )
                        if UVU:
                            parameter = tl.load(
                                W + k[:, None] * W_DIM + wi[None, :],
                                (k[:, None] < R_DIM) & mask_w[None, :],
                                0,
                            )
                            if USE_DOT:
                                w += tl.dot(radial, parameter, input_precision="ieee")
                            else:
                                w += tl.sum(
                                    radial[:, :, None] * parameter[None, :, :], 1
                                )
                        else:
                            parameter = tl.load(
                                W + k[:, None, None] * W_DIM + wi[None, :, :],
                                (k[:, None, None] < R_DIM) & mask_w[None, :, :],
                                0,
                            )
                            w += tl.sum(
                                radial[:, :, None, None] * parameter[None, :, :, :], 1
                            )
                else:
                    if UVU:
                        w = tl.load(
                            R + er[:, None] * R_DIM + wi[None, :],
                            mask_e[:, None] & mask_w[None, :],
                            0,
                        )
                    else:
                        w = tl.load(
                            R + er[:, None, None] * R_DIM + wi[None, :, :],
                            mask_e[:, None, None] & mask_w[None, :, :],
                            0,
                        )

        if WRITE_DO or WRITE_Y:
            if UVU:
                local = cx * w[:, None, :] * amplitude[:, None, None]
            else:
                local = (
                    tl.sum(cx[:, :, :, None] * w[:, None, :, :], 2)
                    * amplitude[:, None, None]
                )
            if WRITE_Y:
                value = _rotate(tl.permute(dout, (0, 2, 1)), local)
                tl.atomic_add(
                    GY
                    + dst[:, None, None] * Y_DIM
                    + end
                    + a[None, :, None] * mul_out
                    + v[None, None, :],
                    value,
                    mask_e[:, None, None]
                    & (a[None, :, None] < dim_out)
                    & (v[None, None, :] < mul_out),
                    sem="relaxed",
                )
            if WRITE_DO:
                value = _outer(local, y)
                tl.atomic_add(
                    GDO
                    + eo[:, None, None] * D_DIM
                    + dend
                    + a[None, :, None] * dim_out
                    + a[None, None, :],
                    value,
                    mask_e[:, None, None]
                    & (a[None, :, None] < dim_out)
                    & (a[None, None, :] < dim_out),
                    sem="relaxed",
                )
        if WRITE_X or WRITE_DI:
            index = tl.load(
                Indices + path * 2 * CG_WIDTH + CG_WIDTH + a, a < CG_WIDTH, 0
            )
            coefficient = tl.load(
                Coefficients + path * 2 * CG_WIDTH + CG_WIDTH + a, a < CG_WIDTH, 0
            ).to(dtype)
            if WEIGHTED_ONLY:
                coefficient = tl.where(weight >= 0, coefficient, 0)
            cy = (
                tl.gather(
                    local_y,
                    index[None, :, None] + tl.zeros((B_E, B_D, B_C), tl.int32),
                    1,
                )
                * coefficient[None, :, None]
            )
            if UVU:
                local = cy * w[:, None, :] * amplitude[:, None, None]
            else:
                local = (
                    tl.sum(cy[:, :, None, :] * w[:, None, :, :], 3)
                    * amplitude[:, None, None]
                )
            local_grad += local
        if WRITE_R or WRITE_W or WRITE_S:
            if UVU:
                q = tl.sum(cx * local_y, 1)
            else:
                q = tl.sum(cx[:, :, :, None] * local_y[:, :, None, :], 1)
            if WRITE_S:
                value = tl.sum(q * w, 1) if UVU else tl.sum(tl.sum(q * w, 1), 1)
                tl.atomic_add(GS + es * S_DIM + harmonic, value, mask_e, sem="relaxed")
            if PROJECTED and (WRITE_R or WRITE_W):
                if weight >= 0:
                    for k0 in range(tl.cdiv(R_DIM, B_K)):
                        k = k0 * B_K + tl.arange(0, B_K)
                        if WRITE_R:
                            if UVU:
                                parameter = tl.load(
                                    W + k[:, None] * W_DIM + wi[None, :],
                                    (k[:, None] < R_DIM) & mask_w[None, :],
                                    0,
                                )
                                if USE_DOT:
                                    grad_radial = (
                                        tl.dot(
                                            q,
                                            tl.trans(parameter),
                                            input_precision="ieee",
                                        )
                                        * amplitude[:, None]
                                    )
                                else:
                                    grad_radial = (
                                        tl.sum(parameter[None, :, :] * q[:, None, :], 2)
                                        * amplitude[:, None]
                                    )
                            else:
                                parameter = tl.load(
                                    W + k[:, None, None] * W_DIM + wi[None, :, :],
                                    (k[:, None, None] < R_DIM) & mask_w[None, :, :],
                                    0,
                                )
                                grad_radial = (
                                    tl.sum(
                                        tl.sum(
                                            parameter[None, :, :, :] * q[:, None, :, :],
                                            3,
                                        ),
                                        2,
                                    )
                                    * amplitude[:, None]
                                )
                            tl.atomic_add(
                                GR + er[:, None] * R_DIM + k[None, :],
                                grad_radial,
                                mask_e[:, None] & (k[None, :] < R_DIM),
                                sem="relaxed",
                            )
                        if WRITE_W:
                            radial = tl.load(
                                R + er[:, None] * R_DIM + k[None, :],
                                mask_e[:, None] & (k[None, :] < R_DIM),
                                0,
                            )
                            if UVU:
                                if USE_DOT:
                                    grad_projection = tl.dot(
                                        tl.trans(radial),
                                        q * amplitude[:, None],
                                        input_precision="ieee",
                                    )
                                else:
                                    grad_projection = tl.sum(
                                        radial[:, :, None]
                                        * q[:, None, :]
                                        * amplitude[:, None, None],
                                        0,
                                    )
                                tl.atomic_add(
                                    GW + k[:, None] * W_DIM + wi[None, :],
                                    grad_projection,
                                    (k[:, None] < R_DIM) & mask_w[None, :],
                                    sem="relaxed",
                                )
                            else:
                                grad_projection = tl.sum(
                                    radial[:, :, None, None]
                                    * q[:, None, :, :]
                                    * amplitude[:, None, None, None],
                                    0,
                                )
                                tl.atomic_add(
                                    GW + k[:, None, None] * W_DIM + wi[None, :, :],
                                    grad_projection,
                                    (k[:, None, None] < R_DIM) & mask_w[None, :, :],
                                    sem="relaxed",
                                )
            elif not PROJECTED and WRITE_R:
                if weight >= 0:
                    if UVU:
                        pointer = GR + er[:, None] * R_DIM + wi[None, :]
                        grad_weight = q * amplitude[:, None]
                        mask = mask_e[:, None] & mask_w[None, :]
                    else:
                        pointer = GR + er[:, None, None] * R_DIM + wi[None, :, :]
                        grad_weight = q * amplitude[:, None, None]
                        mask = mask_e[:, None, None] & mask_w[None, :, :]
                    if R_SHARED or ACCUMULATE:
                        tl.atomic_add(pointer, grad_weight, mask, sem="relaxed")
                    else:
                        # Each edge/path/channel weight has exactly one owner.
                        tl.store(pointer, grad_weight, mask)

    if WRITE_X:
        value = _rotate(tl.permute(din, (0, 2, 1)), local_grad)
        tl.atomic_add(
            GX
            + src[:, None, None] * X_DIM
            + start
            + a[None, :, None] * mul
            + u[None, None, :],
            value,
            mask_e[:, None, None] & (a[None, :, None] < dim) & (u[None, None, :] < mul),
            sem="relaxed",
        )
    if WRITE_DI:
        value = _outer(local_grad, x)
        tl.atomic_add(
            GDI
            + ei[:, None, None] * D_DIM
            + dstart
            + a[None, :, None] * dim
            + a[None, None, :],
            value,
            mask_e[:, None, None] & (a[None, :, None] < dim) & (a[None, None, :] < dim),
            sem="relaxed",
        )


def contract_many(plan, source, target, calls):
    """Evaluate a derivative program, reusing its radial projections per chunk."""
    source, target = source.contiguous(), target.contiguous()
    projected = []
    contiguous = {}
    for outputs, operands, results, weighted_only in calls:
        if operands[0].dtype not in (torch.float32, torch.float64):
            raise TypeError(
                "CUDA O3TensorProduct convolution supports float32 and float64."
            )
        values = []
        for index, value in enumerate(operands):
            if outputs != (index,) and not value.is_contiguous():
                if id(value) not in contiguous:
                    contiguous[id(value)] = value.contiguous()
                value = contiguous[id(value)]
            values.append(value)
        values = tuple(values)
        if values[2].numel() >= PROJECTION_MIN_NUMEL and values[1].size(1) >= 8:
            projected.append((outputs, values, results, weighted_only))
        else:
            order = None
            if plan.channelwise and not values[2].numel() and source.numel() >= 256:
                if 6 in outputs or 0 in outputs:
                    order = edge_order(
                        target if 6 in outputs else source, source.numel()
                    )
            contract_tiles(
                plan,
                outputs,
                source,
                target,
                values,
                results,
                weighted_only,
                True,
                order,
            )
    if not projected:
        return

    # Group mixed adjoints by their radial factors and destinations. Their
    # scalar path cotangents can be added before either transpose GEMM.
    groups = {}
    for outputs, values, results, weighted_only in projected:
        destinations = dict(zip(outputs, results))
        key = (
            id(values[1]),
            id(values[2]),
            id(destinations.get(1)),
            id(destinations.get(2)),
        )
        groups.setdefault(key, []).append(
            (outputs, values, destinations, weighted_only)
        )
    radial = projected[0][1][1]
    need_gradient = any(1 in outputs or 2 in outputs for outputs, _, _, _ in projected)
    chunk_size = min(
        PROJECTION_CHUNK_SIZE,
        max(
            1,
            (128 << 20)
            // ((1 + need_gradient) * plan.weight_numel * radial.element_size()),
        ),
    )
    if chunk_size >= 32:
        chunk_size = chunk_size // 32 * 32
    chunk_size = min(chunk_size, source.numel())
    weights = radial.new_empty(chunk_size, plan.weight_numel)
    weight_gradient = (
        radial.new_empty(chunk_size, plan.weight_numel) if need_gradient else None
    )
    orders = {}
    if plan.channelwise and source.numel() >= 256:
        for outputs, _, _, _ in projected:
            role = 6 if 6 in outputs else 0 if 0 in outputs else None
            if role is not None and role not in orders:
                orders[role] = edge_order(target if role == 6 else source, chunk_size)
    for start in range(0, source.numel(), chunk_size):
        stop = min(start + chunk_size, source.numel())
        ready = None
        for key, terms in groups.items():
            radial, projection = terms[0][1][1:3]
            destinations = terms[0][2]
            rows = 1 if radial.size(0) == 1 else stop - start
            r = radial if radial.size(0) == 1 else radial[start:stop]
            need_weights = any(
                any(i in outputs for i in (0, 3, 4, 5, 6)) for outputs, _, _, _ in terms
            )
            need_gradient = 1 in destinations or 2 in destinations
            if need_weights and ready != key[:2]:
                torch.mm(r, projection, out=weights[:rows])
                ready = key[:2]
            if need_gradient:
                weight_gradient[:rows].zero_()
            for term_index, (
                outputs,
                values,
                term_destinations,
                weighted_only,
            ) in enumerate(terms):
                x, _, _, din, dout, amplitudes, y = values
                gradients = {
                    i: value
                    for i, value in term_destinations.items()
                    if i not in (1, 2)
                }
                if need_gradient:
                    gradients[1] = weight_gradient[:rows]
                for i in (3, 4, 5):
                    if i in term_destinations and term_destinations[i].size(0) != 1:
                        gradients[i] = term_destinations[i][start:stop]
                contract_tiles(
                    plan,
                    tuple(gradients),
                    source[start:stop],
                    target[start:stop],
                    (
                        x,
                        weights[:rows],
                        projection[:0],
                        din if din.size(0) == 1 else din[start:stop],
                        dout if dout.size(0) == 1 else dout[start:stop],
                        amplitudes
                        if amplitudes.size(0) == 1
                        else amplitudes[start:stop],
                        y,
                    ),
                    tuple(gradients.values()),
                    weighted_only,
                    term_index != 0,
                    orders[6 if 6 in outputs else 0][start:stop]
                    if (6 in outputs or 0 in outputs) and orders
                    else None,
                )
            if 1 in destinations:
                grad_radial = destinations[1]
                if radial.size(0) != 1:
                    grad_radial = grad_radial[start:stop]
                grad_radial.addmm_(weight_gradient[:rows], projection.T)
            if 2 in destinations:
                destinations[2].addmm_(r.T, weight_gradient[:rows])


def contract_tiles(
    plan,
    outputs,
    source,
    target,
    operands,
    results,
    weighted_only=False,
    accumulate=False,
    order=None,
):
    x, radial, projection, din, dout, amplitudes, y = operands
    destinations = [None] * 7
    mask = 0
    for index, result in zip(outputs, results):
        if result.numel():
            destinations[index] = result
            mask |= 1 << index
    if not projection.numel():
        mask &= ~(1 << 2)
    if not mask:
        return
    if (
        plan.channelwise
        and not projection.numel()
        and x.dtype == torch.float32
        and min(path[2] for _, path in plan.path_data) >= 16
    ):
        for paths in plan.static_groups:
            specification = tuple(
                tl.constexpr(path + tuple(c for entry in cg for c in entry))
                for path, cg in paths
            )
            channels = min(CHANNEL_TILE, triton.next_power_of_2(paths[0][0][2]))
            degree = max(max(path[4], path[5]) for path, _ in paths)
            degree = triton.next_power_of_2(degree)
            edge_width = min(
                EDGE_TILE, max(1, TILE_BUDGET // (degree * max(channels, degree)))
            )
            # Forward has no channel-reduced rotation gradients. Wider channel
            # tiles help here, but increase register pressure in the transposes.
            if mask == 64 and degree <= 8 and paths[0][0][2] >= 128:
                channels, edge_width = 128, 2
            _channelwise[
                (
                    triton.cdiv(source.numel(), edge_width),
                    triton.cdiv(paths[0][0][2], channels),
                )
            ](
                *operands,
                *destinations,
                source,
                target,
                order,
                source.numel(),
                x.size(1),
                y.size(1),
                radial.size(1),
                din.size(1),
                amplitudes.size(1),
                radial.size(0) == 1,
                din.size(0) == 1,
                dout.size(0) == 1,
                amplitudes.size(0) == 1,
                mask,
                specification,
                weighted_only,
                edge_width,
                channels,
                accumulate,
                order is not None,
                num_warps=4,
                num_stages=1,
                enable_fp_fusion=True,
            )
        return
    for index, (mode, degree, channels) in enumerate(plan.tile_groups):
        tiles = getattr(plan, f"tiles_{index}")
        offsets = getattr(plan, f"tile_offsets_{index}")
        edge_width = 16 if mode == "uvu" and x.dtype == torch.float32 else 4
        edge_width = min(edge_width, max(1, 4096 // (degree * max(degree, channels))))
        _kernel[(triton.cdiv(source.numel(), edge_width), offsets.numel() - 1)](
            *operands,
            *destinations,
            source,
            target,
            plan.paths,
            tiles,
            offsets,
            plan.indices,
            plan.coefficients,
            source.numel(),
            x.size(1),
            y.size(1),
            plan.weight_numel,
            radial.size(1),
            din.size(1),
            amplitudes.size(1),
            radial.size(0) == 1,
            din.size(0) == 1,
            dout.size(0) == 1,
            amplitudes.size(0) == 1,
            projection.numel() != 0,
            mode == "uvu",
            mask,
            edge_width,
            degree,
            channels,
            32,
            plan.degree_width,
            weighted_only,
            accumulate,
            num_warps=8 if edge_width * degree >= 256 else 4,
            num_stages=1,
            enable_fp_fusion=True,
        )

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""CUDA contractions for the explicitly selected Triton backend."""

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


@triton.jit(do_not_specialize=["EDGES"])
def rotation_kernel(
    A,
    B,
    C,
    Indices,
    Coefficients,
    EDGES,
    AS0: tl.constexpr,
    AS1: tl.constexpr,
    BS0: tl.constexpr,
    BS1: tl.constexpr,
    WIDTH: tl.constexpr,
    TERMS: tl.constexpr,
    BE: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64) * BE + tl.arange(0, BE)
    component = tl.program_id(1)
    term = tl.arange(0, TERMS)
    coefficient = tl.load(Coefficients + component * TERMS + term)
    a = tl.load(Indices + 2 * (component * TERMS + term))
    b = tl.load(Indices + 2 * (component * TERMS + term) + 1)
    valid = (edge[:, None] < EDGES) & (coefficient[None, :] != 0)
    left = tl.load(A + edge[:, None] * AS0 + a[None, :] * AS1, valid, 0)
    right = tl.load(B + edge[:, None] * BS0 + b[None, :] * BS1, valid, 0)
    value = tl.sum(left * right * coefficient[None, :], 1)
    tl.store(C + edge * WIDTH + component, value, edge < EDGES)


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
    fused = []
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
        if plan.channelwise:
            fused.append((outputs, values, results, weighted_only))
            continue
        if values[2].numel() >= PROJECTION_MIN_NUMEL and values[1].size(1) >= 8:
            projected.append((outputs, values, results, weighted_only))
        else:
            contract_tiles(
                plan,
                outputs,
                source,
                target,
                values,
                results,
                weighted_only,
                True,
            )
    if fused:
        from .fused import contract, contract_projected

        if fused[0][1][2].numel():
            contract_projected(plan, source, target, fused, PROJECTION_CHUNK_SIZE)
        else:
            contract(plan, source, target, fused)
        return
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

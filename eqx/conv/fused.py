################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Register-resident channelwise contractions and mixed adjoints."""

from collections import OrderedDict, defaultdict, deque
from dataclasses import dataclass
from weakref import WeakKeyDictionary

import torch
import triton
import triton.language as tl

CHANNEL_TILE = 32
REGISTER_TARGET = 128
WORKSPACE_BYTES = 512 << 20
_KERNELS = WeakKeyDictionary()


@dataclass(frozen=True)
class Phase:
    """Compiled contraction phase and its bindings to a derivative program."""

    operands: tuple
    results: tuple
    paths: tuple
    operations: tuple
    edge_tile: int
    channel_tile: int
    kernel: object


@triton.jit
def load_vector(
    P,
    nodes,
    channels,
    valid,
    WIDTH: tl.constexpr,
    START: tl.constexpr,
    DIM: tl.constexpr,
    MUL: tl.constexpr,
):
    result = ()
    for a in tl.static_range(DIM):
        value = tl.load(
            P + nodes[:, None] * WIDTH + START + a * MUL + channels[None, :], valid, 0
        )
        result += (value,)
    return result


@triton.jit
def rotate_vector(
    P,
    edge,
    vector,
    valid,
    WIDTH: tl.constexpr,
    START: tl.constexpr,
    DIM: tl.constexpr,
    TRANSPOSE: tl.constexpr = False,
):
    result = ()
    for a in tl.static_range(DIM):
        value = tl.full(vector[0].shape, 0, vector[0].dtype)
        for b in tl.static_range(DIM):
            offset = b * DIM + a if TRANSPOSE else a * DIM + b
            coefficient = tl.load(
                P
                + edge[:, None] * WIDTH
                + START
                + offset
                + tl.zeros(vector[0].shape, tl.int32),
                valid,
                0,
            )
            value = tl.fma(coefficient, vector[b], value)
        result += (value,)
    return result


@triton.jit
def add_vector(
    P,
    nodes,
    channels,
    valid,
    vector,
    WIDTH: tl.constexpr,
    START: tl.constexpr,
    MUL: tl.constexpr,
):
    for a in tl.static_range(len(vector)):
        tl.atomic_add(
            P + nodes[:, None] * WIDTH + START + a * MUL + channels[None, :],
            vector[a],
            valid,
            sem="relaxed",
        )


@triton.jit
def add_outer(P, edge, left, right, valid, WIDTH: tl.constexpr, START: tl.constexpr):
    dim: tl.constexpr = len(left)
    for a in tl.static_range(dim):
        for b in tl.static_range(dim):
            value = tl.sum(left[a] * right[b], 1)
            tl.atomic_add(
                P + edge * WIDTH + START + a * dim + b, value, valid, sem="relaxed"
            )


@triton.jit
def path_contraction(
    P,
    G,
    src,
    dst,
    channel,
    valid,
    mask,
    ei,
    eo,
    er,
    es,
    zero,
    zero_in,
    local_x,
    local_grad,
    XDIM: tl.constexpr,
    YDIM: tl.constexpr,
    DDIM: tl.constexpr,
    RDIM: tl.constexpr,
    SDIM: tl.constexpr,
    mul: tl.constexpr,
    dim: tl.constexpr,
    path: tl.constexpr,
    XP: tl.constexpr,
    YP: tl.constexpr,
    MSG: tl.constexpr,
    ADJ: tl.constexpr,
    Q: tl.constexpr,
    ACC: tl.constexpr,
    GY: tl.constexpr,
    GDO: tl.constexpr,
    GR: tl.constexpr,
    GS: tl.constexpr,
    BE: tl.constexpr,
    BC: tl.constexpr,
):
    out: tl.constexpr = path[1]
    dout_dim: tl.constexpr = path[5]
    dout_start: tl.constexpr = path[7]
    weight: tl.constexpr = path[8]
    harmonic: tl.constexpr = path[9]
    zero_out = ()
    for a in tl.static_range(dout_dim):
        zero_out += (zero,)
    local_y = ()
    for i in tl.static_range(len(YP)):
        y = load_vector(P[YP[i][0]], dst, channel, valid, YDIM, out, dout_dim, mul)
        local_y += (
            rotate_vector(P[YP[i][1]], eo, y, valid, DDIM, dout_start, dout_dim),
        )
    cx = ()
    for i in tl.static_range(len(XP)):
        value = zero_out
        for c in tl.static_range((len(path) - 10) // 3):
            coefficient = tl.full((), path[12 + 3 * c], zero.dtype)
            updated = ()
            for k in tl.static_range(dout_dim):
                updated += (
                    local_x[i][path[10 + 3 * c]] * coefficient
                    if k == path[11 + 3 * c]
                    else value[k],
                )
            value = updated
        cx += (value,)
    messages = ()
    for i in tl.static_range(len(MSG)):
        if weight >= 0 or not MSG[i][3]:
            w = (
                tl.load(
                    P[MSG[i][1]] + er[:, None] * RDIM + weight + channel[None, :],
                    valid,
                    0,
                )
                if weight >= 0
                else tl.full((BE, BC), 1, zero.dtype)
            )
            s = tl.load(
                P[MSG[i][2]]
                + es[:, None] * SDIM
                + harmonic
                + tl.zeros((BE, BC), tl.int32),
                valid,
                0,
            )
            value = ()
            for k in tl.static_range(dout_dim):
                value += (cx[MSG[i][0]][k] * w * s,)
            messages += (value,)
        else:
            messages += (zero_out,)
    for i in tl.static_range(len(GY)):
        local = zero_out
        for j in tl.static_range(len(GY[i]) - 2):
            updated = ()
            for k in tl.static_range(len(zero_out)):
                updated += (local[k] + messages[GY[i][2 + j]][k],)
            local = updated
        value = rotate_vector(
            P[GY[i][0]], eo, local, valid, DDIM, dout_start, dout_dim, True
        )
        add_vector(G[GY[i][1]], dst, channel, valid, value, YDIM, out, mul)
    for i in tl.static_range(len(GDO)):
        local = zero_out
        for j in tl.static_range(len(GDO[i]) - 2):
            updated = ()
            for k in tl.static_range(len(zero_out)):
                updated += (local[k] + messages[GDO[i][2 + j]][k],)
            local = updated
        y = load_vector(P[GDO[i][0]], dst, channel, valid, YDIM, out, dout_dim, mul)
        add_outer(G[GDO[i][1]], eo, local, y, mask, DDIM, dout_start)
    for i in tl.static_range(len(ADJ)):
        if weight >= 0 or not ADJ[i][3]:
            w = (
                tl.load(
                    P[ADJ[i][1]] + er[:, None] * RDIM + weight + channel[None, :],
                    valid,
                    0,
                )
                if weight >= 0
                else tl.full((BE, BC), 1, zero.dtype)
            )
            s = tl.load(
                P[ADJ[i][2]]
                + es[:, None] * SDIM
                + harmonic
                + tl.zeros((BE, BC), tl.int32),
                valid,
                0,
            )
            value = zero_in
            for c in tl.static_range((len(path) - 10) // 3):
                coefficient = tl.full((), path[12 + 3 * c], zero.dtype)
                updated = ()
                for k in tl.static_range(dim):
                    updated += (
                        value[k]
                        + local_y[ADJ[i][0]][path[11 + 3 * c]] * coefficient * w * s
                        if k == path[10 + 3 * c]
                        else value[k],
                    )
                value = updated
            # Only destination accumulators persist between paths; individual
            # mixed adjoints are consumed immediately.
            for j in tl.static_range(len(ACC)):
                for term in tl.static_range(len(ACC[j])):
                    if ACC[j][term] == i:
                        updated = ()
                        for k in tl.static_range(len(ACC)):
                            if k == j:
                                combined = ()
                                for a in tl.static_range(dim):
                                    combined += (local_grad[k][a] + value[a],)
                                updated += (combined,)
                            else:
                                updated += (local_grad[k],)
                        local_grad = updated
    q = ()
    for i in tl.static_range(len(Q)):
        value = zero
        for a in tl.static_range(dout_dim):
            value = tl.fma(cx[Q[i][0]][a], local_y[Q[i][1]][a], value)
        q += (value,)
    if weight >= 0:
        for i in tl.static_range(len(GR)):
            value = zero
            for j in tl.static_range((len(GR[i]) - 1) // 2):
                s = tl.load(
                    P[GR[i][2 + 2 * j]]
                    + es[:, None] * SDIM
                    + harmonic
                    + tl.zeros((BE, BC), tl.int32),
                    valid,
                    0,
                )
                value += q[GR[i][1 + 2 * j]] * s
            tl.atomic_add(
                G[GR[i][0]] + er[:, None] * RDIM + weight + channel[None, :],
                value,
                valid,
                sem="relaxed",
            )
    for i in tl.static_range(len(GS)):
        value = zero
        for j in tl.static_range((len(GS[i]) - 1) // 3):
            if weight >= 0 or not GS[i][3 + 3 * j]:
                w = (
                    tl.load(
                        P[GS[i][2 + 3 * j]]
                        + er[:, None] * RDIM
                        + weight
                        + channel[None, :],
                        valid,
                        0,
                    )
                    if weight >= 0
                    else tl.full((BE, BC), 1, zero.dtype)
                )
                value += q[GS[i][1 + 3 * j]] * w
        tl.atomic_add(
            G[GS[i][0]] + es * SDIM + harmonic, tl.sum(value, 1), mask, sem="relaxed"
        )
    return local_grad


@triton.jit(do_not_specialize=["EDGES"])
def kernel(
    P,
    G,
    Source,
    Target,
    EDGES,
    XDIM: tl.constexpr,
    YDIM: tl.constexpr,
    DDIM: tl.constexpr,
    RDIM: tl.constexpr,
    SDIM: tl.constexpr,
    RSHARED: tl.constexpr,
    DISHARED: tl.constexpr,
    DOSHARED: tl.constexpr,
    SSHARED: tl.constexpr,
    PATHS: tl.constexpr,
    XP: tl.constexpr,
    YP: tl.constexpr,
    MSG: tl.constexpr,
    ADJ: tl.constexpr,
    Q: tl.constexpr,
    ACC: tl.constexpr,
    GX: tl.constexpr,
    GDI: tl.constexpr,
    GY: tl.constexpr,
    GDO: tl.constexpr,
    GR: tl.constexpr,
    GS: tl.constexpr,
    BE: tl.constexpr,
    BC: tl.constexpr,
):
    edge = tl.program_id(0).to(tl.int64) * BE + tl.arange(0, BE)
    mask = edge < EDGES
    src = tl.load(Source + edge, mask, 0).to(tl.int64)
    dst = tl.load(Target + edge, mask, 0).to(tl.int64)
    start: tl.constexpr = PATHS[0][0]
    mul: tl.constexpr = PATHS[0][2]
    dim: tl.constexpr = PATHS[0][4]
    dstart: tl.constexpr = PATHS[0][6]
    channel = tl.program_id(1) * BC + tl.arange(0, BC)
    valid = mask[:, None] & (channel[None, :] < mul)
    ei = tl.full((BE,), 0, tl.int64) if DISHARED else edge
    eo = tl.full((BE,), 0, tl.int64) if DOSHARED else edge
    er = tl.full((BE,), 0, tl.int64) if RSHARED else edge
    es = tl.full((BE,), 0, tl.int64) if SSHARED else edge
    zero = tl.full((BE, BC), 0, P[0].dtype.element_ty)
    zero_in = ()
    for a in tl.static_range(dim):
        zero_in += (zero,)
    local_x = ()
    for i in tl.static_range(len(XP)):
        x = load_vector(P[XP[i][0]], src, channel, valid, XDIM, start, dim, mul)
        local_x += (rotate_vector(P[XP[i][1]], ei, x, valid, DDIM, dstart, dim),)
    local_grad = ()
    for i in tl.static_range(len(ACC)):
        local_grad += (zero_in,)

    for path_index in tl.static_range(len(PATHS)):
        local_grad = path_contraction(
            P,
            G,
            src,
            dst,
            channel,
            valid,
            mask,
            ei,
            eo,
            er,
            es,
            zero,
            zero_in,
            local_x,
            local_grad,
            XDIM,
            YDIM,
            DDIM,
            RDIM,
            SDIM,
            mul,
            dim,
            PATHS[path_index],
            XP,
            YP,
            MSG,
            ADJ,
            Q,
            ACC,
            GY,
            GDO,
            GR,
            GS,
            BE,
            BC,
        )
    for i in tl.static_range(len(GX)):
        local = local_grad[GX[i][2]]
        value = rotate_vector(P[GX[i][0]], ei, local, valid, DDIM, dstart, dim, True)
        add_vector(G[GX[i][1]], src, channel, valid, value, XDIM, start, mul)
    for i in tl.static_range(len(GDI)):
        local = local_grad[GDI[i][2]]
        x = load_vector(P[GDI[i][0]], src, channel, valid, XDIM, start, dim, mul)
        add_outer(G[GDI[i][1]], ei, local, x, mask, DDIM, dstart)


def schedule(calls):
    """Intern shared factors and collect adjoints before each rotation."""
    pointers, results = [], []
    pointer_ids, result_ids = {}, {}
    xp, yp, messages, adjoints, q = {}, {}, {}, {}, {}
    gx, gdi, gy, gdo, gr, gs = (defaultdict(list) for _ in range(6))
    for outputs, operands, destinations, weighted in calls:
        ids = []
        for value in operands:
            if id(value) not in pointer_ids:
                pointer_ids[id(value)] = len(pointers)
                pointers.append(value)
            ids.append(pointer_ids[id(value)])
        output_ids = {}
        for role, value in zip(outputs, destinations):
            if id(value) not in result_ids:
                result_ids[id(value)] = len(results)
                results.append(value)
            output_ids[role] = result_ids[id(value)]
        x, w, _, di, do, s, y = ids
        if any(i in outputs for i in (1, 4, 5, 6)):
            xi = xp.setdefault((x, di), len(xp))
        if any(i in outputs for i in (0, 1, 3, 5)):
            yi = yp.setdefault((y, do), len(yp))
        if 6 in outputs or 4 in outputs:
            mi = messages.setdefault((xi, w, s, weighted), len(messages))
            if 6 in outputs:
                gy[do, output_ids[6]].append(mi)
            if 4 in outputs:
                gdo[y, output_ids[4]].append(mi)
        if 0 in outputs or 3 in outputs:
            ai = adjoints.setdefault((yi, w, s, weighted), len(adjoints))
            if 0 in outputs:
                gx[di, output_ids[0]].append(ai)
            if 3 in outputs:
                gdi[x, output_ids[3]].append(ai)
        if 1 in outputs or 5 in outputs:
            qi = q.setdefault((xi, yi), len(q))
            if 1 in outputs:
                gr[output_ids[1]].append((qi, s))
            if 5 in outputs:
                gs[output_ids[5]].append((qi, w, weighted))
    accumulators = {}
    input_gradients = []
    for group in (gx, gdi):
        entries = []
        for key, value in group.items():
            terms = tuple(sorted(value))
            accumulator = accumulators.setdefault(terms, len(accumulators))
            entries.append((*key, accumulator))
        input_gradients.append(tuple(entries))
    return (
        tuple(pointers),
        tuple(results),
        tuple(xp),
        tuple(yp),
        tuple(messages),
        tuple(adjoints),
        tuple(q),
        tuple(accumulators),
        *input_gradients,
        *(
            tuple((*key, *value) for key, value in group.items())
            for group in (gy, gdo)
        ),
        *(
            tuple(
                (key, *(entry for term in value for entry in term))
                for key, value in group.items()
            )
            for group in (gr, gs)
        ),
    )


def register_estimate(paths, operations, itemsize):
    """Estimate live 32-bit words per channel, including rotation temporaries."""
    xp, yp, messages, _, q, accumulators, *_ = operations
    dim = paths[0][0][4]
    dim_out = max(path[5] for path, _ in paths)
    live = dim * (len(xp) + len(accumulators))
    live += dim_out * (len(yp) + len(xp) + len(messages))
    live += 2 * max(dim, dim_out) + len(q) + 8
    return 24 + live * (itemsize // 4)


def split_program(calls):
    """Partition output instructions without changing their destinations."""
    terms = [
        ((role,), values, (destination,), weighted)
        for outputs, values, destinations, weighted in calls
        for role, destination in zip(outputs, destinations)
    ]
    middle = len(terms) // 2
    return terms[:middle], terms[middle:]


def compile_schedule(paths, calls, pointers, results, source, target, shared):
    """Compile resource-bounded phases without executing candidate kernels.

    All angular degrees use the same instruction generator. Register estimates
    bound large derivative programs before compilation; compiler resource
    reports refine the subdivision and edge/channel tile sizes. Cached phases
    contain operand indices, never the tensors of a particular graph.
    """
    x, radial, _, din, _, amplitudes, y = calls[0][1]
    pointer_indices = {id(value): i for i, value in enumerate(pointers)}
    result_indices = {id(value): i for i, value in enumerate(results)}
    specification = tuple(
        tl.constexpr(path + tuple(c for entry in cg for c in entry))
        for path, cg in paths
    )
    channels = min(CHANNEL_TILE, triton.next_power_of_2(paths[0][0][2]))
    edges = min(32, 128 // channels)
    warps = max(1, edges * channels // 32)
    configurations = tuple(
        dict.fromkeys(
            (
                (edges, channels, warps),
                (max(1, edges // 2), channels, warps),
                (max(1, edges // 2), channels, max(1, warps // 2)),
                (edges, max(1, channels // 2), warps),
            )
        )
    )
    pending = deque([calls])
    phases = []
    with torch.cuda.device(x.device):
        properties = triton.runtime.driver.active.utils.get_device_properties(
            x.device.index
        )
        register_limit = min(
            REGISTER_TARGET, properties["max_num_regs"] // (2 * 32 * warps)
        )
        while pending:
            program = pending.popleft()
            inputs, destinations, *operations = schedule(program)
            can_split = sum(len(outputs) for outputs, _, _, _ in program) > 1
            if (
                register_estimate(paths, operations, x.element_size())
                > 2 * register_limit
                and can_split
            ):
                pending.extendleft(reversed(split_program(program)))
                continue
            operations = tuple(
                tuple(tl.constexpr(value) for value in group) for group in operations
            )
            arguments = (
                inputs,
                destinations,
                source,
                target,
                source.numel(),
                x.size(1),
                y.size(1),
                din.size(1),
                radial.size(1),
                amplitudes.size(1),
                *shared,
                specification,
                *operations,
            )
            best = None
            resource_error = None
            for edge_tile, channel_tile, num_warps in configurations:
                try:
                    compiled = kernel.warmup(
                        *arguments,
                        edge_tile,
                        channel_tile,
                        grid=(1, 1, 1),
                        num_warps=num_warps,
                        num_stages=1,
                    )
                    if compiled.metadata.shared > properties["max_shared_mem"]:
                        raise triton.OutOfResources(
                            compiled.metadata.shared,
                            properties["max_shared_mem"],
                            "shared memory",
                        )
                    # Load resource metadata without launching the contraction.
                    compiled._init_handles()
                    max_threads = getattr(compiled, "n_max_threads", 1024)
                    if num_warps * 32 > max_threads:
                        raise triton.OutOfResources(
                            num_warps * 32, max_threads, "threads"
                        )
                except triton.OutOfResources as error:
                    resource_error = error
                    continue
                score = (
                    compiled.n_spills,
                    max(0, compiled.n_regs - register_limit),
                    max(
                        0,
                        compiled.metadata.shared - properties["max_shared_mem"] // 2,
                    ),
                )
                if best is None or score < best[0]:
                    best = score, compiled, edge_tile, channel_tile
                if score == (0, 0, 0):
                    break
            if can_split and (best is None or best[0] != (0, 0, 0)):
                pending.extendleft(reversed(split_program(program)))
                continue
            if best is None:
                raise resource_error
            _, compiled, edge_tile, channel_tile = best
            phases.append(
                Phase(
                    tuple(pointer_indices[id(value)] for value in inputs),
                    tuple(result_indices[id(value)] for value in destinations),
                    specification,
                    operations,
                    edge_tile,
                    channel_tile,
                    compiled,
                )
            )
    return tuple(phases)


def contract(plan, source, target, calls, shared=None):
    pointers, results, *operations = schedule(calls)
    signature = tuple(operations)
    x, r, _, di, do, s, y = calls[0][1]
    if shared is None:
        shared = tuple(value.size(0) == 1 for value in (r, di, do, s))
    layouts = (
        x.device,
        tuple(
            (value.dtype, value.data_ptr() % 16)
            for value in (*pointers, *results, source, target)
        ),
        *shared,
        x.size(1),
        y.size(1),
        di.size(1),
        r.size(1),
        s.size(1),
        source.numel() > 2**31 - 1,
    )
    cache = _KERNELS.setdefault(plan, OrderedDict())
    for path_index, paths in enumerate(plan.path_groups):
        key = (signature, layouts, path_index)
        phases = cache.get(key)
        if phases is None:
            phases = compile_schedule(
                paths, calls, pointers, results, source, target, shared
            )
            cache[key] = phases
            if len(cache) > 256:
                cache.popitem(last=False)
        else:
            cache.move_to_end(key)
        for phase in phases:
            grid = (
                triton.cdiv(source.numel(), phase.edge_tile),
                triton.cdiv(paths[0][0][2], phase.channel_tile),
                1,
            )
            phase.kernel[grid](
                tuple(pointers[i] for i in phase.operands),
                tuple(results[i] for i in phase.results),
                source,
                target,
                source.numel(),
                x.size(1),
                y.size(1),
                di.size(1),
                r.size(1),
                s.size(1),
                *shared,
                phase.paths,
                *phase.operations,
                phase.edge_tile,
                phase.channel_tile,
            )


def contract_projected(plan, source, target, calls, chunk_size=16384):
    """Fuse a mixed-adjoint program with bounded radial workspaces."""
    # Preserve the original broadcasting layout even for a one-edge tail.
    shared = tuple(calls[0][1][i].size(0) == 1 for i in (1, 3, 4, 5))
    groups = {}
    factors = {}
    for outputs, values, results, weighted in calls:
        destinations = dict(zip(outputs, results))
        key = (
            id(values[1]),
            id(values[2]),
            id(destinations.get(1)),
            id(destinations.get(2)),
        )
        groups.setdefault(key, []).append((outputs, values, destinations, weighted))
        if any(i in outputs for i in (0, 3, 4, 5, 6)):
            factors[key[:2]] = values[1:3]
    radial = calls[0][1][1]
    count = max(1, len(factors)) + sum(
        1 in terms[0][2] or 2 in terms[0][2] for terms in groups.values()
    )
    chunk = max(
        1,
        WORKSPACE_BYTES // (count * plan.weight_numel * radial.element_size()),
    )
    if chunk >= 32:
        chunk = chunk // 32 * 32
    chunk = min(chunk_size, chunk, source.numel())
    workspaces = {key: radial.new_empty(chunk, plan.weight_numel) for key in factors}
    gradients = {
        key: radial.new_empty(chunk, plan.weight_numel)
        for key, terms in groups.items()
        if 1 in terms[0][2] or 2 in terms[0][2]
    }
    unused = (
        radial.new_empty(chunk, plan.weight_numel)
        if not workspaces
        else next(iter(workspaces.values()))
    )
    empty = radial.new_empty(0, plan.weight_numel)
    for start in range(0, source.numel(), chunk):
        stop = min(start + chunk, source.numel())
        views = {}

        def edge_view(value):
            if id(value) not in views:
                views[id(value)] = value if value.size(0) == 1 else value[start:stop]
            return views[id(value)]

        weights = {}
        for key, (r, projection) in factors.items():
            r = edge_view(r)
            weights[key] = workspaces[key][: r.size(0)]
            torch.mm(r, projection, out=weights[key])
        direct = []
        weight_gradients = {}
        for key, terms in groups.items():
            r, projection = terms[0][1][1:3]
            rows = 1 if r.size(0) == 1 else stop - start
            if key in gradients:
                weight_gradients[key] = gradients[key][:rows].zero_()
            w = weights.get(key[:2], unused[:rows])
            for outputs, values, destinations, weighted in terms:
                x, _, _, din, dout, amplitudes, y = values
                result = {
                    i: edge_view(value) if i in (3, 4, 5) else value
                    for i, value in destinations.items()
                    if i not in (1, 2)
                }
                if key in weight_gradients:
                    result[1] = weight_gradients[key]
                direct.append(
                    (
                        tuple(result),
                        (
                            x,
                            w,
                            empty,
                            edge_view(din),
                            edge_view(dout),
                            edge_view(amplitudes),
                            y,
                        ),
                        tuple(result.values()),
                        weighted,
                    )
                )
        contract(plan, source[start:stop], target[start:stop], direct, shared)
        for key, gradient in weight_gradients.items():
            r, projection = groups[key][0][1][1:3]
            destinations = groups[key][0][2]
            if 1 in destinations:
                edge_view(destinations[1]).addmm_(gradient, projection.T)
            if 2 in destinations:
                destinations[2].addmm_(edge_view(r).T, gradient)

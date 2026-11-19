"""Cartesian frame and channel paths for streamed planar convolutions."""

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ...utils.program import Program
from .. import config
from ..precision import validate_module_precision
from .convolution import radial_input
from .radial import (
    materialize,
    radial_expression,
    radial_features,
    radial_parts,
    radial_reference,
    split_radial,
)
from .weights import IndexedProjection, radial_policy


def frame_rows(program, rows, rotation, rank, count):
    """Rotate a fixed set of Cartesian rows once for all multiplicities."""
    width = 3**rank
    for _ in range(rank):
        rows = program.matmul(rows, rotation, count * (width // 3), 3, 3)
        if rank > 1:
            rows = program.gather(
                rows,
                (
                    i * width + 3 * j + a
                    for i in range(count)
                    for a in range(3)
                    for j in range(width // 3)
                ),
            )
    return rows


def orient(program, value, mul, dim, inverse):
    """Apply the reflected-sector quarter turn to the first planar index."""
    half = dim // 2
    value = program.gather(
        value, (c * dim + (a + half) % dim for c in range(mul) for a in range(dim))
    )
    signs = program.concatenate(
        [
            program.constant(half, 1 if inverse else -1),
            program.constant(half, -1 if inverse else 1),
        ]
    )
    return program.binary("mul", value, program.gather(signs, tuple(range(dim)) * mul))


def repeated_rows(p, value, codes, copies, inverse=False):
    """Expand equal or opposite rows, or apply the transpose of that expansion."""
    if not codes:
        return value
    width = max(map(abs, codes))
    indices = tuple(c * width + abs(k) - 1 for c in range(copies) for k in codes)
    signs = p.concatenate([p.constant(1, 1), p.constant(1, -1)])
    signs = p.gather(signs, tuple(int(k < 0) for k in codes) * copies)
    if inverse:
        return p.scatter(p.binary("mul", value, signs), indices, copies * width)
    return p.binary("mul", p.gather(value, indices), signs)


def analysis(p, frame, source, maps):
    local = [[] for _ in frame.irreps_out]
    for section, mul, rank, odd, paths in frame.entries:
        width, cols = 3**rank, 2 ** len(paths) - 1
        rows, size, codes = maps[rank]
        value = p.matmul(
            p.slice(source, section.start, mul * width),
            p.transpose(rows, size, width),
            mul,
            width,
            size,
        )
        value = repeated_rows(p, value, codes, mul)
        for m, (index, _) in enumerate(paths):
            dim = 2**m
            part = p.gather(
                value, (c * cols + dim - 1 + a for c in range(mul) for a in range(dim))
            )
            if odd and m and frame.basis_change:
                part = orient(p, part, mul, dim, False)
            local[index].append(part)
    return [p.concatenate(parts) for parts in local]


def uu_paths(p, linear, local, weights):
    outputs = [[] for _ in linear.irreps_out]
    offset = 0
    for ins in linear.instructions:
        ni, no, channels = ins.path_shape
        dim = linear.irreps_in[ins.i_in].ir.dim
        for i in range(ni):
            x = p.slice(local[ins.i_in], i * channels * dim, channels * dim)
            w = p.slice(weights, offset + i * no * channels, no * channels)
            w = p.gather(w, (c for c in range(no * channels) for _ in range(dim)))
            outputs[ins.i_out].append(
                p.scale(
                    p.binary("mul", p.gather(x, tuple(range(channels * dim)) * no), w),
                    ins.path_weight,
                )
            )
        offset += ni * no * channels
    return [
        p.sum(parts, mul * ir.dim)
        for (mul, ir), parts in zip(linear.irreps_out, outputs)
    ]


def build_plan(module):
    frame_in, frame_out = module.frame_in, module.frame_out
    p = Program()
    source = p.input(0, "source", frame_in.input_dim)
    weights = p.input(1, "edge", module.weight_numel)
    rotation = p.input(2, "edge", 9)
    cutoff = p.input(3, "edge", 1)
    tables = []
    plan = CartesianEdgeProgram(
        module.backend, frame_in.input_dim, module.weight_numel, module.radial_mode
    )

    def matrices(frame, inverse):
        maps = {}
        for _, _, rank, _, paths in frame.entries:
            if rank in maps:
                continue
            rows = (
                getattr(frame, f"synthesis_{rank}").T
                if inverse
                else frame.analysis_matrix(rank, len(paths) - 1).to(
                    getattr(frame, f"basis_{rank}_0")
                )
            )
            rows = rows.contiguous()
            name = f"{'synthesis' if inverse else 'analysis'}_{rank}"
            plan.register_buffer(name, rows, persistent=False)
            table = p.input(4 + len(tables), "shared", rows.numel())
            tables.append((name, rank, len(paths) - 1, inverse))
            count, width = rows.shape
            codes = frame.frame_codes.get(name, ())
            # The eager Torch gather/scatter only pays off for denser repetitions.
            reuse = 2 if module.backend == "cuda" else 3
            if codes and reuse * max(map(abs, codes)) <= count:
                # Choose one representative per class from the original table.
                first = [
                    next(i for i, k in enumerate(codes) if abs(k) == j)
                    for j in range(1, 1 + max(map(abs, codes)))
                ]
                table = p.gather(
                    table, (i * width + j for i in first for j in range(width))
                )
                codes = tuple(k if codes[first[abs(k) - 1]] > 0 else -k for k in codes)
                count = len(first)
            else:
                codes = ()
            maps[rank] = frame_rows(p, table, rotation, rank, count), count, codes
        return maps

    maps = matrices(frame_in, False)
    global_maps = matrices(frame_out, True)
    plan.tables = tuple(tables)
    local = analysis(p, frame_in, source, maps)
    outputs = module.local_program(p, local, maps, weights, 4 + len(tables))
    result = []
    for _, mul, rank, odd, paths in frame_out.entries:
        parts = []
        for m, (i, section) in enumerate(paths):
            dim = 2**m
            x = p.slice(outputs[i], section.start * dim, mul * dim)
            if odd and m and frame_out.basis_change:
                x = orient(p, x, mul, dim, True)
            parts.append(p.transpose(x, mul, dim))
        cols = 2 ** len(paths) - 1
        value = p.transpose(p.concatenate(parts), cols, mul)
        rows, size, codes = global_maps[rank]
        result.append(
            p.matmul(
                repeated_rows(p, value, codes, mul, inverse=True),
                rows,
                mul,
                size,
                3**rank,
            )
        )
    result = p.concatenate(result)
    result = p.binary("mul", result, p.gather(cutoff, (0,) * p.size(result)))
    plan.metadata = repr((tuple(p.nodes), ((result, 0, "target"),)))
    return plan


class CartesianEdgeProgram(torch.nn.Module):
    """Static Cartesian edge schedule with externally supplied learnable operands."""

    def __init__(self, backend, input_dim, weight_numel, radial_mode):
        super().__init__()
        self.backend, self.input_dim, self.weight_numel = (
            backend,
            input_dim,
            weight_numel,
        )
        if radial_mode not in ("last", "full"):
            raise ValueError("radial_mode must be 'last' or 'full'.")
        self.radial_mode = radial_mode

    def _apply(self, fn, recurse=True):
        from ...co2.local_frame import LocalFrame

        previous = {name: getattr(self, name).dtype for name, *_ in self.tables}
        super()._apply(fn, recurse)
        for name, rank, order, inverse in self.tables:
            table = getattr(self, name)
            if torch.finfo(table.dtype).eps < torch.finfo(previous[name]).eps:
                rows = (
                    LocalFrame.synthesis_matrix(rank, order).T
                    if inverse
                    else LocalFrame.analysis_matrix(rank, order)
                )
                self._buffers[name] = rows.contiguous().to(table)
        return self

    def forward(
        self,
        features,
        weights,
        edge_index,
        rotation,
        cutoff,
        operands=(),
        *,
        radial_network=None,
        projection=None,
        edge_pairs=None,
    ):
        if edge_pairs is not None:
            from .edge_pairs import reciprocal_weights

            weights, projection, radial_network = reciprocal_weights(
                self, edge_pairs, weights, projection, radial_network, edge_index
            )
            if projection is not None and not projection.shape[0]:
                projection = None
        validate_module_precision(
            self, (features, weights, rotation, cutoff, projection, operands)
        )
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError("Node features must match the input frame.")
        if (
            edge_index.ndim != 2
            or edge_index.shape[0] != 2
            or edge_index.dtype != torch.int64
        ):
            raise ValueError("Expected integer edge indices of shape (2, edges).")
        n, e = features.shape[0], edge_index.shape[1]
        if (
            edge_index.device != features.device
            or rotation.shape != (e, 3, 3)
            or cutoff.shape != (e, 1)
        ):
            raise ValueError("Expected one rotation and one scalar cutoff per edge.")
        parts = radial_parts(weights)
        for v, kind in parts:
            if (kind == "edge" and v.shape[0] != e) or (
                kind in ("source", "target") and v.shape[0] != n
            ):
                raise ValueError("Radial input rows do not match edge or node count.")
        if projection is not None and (
            projection.ndim != 2 or projection.shape[1] != self.weight_numel
        ):
            raise ValueError("Projection output must match the local weights.")
        if (
            projection is None
            and radial_network is None
            and sum(v.shape[1] for v, _ in parts) != self.weight_numel
        ):
            raise ValueError("Wrong number of convolution weights.")

        from .edge import evaluate
        from .torch.edge import evaluate_recomputed, evaluate_tiled

        if (
            radial_policy(self, radial_network) == "last"
            and not isinstance(radial_network, IndexedProjection)
            and (radial_network is not None or projection is not None)
        ):
            hidden, tail = (
                (radial_network, ())
                if projection is not None
                else split_radial(radial_network)
            )
            if projection is not None:
                width = projection.shape[0]
            elif tail:
                last = tail[0]
                width = getattr(
                    last,
                    "in_features",
                    getattr(last, "in_dim", getattr(last, "h_in", 0)),
                )
            else:
                width = 0
            # Wide GEMMs have several consumers in a force-loss adjoint.
            # Reuse one bounded projection instead of replaying it per tile.
            if (
                self.backend == "cuda"
                and features.is_cuda
                and torch.is_grad_enabled()
                and config.RADIAL_PROJECT_MIN_WIDTH > 0
                and statically_known_true(width >= config.RADIAL_PROJECT_MIN_WIDTH)
                and statically_known_true(
                    e * (width + self.weight_numel) * features.element_size()
                    <= config.RADIAL_PROJECT_MAX_BYTES
                )
            ):
                weights = materialize(weights, edge_index)
                if radial_network is not None:
                    weights = radial_reference(radial_network, weights)
                if projection is not None:
                    bias = (
                        projection[-1]
                        if weights.shape[-1] + 1 == projection.shape[0]
                        else None
                    )
                    matrix = projection[:-1] if bias is not None else projection
                    weights = torch.nn.functional.linear(weights, matrix.T, bias)
                radial_network = projection = None
            else:
                if hidden:
                    weights = radial_features(
                        hidden, weights, edge_index, reference=False
                    )
                radial_network = tail or None

        inputs = [features, weights, rotation.to(features).reshape(-1, 9), cutoff]
        inputs.extend(
            getattr(self, name).reshape(1, -1).to(features) for name, *_ in self.tables
        )
        inputs.extend(x.reshape(1, -1) for x in operands)
        metadata = self.metadata
        if (
            radial_network is not None
            or projection is not None
            or isinstance(weights, tuple)
            or weights.shape[0] == 1
        ):
            p = Program()
            inputs[1] = features.new_empty((1, 0))
            value = radial_input(p, inputs, weights, radial_network)
            if projection is not None:
                rows, cols = projection.shape
                if rows == p.size(value) + 1:
                    value = p.concatenate([value, p.constant(1, 1)])
                if rows != p.size(value):
                    raise ValueError(
                        "Projection width does not match the radial inputs."
                    )
                w = p.input(len(inputs), "shared", rows * cols)
                inputs.append(projection.reshape(1, -1))
                value = p.matmul(value, w, 1, rows, cols)
            lower = (
                radial_expression.__wrapped__
                if torch.compiler.is_compiling()
                else radial_expression
            )
            metadata = lower(metadata, tuple(p.nodes), 1, value)
        if self.backend == "cuda" and features.is_cuda:
            return evaluate(metadata, inputs, *edge_index, n, inline=False)[0]
        run = evaluate_recomputed if config.TORCH_RECOMPUTE else evaluate_tiled
        return run(metadata, inputs, *edge_index, n)[0]

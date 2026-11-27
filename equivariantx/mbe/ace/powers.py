"""Shared-input angular products with independent channel coefficient maps."""

from collections import Counter, defaultdict
from fractions import Fraction
from itertools import accumulate

import torch

from ...ops import config
from ...utils.autograd import forward_ad_active
from .generalized import CouplingBlock, angular_offsets, channel_permutation


class ProductExpressions:
    """Identify equal scalar products up to a constant without sampling features."""

    def __init__(self, width):
        self.base = [((i,), Fraction(1)) for i in range(width)]
        self.sums = {}

    def expression(self, terms):
        terms = sorted((key, val) for key, val in terms.items() if val)
        if not terms:
            return (), Fraction(0)
        factors, scale = terms[0]
        if len(terms) == 1:
            return factors, scale
        common = Counter(factors)
        for key, _ in terms[1:]:
            common &= Counter(key)
            if not common:
                break
        key = tuple(
            (
                tuple(sorted((Counter(k) - common).elements())) if common else k,
                v / scale,
            )
            for k, v in terms
        )
        symbol = self.sums.setdefault(key, len(self.base) + len(self.sums))
        return tuple(sorted((*common.elements(), symbol))), scale

    def reduce(self, columns, previous, locations, first):
        expressions, aliases, representatives, output = [], [], {}, []
        for column in columns:
            terms, rows = defaultdict(Fraction), defaultdict(Fraction)
            for (i, j), value in column.items():
                factors, scale = previous[i]
                if not value or not scale:
                    continue
                w = Fraction(value)
                terms[tuple(sorted((*factors, j)))] += w * scale
                index, ratio = locations[i]
                pair = tuple(sorted((index, j))) if first else (index, j)
                rows[pair] += w * ratio
            factors, scale = self.expression(terms)
            expressions.append((factors, scale))
            if not scale:
                aliases.append((0, Fraction(0)))
                continue
            if factors not in representatives:
                representatives[factors] = len(output), scale
                output.append({key: float(v) for key, v in rows.items() if v})
            index, original = representatives[factors]
            aliases.append((index, scale / original))
        return output or [{}], expressions, aliases


class CoefficientGroup(torch.nn.Module):
    """Apply independent path weights to the shared angular coordinates."""

    def __init__(self, pieces, channels, dimension, outputs, backend):
        super().__init__()
        self.channels, self.dimension, self.outputs = channels, dimension, outputs
        self.slices = tuple((stage, start, stop) for stage, start, stop, _ in pieces)
        coordinates = [pair for *_, indices in pieces for pair in indices]
        paths = [
            coordinates[i : i + dimension]
            for i in range(0, len(coordinates), dimension)
        ]
        representatives, unique, columns = {}, [], []
        for p, path in enumerate(paths):
            lead = next((w for _, w in path if w), None)
            if lead is None:
                continue
            signature = tuple((i, w / lead) if w else (-1, 0) for i, w in path)
            if signature not in representatives:
                representatives[signature] = len(unique), lead
                unique.append(path)
                columns.append({})
            k, first = representatives[signature]
            columns[k][p,] = float(lead / first)
        self.fold_weights = None
        if unique and len(unique) * 5 <= len(paths) * 4:
            self.fold_weights = CouplingBlock(columns, (len(paths),), 0, backend, False)
            paths = unique
        self.rows = tuple((i, float(w)) for column in zip(*paths) for i, w in column)
        self.width = len(self.rows)
        self.paths = len(coordinates) // dimension

    def forward(self, features, tables, node_type, counts=()):
        n, z = node_type.shape[0], tables[0][0].shape[0]
        x = features.reshape(
            n, self.dimension, self.width * self.channels // self.dimension
        )
        matrices = [tables[s][i] for s, i in self.blocks]
        w = matrices[0] if len(matrices) == 1 else torch.cat(matrices, 1)
        if self.fold_weights is not None:
            w = self.fold_weights(
                (w.reshape(z, self.paths, self.channels * self.outputs),)
            )
        w = w.reshape(z, -1, self.outputs)
        if counts:
            result = torch.cat(
                [block @ matrix for block, matrix in zip(x.split(counts), w.unbind(0))]
            )
        elif z == 1:
            result = x @ w[0]
        else:
            result = (
                (x.reshape(-1, w.shape[1]) @ w)
                .reshape(z, n, self.dimension, self.outputs)
                .permute(1, 2, 0, 3)
            )
            result = result.gather(
                2, node_type.view(n, 1, 1, 1).expand(n, self.dimension, 1, self.outputs)
            ).squeeze(2)
        return result.transpose(1, 2).flatten(1)


class SharedPowers(torch.nn.Module):
    """Evaluate repeated binary couplings and their original coefficient maps."""

    def __init__(self, module, first, channels):
        super().__init__()
        self.channels = channels
        self.register_buffer(
            "permutation", channel_permutation(first), persistent=False
        )
        width = sum(ir.dim for _, ir in first)
        planner = ProductExpressions(width)
        expr = planner.base
        locations = [(i, Fraction(1)) for i in range(width)]
        aliases = [locations]
        sizes = [width]
        self.layers = torch.nn.ModuleList()
        for stage, tp in enumerate(module.tensor_products):
            a, b, c = (
                angular_offsets(irs)
                for irs in (tp.irreps_in1, tp.irreps_in2, tp.irreps_out)
            )
            columns = [defaultdict(float) for _ in range(c[-1])]
            for k, ins in enumerate(tp.instructions):
                for i, j, o, w in module.coupling_entries(tp, k):
                    columns[c[ins.i_out] + o][a[ins.i_in1] + i, b[ins.i_in2] + j] += w
            if hasattr(tp, "projection"):
                from ...ops.torch.mbe.ace.ictd_symmetric_contraction import (
                    project_columns,
                )

                projection = tp.projection
                if (
                    module.backend == "cuda"
                    and stage == len(module.tensor_products) - 1
                ):
                    from ...co3.basis import Projector

                    projection = Projector(tp.irreps_out, backend="torch")
                project_columns(projection, columns, c)
            columns, expr, locations = planner.reduce(
                columns, expr, locations, stage == 0
            )
            self.layers.append(
                CouplingBlock(columns, (sizes[-1], width), 0, module.backend, False)
            )
            aliases.append(locations)
            sizes.append(len(columns))
        self.widths = tuple(sizes)
        offsets = tuple(accumulate(sizes, initial=0))
        groups = defaultdict(list)
        for stage, coef in enumerate(module.coefficient_maps):
            for sx, sw, sy, ci, co, dim, scale in coef.paths:
                lo = sx // channels
                values = aliases[stage][lo : lo + (ci // channels) * dim]
                indices = [(offsets[stage] + i, r * Fraction(scale)) for i, r in values]
                groups[sy, dim, co].append((stage, sw, sw + ci * co, indices))
        self.groups = torch.nn.ModuleList(
            CoefficientGroup(parts, channels, d, co, module.backend)
            for (_, d, co), parts in sorted(groups.items())
        )
        boundaries = [{0, n} for n in module.weight_numels]
        for group in self.groups:
            for stage, lo, hi in group.slices:
                boundaries[stage].update((lo, hi))
        points = [sorted(b) for b in boundaries]
        self.table_splits = tuple(
            tuple(b - a for a, b in zip(row, row[1:])) for row in points
        )
        for group in self.groups:
            group.blocks = tuple(
                (s, k)
                for s, lo, hi in group.slices
                for k in range(points[s].index(lo), points[s].index(hi))
            )
        self.group_widths = tuple(g.width for g in self.groups)
        all_rows = [row for group in self.groups for row in group.rows]
        index = torch.tensor([j for j, _ in all_rows], dtype=torch.int32)
        self.row_weights = torch.tensor([w for _, w in all_rows], dtype=torch.float64)
        reverse = torch.argsort(index, stable=True)
        counts = torch.bincount(index.long(), minlength=sum(sizes))
        self.register_buffer("row_index", index, persistent=False)
        self.register_buffer(
            "row_scale",
            self.row_weights.to(torch.get_default_dtype()),
            persistent=False,
        )
        self.register_buffer(
            "row_ptr",
            torch.nn.functional.pad(counts.cumsum(0), (1, 0)).int(),
            persistent=False,
        )
        self.register_buffer("row_reverse", reverse.int(), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.row_scale = self.row_weights.to(self.row_scale)
        return self

    def forward(self, features, weights, node_type):
        counts, inverse = (), None
        if (
            features.device.type == "cpu"
            and weights[0].shape[0] > 1
            and features.shape[0]
            and not (forward_ad_active() or torch.compiler.is_compiling())
        ):
            sorted_types, order = node_type.sort()
            elements, lengths = sorted_types.unique_consecutive(return_counts=True)
            counts = tuple(lengths.tolist())
            inverse = order.argsort()
            features = features.index_select(0, order)
            weights = [w.index_select(0, elements) for w in weights]
        x = features.index_select(1, self.permutation).reshape(
            features.shape[0], self.widths[0], self.channels
        )
        powers = [x]
        for layer in self.layers:
            powers.append(layer((powers[-1], x)))
        if (
            self.layers[0].backend == "cuda"
            and features.is_cuda
            and not forward_ad_active()
        ):
            from ...ops.cuda.mbe.ace.generalized import PackRows

            packed = PackRows.apply(
                self.widths,
                self.group_widths,
                (self.row_index, self.row_scale, self.row_ptr, self.row_reverse),
                *powers,
            )
            blocks = packed.split(
                tuple(features.shape[0] * w for w in self.group_widths), dim=0
            )
        else:
            packed = torch.cat(powers, 1).index_select(1, self.row_index)
            blocks = (packed * self.row_scale.view(1, -1, 1)).split(
                self.group_widths, dim=1
            )
        tables = [w.split(parts, 1) for w, parts in zip(weights, self.table_splits)]
        parts = [
            group(block, tables, node_type, counts)
            for group, block in zip(self.groups, blocks)
        ]
        result = parts[0] if len(parts) == 1 else torch.cat(parts, 1)
        return result if inverse is None else result.index_select(0, inverse)

    def fits(self, features, elements):
        from torch.fx.experimental.symbolic_shapes import statically_known_true

        width = sum(self.widths) + sum(self.group_widths)
        scratch = features.shape[0] * self.channels * width * features.element_size()
        return (
            self.layers[0].w_2.dtype == features.dtype
            and self.permutation.device == features.device
            and statically_known_true(elements > 0)
            and (
                features.device.type == "cpu"
                and not forward_ad_active()
                or statically_known_true(elements <= 4)
            )
            and statically_known_true(features.shape[0] > 0)
            and statically_known_true(scratch <= config.ACE_SHARED_POWER_WORKSPACE)
        )


def shared_powers(module):
    """Build a bounded shared-input schedule for uniform diagonal channels."""
    tps = module.tensor_products
    first = tps[0].irreps_in1
    if not first or len(tps) not in (1, 2, 3) or first[0].mul < 32:
        return None
    if module.backend == "cuda" and len(tps) < 2:
        return None
    channels = first[0].mul
    if sum(ir.dim for _, ir in first) > 48:
        return None
    prev = first
    for tp in tps:
        if (
            tp.irreps_in1 != prev
            or tp.irreps_in2 != first
            or any(m != channels for irreps in (prev, tp.irreps_out) for m, _ in irreps)
            or sum(ir.dim for _, ir in tp.irreps_out) > 2048
            or not getattr(tp, "project", True)
            or (hasattr(tp, "projection") and any(ir.l > 3 for _, ir in tp.irreps_out))
        ):
            return None
        prev = tp.irreps_out
    if any(c is None or not c.paths for c in module.coefficient_maps):
        return None
    if any(
        sx % channels or ci % channels
        for c in module.coefficient_maps
        for sx, _, _, ci, _, _, _ in c.paths
    ):
        return None
    if any(
        getattr(getattr(linear, "projection", None), "groups", ())
        for linear in module.linears
    ):
        return None
    outputs = sorted(
        {
            (sy, d * co)
            for c in module.coefficient_maps
            for _, _, sy, _, co, d, _ in c.paths
        }
    )
    if (
        tuple(start for start, _ in outputs)
        != tuple(accumulate((n for _, n in outputs), initial=0))[:-1]
        or sum(n for _, n in outputs) != module.contraction_dim
    ):
        return None
    return SharedPowers(module, first, channels)

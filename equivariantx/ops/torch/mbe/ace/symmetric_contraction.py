"""Symmetric polynomial contractions and an independent monomial reference."""

from collections import defaultdict
from itertools import accumulate

import torch
from torch._subclasses.fake_tensor import is_fake
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from .....utils.autograd import forward_ad_active, torch_gradient_mask
from .....utils.metadata import parse_metadata
from .... import config, get_kernel
from ....common.ace import ace_inputs
from ....common.symmetric import symmetric_inputs, symmetric_result
from ....common.torch.grouped import channel_groups, indexed_channel_product
from ....common.torch.linear import grouped_linear_available
from ....common.torch.node_polynomial import AngularPolynomial
from ....common.torch.symmetric import evaluate_prefix, prefix_plan
from .ictd_symmetric_contraction import project


def angular_columns(module, tp, previous, same_input):
    """Multiply component polynomials by the original input without coupling channels."""
    starts = [
        tuple(accumulate((ir.dim for _, ir in irreps), initial=0))
        for irreps in (tp.irreps_in1, tp.irreps_in2, tp.irreps_out)
    ]
    columns = [defaultdict(float) for _ in range(starts[2][-1])]
    for i, path in enumerate(tp.instructions):
        x, y, z = (s[k] for s, k in zip(starts, (path.i_in1, path.i_in2, path.i_out)))
        for a, b, c, scale in module.coupling_entries(tp, i):
            if same_input:
                columns[z + c][tuple(sorted((x + a, y + b)))] += scale
            else:
                for term, value in previous[x + a].items():
                    columns[z + c][tuple(sorted((*term, y + b)))] += value * scale
    projector = getattr(tp, "projection", None)
    if projector is not None:
        from .ictd_symmetric_contraction import project_columns

        project_columns(projector, columns, starts[2])
    return columns


class CoefficientPolynomials(torch.nn.Module):
    """Mix each independent path into one output irrep with its original coefficients."""

    def __init__(self, orders, offsets):
        super().__init__()
        self.maps = torch.nn.ModuleList(
            AngularPolynomial(columns, stage + 1) for stage, columns, _, _, _ in orders
        )
        self.blocks = tuple(
            (
                stage,
                tuple(
                    i
                    for start, stop in slices
                    for i in range(offsets[stage][start], offsets[stage][stop])
                ),
            )
            for stage, _, slices, _, _ in orders
        )
        self.dim, self.outputs = orders[0][-2:]

    def forward(self, features, weights, index, elements, grouping):
        nodes, channels = features.shape[:2]
        inputs, matrices = [], []
        for polynomial, (stage, indices) in zip(self.maps, self.blocks):
            values = polynomial(features, config.ACE_POLYNOMIAL_WORKSPACE)
            paths = values.shape[-1] // self.dim
            width = paths * channels
            values = (
                values.reshape(nodes, channels, paths, self.dim)
                .permute(0, 3, 2, 1)
                .reshape(nodes, self.dim, width)
            )
            blocks = [weights[stage][i] for i in indices]
            if grouping is not None:
                inputs.append(values)
                matrices.extend(
                    b.reshape(elements, b.shape[1] // self.outputs, self.outputs)
                    for b in blocks
                )
                continue
            matrix = (
                blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
            ).reshape(elements, width, self.outputs)
            inputs.append(values)
            matrices.append(matrix)
        values = inputs[0] if len(inputs) == 1 else torch.cat(inputs, dim=-1)
        if grouping is not None:
            return (
                indexed_channel_product(values, matrices, grouping)
                .transpose(1, 2)
                .flatten(1)
            )
        matrix = matrices[0] if len(matrices) == 1 else torch.cat(matrices, dim=1)
        matrix = matrix.transpose(0, 1).reshape(
            values.shape[-1], elements * self.outputs
        )
        result = (values @ matrix).reshape(nodes, self.dim, elements, self.outputs)
        result = result.gather(
            2, index.expand(nodes, self.dim, 1, self.outputs)
        ).squeeze(2)
        return result.transpose(1, 2).flatten(1)


class ChannelPolynomials(torch.nn.Module):
    """Evaluate node polynomials grouped by output irrep in Cartesian or spherical storage."""

    def __init__(self, irreps, outputs, weight_numels):
        super().__init__()
        self.channels = irreps[0].mul
        self.dimensions = tuple(ir.dim for _, ir in irreps)
        self.sizes = tuple(self.channels * dim for dim in self.dimensions)
        # One split per original table gives its transpose a single gradient
        # assembly rather than a full zero-padded gradient for every path slice.
        boundaries = [{0, n} for n in weight_numels]
        for orders in outputs:
            for stage, _, slices, _, _ in orders:
                for start, stop in slices:
                    boundaries[stage].update((start, stop))
        boundaries = [sorted(b) for b in boundaries]
        self.weight_splits = tuple(
            tuple(b - a for a, b in zip(row, row[1:])) for row in boundaries
        )
        offsets = [{v: i for i, v in enumerate(row)} for row in boundaries]
        self.groups = torch.nn.ModuleList(
            CoefficientPolynomials(orders, offsets) for orders in outputs
        )

    def forward(self, features, weights, node_type):
        nodes = features.shape[0]
        blocks = [
            x.reshape(nodes, self.channels, d)
            for x, d in zip(features.split(self.sizes, -1), self.dimensions)
        ]
        value = blocks[0] if len(blocks) == 1 else torch.cat(blocks, dim=-1)
        index = node_type.reshape(nodes, 1, 1, 1)
        elements = weights[0].shape[0]
        grouping = channel_groups(node_type, elements) if elements > 4 else None
        tables = [
            w.split(sizes, 1) if len(sizes) > 1 else (w,)
            for w, sizes in zip(weights, self.weight_splits)
        ]
        parts = [
            group(value, tables, index, elements, grouping) for group in self.groups
        ]
        return parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)


def build_channel_polynomial(module):
    """Prepare low-order products for a bounded shared angular basis."""
    if (
        not config.TORCH_ACE_CHANNEL_POLYNOMIAL
        or not 1 <= len(module.tensor_products) <= 2
    ):
        return None
    first = module.tensor_products[0].irreps_in1
    if not first or first[0].mul < 32 or sum(ir.dim for _, ir in first) > 16:
        return None
    channels = first[0].mul
    previous = first
    for tp in module.tensor_products:
        if (
            tp.irreps_in1 != previous
            or tp.irreps_in2 != first
            or any(
                mul != channels
                for irreps in (previous, tp.irreps_out)
                for mul, _ in irreps
            )
            or sum(ir.dim for _, ir in tp.irreps_out) > 256
            or not getattr(tp, "project", True)
        ):
            return None
        previous = tp.irreps_out
    if any(m is None or not m.paths for m in module.coefficient_maps) or any(
        getattr(getattr(linear, "projection", None), "groups", ())
        for linear in module.linears
    ):
        return None
    if any(
        sx % channels or ci % channels or not ci or not co
        for coefficient in module.coefficient_maps
        for sx, _, _, ci, co, _, _ in coefficient.paths
    ):
        return None
    if (
        sum(
            ci // channels * dim
            for coefficient in module.coefficient_maps
            for _, _, _, ci, _, dim, _ in coefficient.paths
        )
        > 512
    ):
        return None
    orders = defaultdict(list)
    columns = [{(i,): 1.0} for i in range(sum(ir.dim for _, ir in first))]
    for stage, coefficient in enumerate(module.coefficient_maps):
        if stage:
            columns = angular_columns(
                module, module.tensor_products[stage - 1], columns, stage == 1
            )
        groups = defaultdict(list)
        for sx, sw, sy, ci, co, dim, scale in coefficient.paths:
            groups[sy, dim, co].append((sx, sw, ci, scale))
        for (sy, dim, co), paths in groups.items():
            angular, slices = [], []
            for sx, sw, ci, scale in paths:
                start = sx // channels
                angular.extend(
                    {key: value * scale for key, value in col.items()}
                    for col in columns[start : start + ci // channels * dim]
                )
                stop = sw + ci * co
                if slices and slices[-1][1] == sw:
                    slices[-1] = (slices[-1][0], stop)
                else:
                    slices.append((sw, stop))
            orders[sy].append((stage, angular, tuple(slices), dim, co))
    offset = 0
    for start in sorted(orders):
        if start != offset:
            return None
        dim, co = orders[start][0][-2:]
        offset += dim * co
    if not offset or offset != module.contraction_dim:
        return None
    return ChannelPolynomials(
        first, [orders[key] for key in sorted(orders)], module.weight_numels
    )


def register_monomials(module, rows):
    """Group equal angular monomials without identifying their path weights."""
    terms = sorted(
        {indices for indices, _, _, _ in rows},
        key=lambda indices: (len(indices), indices),
    )
    if len(terms) * module.output_dim > 2048:
        return
    module.monomial_count = len(terms)
    positions = {indices: i for i, indices in enumerate(terms)}
    orders = sorted({len(indices) for indices in terms})
    module.monomial_orders = tuple(orders)
    for order in orders:
        indices = [term for term in terms if len(term) == order]
        module.register_buffer(
            f"monomial_indices_{order}",
            torch.tensor(indices, dtype=torch.long).T.contiguous(),
            persistent=False,
        )
    for name, values, dtype in (
        ("monomial_paths", [p for _, p, _, _ in rows], torch.long),
        (
            "monomial_outputs",
            [positions[s] * module.output_dim + a for s, _, a, _ in rows],
            torch.long,
        ),
        ("monomial_factors", [c for _, _, _, c in rows], torch.float64),
    ):
        module.register_buffer(
            name, torch.tensor(values, dtype=dtype), persistent=False
        )
    columns = module.num_paths * module.output_dim
    if (
        columns <= len(rows)
        and len(terms) * columns <= config.SYMMETRIC_PROJECTION_MAX_ENTRIES
    ):
        projection = torch.zeros(len(terms), columns, dtype=torch.float64)
        row = torch.tensor([positions[s] for s, _, _, _ in rows])
        column = torch.tensor([a * module.num_paths + p for _, p, a, _ in rows])
        projection[row, column] = module.monomial_factors
        module.register_buffer("monomial_projection", projection, persistent=False)
        if module.output_dim == 1 and len(orders) > 1:
            padded = [s + (module.input_dim,) * (orders[-1] - len(s)) for s in terms]
            module.register_buffer(
                "scalar_monomial_indices",
                torch.tensor(padded).T.contiguous(),
                persistent=False,
            )


def monomial_basis(module, features):
    """Evaluate each distinct angular monomial once."""
    monomials = []
    for order in module.monomial_orders:
        indices = getattr(module, f"monomial_indices_{order}")
        product = features.index_select(2, indices[0])
        for axis in range(1, order):
            product = product * features.index_select(2, indices[axis])
        monomials.append(product)
    return torch.cat(monomials, dim=2) if len(monomials) > 1 else monomials[0]


def projected_polynomial(module, features, coefficients):
    """Project shared monomials and apply independent per-sample coefficients."""
    n, channels = features.shape[:2]
    selected = coefficients.transpose(1, 2)
    if hasattr(module, "scalar_monomial_indices"):
        x = torch.nn.functional.pad(features, (0, 1), value=1)
        indices = module.scalar_monomial_indices
        basis = x.index_select(2, indices[0])
        for k in range(1, module.correlation):
            basis = basis * x.index_select(2, indices[k])
        return ((basis @ module.monomial_projection.to(features)) * selected).sum(2)
    basis = monomial_basis(module, features)
    paths = (basis @ module.monomial_projection.to(features)).reshape(
        n, channels, module.output_dim, module.num_paths
    )
    result = (paths @ selected.unsqueeze(-1)).squeeze(-1).transpose(1, 2)
    return symmetric_result(result, module.layout, module.output_dims)


class TiledPolynomial(torch.autograd.Function):
    """Replay disjoint node tiles without retaining their angular products."""

    @staticmethod
    def forward(ctx, module, width, features, weight, node_type):
        ctx.module, ctx.width = module, width
        ctx.save_for_backward(features, weight, node_type)
        result = features.new_empty(
            (features.size(0), module.output_dim * module.num_channels)
        )
        for start in range(0, features.size(0), width):
            rows = slice(start, start + width)
            result[rows] = projected_polynomial(
                module, features[rows], weight.index_select(0, node_type[rows])
            )
        return result

    @staticmethod
    def backward(ctx, gradient):
        features, weight, node_type = ctx.saved_tensors
        dx, dw, _ = torch_gradient_mask(ctx, ctx.needs_input_grad[2:])
        if not (dx or dw):
            return None, None, None, None, None
        higher = torch.is_grad_enabled()
        feature_grad = [] if dx else None
        weight_grad = torch.zeros_like(weight) if dw else None
        for start in range(0, features.size(0), ctx.width):
            rows = slice(start, start + ctx.width)
            types = node_type[rows]
            with torch.enable_grad():
                x, w = features[rows], weight.index_select(0, types)
                y = projected_polynomial(ctx.module, x, w)
                active = (x, w) if dx and dw else (x,) if dx else (w,)
                deriv = torch.autograd.grad(
                    y, active, gradient[rows], create_graph=higher
                )
            if dx:
                feature_grad.append(deriv[0])
            if dw:
                weight_grad = weight_grad.index_add_(0, types, deriv[-1])
            del y, w, x, deriv
        return None, None, torch.cat(feature_grad) if dx else None, weight_grad, None


def indexed_polynomial(module, features, weight, node_type):
    """Use a bounded node schedule for large eager evaluations."""
    types = node_type.long()
    if not (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or is_fake(features)
        or forward_ad_active()
    ):
        entries = 3 * module.monomial_count + (module.output_dim + 1) * module.num_paths
        width = max(
            1,
            config.SYMMETRIC_MONOMIAL_WORKSPACE
            // (features.element_size() * module.num_channels * entries),
        )
        if features.size(0) > width:
            return TiledPolynomial.apply(module, width, features, weight, types)
    return projected_polynomial(module, features, weight.index_select(0, types))


def channel_polynomial(module, features, weight, node_type):
    """Combine coefficients per element, then contract each channel by GEMM."""
    basis = monomial_basis(module, features)
    n, channels, count = basis.shape
    elements, outputs = weight.shape[0], module.output_dim
    values = (
        weight.index_select(1, module.monomial_paths)
        * module.monomial_factors.to(weight)[None, :, None]
    )
    matrices = weight.new_zeros((elements, count * outputs, channels)).index_add(
        1, module.monomial_outputs, values
    )
    matrices = matrices.reshape(elements, count, outputs, channels).permute(3, 1, 0, 2)
    result = torch.bmm(
        basis.transpose(0, 1), matrices.reshape(channels, count, elements * outputs)
    )
    result = result.reshape(channels, n, elements, outputs).permute(1, 2, 3, 0)
    indices = node_type.long()[:, None, None, None].expand(n, 1, outputs, channels)
    return symmetric_result(
        result.gather(1, indices).squeeze(1), module.layout, module.output_dims
    )


def reference(module, features, weight, node_type):
    """Multiply each declared monomial and scatter its coefficient separately."""
    value = features.transpose(1, 2)
    output = value.new_zeros((value.shape[0], module.output_dim, module.num_channels))
    selected = weight[node_type.long()]
    for degree in range(1, module.correlation + 1):
        indices = getattr(module, f"indices_{degree}")
        if not indices.shape[0]:
            continue
        product = value[:, indices[:, 0], :]
        for k in range(1, degree):
            product = product * value[:, indices[:, k], :]
        product = product * selected[:, getattr(module, f"weights_{degree}"), :]
        product = (
            product
            * getattr(module, f"coefficients_{degree}").to(value.dtype)[None, :, None]
        )
        output = output.index_add(1, getattr(module, f"outputs_{degree}"), product)
    if not module.has_terms:
        output = output + (value.sum() + weight.sum()) * 0
    return symmetric_result(output, module.layout, module.output_dims)


def prefix_enabled(module, features, program):
    """Select the measured CPU schedules using static model limits in graphs."""
    if (
        not config.TORCH_SYMMETRIC_PREFIX
        or config.TORCH_RECOMPUTE
        or features.device.type != "cpu"
        or not 2 <= module.correlation <= 4
    ):
        return False
    terms = len(parse_metadata(module.metadata)[-1])
    if terms < 8:
        return False
    if (
        torch.compiler.is_compiling()
        or get_proxy_mode() is not None
        or is_fake(features)
    ):
        if module.num_channels < 64:
            return False
        if terms <= config.SYMMETRIC_PREFIX_MAX_TERMS:
            return True
        budget = config.SYMMETRIC_PREFIX_MAX_NODES
        return (
            terms <= budget and len(prefix_plan(module.metadata, program)[0]) <= budget
        )
    return (
        terms <= config.SYMMETRIC_PREFIX_MAX_TERMS
        and module.correlation >= 3
        and features.shape[0] * module.num_channels * terms
        >= config.SYMMETRIC_PREFIX_MIN_WORK
    )


def uuu(module, features, weight, node_type):
    """Evaluate the scheduled polynomial without materialized tensor powers."""
    if config.TORCH_REFERENCE:
        return reference(module, features, weight, node_type)
    if (
        config.TORCH_SYMMETRIC_MONOMIAL_GEMM
        and not config.TORCH_RECOMPUTE
        and module.monomial_count
        and features.is_cuda
    ):
        if weight.shape[0] >= 16 and hasattr(module, "monomial_projection"):
            return indexed_polynomial(module, features, weight, node_type)
        if weight.shape[0] * module.output_dim <= 128:
            return channel_polynomial(module, features, weight, node_type)
    program, operands = symmetric_inputs(
        features, weight, module.correlation, module.output_dim
    )
    kernel = (
        evaluate_prefix
        if prefix_enabled(module, features, program)
        else get_kernel("symmetric_contract", "torch")
    )
    output = kernel(module.metadata, program, node_type, operands)[0]
    return symmetric_result(output, module.layout, module.output_dims)


def linear(module, features, weight, node_type, stage):
    """Share element matrices across nodes instead of gathering dense weights."""
    coefficient = module.coefficient_maps[stage]
    if coefficient is not None and (
        hasattr(module.linears[stage], "output_basis")
        or module.linears[stage].shared_weights
        or (
            not config.TORCH_REFERENCE
            and (
                features.is_cuda
                or grouped_linear_available(features, weight, coefficient.paths)
            )
        )
    ):
        output = coefficient(features, weight, node_type)
        projection = getattr(module.linears[stage], "projection", None)
        return output if projection is None else project(projection, output)
    return module.linears[stage](features, weight[node_type])


def uuu_fully_connected(module, features, weights, node_type):
    """Accumulate each body order while reusing the preceding product."""
    polynomial = module.channel_polynomial
    if (
        config.TORCH_ACE_CHANNEL_POLYNOMIAL
        and not config.TORCH_REFERENCE
        and not config.TORCH_RECOMPUTE
        and features.is_cuda
        and polynomial is not None
        and weights[0].shape[0] > 0
        and all(w.shape[0] == weights[0].shape[0] for w in weights)
    ):
        return polynomial(features, weights, node_type)
    correlation = features
    output = linear(module, features, weights[0], node_type, 0)
    for stage, weight in enumerate(weights[1:]):
        if getattr(module.tensor_products[stage], "backend", "torch") != "torch":
            final = stage == len(module.product_dims) - 1
            program, operands = ace_inputs(
                correlation,
                features,
                weight if final else None,
                module.contraction_dim if final else module.product_dims[stage],
            )
            value = get_kernel("ace_contract", "torch")(
                module.metadata[stage], program, node_type, operands
            )[0]
            if final:
                output = output + module.project_output(value)
                continue
            correlation = module.project_product(value, stage)
        else:
            correlation = module.tensor_products[stage](correlation, features)
        output = output + linear(module, correlation, weight, node_type, stage + 1)
    return output

"""Shared channel projections and exact angular transposes for Racah products."""

import math

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from .... import config
from ....common.sparse_coefficients import AngularMap


def channel_projection(x, weight):
    """Use multiplication for singleton channels and GEMM for shared weights."""
    if x.shape[-2] == 1:
        return weight * x
    if all(size == 1 for size in weight.shape[:-2]):
        weight = weight.reshape(weight.shape[-2:])
        return torch.matmul(x.transpose(-1, -2), weight.T).transpose(-1, -2)
    return torch.matmul(weight, x)


def gather_nodes(values, source):
    """Gather all node sections together, retaining one node-gradient reduction."""
    if not values or source is None:
        return values
    dimensions = [value.shape[-2:] for value in values]
    packed = torch.cat([value.flatten(-2) for value in values], dim=-1)[source]
    return [
        value.unflatten(-1, shape)
        for value, shape in zip(
            packed.split([math.prod(shape) for shape in dimensions], dim=-1), dimensions
        )
    ]


class Schedule(torch.nn.Module):
    """Project edge channels once and reuse them across recoupled paths.

    The first weight acts before angular coupling. The second weight acts
    afterward; no expanded product of the two path-weight vectors is stored.
    """

    def __init__(self, module):
        super().__init__()
        from .....o3.wigner6j.tensor_product import recoupling_coefficient

        self.angular = torch.nn.ModuleDict()
        self.input_sections = tuple(
            tuple(
                (s.start, mul, ir.dim) for (mul, ir), s in zip(irreps, irreps.slices())
            )
            for irreps in (module.irreps_in1, module.irreps_in2, module.irreps_in3)
        )
        self.output_sections = tuple((mul, ir.dim) for mul, ir in module.irreps_out)
        offsets = []
        for product in (module.tp1, module.tp2):
            row, offset = [], 0
            for instruction in product.instructions:
                row.append(offset if instruction.has_weight else -1)
                if instruction.has_weight:
                    offset += math.prod(instruction.path_shape)
            offsets.append(row)

        def angular(degrees):
            key = "_".join(map(str, degrees))
            if key not in self.angular:
                self.angular[key] = AngularMap(degrees)
            return key

        nodes, projections, routes = {}, {}, {}
        for j, second in enumerate(module.tp2.instructions):
            mul3, ir3 = module.irreps_in3[second.i_in2]
            ir_out = module.irreps_out[second.i_out].ir
            for i, first in enumerate(module.tp1.instructions):
                if first.i_out != second.i_in1:
                    continue
                mul1, ir1 = module.irreps_in1[first.i_in1]
                mul2, ir2 = module.irreps_in2[first.i_in2]
                ir12 = module.irreps_mid[first.i_out].ir
                if not (
                    mul1 and mul2 and mul3 and first.path_weight and second.path_weight
                ):
                    continue
                projection = (first.i_in2, mul1, offsets[0][i])
                projection = projections.setdefault(projection, len(projections))
                for ir13 in ir1 * ir3:
                    if ir_out not in ir13 * ir2:
                        continue
                    coefficient = (
                        recoupling_coefficient(
                            ir1.l, ir2.l, ir3.l, ir12.l, ir13.l, ir_out.l
                        )
                        * first.path_weight
                        * second.path_weight
                    )
                    if not coefficient:
                        continue
                    for channel in range(mul3):
                        node = (
                            first.i_in1,
                            second.i_in2,
                            channel,
                            angular((ir1.l, ir3.l, ir13.l)),
                        )
                        node = nodes.setdefault(node, len(nodes))
                        key = (
                            node,
                            projection,
                            second.i_out,
                            offsets[1][j],
                            mul1,
                            mul3,
                            channel,
                            angular((ir13.l, ir2.l, ir_out.l)),
                        )
                        routes[key] = routes.get(key, 0.0) + coefficient
        first_paths, second_paths = [], []
        for i, first in enumerate(module.tp1.instructions):
            mul1, ir1 = module.irreps_in1[first.i_in1]
            mul2, ir2 = module.irreps_in2[first.i_in2]
            ir12 = module.irreps_mid[first.i_out].ir
            if not (mul1 and mul2 and first.path_weight):
                continue
            projection = first.i_in2, mul1, offsets[0][i]
            projection = projections.setdefault(projection, len(projections))
            first_paths.append(
                (
                    first.i_in1,
                    projection,
                    first.i_out,
                    angular((ir1.l, ir2.l, ir12.l)),
                    first.path_weight,
                )
            )
        for j, second in enumerate(module.tp2.instructions):
            channels, ir12 = module.irreps_mid[second.i_in1]
            mul3, ir3 = module.irreps_in3[second.i_in2]
            ir_out = module.irreps_out[second.i_out].ir
            if not (channels and mul3 and second.path_weight):
                continue
            second_paths.append(
                (
                    second.i_in1,
                    second.i_in2,
                    second.i_out,
                    channels,
                    offsets[1][j],
                    angular((ir12.l, ir3.l, ir_out.l)),
                    second.path_weight,
                )
            )
        self.first_paths, self.second_paths = tuple(first_paths), tuple(second_paths)
        self.nodes = tuple(nodes)
        self.projections = tuple(projections)
        self.routes = tuple(
            (*key, coefficient) for key, coefficient in routes.items() if coefficient
        )
        scalar = self.projections[0][0] if self.projections else None
        self.scalar_edge = (
            scalar
            if scalar is not None
            and all(
                index == scalar and self.input_sections[1][index][1:] == (1, 1)
                for index, _, _ in self.projections
            )
            and all(
                self.angular[key].scalar_factor is not None
                for *_, key, scale in self.routes
            )
            else None
        )
        self.node_cost = sum(
            self.input_sections[0][i][1] * len(self.angular[key].entries)
            for i, _, _, key in self.nodes
        )
        self.recoupled_cost = sum(
            channels
            * (
                len(self.angular[key].entries)
                + (offset >= 0) * self.angular[key].dimensions[-1]
            )
            for _, _, _, offset, channels, _, _, key, _ in self.routes
        )
        self.standard_cost = sum(
            self.input_sections[0][i][1] * len(self.angular[key].entries)
            for i, _, _, key, _ in self.first_paths
        ) + sum(
            channels
            * (
                len(self.angular[key].entries)
                + (offset >= 0) * math.prod(self.input_sections[2][third][1:])
            )
            for _, third, _, channels, offset, key, _ in self.second_paths
        )

    def node_values(self, x1, x3, backend):
        values = []
        for first, third, channel, key in self.nodes:
            start, mul, dim = self.input_sections[0][first]
            x = x1[..., start : start + mul * dim].unflatten(-1, (mul, dim))
            start, _, dim = self.input_sections[2][third]
            y = x3[..., start + channel * dim : start + (channel + 1) * dim].unsqueeze(
                -2
            )
            values.append(self.angular[key](x, y, backend))
        return values

    def project_inputs(self, x2, weight):
        values = []
        for second, channels, offset in self.projections:
            start, mul, dim = self.input_sections[1][second]
            x = x2[..., start : start + mul * dim].unflatten(-1, (mul, dim))
            if offset >= 0:
                w = weight[..., offset : offset + channels * mul].unflatten(
                    -1, (channels, mul)
                )
                x = channel_projection(x, w)
            else:
                x = x.sum(-2, keepdim=True)
            values.append(x)
        return values

    def recoupled(self, values, backend, source=None):
        x1, x2, x3, weight1, weight2 = values
        shared1 = weight1.ndim == 1 or weight1.shape[0] == 1
        shared2 = weight2.ndim == 1 or weight2.shape[0] == 1
        if source is not None and shared1 and shared2 and self.scalar_edge is not None:
            return self.scalar_nodes(values, backend, source)
        nodes = self.node_values(x1, x3, backend)
        projected, weights, weighted_nodes, operations = {}, {}, {}, []
        for (
            node,
            projection,
            target,
            offset,
            channels,
            multiplicity,
            channel,
            key,
            scale,
        ) in self.routes:
            second, _, first_offset = self.projections[projection]
            start, mul, dim = self.input_sections[1][second]
            first_on_nodes = (
                source is not None and shared1 and mul == 1 and first_offset >= 0
            )
            second_on_nodes = source is not None and shared2 and offset >= 0
            if projection not in projected:
                value = x2[..., start : start + mul * dim].unflatten(-1, (mul, dim))
                if first_offset >= 0 and not first_on_nodes:
                    weight = weight1[
                        ..., first_offset : first_offset + channels * mul
                    ].unflatten(-1, (channels, mul))
                    value = channel_projection(value, weight)
                elif first_offset < 0:
                    value = value.sum(-2, keepdim=True)
                projected[projection] = value
            spec = offset, channels, multiplicity, channel
            if offset >= 0 and spec not in weights:
                weights[spec] = weight2[
                    ..., offset : offset + channels * multiplicity
                ].unflatten(-1, (channels, multiplicity))[..., channel : channel + 1]
            before = first_on_nodes or second_on_nodes
            node_key = (
                node,
                first_offset if first_on_nodes else -1,
                spec if second_on_nodes else None,
                scale if before else 1,
            )
            if node_key not in weighted_nodes:
                weighted = nodes[node]
                if first_on_nodes:
                    weighted = weighted * weight1[
                        ..., first_offset : first_offset + channels
                    ].unsqueeze(-1)
                if second_on_nodes:
                    weighted = weighted * weights[spec]
                weighted_nodes[node_key] = weighted * scale if before else weighted
            operations.append(
                (
                    node_key,
                    projection,
                    target,
                    key,
                    spec,
                    offset >= 0 and not second_on_nodes,
                    1 if before else scale,
                )
            )
        gathered = dict(
            zip(weighted_nodes, gather_nodes(list(weighted_nodes.values()), source))
        )
        outputs = {}
        for node, projection, target, key, spec, late_weight, scale in operations:
            value = self.angular[key](gathered[node], projected[projection], backend)
            if late_weight:
                value = value * weights[spec]
            if scale != 1:
                value = value * scale
            outputs[target] = (
                value if target not in outputs else outputs[target] + value
            )
        return self.assemble(values, outputs, source)

    def scalar_nodes(self, values, backend, source):
        """Move all shared path mixing before scalar edge propagation."""
        x1, x2, x3, weight1, weight2 = values
        nodes = self.node_values(x1, x3, backend)
        outputs = {}
        for (
            node,
            projection,
            target,
            offset,
            channels,
            multiplicity,
            channel,
            key,
            scale,
        ) in self.routes:
            first_offset = self.projections[projection][2]
            value = nodes[node]
            if first_offset >= 0:
                value = value * weight1[
                    ..., first_offset : first_offset + channels
                ].unsqueeze(-1)
            if offset >= 0:
                value = (
                    value
                    * weight2[..., offset : offset + channels * multiplicity].unflatten(
                        -1, (channels, multiplicity)
                    )[..., channel : channel + 1]
                )
            value = value * (scale * self.angular[key].scalar_factor)
            outputs[target] = (
                value if target not in outputs else outputs[target] + value
            )
        features = self.assemble(values, outputs, None, leading=(x1.shape[0],))
        start = self.input_sections[1][self.scalar_edge][0]
        return features[source] * x2[:, start : start + 1]

    def standard(self, values, backend, source=None):
        """Contract the standard tree without expanding channel-pair products."""
        x1, x2, x3, weight1, weight2 = values
        projected = self.project_inputs(x2, weight1)
        inputs, intermediates = {}, {}
        for first, projection, target, key, scale in self.first_paths:
            if first not in inputs:
                start, channels, dim = self.input_sections[0][first]
                value = x1[..., start : start + channels * dim].unflatten(
                    -1, (channels, dim)
                )
                inputs[first] = value if source is None else value[source]
            value = (
                self.angular[key](inputs[first], projected[projection], backend) * scale
            )
            intermediates[target] = (
                value if target not in intermediates else intermediates[target] + value
            )
        outputs, projections = {}, {}
        for mid, third, target, channels, offset, key, scale in self.second_paths:
            if mid not in intermediates:
                continue
            spec = third, channels, offset
            if spec not in projections:
                start, mul, dim = self.input_sections[2][third]
                other = x3[..., start : start + mul * dim].unflatten(-1, (mul, dim))
                shared = offset < 0 or weight2.ndim == 1 or weight2.shape[0] == 1
                if source is not None and not shared:
                    other = other[source]
                if offset >= 0:
                    weight = weight2[..., offset : offset + channels * mul].unflatten(
                        -1, (channels, mul)
                    )
                    other = channel_projection(other, weight)
                else:
                    other = other.sum(-2, keepdim=True)
                projections[spec] = (
                    other[source] if source is not None and shared else other
                )
            value = (
                self.angular[key](intermediates[mid], projections[spec], backend)
                * scale
            )
            outputs[target] = (
                value if target not in outputs else outputs[target] + value
            )
        return self.assemble(values, outputs, source)

    def assemble(self, values, outputs, source, leading=None):
        if leading is None:
            leading = (
                torch.broadcast_shapes(*(v.shape[:-1] for v in values))
                if source is None
                else (values[1].shape[0],)
            )
        zero = sum(value[..., :0].sum() for value in values)
        blocks = [
            outputs[i].expand(*leading, mul, dim).flatten(-2)
            if i in outputs
            else values[0].new_zeros((*leading, mul * dim))
            for i, (mul, dim) in enumerate(self.output_sections)
        ]
        return (
            torch.cat(blocks, -1) if blocks else values[0].new_zeros((*leading, 0))
        ) + zero

    def use_recoupling(self, values, source=None):
        """Compare sparse arithmetic before and after node reuse."""
        recouple = config.WIGNER6J_RECOUPLE
        if recouple is None:
            saving = self.standard_cost - self.recoupled_cost
            recouple = saving > 0 and statically_known_true(
                self.node_cost < saving
                if source is None
                else self.node_cost * values[0].shape[0] < saving * values[1].shape[0]
            )
        return recouple

    def evaluate(self, values, backend, source=None):
        operation = (
            self.recoupled if self.use_recoupling(values, source) else self.standard
        )
        return operation(values, backend, source)


def recoupled(module, values):
    """Keep the original e3nn recoupled tree as a performance baseline."""
    x1, x2, x3, weight1, weight2 = values
    shape = torch.broadcast_shapes(*(v.shape[:-1] for v in values))
    count = math.prod(shape)
    x1, x2, x3 = [
        v.expand(shape + (v.shape[-1],)).reshape(count, v.shape[-1]) for v in values[:3]
    ]
    weight = (
        module.combine_weights(weight1, weight2)
        .expand(shape + (module.tp132.weight_numel,))
        .reshape(count, module.tp132.weight_numel)
    )
    return module.tp132(module.tp13(x1, x3), x2, weight).reshape(
        shape + (module.output_dim,)
    )


def evaluate(module, values):
    """Select the independent reference or measured shared-projection schedule."""
    if config.TORCH_REFERENCE:
        return module.forward_reference(*values)
    if not config.TORCH_WIGNER6J:
        return recoupled(module, values)
    return module.schedule.evaluate(values, "torch")

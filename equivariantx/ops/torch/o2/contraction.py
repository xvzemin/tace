"""Batched real O(2) coupling paths."""

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from equivariantx.o2.clebsch_gordan import coupling_entries

from ... import config
from ...common.sparse_coefficients import SparseMap


class Coupling(torch.nn.Module):
    """Share real scalar products across all coupling paths of one order."""

    def __init__(self, entries, output_dim, input_dim=None, other_dim=None):
        super().__init__()
        if input_dim is not None:
            self.sparse = SparseMap((input_dim, other_dim, output_dim), entries)
        products, rows = {}, [[] for _ in range(output_dim)]
        for i, j, k, coefficient in entries:
            index = products.setdefault((i, j), len(products))
            rows[k].append((index, coefficient))
        columns = max(map(len, rows), default=0)
        self.output_dim = output_dim
        self.coefficients = tuple(
            tuple(row[i][1] if i < len(row) else 0.0 for row in rows)
            for i in range(columns)
        )
        indices = (
            [i for i, _ in products],
            [j for _, j in products],
            *[
                [row[i][0] if i < len(row) else 0 for row in rows]
                for i in range(columns)
            ],
        )
        self.input_identity = (
            indices[0] == list(range(input_dim or 0)),
            indices[1] == list(range(other_dim or 0)),
        )
        self.output_identity = tuple(
            index == list(range(len(products))) for index in indices[2:]
        )
        self.unit_scale = tuple(all(v == 1.0 for v in row) for row in self.coefficients)
        for name, value in zip(
            ("input1", "input2", *(f"output_{i}" for i in range(columns))), indices
        ):
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.long), persistent=False
            )
        for i, values in enumerate(self.coefficients):
            self.register_buffer(
                f"scale_{i}",
                torch.tensor(values, dtype=torch.float64),
                persistent=False,
            )
            self.register_buffer(
                f"float_scale_{i}",
                torch.tensor(values, dtype=torch.float32),
                persistent=False,
            )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for i, values in enumerate(self.coefficients):
            name = f"scale_{i}"
            master = torch.tensor(
                values, dtype=torch.float64, device=getattr(self, name).device
            )
            self._buffers[name] = master
            self._buffers[f"float_scale_{i}"] = master.float()
        return self

    def forward(self, input1, input2, backend="torch", optimized=False):
        if optimized and hasattr(self, "sparse"):
            return self.sparse(input1, input2, backend)
        a = input1 if self.input_identity[0] else input1.index_select(-2, self.input1)
        b = input2 if self.input_identity[1] else input2.index_select(-2, self.input2)
        products = a * b
        if not self.output_dim:
            return products
        if not self.coefficients:
            return products.sum(-2, keepdim=True).expand(
                *products.shape[:-2], self.output_dim, products.shape[-1]
            )
        prefix = "float_scale_" if products.dtype == torch.float32 else "scale_"
        result = None
        for i, identity in enumerate(self.output_identity):
            part = (
                products
                if identity
                else products.index_select(-2, getattr(self, f"output_{i}"))
            )
            if not self.unit_scale[i]:
                part = part * getattr(self, f"{prefix}{i}").unsqueeze(-1)
            result = part if result is None else result + part
        return result


class Readout(torch.nn.Module):
    """Apply independent path weights and reduce angular outputs."""

    def __init__(self, indices, scales, output_dim, input_dim, *, sparse=True):
        super().__init__()
        if sparse:
            self.sparse = SparseMap(
                (input_dim, max(indices[1], default=-1) + 1, output_dim),
                tuple((*index, value) for index, value in zip(zip(*indices), scales)),
            )
        self.output_dim = output_dim
        self.scale_values = tuple(scales)
        self.input_is_identity = indices[0] == list(range(input_dim))
        self.output_is_identity = indices[2] == list(range(output_dim))
        for name, value in zip(("input", "weight", "output"), indices):
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.long), persistent=False
            )
        self.register_buffer(
            "scales", torch.tensor(scales, dtype=torch.float64), persistent=False
        )
        self.register_buffer("float_scales", self.scales.float(), persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.scales = torch.tensor(
            self.scale_values, dtype=torch.float64, device=self.scales.device
        )
        self.float_scales = self.scales.float()
        return self

    def forward(self, features, weight, backend="torch", optimized=False):
        if optimized and hasattr(self, "sparse"):
            return self.sparse(features, weight, backend)
        values = (
            features
            if self.input_is_identity
            else features.index_select(-2, self.input)
        )
        scale = self.float_scales if features.dtype == torch.float32 else self.scales
        weights = weight.index_select(-2, self.weight) * scale.unsqueeze(-1)
        values = values * weights
        if self.output_is_identity:
            return values
        output = values.new_zeros(
            (*values.shape[:-2], self.output_dim, values.shape[-1])
        )
        return output.index_add_(-2, self.output, values)


class Contraction(torch.nn.Module):
    """Evaluate shared coupling prefixes without generalized CG tensors."""

    def __init__(
        self,
        irreps_in,
        paths_by_order,
        scales_by_order,
        input_slices,
        output_slices,
        output_dim,
        *,
        shared=False,
        coupling=coupling_entries,
    ):
        super().__init__()
        prefixes = [{} for _ in paths_by_order]
        for paths in paths_by_order:
            for path in paths:
                for i in range(len(path.leaves)):
                    prefixes[i].setdefault(
                        (path.leaves[: i + 1], path.intermediates[: i + 1]), None
                    )
        locations = {key: input_slices[key[0][0]].start for key in prefixes[0]}
        previous_width = sum(ir.dim for ir in irreps_in)
        planner = None
        if shared and len(paths_by_order) <= 4:
            from ....mbe.ace.powers import ProductExpressions

            planner = ProductExpressions(previous_width)
            expressions = planner.base
        aliases = [(i, 1) for i in range(previous_width)]
        self.layers = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        join = (
            shared
            and 1 < len(paths_by_order) <= 4
            and sum(p.intermediates[-1].dim for paths in paths_by_order for p in paths)
            <= 256
        )
        joint = [[], [], []]
        joint_scales = []
        input_start = weight_start = 0
        for order, paths in enumerate(paths_by_order):
            if order:
                entries, next_locations, offset = [], {}, 0
                for leaves, intermediate in prefixes[order]:
                    first = locations[leaves[:-1], intermediate[:-1]]
                    second = input_slices[leaves[-1]].start
                    for i, j, k, coefficient in coupling(
                        intermediate[-2], irreps_in[leaves[-1]], intermediate[-1]
                    ):
                        entries.append((first + i, second + j, offset + k, coefficient))
                    next_locations[leaves, intermediate] = offset
                    offset += intermediate[-1].dim
                if planner is not None and offset:
                    columns = [{} for _ in range(offset)]
                    for a, b, c, w in entries:
                        columns[c][a, b] = columns[c].get((a, b), 0.0) + w
                    columns, expressions, aliases = planner.reduce(
                        columns, expressions, aliases, order == 1
                    )
                    entries = [
                        (a, b, c, w)
                        for c, col in enumerate(columns)
                        for (a, b), w in col.items()
                    ]
                    offset = len(columns)
                else:
                    aliases = [(i, 1) for i in range(offset)]
                self.layers.append(
                    Coupling(
                        entries, offset, previous_width, sum(ir.dim for ir in irreps_in)
                    )
                )
                previous_width = offset
                locations = next_locations
            else:
                offset = sum(ir.dim for ir in irreps_in)
            indices, scales = [[], [], []], []
            for i, (path, scale) in enumerate(zip(paths, scales_by_order[order])):
                start = locations[path.leaves, path.intermediates]
                output = output_slices[path.output_index].start
                for k in range(path.intermediates[-1].dim):
                    index, factor = aliases[start + k]
                    indices[0].append(index)
                    indices[1].append(i)
                    indices[2].append(output + k)
                    scales.append(scale * float(factor))
            self.readouts.append(Readout(indices, scales, output_dim, offset))
            if join:
                joint[0].extend(input_start + i for i in indices[0])
                joint[1].extend(weight_start + i for i in indices[1])
                joint[2].extend(indices[2])
                joint_scales.extend(scales)
                input_start += offset
                weight_start += len(paths)
        self.joint = (
            Readout(joint, joint_scales, output_dim, input_start, sparse=False)
            if join
            else None
        )
        self.joint_width = input_start + 3 * len(joint_scales)

    def forward(
        self, inputs, weights, backend="torch", optimized=False, *, all_weights=None
    ):
        if (
            self.joint is not None
            and all_weights is not None
            and not optimized
            and not (config.TORCH_REFERENCE or config.TORCH_RECOMPUTE)
            and statically_known_true(
                inputs[0].numel() * self.joint_width * inputs[0].element_size()
                <= config.ACE_SHARED_POWER_WORKSPACE * inputs[0].shape[-2]
            )
        ):
            powers = [inputs[0]]
            for order, layer in enumerate(self.layers, 1):
                powers.append(layer(powers[-1], inputs[order], backend, optimized))
            return self.joint(torch.cat(powers, dim=-2), all_weights)
        features = inputs[0]
        output = self.readouts[0](features, weights[0], backend, optimized)
        for order, layer in enumerate(self.layers, 1):
            features = layer(features, inputs[order], backend, optimized)
            output = output + self.readouts[order](
                features, weights[order], backend, optimized
            )
        return output


def dense(module, inputs, order_weights):
    letters = "abdefghijklmnqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if module.correlation > len(letters):
        raise ValueError("correlation exceeds the number of einsum indices.")
    output = None
    for order_index in range(module.correlation):
        order = order_index + 1
        indices = letters[:order]
        coefficient = getattr(module, f"generalized_cg_{order}").to(inputs[0])
        # Weight the generalized tensor, then contract one input at a time.
        contribution = torch.einsum(
            f"o{indices}p,...pc->...o{indices}c",
            coefficient,
            order_weights[order_index],
        )
        for i in reversed(range(order)):
            contribution = torch.einsum(
                f"...o{indices[: i + 1]}c,...{indices[i]}c->...o{indices[:i]}c",
                contribution,
                inputs[i],
            )
        output = contribution if output is None else output + contribution
    return module._flatten_output(output)


def prepare(module, inputs, weight, leading_shape):
    """Keep flattened irrep layouts and broadcasts outside contraction kernels."""
    inputs = tuple(
        features.expand(*leading_shape, module.input_dim).reshape(
            *leading_shape, module._base_input_dim, module.num_channels
        )
        for features in (inputs[:1] if module.shared_inputs else inputs)
    )
    if module.shared_inputs:
        inputs = inputs * module.correlation
    order_weights = tuple(
        weight[..., section].reshape(*weight.shape[:-1], count, module.num_channels)
        for section, count in zip(module._order_weight_slices, module.order_num_paths)
    )
    return inputs, order_weights


def evaluate(module, inputs, weight, leading_shape, *, backend="torch", optimized=None):
    inputs, order_weights = prepare(module, inputs, weight, leading_shape)
    if module.algorithm == "dense":
        return dense(module, inputs, order_weights)
    if optimized is None:
        optimized = not config.TORCH_REFERENCE and getattr(
            config, "TORCH_O2_CONTRACTION", False
        )
    all_weights = (
        weight.unflatten(-1, (module.num_paths, module.num_channels))
        if module.contraction.joint is not None
        else None
    )
    output = module.contraction(
        inputs, order_weights, backend, optimized, all_weights=all_weights
    )
    return module._flatten_output(output)

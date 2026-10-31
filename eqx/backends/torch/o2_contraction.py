"""Batched real O(2) coupling paths."""

import torch

from ...o2.clebsch_gordan import coupling_entries


class Coupling(torch.nn.Module):
    """Share real scalar products across all coupling paths of one order."""

    def __init__(self, entries, output_dim):
        super().__init__()
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

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for i, values in enumerate(self.coefficients):
            name = f"scale_{i}"
            self._buffers[name] = torch.tensor(values, dtype=torch.float64).to(
                getattr(self, name)
            )
        return self

    def forward(self, input1, input2):
        products = input1.index_select(-2, self.input1) * input2.index_select(
            -2, self.input2
        )
        if not self.output_dim:
            return products
        result = products.index_select(-2, self.output_0) * self.scale_0.to(
            products
        ).unsqueeze(-1)
        for i in range(1, len(self.coefficients)):
            result = result + products.index_select(
                -2, getattr(self, f"output_{i}")
            ) * getattr(self, f"scale_{i}").to(products).unsqueeze(-1)
        return result


class Readout(torch.nn.Module):
    """Apply independent path weights and reduce angular outputs."""

    def __init__(self, indices, scales, output_dim, input_dim):
        super().__init__()
        self.output_dim = output_dim
        self.scale_values = tuple(scales)
        self.input_is_identity = indices[0] == list(range(input_dim))
        for name, value in zip(("input", "weight", "output"), indices):
            self.register_buffer(
                name, torch.tensor(value, dtype=torch.long), persistent=False
            )
        self.register_buffer(
            "scales", torch.tensor(scales, dtype=torch.float64), persistent=False
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.scales = torch.tensor(self.scale_values, dtype=torch.float64).to(
            self.scales
        )
        return self

    def forward(self, features, weight):
        values = (
            features
            if self.input_is_identity
            else features.index_select(-2, self.input)
        )
        weights = weight.index_select(-2, self.weight) * self.scales.to(
            features
        ).unsqueeze(-1)
        values = values * weights
        output = values.new_zeros(
            (*values.shape[:-2], self.output_dim, values.shape[-1])
        )
        return output.index_add(-2, self.output, values)


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
        self.layers = torch.nn.ModuleList()
        self.readouts = torch.nn.ModuleList()
        for order, paths in enumerate(paths_by_order):
            if order:
                entries, next_locations, offset = [], {}, 0
                for leaves, intermediate in prefixes[order]:
                    first = locations[leaves[:-1], intermediate[:-1]]
                    second = input_slices[leaves[-1]].start
                    for i, j, k, coefficient in coupling_entries(
                        intermediate[-2], irreps_in[leaves[-1]], intermediate[-1]
                    ):
                        entries.append((first + i, second + j, offset + k, coefficient))
                    next_locations[leaves, intermediate] = offset
                    offset += intermediate[-1].dim
                self.layers.append(Coupling(entries, offset))
                locations = next_locations
            else:
                offset = sum(ir.dim for ir in irreps_in)
            indices, scales = [[], [], []], []
            for i, (path, scale) in enumerate(zip(paths, scales_by_order[order])):
                start = locations[path.leaves, path.intermediates]
                output = output_slices[path.output_index].start
                for k in range(path.intermediates[-1].dim):
                    indices[0].append(start + k)
                    indices[1].append(i)
                    indices[2].append(output + k)
                    scales.append(scale)
            self.readouts.append(Readout(indices, scales, output_dim, offset))

    def forward(self, inputs, weights):
        features = inputs[0]
        output = self.readouts[0](features, weights[0])
        for order, layer in enumerate(self.layers, 1):
            features = layer(features, inputs[order])
            output = output + self.readouts[order](features, weights[order])
        return output

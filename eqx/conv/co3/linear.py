"""Node-level channel compression of partially symmetric Cartesian tensors."""

from collections import defaultdict

import torch

from ...co3 import Linear as CartesianLinear
from ...co3.basis import path_matrix
from ...utils.layout import _Permute


class Linear(CartesianLinear):
    """Mix compact Cartesian channels before projection to spherical features.

    Parameters
    ----------
    irreps_in, irreps_out : Irreps or str
        Input and output representations, including channel multiplicities.
    input_indices : sequence of int
        Expansion from compact storage to flattened Cartesian ``mul_ir``.
    backend : {"cuda", "torch"}, optional
        CUDA packs contiguous group matrices in one permutation. CPU inputs
        use ordinary PyTorch operations.
    **kwargs
        Weight, bias and normalization arguments of ``co3.Linear``.
    """

    def __init__(
        self, irreps_in, irreps_out, input_indices, *, backend="cuda", **kwargs
    ):
        super().__init__(irreps_in, irreps_out, output_basis="spherical", **kwargs)
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be cuda or torch.")
        self.backend = backend
        indices = torch.as_tensor(input_indices, dtype=torch.long, device="cpu")
        if indices.numel() != self.irreps_in.dim or (indices < 0).any():
            raise ValueError("input_indices must expand to irreps_in.dim entries.")
        self.input_dim = int(indices.max()) + 1 if indices.numel() else 0
        groups = defaultdict(list)
        for group in self.groups:
            for ins, section in group:
                if ins.i_in < 0 or not all(ins.path_shape):
                    continue
                mul, ir = self.irreps_in[ins.i_in]
                rows = indices[self.slices_in[ins.i_in]].reshape(mul, ir.dim).tolist()
                for u, row in enumerate(rows):
                    unique = tuple(dict.fromkeys(row))
                    lookup = {index: i for i, index in enumerate(unique)}
                    expansion = tuple(lookup[index] for index in row)
                    groups[ins.i_out, expansion].append(
                        (unique, section.start + u * ins.path_shape[1], ins.path_weight)
                    )
        self.compact_groups = []
        self.expansions = []
        self.scales = []
        self.input_sections = []
        self.weight_sections = []
        self.weight_rows = []
        self.weight_identity = []
        self.uniform_scales = []
        input_indices = []
        for i, ((dest, expansion), entries) in enumerate(groups.items()):
            mul, ir = self.cartesian_out[dest]
            width = len(entries[0][0])
            self.compact_groups.append((dest, width, len(entries), mul))
            self.expansions.append((ir.l, expansion))
            self.scales.append(tuple(scale for _, _, scale in entries))
            self.uniform_scales.append(len(set(self.scales[-1])) == 1)
            gathered = (
                torch.tensor([entry[0] for entry in entries], dtype=torch.long)
                .T.flatten()
                .tolist()
            )
            self.input_sections.append(
                slice(len(input_indices), len(input_indices) + len(gathered))
            )
            input_indices.extend(gathered)
            start = min(entry[1] for entry in entries)
            stop = max(entry[1] for entry in entries) + mul
            rows = all((offset - start) % mul == 0 for _, offset, _ in entries)
            weight_indices = (
                [(offset - start) // mul for _, offset, _ in entries]
                if rows
                else [
                    offset - start + v for _, offset, _ in entries for v in range(mul)
                ]
            )
            self.weight_sections.append(slice(start, stop))
            self.weight_rows.append(rows)
            self.weight_identity.append(
                weight_indices
                == list(range((stop - start) // mul if rows else stop - start))
            )
            self.register_buffer(
                f"weight_index_{i}",
                torch.tensor(weight_indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"weight_scale_{i}",
                torch.tensor([scale for _, _, scale in entries]).unsqueeze(-1),
                persistent=False,
            )
            matrix = path_matrix(ir.l)
            packed = matrix.new_zeros(width, 2 * ir.l + 1).index_add(
                0, torch.tensor(expansion, dtype=torch.long), matrix
            )
            if self.uniform_scales[-1]:
                packed = packed * self.scales[-1][0]
            self.register_buffer(
                f"basis_{i}", packed.to(torch.get_default_dtype()), persistent=False
            )
        self.input_identity = input_indices == list(range(self.input_dim))
        self.input_permutation = sorted(input_indices) == list(range(self.input_dim))
        index = torch.tensor(input_indices, dtype=torch.long)
        self.register_buffer("input_index", index, persistent=False)
        self.register_buffer("input_inverse", index.argsort(), persistent=False)
        self.group_widths = repr(tuple(s.stop - s.start for s in self.input_sections))

    def forward(self, features, weight=None, bias=None):
        """Apply channel weights to compact coordinates and then project."""
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                "The compact feature dimension does not match input_indices."
            )
        weight = self.weight if weight is None else weight
        bias = self.bias if bias is None else bias
        if weight.shape[-1] != self.weight_numel or bias.shape[-1] != self.bias_numel:
            raise ValueError("Weight or bias size does not match the instructions.")
        if self.shared_weights and (weight.ndim != 1 or bias.ndim != 1):
            raise ValueError("Shared weights and biases must be one-dimensional.")
        shape = torch.broadcast_shapes(
            features.shape[:-1], weight.shape[:-1], bias.shape[:-1]
        )
        zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
        grouped = (
            self.backend == "cuda"
            and features.is_cuda
            and features.ndim == 2
            and self.input_permutation
        )
        if grouped:
            from ...kernels.layout import grouped_permute

            features = grouped_permute(
                features, self.input_index, self.input_inverse, self.group_widths
            )
        elif not self.input_identity:
            features = (
                _Permute.apply(features, self.input_index, self.input_inverse)
                if self.input_permutation
                else features.index_select(-1, self.input_index)
            )
        outputs = [None] * len(self.cartesian_out)
        for i, (dest, dim, mul_in, mul_out) in enumerate(self.compact_groups):
            section = self.input_sections[i]
            x = (
                features.flatten()[
                    features.size(0) * section.start : features.size(0) * section.stop
                ]
                if grouped
                else features[..., section]
            )
            x = x.reshape(*features.shape[:-1], dim, mul_in)
            w = weight[..., self.weight_sections[i]]
            if self.weight_rows[i]:
                rows = (
                    self.weight_sections[i].stop - self.weight_sections[i].start
                ) // mul_out
                w = w.reshape(*weight.shape[:-1], rows, mul_out)
            if not self.weight_identity[i]:
                w = w.index_select(
                    -2 if self.weight_rows[i] else -1,
                    getattr(self, f"weight_index_{i}"),
                )
            w = w.reshape(*weight.shape[:-1], mul_in, mul_out)
            if not self.uniform_scales[i]:
                w = w * getattr(self, f"weight_scale_{i}")
            mixed = x @ w
            # Project only after reducing the input channel multiplicity.
            value = mixed.transpose(-1, -2) @ getattr(self, f"basis_{i}")
            outputs[dest] = value if outputs[dest] is None else outputs[dest] + value
        for i, ((mul, ir), group) in enumerate(zip(self.irreps_out, self.groups)):
            if outputs[i] is None:
                outputs[i] = features.new_zeros((*shape, mul, ir.dim)) + zero
            for ins, section in group:
                if ins.i_in < 0:
                    outputs[i] = outputs[i] + bias[..., section].unsqueeze(-1)
        return (
            torch.cat([value.flatten(-2) for value in outputs], -1) + zero
            if outputs
            else features[..., :0] + zero
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for i, (degree, expansion) in enumerate(self.expansions):
            matrix = path_matrix(degree)
            indices = torch.tensor(expansion, dtype=torch.long, device="cpu")
            packed = matrix.new_zeros(int(indices.max()) + 1, matrix.size(1)).index_add(
                0, indices, matrix
            )
            if self.uniform_scales[i]:
                packed = packed * self.scales[i][0]
            name = f"basis_{i}"
            self._buffers[name] = packed.to(self._buffers[name]).clone()
            name = f"weight_scale_{i}"
            self._buffers[name] = (
                torch.tensor(self.scales[i], dtype=torch.float64, device="cpu")
                .unsqueeze(-1)
                .to(self._buffers[name])
            )
        return self

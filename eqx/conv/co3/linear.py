"""Node-level channel compression of partially symmetric Cartesian tensors."""

from collections import defaultdict

import torch

from ...co3 import Linear as CartesianLinear
from ...co3.basis import path_matrix, stf_projection
from ...co3.trace_free import trace_free_basis
from ...utils.layout import _Permute


class Linear(CartesianLinear):
    """Mix channels through compact irreducible coordinates.

    Parameters
    ----------
    irreps_in, irreps_out : Irreps or str
        Input and output representations, including channel multiplicities.
    input_indices : sequence of int
        Expansion from compact storage to flattened Cartesian ``mul_ir``.
    backend : {"cuda", "torch"}, optional
        CUDA packs group matrices and fuses their final projections and sum.
        CPU inputs use ordinary PyTorch operations.
    projected_inputs : sequence of bool, optional
        Channels already in an STF subspace. These need extraction, not
        projection. Defaults to False for every input channel.
    **kwargs
        Weight, bias and normalization arguments of ``co3.Linear``.
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        input_indices,
        *,
        backend="cuda",
        projected_inputs=None,
        **kwargs,
    ):
        super().__init__(irreps_in, irreps_out, **kwargs)
        if backend not in ("cuda", "torch"):
            raise ValueError("backend must be cuda or torch.")
        self.backend = backend
        indices = torch.as_tensor(input_indices, dtype=torch.long, device="cpu")
        if indices.numel() != self.irreps_in.dim or (indices < 0).any():
            raise ValueError("input_indices must expand to irreps_in.dim entries.")
        self.input_dim = int(indices.max()) + 1 if indices.numel() else 0
        channels = sum(mul for mul, _ in self.irreps_in)
        projected_inputs = (
            (False,) * channels if projected_inputs is None else tuple(projected_inputs)
        )
        if len(projected_inputs) != channels:
            raise ValueError("projected_inputs must contain one value per channel.")
        offsets, offset = [], 0
        for mul, _ in self.irreps_in:
            offsets.append(offset)
            offset += mul
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
                    stf = projected_inputs[offsets[ins.i_in] + u]
                    groups[ins.i_out, expansion, stf].append(
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
        self.reduce_input = []
        self.select_input = []
        input_indices = []
        destinations = defaultdict(list)
        for i, ((dest, expansion, stf), entries) in enumerate(groups.items()):
            mul, ir = self.cartesian_out[dest]
            width = len(entries[0][0])
            self.compact_groups.append((dest, width, len(entries), mul))
            destinations[dest].append((i, entries))
            self.expansions.append((ir.l, expansion, stf))
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
            packed, _ = self.matrices(ir.l, expansion, stf)
            dim = packed.shape[1]
            ci = len(entries)
            selection = stf and self.output_basis == "cartesian" and ir.l <= 4
            self.select_input.append(selection)
            if selection:
                self.register_buffer(
                    f"selection_{i}", packed.argmax(0), persistent=False
                )
            self.reduce_input.append(
                selection
                or (dim < width and ci * dim * (width + mul) < mul * width * (ci + dim))
            )
            if self.uniform_scales[-1]:
                packed = packed * self.scales[-1][0]
            self.register_buffer(
                f"projection_{i}",
                packed.to(torch.get_default_dtype()),
                persistent=False,
            )
        for l in {ir.l for _, ir in self.cartesian_out}:
            _, matrix = self.matrices(l, tuple(range(3**l)))
            self.register_buffer(
                f"expansion_{l}",
                matrix.to(torch.get_default_dtype()).clone(),
                persistent=False,
            )
        self.input_identity = input_indices == list(range(self.input_dim))
        self.input_permutation = sorted(input_indices) == list(range(self.input_dim))
        index = torch.tensor(input_indices, dtype=torch.long)
        self.register_buffer("input_index", index, persistent=False)
        self.register_buffer("input_inverse", index.argsort(), persistent=False)
        self.cuda_groups = []
        self.cuda_scales = []
        self.cuda_scaled = []
        self.cuda_weight_rows = []
        projections, extractions, start = [], [], 0
        for dest, entries in destinations.items():
            cout, ir = self.cartesian_out[dest]
            cin = sum(len(rows) for _, rows in entries)
            dim = 2 * ir.l + 1
            offsets, scales, channel = [], [], 0
            for i, rows in entries:
                # Kernel constants are built directly in float64, including
                # normalization, rather than from dtype-converted buffers.
                matrix, _ = self.matrices(*self.expansions[i])
                if self.uniform_scales[i]:
                    matrix = matrix * self.scales[i][0]
                extractions.append(
                    (
                        self.input_sections[i].start,
                        len(rows),
                        matrix.shape[0],
                        dim,
                        tuple(
                            (a, b, float(matrix[a, b]))
                            for a, b in matrix.nonzero().tolist()
                        ),
                        channel,
                        cin,
                        start,
                    )
                )
                channel += len(rows)
                offsets.extend(offset for _, offset, _ in rows)
                scales.extend(
                    1.0 if self.uniform_scales[i] else scale for _, _, scale in rows
                )
            lower, upper = min(offsets), max(offsets) + cout
            rows = all((offset - lower) % cout == 0 for offset in offsets)
            indices = (
                [(offset - lower) // cout for offset in offsets]
                if rows
                else [offset - lower + v for offset in offsets for v in range(cout)]
            )
            i = len(self.cuda_groups)
            identity = indices == list(
                range((upper - lower) // cout if rows else upper - lower)
            )
            self.cuda_groups.append((dest, dim, cin, cout, lower, upper, identity))
            self.cuda_weight_rows.append(rows)
            self.cuda_scales.append(tuple(scales))
            self.cuda_scaled.append(any(scale != 1.0 for scale in scales))
            self.register_buffer(
                f"cuda_weight_index_{i}",
                torch.tensor([] if identity else indices, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                f"cuda_weight_scale_{i}",
                torch.tensor(scales).unsqueeze(-1),
                persistent=False,
            )
            matrix = self.matrices(ir.l, tuple(range(ir.dim)))[1]
            projections.append(
                (
                    self.irreps_out.slices()[dest].start,
                    cout,
                    dim,
                    self.irreps_out[dest].ir.dim,
                    tuple(
                        (a, b, float(matrix[a, b]))
                        for a, b in matrix.nonzero().tolist()
                    ),
                )
            )
            start += cin * dim
        self.packing_metadata = repr((self.input_dim, tuple(extractions)))
        self.projection_metadata = repr((self.irreps_out.dim, tuple(projections)))

    def matrices(self, l, expansion, stf=False):
        """Return extraction and reconstruction matrices in CPU float64."""
        if self.output_basis == "spherical" or l > 4:
            matrix = path_matrix(l)
            reconstruction = (
                torch.eye(2 * l + 1, dtype=torch.float64, device="cpu")
                if self.output_basis == "spherical"
                else matrix.T
            )
        else:
            indices, reconstruction = trace_free_basis(l)
            if stf:
                matrix = torch.zeros(3**l, 2 * l + 1, dtype=torch.float64, device="cpu")
                matrix[indices, torch.arange(2 * l + 1, device="cpu")] = 1
            else:
                matrix = stf_projection(l).index_select(1, indices)
        index = torch.tensor(expansion, dtype=torch.long, device="cpu")
        return matrix.new_zeros(int(index.max()) + 1, matrix.shape[1]).index_add(
            0, index, matrix
        ), reconstruction

    def forward(self, features, weight=None, bias=None):
        """Extract irreducible coordinates, mix channels and reconstruct."""
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
        fused = (
            self.backend == "cuda"
            and features.is_cuda
            and features.ndim == 2
            and self.input_permutation
            and weight.ndim == 1
            and not self.bias_numel
            and not bias.requires_grad
            and bool(self.cuda_groups)
        )
        if fused:
            return self.forward_cuda(features, weight)
        if not self.input_identity:
            features = (
                _Permute.apply(features, self.input_index, self.input_inverse)
                if self.input_permutation
                else features.index_select(-1, self.input_index)
            )
        outputs = [None] * len(self.cartesian_out)
        widths = tuple(s.stop - s.start for s in self.input_sections)
        blocks = features.split(widths, dim=-1)
        for i, (dest, dim, mul_in, mul_out) in enumerate(self.compact_groups):
            x = blocks[i].reshape(*features.shape[:-1], dim, mul_in)
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
            projection = getattr(self, f"projection_{i}")
            if self.select_input[i]:
                x = x.index_select(-2, getattr(self, f"selection_{i}"))
                if self.uniform_scales[i]:
                    w = w * self.scales[i][0]
            elif self.reduce_input[i]:
                x = projection.T @ x
            mixed = x @ w
            value = mixed.transpose(-1, -2)
            if not self.reduce_input[i]:
                value = value @ projection
            outputs[dest] = value if outputs[dest] is None else outputs[dest] + value
        zero = None
        if not self.has_weighted_paths:
            zero = features[..., :0].sum() + weight[..., :0].sum() + bias[..., :0].sum()
        elif not self.bias_numel and bias.requires_grad:
            zero = bias.sum()
        for i, ((mul, ir), group) in enumerate(zip(self.cartesian_out, self.groups)):
            if outputs[i] is None:
                outputs[i] = features.new_zeros((*shape, mul, 2 * ir.l + 1))
            if self.output_basis == "cartesian" and ir.l >= 2:
                outputs[i] = outputs[i] @ getattr(self, f"expansion_{ir.l}")
            for ins, section in group:
                if ins.i_in < 0:
                    outputs[i] = outputs[i] + bias[..., section].unsqueeze(-1)
        output = (
            torch.cat([value.flatten(-2) for value in outputs], -1)
            if outputs
            else features.new_zeros((*shape, 0))
        )
        return output if zero is None else output + zero

    def forward_cuda(self, features, weight):
        """Pack all paths of each output irrep for one channel contraction."""
        from .projection import pack, project

        packed = pack(self.packing_metadata, features, self.input_index)
        widths = tuple(
            features.size(0) * dim * cin for _, dim, cin, _, _, _, _ in self.cuda_groups
        )
        outputs = []
        for i, (block, (_, dim, cin, cout, start, stop, identity)) in enumerate(
            zip(packed.flatten().split(widths), self.cuda_groups)
        ):
            w = weight[start:stop]
            if self.cuda_weight_rows[i]:
                w = w.reshape(-1, cout)
            if not identity:
                w = w.index_select(0, getattr(self, f"cuda_weight_index_{i}"))
            w = w.reshape(cin, cout)
            if self.cuda_scaled[i]:
                w = w * getattr(self, f"cuda_weight_scale_{i}")
            outputs.append(block.reshape(features.size(0), dim, cin) @ w)
        return project(self.projection_metadata, outputs, False)[0]

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for i, (degree, expansion, stf) in enumerate(self.expansions):
            packed, matrix = self.matrices(degree, expansion, stf)
            if self.uniform_scales[i]:
                packed = packed * self.scales[i][0]
            name = f"projection_{i}"
            self._buffers[name] = packed.to(self._buffers[name]).clone()
            name = f"weight_scale_{i}"
            self._buffers[name] = (
                torch.tensor(self.scales[i], dtype=torch.float64, device="cpu")
                .unsqueeze(-1)
                .to(self._buffers[name])
            )
        for l in {ir.l for _, ir in self.cartesian_out}:
            name = f"expansion_{l}"
            matrix = self.matrices(l, tuple(range(3**l)))[1]
            self._buffers[name] = matrix.to(self._buffers[name]).clone()
        for i, scales in enumerate(self.cuda_scales):
            name = f"cuda_weight_scale_{i}"
            self._buffers[name] = (
                torch.tensor(scales, dtype=torch.float64, device="cpu")
                .unsqueeze(-1)
                .to(self._buffers[name])
            )
        return self

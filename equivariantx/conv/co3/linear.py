"""Node-level channel compression of partially symmetric Cartesian tensors."""

from collections import defaultdict

import torch

from ...co3.basis import path_matrix
from ...co3.linear import Linear as CartesianLinear
from ...ops import resolve_backend
from ...ops.precision import validate_module_precision
from ...ops.torch.conv.co3 import linear as torch_kernel


class Linear(CartesianLinear):
    """Mix and project compact Cartesian channels.

    Parameters
    ----------
    irreps_in, irreps_out : equivariantx.co3.Irreps or str
        Input and output representations, including channel multiplicities.
    input_indices : sequence of int
        Expansion from compact storage to flattened Cartesian ``mul_ir``.
    backend : {None, "cuda", "torch"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        CUDA packs contiguous group matrices in one permutation. CPU inputs
        use ordinary PyTorch operations.
    output_basis : {"spherical", "cartesian"}, optional
        Angular storage after projection, in flattened ``mul_ir`` layout.
    **kwargs
        Weight, bias and normalization arguments of :class:`equivariantx.co3.Linear`.

    Examples
    --------
    >>> import torch
    >>> from equivariantx.conv.co3 import Linear
    >>> indices = [0, 1, 2, 1, 3, 4, 2, 4, 5]
    >>> layer = Linear("1x2e", "2x2e", indices)
    >>> layer(torch.randn(4, 6)).shape
    torch.Size([4, 10])
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        input_indices,
        *,
        backend=None,
        output_basis="spherical",
        **kwargs,
    ):
        backend = resolve_backend(backend)
        super().__init__(
            irreps_in, irreps_out, output_basis=output_basis, backend=backend, **kwargs
        )
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
            if output_basis == "cartesian":
                packed = packed @ matrix.T
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
        self.set_reference(indices)

    def set_reference(self, indices=None):
        """Retain ordinary Cartesian coordinates, including older cached modules."""
        if indices is None and not hasattr(self, "input_expansion"):
            indices = torch.zeros(self.irreps_in.dim, dtype=torch.long, device="cpu")
            packed = self.input_index.detach().cpu()
            for i, (dest, dim, count, mul_out) in enumerate(self.compact_groups):
                rows = packed[self.input_sections[i]].reshape(dim, count).T
                section = self.weight_sections[i]
                offsets = getattr(self, f"weight_index_{i}").detach().cpu()
                offsets = (
                    offsets * mul_out if self.weight_rows[i] else offsets[::mul_out]
                ) + section.start
                expansion = self.expansions[i][1]
                for row, offset in zip(rows, offsets.tolist()):
                    for ins, path in self.groups[dest]:
                        if ins.i_in >= 0 and path.start <= offset < path.stop:
                            width = self.irreps_in[ins.i_in].ir.dim
                            channel = (offset - path.start) // mul_out
                            start = self.slices_in[ins.i_in].start + channel * width
                            indices[start : start + width] = row[list(expansion)]
                            break
        if indices is not None:
            self.register_buffer(
                "input_expansion",
                indices.to(self.input_index).clone(),
                persistent=False,
            )
        degrees = {self.cartesian_out[dest].ir.l for dest, *_ in self.compact_groups}
        for degree in sorted(degrees):
            self.register_buffer(
                f"reference_basis_{degree}",
                path_matrix(degree).to(self.weight).clone(),
                persistent=False,
            )

    def __setstate__(self, state):
        super().__setstate__(state)
        if not hasattr(self, "input_expansion"):
            self.set_reference()

    def forward(self, features, weight=None, bias=None):
        """Apply channel weights and irreducible projection to compact coordinates."""
        validate_module_precision(self, (features, weight, bias))
        if features.shape[-1] != self.input_dim:
            raise ValueError(
                "The compact feature dimension does not match input_indices."
            )
        weight = self.weight if weight is None else weight
        if bias is None:
            bias = self.bias if self.bias_numel else features.new_empty(0)
        if weight.shape[-1] != self.weight_numel or bias.shape[-1] != self.bias_numel:
            raise ValueError("Weight or bias size does not match the instructions.")
        if self.shared_weights and (weight.ndim != 1 or bias.ndim != 1):
            raise ValueError("Shared weights and biases must be one-dimensional.")
        if self.backend == "cuda" and features.is_cuda:
            from ...ops.cuda.conv.co3 import linear as cuda_kernel

            return cuda_kernel.evaluate(self, features, weight, bias)
        return torch_kernel.evaluate(self, features, weight, bias)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for i, (degree, expansion) in enumerate(self.expansions):
            matrix = path_matrix(degree)
            indices = torch.tensor(expansion, dtype=torch.long, device="cpu")
            packed = matrix.new_zeros(int(indices.max()) + 1, matrix.size(1)).index_add(
                0, indices, matrix
            )
            if self.output_basis == "cartesian":
                packed = packed @ matrix.T
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
        self.set_reference()
        return self

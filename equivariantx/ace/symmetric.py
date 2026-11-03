"""Channel-wise symmetric contractions with element-dependent coefficients."""

from collections import defaultdict

import torch
from e3nn import o3

from ..ops.symmetric import symmetric_contract


class SymmetricContraction(torch.nn.Module):
    """Contract repeated features with sparse symmetric coupling tensors.

    Parameters
    ----------
    irreps_in, irreps_out : o3.Irreps
        Input and output representations with equal multiplicities.
    coupling_tensors : sequence of tuple
        Entries ``(i_out, degree, tensor)``. A tensor has shape
        ``(irrep_out.dim, input_dim, ..., input_dim, num_paths)`` with
        ``degree`` input axes. Scalar output axes may be omitted.
        ``input_dim`` counts angular components without multiplicities.
        Weight blocks follow the supplied order.
    backend : {"cuda", "torch"}, optional
        Execution backend. CPU tensors use PyTorch operations.
    layout : {"mul_ir", "ir_mul"}, optional
        Output layout. Defaults to ``"mul_ir"``.

    Examples
    --------
    >>> import torch
    >>> from equivariantx.ace import SymmetricContraction
    >>> coupling = torch.ones(1, 1, 1, 1)
    >>> layer = SymmetricContraction("2x0e", "2x0e", [(0, 2, coupling)])
    >>> x = torch.randn(3, 2, 1)
    >>> weight = torch.randn(2, 1, 2)
    >>> layer(x, weight, torch.tensor([0, 1, 0])).shape
    torch.Size([3, 2])
    """

    def __init__(
        self,
        irreps_in,
        irreps_out,
        coupling_tensors,
        backend="cuda",
        layout="mul_ir",
    ):
        super().__init__()
        irreps_in, irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        input_dim = sum(ir.dim for _, ir in irreps_in)
        offsets, offset = [], 0
        for _, ir in irreps_out:
            offsets.append(offset)
            offset += ir.dim
        entries, num_paths = [], 0
        for output, degree, tensor in coupling_tensors:
            if degree < 1 or not 0 <= output < len(irreps_out):
                raise ValueError("Provide a positive degree and a valid output index.")
            tensor = tensor.detach().to(device="cpu", dtype=torch.float64)
            paths = tensor.shape[-1]
            tensor = tensor.reshape(
                irreps_out[output].ir.dim, *([input_dim] * degree), paths
            )
            for index in tensor.nonzero().tolist():
                entries.append(
                    (
                        tuple(index[1:-1]),
                        num_paths + index[-1],
                        offsets[output] + index[0],
                        tensor[tuple(index)].item(),
                    )
                )
            num_paths += paths
        self.initialize(
            irreps_in, irreps_out, entries, num_paths, backend, layout
        )

    @classmethod
    def from_entries(
        cls, irreps_in, irreps_out, entries, num_paths, backend="cuda", layout="mul_ir"
    ):
        """Construct from sparse coupling coefficients.

        Parameters
        ----------
        entries : sequence of tuple
            ``(input_indices, weight_index, output_index, value)`` entries.
            Angular indices exclude channel multiplicities.
        num_paths : int
            Number of independent coefficient vectors.

        Returns
        -------
        SymmetricContraction
            Module with the supplied representations, backend and layout.
        """
        module = cls.__new__(cls)
        torch.nn.Module.__init__(module)
        module.initialize(irreps_in, irreps_out, entries, num_paths, backend, layout)
        return module

    def initialize(self, irreps_in, irreps_out, entries, num_paths, backend, layout):
        """Register the symmetric polynomial and its reference tensors."""
        self.irreps_in, self.irreps_out = o3.Irreps(irreps_in), o3.Irreps(irreps_out)
        multiplicities = {mul for mul, _ in (*self.irreps_in, *self.irreps_out)}
        if len(multiplicities) != 1 or min(multiplicities) < 1:
            raise ValueError("All multiplicities must agree and be positive.")
        if backend not in ("cuda", "torch") or layout not in ("mul_ir", "ir_mul"):
            raise ValueError("Invalid contraction backend or output layout.")
        self.backend, self.layout = backend, layout
        self.num_channels = multiplicities.pop()
        self.input_dim = sum(ir.dim for _, ir in self.irreps_in)
        self.output_dims = tuple(ir.dim for _, ir in self.irreps_out)
        self.output_dim = sum(self.output_dims)
        self.num_paths = num_paths
        self.weight_numel = num_paths * self.num_channels
        grouped = defaultdict(float)
        for indices, weight, output, value in entries:
            if not indices or not all(0 <= i < self.input_dim for i in indices):
                raise ValueError("Input angular indices are out of range.")
            if not 0 <= weight < num_paths or not 0 <= output < self.output_dim:
                raise ValueError("Weight or output indices are out of range.")
            grouped[tuple(sorted(indices)), weight, output] += value
        rows = tuple((*key, value) for key, value in sorted(grouped.items()) if value)
        self.correlation = max((len(row[0]) for row in rows), default=1)
        self.has_terms = bool(rows)
        self.metadata = repr(
            (self.input_dim, num_paths, self.output_dim, self.correlation, rows)
        )
        for degree in range(1, self.correlation + 1):
            terms = [row for row in rows if len(row[0]) == degree]
            self.register_buffer(
                f"indices_{degree}",
                torch.tensor([row[0] for row in terms], dtype=torch.long).reshape(
                    -1, degree
                ),
            )
            self.register_buffer(
                f"weights_{degree}",
                torch.tensor([row[1] for row in terms], dtype=torch.long),
            )
            self.register_buffer(
                f"outputs_{degree}",
                torch.tensor([row[2] for row in terms], dtype=torch.long),
            )
            self.register_buffer(
                f"coefficients_{degree}",
                torch.tensor([row[3] for row in terms], dtype=torch.float64),
            )

    def forward(self, features, weight, node_type):
        """Evaluate element-dependent symmetric contractions.

        Parameters
        ----------
        features : torch.Tensor
            Shape ``(nodes, channels, input_dim)``.
        weight : torch.Tensor
            Shape ``(elements, num_paths, channels)``.
        node_type : torch.Tensor
            Integer element indices of shape ``(nodes,)``.

        Returns
        -------
        torch.Tensor
            Shape ``(nodes, irreps_out.dim)`` in the requested output layout.
        """
        if features.ndim != 3 or features.shape[1:] != (
            self.num_channels,
            self.input_dim,
        ):
            raise ValueError("Features must have shape (nodes, channels, input_dim).")
        if weight.ndim != 3 or weight.shape[1:] != (self.num_paths, self.num_channels):
            raise ValueError("Weights must have shape (elements, num_paths, channels).")
        if weight.dtype != features.dtype or weight.device != features.device:
            raise ValueError("Features and weights must share a dtype and device.")
        if node_type.shape != features.shape[:1] or node_type.device != features.device:
            raise ValueError(
                "Provide one element index per node on the feature device."
            )
        if node_type.dtype not in (torch.int32, torch.int64):
            raise TypeError("Element indices must be integers.")
        x = features.transpose(1, 2)
        if features.is_cuda and self.backend == "cuda":
            role = self.correlation + 1
            mapping = (*([0] * self.correlation), 1, 2)
            placeholder = x.new_empty((0, self.output_dim, self.num_channels))
            output = symmetric_contract(
                self.metadata,
                repr(((mapping, role, 0, 0),)),
                node_type,
                [x, weight, placeholder],
            )[0]
        else:
            output = x.new_zeros((x.shape[0], self.output_dim, self.num_channels))
            selected = weight[node_type.long()]
            for degree in range(1, self.correlation + 1):
                indices = getattr(self, f"indices_{degree}")
                if not indices.shape[0]:
                    continue
                values = x[:, indices[:, 0], :]
                for k in range(1, degree):
                    values = values * x[:, indices[:, k], :]
                values = values * selected[:, getattr(self, f"weights_{degree}"), :]
                values = (
                    values
                    * getattr(self, f"coefficients_{degree}").to(x.dtype)[None, :, None]
                )
                output = output.index_add(1, getattr(self, f"outputs_{degree}"), values)
            if not self.has_terms:
                output = output + (x.sum() + weight.sum()) * 0
        if self.layout == "ir_mul" or not self.output_dims:
            return output.flatten(1)
        return torch.cat(
            [
                part.transpose(1, 2).flatten(1)
                for part in output.split(self.output_dims, dim=1)
            ],
            dim=1,
        )

    def extra_repr(self):
        return f"{self.irreps_in} -> {self.irreps_out}, correlation={self.correlation}, {self.num_paths} paths"

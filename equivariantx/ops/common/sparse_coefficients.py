"""Sparse fixed-coefficient tables without dense Cartesian intermediates."""

from functools import lru_cache

import torch

from .. import config
from .torch import literal_bilinear, sparse_bilinear


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def bilinear_coefficients(dimensions, entries):
    """Pack all three transposes of ``(left, right, output, coefficient)`` entries."""
    result = []
    for role in range(3):
        rows = [{} for _ in range(dimensions[role])]
        for *indices, coefficient in entries:
            other = tuple(indices[k] for k in range(3) if k != role)
            row = rows[indices[role]]
            row[other] = row.get(other, 0.0) + coefficient
        rows = [
            tuple((*pair, value) for pair, value in row.items() if value)
            for row in rows
        ]
        width = max(1, max((len(row) for row in rows), default=0))
        result.extend(
            tuple(
                tuple(row[k][i] if k < len(row) else 0 for k in range(width))
                for row in rows
            )
            for i in range(3)
        )
    return tuple(result)


def bilinear_tables(dimensions, entries, *, dtype=torch.float64, device="cpu"):
    """Create immutable gather indices and coefficients on the requested device."""
    return tuple(
        torch.tensor(
            data, dtype=dtype if i % 3 == 2 else torch.long, device=device
        ).reshape(dimensions[i // 3], -1)
        if dimensions[i // 3]
        else torch.empty(
            (0, 1), dtype=dtype if i % 3 == 2 else torch.long, device=device
        )
        for i, data in enumerate(bilinear_coefficients(dimensions, entries))
    )


class SparseMap(torch.nn.Module):
    """Store a fixed sparse bilinear map and its two exact transposes."""

    def __init__(self, dimensions, entries):
        super().__init__()
        self.dimensions, self.entries = dimensions, tuple(entries)
        for index, value in enumerate(bilinear_tables(dimensions, self.entries)):
            self.register_buffer(f"table_{index}", value, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        data = bilinear_coefficients(self.dimensions, self.entries)
        for index in (2, 5, 8):
            value = getattr(self, f"table_{index}")
            setattr(
                self,
                f"table_{index}",
                torch.tensor(data[index], dtype=torch.float64)
                .reshape(value.shape)
                .to(value),
            )
        return self

    def forward(self, x, y, backend="torch"):
        shape = torch.broadcast_shapes(x.shape[:-2], y.shape[:-2])
        if not self.entries:
            return (
                x.new_zeros((*shape, self.dimensions[-1], x.shape[-1]))
                + x[..., :0, :].sum()
                + y[..., :0, :].sum()
            )
        if backend == "cuda":
            from .cuda.sparse_bilinear import evaluate
        else:
            evaluate = (
                sparse_bilinear.evaluate
                if config.TORCH_O2_CONTRACTION_ADJOINT
                else sparse_bilinear.primal
            )
        tables = tuple(
            getattr(self, f"table_{i}").to(dtype=x.dtype)
            if i % 3 == 2
            else getattr(self, f"table_{i}")
            for i in range(9)
        )
        return evaluate(x.transpose(-1, -2), y.transpose(-1, -2), tables).transpose(
            -1, -2
        )


class AngularMap(torch.nn.Module):
    """Store one normalized angular map and its two exact transposes."""

    def __init__(self, degrees):
        super().__init__()
        from e3nn import o3

        coefficient = o3.wigner_3j(*degrees, dtype=torch.float64)
        self.dimensions = tuple(coefficient.shape)
        self.entries = tuple(
            (*index, float(coefficient[tuple(index)]))
            for index in coefficient.nonzero().tolist()
        )
        first, second, output = self.dimensions
        self.scalar_factor = None
        if min(first, second) == 1 and output == max(first, second):
            scale = self.entries[0][-1]
            expected = tuple(
                (i if first > 1 else 0, i if second > 1 else 0, i, scale)
                for i in range(output)
            )
            if self.entries == expected:
                self.scalar_factor = scale
        for i, table in enumerate(bilinear_tables(self.dimensions, self.entries)):
            self.register_buffer(f"table_{i}", table, persistent=False)
        self.register_buffer("dense", coefficient, persistent=False)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        rows = bilinear_coefficients(self.dimensions, self.entries)
        for i in (2, 5, 8):
            table = getattr(self, f"table_{i}")
            setattr(self, f"table_{i}", table.new_tensor(rows[i]).reshape(table.shape))
        dense = torch.zeros(self.dimensions, dtype=torch.float64)
        for a, b, c, value in self.entries:
            dense[a, b, c] = value
        self.dense = dense.to(self.dense)
        return self

    def forward(self, x, y, backend):
        if self.scalar_factor is not None:
            value = x * y
            return value if self.scalar_factor == 1 else value * self.scalar_factor
        if backend == "cuda":
            from .cuda.sparse_bilinear import evaluate

            tables = tuple(
                getattr(self, f"table_{i}").to(dtype=x.dtype)
                if i % 3 == 2
                else getattr(self, f"table_{i}")
                for i in range(9)
            )
            return evaluate(x, y, tables)
        if len(self.entries) <= 256:
            operation = (
                literal_bilinear.evaluate
                if config.TORCH_WIGNER6J_ADJOINT
                else literal_bilinear.primal
            )
            return operation(x, y, self.dimensions, self.entries)
        return torch.einsum("...i,...j,ijk->...k", x, y, self.dense.to(x))

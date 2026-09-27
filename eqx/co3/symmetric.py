"""Normalized storage and contractions for symmetric Cartesian tensors."""

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=None)
def symmetric_powers(rank):
    """Return Cartesian exponent triples for the specified tensor rank."""
    return tuple((rank - k, k - z, z) for k in range(rank + 1) for z in range(k + 1))


@lru_cache(maxsize=None)
def symmetric_basis(rank):
    """Return CPU float64 coefficients and integer contraction indices."""
    powers = symmetric_powers(rank)
    lookup = {p: i for i, p in enumerate(powers)}
    integers = torch.arange(3**rank, device="cpu")
    y, z = torch.zeros_like(integers), torch.zeros_like(integers)
    for _ in range(rank):
        y = y + (integers % 3 == 1)
        z = z + (integers % 3 == 2)
        integers = integers // 3
    data = {
        "indices": (y + z) * (y + z + 1) // 2 + z,
        "norm": torch.tensor(
            [
                math.sqrt(math.comb(rank, p[1]) * math.comb(rank - p[1], p[2]))
                for p in powers
            ],
            dtype=torch.float64,
            device="cpu",
        ),
    }

    def add(name, indices, weights):
        data[f"{name}_indices"] = torch.tensor(indices, dtype=torch.long, device="cpu")
        data[f"{name}_weights"] = torch.tensor(
            weights, dtype=torch.float64, device="cpu"
        )

    if rank:
        indices, weights = [], []
        for p in symmetric_powers(rank - 1):
            indices.append(
                [lookup[tuple(v + (a == b) for b, v in enumerate(p))] for a in range(3)]
            )
            weights.append([math.sqrt((v + 1) / rank) for v in p])
        add("contraction", indices, weights)
    if rank >= 2:
        indices, weights = [], []
        for p in symmetric_powers(rank - 2):
            indices.append(
                [
                    lookup[tuple(v + 2 * (a == b) for b, v in enumerate(p))]
                    for a in range(3)
                ]
            )
            weights.append(
                [math.sqrt((v + 1) * (v + 2) / (rank * (rank - 1))) for v in p]
            )
        add("trace", indices, weights)
        lower = {p: i for i, p in enumerate(symmetric_powers(rank - 2))}
        indices, weights = [], []
        for p in powers:
            row, scale = [], []
            for a, b in ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)):
                q = tuple(v - (c == a) - (c == b) for c, v in enumerate(p))
                row.append(lower.get(q, 0))
                scale.append(
                    (1 if a == b else 2)
                    * math.sqrt(p[a] * (p[b] - (a == b)) / (rank * (rank - 1)))
                    if q in lower
                    else 0.0
                )
            indices.append(row)
            weights.append(scale)
        add("product", indices, weights)

    # After k contractions, the two index groups have ranks k and rank-k.
    # Both remain symmetric, so no 3**rank intermediate is required.
    for k in range(1, rank + 1):
        left = {p: i for i, p in enumerate(symmetric_powers(k - 1))}
        right = {p: i for i, p in enumerate(symmetric_powers(rank - k + 1))}
        indices, weights = [], []
        for q in symmetric_powers(rank - k):
            indices.append(
                [right[tuple(v + (a == b) for b, v in enumerate(q))] for a in range(3)]
            )
            weights.append([math.sqrt((v + 1) / (rank - k + 1)) for v in q])
        add(f"input_{k}", indices, weights)
        indices, weights = [], []
        for p in symmetric_powers(k):
            indices.append(
                [
                    3 * left.get(tuple(v - (a == b) for b, v in enumerate(p)), 0) + a
                    for a in range(3)
                ]
            )
            weights.append([math.sqrt(v / k) for v in p])
        data[f"output_{k}"] = torch.zeros(
            len(indices), 3 * len(left), dtype=torch.float64, device="cpu"
        ).scatter_add_(
            1,
            torch.tensor(indices, dtype=torch.long, device="cpu"),
            torch.tensor(weights, dtype=torch.float64, device="cpu"),
        )
    return data


class SymmetricBasis(torch.nn.Module):
    """Store symmetric tensors with orthonormal permutation weights.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian rank.
    """

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.dim = (rank + 1) * (rank + 2) // 2
        for name, value in symmetric_basis(rank).items():
            self.register_buffer(
                name,
                value.to(torch.get_default_dtype()).clone()
                if value.is_floating_point()
                else value.clone(),
                persistent=False,
            )

    def pack(self, tensor):
        """Symmetrize full Cartesian entries and return normalized coordinates."""
        if self.rank < 2:
            return tensor
        # Each row is a normalized sum of at most three coordinates.
        # This avoids long atomic sums over all equivalent index permutations.
        for k in range(2, self.rank + 1):
            tensor = tensor.unflatten(-1, (3 * k * (k + 1) // 2, 3 ** (self.rank - k)))
            tensor = (getattr(self, f"output_{k}") @ tensor).flatten(-2)
        return tensor

    def unpack(self, tensor):
        """Expand normalized coordinates to full Cartesian storage."""
        if self.rank < 2:
            return tensor
        return (tensor / self.norm).index_select(-1, self.indices)

    def contract(self, tensor, vector):
        """Contract one index with a vector, lowering the rank by one."""
        if self.rank == 1:
            return (tensor * vector).sum(-1, keepdim=True)
        value = tensor[..., self.contraction_indices]
        return (value * (self.contraction_weights * vector.unsqueeze(-2))).sum(-1)

    def trace(self, tensor):
        """Contract two indices, lowering the rank by two."""
        return (tensor[..., self.trace_indices] * self.trace_weights).sum(-1)

    def multiply(self, tensor, matrix):
        """Multiply symmetrically by matrix entries (xx, xy, xz, yy, yz, zz)."""
        if self.rank == 2:
            return tensor * (matrix * self.norm)
        return (
            tensor[..., self.product_indices]
            * (self.product_weights * matrix.unsqueeze(-2))
        ).sum(-1)

    def forward(self, tensor, matrix):
        """Apply a 3-by-3 matrix to every index in normalized storage."""
        if self.rank == 1:
            return torch.einsum("...ij,...j->...i", matrix, tensor)
        for k in range(1, self.rank + 1):
            rank = self.rank - k
            tensor = tensor.unflatten(
                -1, (k * (k + 1) // 2, (rank + 2) * (rank + 3) // 2)
            )
            if rank == 0:
                tensor = torch.einsum("...ij,...aj->...ai", matrix, tensor).flatten(-2)
                tensor = torch.nn.functional.linear(
                    tensor, getattr(self, f"output_{k}")
                )
            else:
                indices = getattr(self, f"input_{k}_indices")
                weights = getattr(self, f"input_{k}_weights")
                tensor = tensor[..., indices] * weights
                tensor = torch.einsum("...ij,...aqj->...aiq", matrix, tensor)
                if k == 1:
                    tensor = tensor.flatten(-3)
                else:
                    tensor = (
                        getattr(self, f"output_{k}") @ tensor.flatten(-3, -2)
                    ).flatten(-2)
        return tensor

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for name, value in symmetric_basis(self.rank).items():
            if value.is_floating_point():
                self._buffers[name] = value.to(self._buffers[name]).clone()
        return self

    def extra_repr(self):
        return f"rank={self.rank}"

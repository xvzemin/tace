"""Normalized storage and contractions for symmetric Cartesian tensors."""

import math
from functools import lru_cache

import torch

from ..ops import resolve_backend
from ..ops.precision import validate_module_precision
from ..ops.torch.co3 import symmetric as torch_kernel


def execution(module, tensor):
    """Resolve a native backend without loading it for CPU execution."""
    if module.backend == "cuda" and tensor.is_cuda:
        from ..ops.cuda.co3 import symmetric as cuda_kernel

        return cuda_kernel
    return torch_kernel


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

    data["permutation"] = torch.argsort(data["indices"], stable=True)
    counts = torch.bincount(data["indices"], minlength=len(powers))
    data["offsets"] = torch.cat((counts.new_zeros(1), counts.cumsum(0)))

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
        add(f"output_{k}", indices, weights)
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
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CPU tensors use Torch.

    Examples
    --------
    >>> from equivariantx.co3.symmetric import SymmetricBasis
    >>> basis = SymmetricBasis(2)
    >>> tensor = torch.eye(3).flatten()
    >>> torch.testing.assert_close(basis.unpack(basis.pack(tensor)), tensor)
    """

    def __init__(self, rank, *, backend=None):
        backend = resolve_backend(backend)
        super().__init__()
        self.backend = backend
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
        if rank in (2, 4, 6):
            self.register_buffer(
                "transform_identity",
                torch.eye(self.dim, dtype=self.norm.dtype, device=self.norm.device),
                persistent=False,
            )

    def pack(self, tensor):
        """Symmetrize full Cartesian entries and return normalized coordinates."""
        validate_module_precision(self, (tensor,))
        return execution(self, tensor).pack(self, tensor)

    def unpack(self, tensor):
        """Expand normalized coordinates to full Cartesian storage."""
        validate_module_precision(self, (tensor,))
        return execution(self, tensor).unpack(self, tensor)

    def contract(self, tensor, vector):
        """Contract one index with a vector, lowering the rank by one."""
        validate_module_precision(self, (tensor, vector))
        return execution(self, tensor).contract(self, tensor, vector)

    def trace(self, tensor):
        """Contract two indices, lowering the rank by two."""
        validate_module_precision(self, (tensor,))
        return execution(self, tensor).trace(self, tensor)

    def multiply(self, tensor, matrix):
        """Multiply symmetrically by matrix entries (xx, xy, xz, yy, yz, zz)."""
        validate_module_precision(self, (tensor, matrix))
        return execution(self, tensor).multiply(self, tensor, matrix)

    def forward(self, tensor, matrix):
        """Apply a 3-by-3 matrix to every index in normalized storage."""
        validate_module_precision(self, (tensor, matrix))
        return execution(self, tensor).transform(self, tensor, matrix)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        for name, value in symmetric_basis(self.rank).items():
            if value.is_floating_point():
                self._buffers[name] = value.to(self._buffers[name]).clone()
        return self

    def extra_repr(self):
        return f"rank={self.rank}"

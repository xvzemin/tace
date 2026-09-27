"""Representation metadata for two-dimensional Cartesian tensors."""

from collections import namedtuple
from typing import NamedTuple

import torch

from .. import o2


class Irrep(o2.Irrep):
    """An O(2) irrep in symmetric traceless Cartesian storage.

    Parameters
    ----------
    m : int, str, tuple or Irrep
        Order, representation name, or existing representation.
    p : {-1, 0, 1}, optional
        Reflection parity. Positive orders use zero.
    t : {-1, 1}, optional
        Time-reversal parity. Defaults to one.

    Notes
    -----
    ``dim`` is the storage size ``2**m``. The irreducible subspace has
    dimension one at order zero and two at every positive order.
    """

    @property
    def dim(self):
        """Number of Cartesian entries."""
        return 2**self.m

    @property
    def circular_dim(self):
        """Dimension of the irreducible subspace."""
        return 1 if self.m == 0 else 2

    def circular(self):
        """Return the same labels in compact O(2) metadata."""
        return o2.Irrep(self.m, self.p, self.t)

    def __mul__(self, other):
        return tuple(Irrep(ir) for ir in self.circular() * Irrep(other).circular())

    def __rmul__(self, mul):
        return Irreps([(mul, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    def D_from_matrix(self, matrix, time_reversal=False):
        """Return Cartesian matrices for (..., 2, 2) orthogonal inputs."""
        if matrix.shape[-2:] != (2, 2):
            raise ValueError("An O(2) matrix must have trailing shape (2, 2).")
        value = matrix.new_ones((*matrix.shape[:-2], 1, 1))
        for _ in range(self.m):
            value = torch.einsum("...ab,...cd->...acbd", value, matrix)
            value = value.flatten(-4, -3).flatten(-2)
        if self.is_odd_scalar():
            value = value * torch.linalg.det(matrix)[..., None, None]
        return value * self.t if time_reversal else value

    def D_from_angle(
        self, angle, reflected=False, time_reversal=False, *, dtype=None, device=None
    ):
        """Return S**reflected R(angle), including optional time reversal."""
        matrix = o2.Irrep("1m").D_from_angle(
            angle, reflected, dtype=dtype, device=device
        )
        return self.D_from_matrix(matrix, time_reversal)


class _MulIr(NamedTuple):
    mul: int
    ir: Irrep

    @property
    def dim(self):
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"


class Irreps(tuple):
    """Direct sum of Cartesian O(2) irreps in flattened mul_ir layout.

    Parameters
    ----------
    irreps : str, Irrep or iterable, optional
        Representation string or (multiplicity, irrep) entries. Compact
        O(2) metadata is accepted and converted without regrouping.
    """

    def __new__(cls, irreps=None):
        if isinstance(irreps, (str, o2.Irreps, o2.Irrep)) or irreps is None:
            entries = [(mul, ir) for ir, mul in o2.Irreps(irreps)]
        else:
            entries = [
                (1, item) if isinstance(item, (str, o2.Irrep)) else item
                for item in irreps
            ]
        result = []
        for mul, ir in entries:
            if not isinstance(mul, int) or mul < 1:
                raise ValueError("Multiplicities must be positive integers.")
            result.append(_MulIr(mul, Irrep(ir)))
        return tuple.__new__(cls, result)

    def __repr__(self):
        return "+".join(map(repr, self))

    def __getitem__(self, item):
        value = tuple.__getitem__(self, item)
        return Irreps(value) if isinstance(item, slice) else value

    def __add__(self, other):
        return Irreps(tuple(self) + tuple(Irreps(other)))

    def __radd__(self, other):
        return self if other == 0 else Irreps(other) + self

    def __mul__(self, mul):
        if not isinstance(mul, int) or mul < 0:
            raise ValueError("Multiplicity must be a non-negative integer.")
        return Irreps([(mul * n, ir) for n, ir in self] if mul else [])

    __rmul__ = __mul__

    def __contains__(self, ir):
        return any(item.ir == Irrep(ir) for item in self)

    @property
    def dim(self):
        """Total number of Cartesian entries."""
        return sum(item.dim for item in self)

    @property
    def num_irreps(self):
        """Total multiplicity."""
        return sum(mul for mul, _ in self)

    @property
    def mmax(self):
        """Maximum order, or -1 for an empty representation."""
        return max((ir.m for _, ir in self), default=-1)

    def circular(self):
        """Return compact O(2) metadata in ir_mul layout."""
        return o2.Irreps([(ir.circular(), mul) for mul, ir in self])

    def slices(self):
        """Return one Cartesian feature slice per entry."""
        result, offset = [], 0
        for item in self:
            result.append(slice(offset, offset + item.dim))
            offset += item.dim
        return result

    def simplify(self):
        """Combine adjacent entries carrying the same irrep."""
        return Irreps(self.circular().simplify())

    def sort(self):
        """Return sorted irreps, the entry permutation, and its inverse."""
        result = self.circular().sort()
        return namedtuple("sort", "irreps p inv")(
            Irreps(result.irreps), result.p, result.inv
        )

    def regroup(self):
        """Sort entries and combine identical irreps."""
        return self.sort().irreps.simplify()

    def count(self, ir):
        """Return the total multiplicity of an irrep."""
        return sum(mul for mul, item in self if item == Irrep(ir))

    def index(self, ir):
        """Return the first entry containing an irrep."""
        for i, (_, item) in enumerate(self):
            if item == Irrep(ir):
                return i
        raise ValueError(f"{ir} is not in {self}.")

    def filter(self, keep=None, *, drop=None, mmax=None):
        """Select entries by irrep, predicate, or maximum order."""
        if keep is not None and drop is not None:
            raise ValueError("Specify keep or drop, not both.")
        result = []
        for item in self:
            if mmax is not None and item.ir.m > mmax:
                continue
            if keep is not None and not (
                keep(item) if callable(keep) else item.ir in Irreps(keep)
            ):
                continue
            if drop is not None and (
                drop(item) if callable(drop) else item.ir in Irreps(drop)
            ):
                continue
            result.append(item)
        return Irreps(result)

    def randn(
        self,
        *size,
        normalization="component",
        requires_grad=False,
        dtype=None,
        device=None,
    ):
        """Sample STF features; -1 selects the representation axis."""
        from .basis import path_matrix

        compact = self.circular().randn(
            *size, normalization=normalization, dtype=dtype, device=device
        )
        axis = size.index(-1)
        compact = compact.movedim(axis, -1)
        values = []
        for (mul, ir), section in zip(self, self.circular().slices()):
            x = compact[..., section].reshape(*compact.shape[:-1], ir.circular_dim, mul)
            values.append((x.transpose(-1, -2) @ path_matrix(ir.m).to(x).T).flatten(-2))
        value = torch.cat(values, -1) if values else compact
        return value.movedim(-1, axis).detach().requires_grad_(requires_grad)

    def D_from_matrix(self, matrix, time_reversal=False):
        """Return the block-diagonal Cartesian representation matrix."""
        result = matrix.new_zeros((*matrix.shape[:-2], self.dim, self.dim))
        offset = 0
        for mul, ir in self:
            value = ir.D_from_matrix(matrix, time_reversal)
            for _ in range(mul):
                result[..., offset : offset + ir.dim, offset : offset + ir.dim] = value
                offset += ir.dim
        return result

    def D_from_angle(
        self, angle, reflected=False, time_reversal=False, *, dtype=None, device=None
    ):
        """Return the direct sum of rotation and reflection matrices."""
        matrix = o2.Irrep("1m").D_from_angle(
            angle, reflected, dtype=dtype, device=device
        )
        return self.D_from_matrix(matrix, time_reversal)

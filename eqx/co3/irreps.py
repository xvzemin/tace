"""Representation metadata for symmetric traceless Cartesian tensors."""

from collections import namedtuple
from typing import NamedTuple

import torch
from e3nn import o3


class Irrep(tuple):
    """An O(3) irrep stored as a rank-l Cartesian tensor.

    Parameters
    ----------
    l : int, str, tuple or Irrep
        Angular degree, an irrep name, or an existing representation.
    p : {-1, 1}, optional
        Spatial inversion parity when l is an integer.

    Notes
    -----
    ``dim`` is the storage size, ``3**l``. The symmetric traceless
    subspace has ``2*l+1`` independent coordinates. Spatial parity is
    independent of tensor rank.
    """

    def __new__(cls, l, p=None):
        return tuple.__new__(cls, [value for value in o3.Irrep(l, p)])

    @property
    def l(self) -> int:  # noqa: E743
        """Angular degree."""
        return self[0]

    @property
    def p(self) -> int:
        """Spatial inversion parity."""
        return self[1]

    @property
    def t(self) -> int:
        """Time-reversal parity, when supplied by the representation labels."""
        return self[2] if len(self) == 3 else 1

    @property
    def dim(self) -> int:
        """Number of Cartesian entries."""
        return 3**self.l

    @property
    def spherical_dim(self) -> int:
        """Dimension of the symmetric traceless subspace."""
        return 2 * self.l + 1

    def __repr__(self):
        return repr(o3.Irrep(tuple(self)))

    def is_scalar(self):
        """Return whether the representation is an invariant scalar."""
        return self.l == 0 and self.p == 1 and self.t == 1

    def __mul__(self, other):
        return (
            Irrep(ir) for ir in o3.Irrep(tuple(self)) * o3.Irrep(tuple(Irrep(other)))
        )

    def __rmul__(self, mul):
        return Irreps([(mul, self)])

    def __add__(self, other):
        return Irreps(self) + Irreps(other)

    @classmethod
    def iterator(cls, lmax=None):
        """Iterate over angular degrees and spatial parities."""
        return (cls(ir) for ir in o3.Irrep.iterator(lmax))

    def D_from_matrix(self, matrix):
        """Return Cartesian transformation matrices of shape (..., dim, dim)."""
        value = matrix.new_ones((*matrix.shape[:-2], 1, 1))
        for _ in range(self.l):
            value = (
                torch.einsum("...ab,...cd->...acbd", value, matrix)
                .flatten(-4, -3)
                .flatten(-2)
            )
        if self.p != (-1) ** self.l:
            value = value * torch.linalg.det(matrix)[..., None, None]
        return value

    def D_from_angles(self, alpha, beta, gamma, k=None):
        """Return the representation of Y-X-Y rotations and optional inversion."""
        value = self.D_from_matrix(o3.angles_to_matrix(alpha, beta, gamma))
        return value if k is None else value * self.p ** k[..., None, None]

    def D_from_quaternion(self, quaternion, k=None):
        """Return the representation of a quaternion and optional inversion."""
        value = self.D_from_matrix(o3.quaternion_to_matrix(quaternion))
        return value if k is None else value * self.p ** k[..., None, None]

    def D_from_axis_angle(self, axis, angle):
        """Return the representation of an axis-angle rotation."""
        return self.D_from_matrix(o3.axis_angle_to_matrix(axis, angle))


class _MulIr(NamedTuple):
    mul: int
    ir: Irrep

    @property
    def dim(self):
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"


class Irreps(tuple):
    """Direct sum of Cartesian irreps in flattened mul_ir layout.

    Parameters
    ----------
    irreps : str, Irrep or iterable, optional
        Representation string or sequence of (multiplicity, irrep) entries.
        Spherical representation metadata is also accepted.

    Notes
    -----
    Each entry occupies ``mul * 3**l`` consecutive values and is viewed
    as ``(..., mul, 3**l)``. Iteration yields ``(mul, ir)``.
    """

    def __new__(cls, irreps=None):
        if isinstance(irreps, Irrep):
            irreps = [(1, tuple(irreps))]
        elif irreps is not None and not isinstance(irreps, str):
            irreps = [
                (1, tuple(item)) if isinstance(item, Irrep) else item for item in irreps
            ]
        return tuple.__new__(
            cls, (_MulIr(mul, Irrep(ir)) for mul, ir in o3.Irreps(irreps))
        )

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
        return Irreps([(mul * n, ir) for n, ir in self])

    __rmul__ = __mul__

    def __contains__(self, ir):
        return any(item.ir == Irrep(ir) for item in self)

    @property
    def dim(self) -> int:
        """Total number of Cartesian entries."""
        return sum(item.dim for item in self)

    @property
    def num_irreps(self) -> int:
        """Total multiplicity."""
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> list:
        """Angular degree of each irrep copy."""
        return [ir.l for mul, ir in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        """Maximum angular degree, or -1 for an empty representation."""
        return max((ir.l for mul, ir in self if mul), default=-1)

    def spherical(self):
        """Return the same labels in spherical representation metadata."""
        return o3.Irreps([(mul, tuple(ir)) for mul, ir in self])

    @staticmethod
    def spherical_harmonics(lmax, p=-1):
        """Return degrees zero through lmax with parity p**l."""
        return Irreps([(1, (l, p**l)) for l in range(lmax + 1)])

    def slices(self):
        """Return the Cartesian feature slice for each entry."""
        result, offset = [], 0
        for item in self:
            result.append(slice(offset, offset + item.dim))
            offset += item.dim
        return result

    def simplify(self):
        """Combine adjacent entries of the same irrep and remove zero multiplicities."""
        return Irreps(self.spherical().simplify())

    def sort(self):
        """Return sorted irreps, the entry permutation, and its inverse."""
        result = self.spherical().sort()
        return namedtuple("sort", "irreps p inv")(
            Irreps(result.irreps), result.p, result.inv
        )

    def regroup(self):
        """Sort entries and combine identical irreps."""
        return self.sort().irreps.simplify()

    def remove_zero_multiplicities(self):
        """Remove entries with zero channels."""
        return Irreps([(mul, ir) for mul, ir in self if mul])

    def count(self, ir):
        """Return the total multiplicity of an irrep."""
        return sum(mul for mul, item in self if item == Irrep(ir))

    def index(self, ir):
        """Return the first entry containing an irrep."""
        for i, (_, item) in enumerate(self):
            if item == Irrep(ir):
                return i
        raise ValueError(f"{ir} is not in {self}.")

    def filter(self, keep=None, drop=None, lmax=None):
        """Select entries by irrep, predicate, or maximum angular degree."""
        if keep is not None and drop is not None:
            raise ValueError("Specify keep or drop, not both.")

        def match(rule, item):
            return rule(item) if callable(rule) else item.ir in Irreps(rule)

        return Irreps(
            [
                item
                for item in self
                if (lmax is None or item.ir.l <= lmax)
                and (keep is None or match(keep, item))
                and (drop is None or not match(drop, item))
            ]
        )

    def D_from_matrix(self, matrix):
        """Return the block-diagonal Cartesian representation of a matrix."""
        result = matrix.new_zeros((*matrix.shape[:-2], self.dim, self.dim))
        offset = 0
        for mul, ir in self:
            value = ir.D_from_matrix(matrix)
            for _ in range(mul):
                result[..., offset : offset + ir.dim, offset : offset + ir.dim] = value
                offset += ir.dim
        return result

    def randn(
        self,
        *size,
        normalization="component",
        requires_grad=False,
        dtype=None,
        device=None,
    ):
        """Generate symmetric traceless features; -1 marks the feature axis.

        Component normalization refers to independent orthonormal coordinates,
        not to the redundant Cartesian entries.
        """
        from .basis import path_matrix

        axis = size.index(-1)
        spherical = self.spherical().randn(
            *size, normalization=normalization, dtype=dtype, device=device
        )
        spherical = spherical.movedim(axis, -1)
        values = []
        for (mul, ir), section in zip(self, self.spherical().slices()):
            x = spherical[..., section].reshape(
                *spherical.shape[:-1], mul, ir.spherical_dim
            )
            values.append((x @ path_matrix(ir.l).to(spherical).T).flatten(-2))
        value = (torch.cat(values, -1) if values else spherical).movedim(-1, axis)
        return value.detach().requires_grad_(requires_grad)

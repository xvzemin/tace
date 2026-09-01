################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple, Union

import torch

Parity = Union[int, str]
IrrepLike = Union["Irrep", str, Tuple[int, Parity]]
IrrepsLike = Union[
    "Irreps",
    "Irrep",
    str,
    Sequence[Union[IrrepLike, Tuple[int, IrrepLike]]],
]


@dataclass(frozen=True, init=False)
class Irrep:
    """An irreducible Cartesian representation of O(3).

    Parameters
    ----------
    l : int, str, tuple, or Irrep
        Non-negative tensor degree, canonical name such as ``"2e"``, a
        ``(degree, parity)`` pair, or an existing instance.
    p : {-1, 1, "o", "e"}, optional
        Inversion parity when ``l`` is supplied as an integer.

    Notes
    -----
    The representation is stored in the ambient rank-``l`` Cartesian space
    of dimension ``3**l``. Its physical subspace is the symmetric traceless
    tensor with ``2*l + 1`` degrees of freedom. Parity is always explicit.
    """

    l: int
    p: int

    def __init__(
        self,
        l: Union[int, str, Tuple[int, Parity], "Irrep"],
        p: Optional[Parity] = None,
    ) -> None:
        if isinstance(l, Irrep):
            if p is not None:
                raise TypeError("Do not provide parity when copying an Irrep.")
            degree, parity = l.l, l.p
        elif isinstance(l, str):
            if p is not None:
                raise TypeError("Do not provide parity with an O(3) irrep string.")
            name = l.strip()
            try:
                degree = int(name[:-1])
                parity = {"e": 1, "o": -1}[name[-1]]
            except (IndexError, KeyError, ValueError):
                raise ValueError(
                    "Cartesian O(3) irreps must be written as <degree>e or "
                    "<degree>o."
                ) from None
        elif isinstance(l, tuple):
            if p is not None or len(l) != 2:
                raise TypeError("An Irrep tuple must be (l, p).")
            degree, parity = l
        else:
            if p is None:
                raise TypeError("Parity is required when constructing Irrep(l, p).")
            degree, parity = l, p

        if isinstance(parity, str):
            try:
                parity = {"e": 1, "o": -1}[parity]
            except KeyError:
                raise ValueError("O(3) parity must be -1/'o' or +1/'e'.") from None
        if (
            not isinstance(parity, int)
            or isinstance(parity, bool)
            or parity not in (-1, 1)
        ):
            raise ValueError("O(3) parity must be -1/'o' or +1/'e'.")
        if not isinstance(degree, int) or isinstance(degree, bool) or degree < 0:
            raise ValueError("Cartesian tensor degree l must be non-negative.")

        object.__setattr__(self, "l", degree)
        object.__setattr__(self, "p", parity)

    @property
    def dim(self) -> int:
        """Ambient Cartesian storage dimension."""
        return 3**self.l

    @property
    def dof(self) -> int:
        """Number of independent components in the irreducible subspace."""
        return 2 * self.l + 1

    @property
    def pseudo(self) -> bool:
        """Whether the Cartesian realization carries a determinant factor."""
        natural_parity = -1 if self.l % 2 else 1
        return self.p != natural_parity

    def __iter__(self) -> Iterator[int]:
        yield self.l
        yield self.p

    def __str__(self) -> str:
        return f"{self.l}{'e' if self.p == 1 else 'o'}"

    def __repr__(self) -> str:
        return str(self)

    def __mul__(self, other):
        try:
            other = Irrep(other)
        except (TypeError, ValueError):
            return NotImplemented
        parity = self.p * other.p
        return tuple(
            Irrep(degree, parity)
            for degree in range(abs(self.l - other.l), self.l + other.l + 1)
        )

    def D_from_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return the ambient Cartesian representation of a matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Orthogonal matrices with shape ``(..., 3, 3)``.

        Returns
        -------
        torch.Tensor
            Representation matrices with shape ``(..., dim, dim)``. A
            determinant factor is included for pseudotensor realizations.
        """
        matrix = torch.as_tensor(matrix)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError("matrix must have trailing shape (3, 3).")
        if not matrix.is_floating_point():
            matrix = matrix.to(torch.get_default_dtype())

        output = matrix.new_ones(matrix.shape[:-2] + (1, 1))
        for _ in range(self.l):
            output = torch.einsum("...ab,...cd->...acbd", output, matrix)
            output = output.reshape(
                matrix.shape[:-2] + (output.shape[-4] * 3, output.shape[-2] * 3)
            )
        if self.pseudo:
            output = output * torch.linalg.det(matrix)[..., None, None]
        return output


class Irreps:
    """An immutable direct sum of Cartesian O(3) irreps.

    Parameters
    ----------
    irreps : IrrepsLike, optional
        Representation specification. Accepted forms include canonical strings,
        individual irreps, ``(multiplicity, irrep)`` pairs, and sequences of
        those pairs.

    Notes
    -----
    Iteration yields ``(multiplicity, irrep)`` pairs. :attr:`dim` counts the
    ambient Cartesian storage, while :attr:`dof` counts independent symmetric
    traceless components.
    """

    __slots__ = ("_groups",)

    def __init__(self, irreps: IrrepsLike = "") -> None:
        if isinstance(irreps, Irreps):
            groups = tuple(irreps)
        elif isinstance(irreps, Irrep):
            groups = ((1, irreps),)
        elif isinstance(irreps, str):
            groups = []
            compact = irreps.replace(" ", "")
            if compact:
                for term in compact.split("+"):
                    try:
                        if "x" in term:
                            multiplicity, name = term.split("x")
                            multiplicity = int(multiplicity)
                        else:
                            multiplicity, name = 1, term
                        irrep = Irrep(name)
                    except (TypeError, ValueError):
                        raise ValueError(
                            f"Invalid Cartesian O(3) irreps term: {term!r}."
                        ) from None
                    if multiplicity < 1:
                        raise ValueError("Irrep multiplicities must be positive.")
                    groups.append((multiplicity, irrep))
            groups = tuple(groups)
        elif (
            isinstance(irreps, tuple)
            and len(irreps) == 2
            and isinstance(irreps[0], int)
        ):
            groups = (self._from_group(irreps),)
        elif isinstance(irreps, Sequence):
            groups = tuple(self._from_group(item) for item in irreps)
        else:
            raise TypeError("Unsupported Irreps input.")
        object.__setattr__(self, "_groups", tuple(groups))

    @staticmethod
    def _from_group(item) -> Tuple[int, Irrep]:
        if isinstance(item, (Irrep, str)):
            return 1, Irrep(item)
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "Each Irreps entry must be an irrep or a (multiplicity, irrep) pair."
            )
        multiplicity, irrep = item
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            raise TypeError("Irrep multiplicity must be an integer.")
        if multiplicity < 1:
            raise ValueError("Irrep multiplicities must be positive.")
        return multiplicity, Irrep(irrep)

    def __setattr__(self, name, value) -> None:
        raise AttributeError("Irreps metadata is immutable.")

    @property
    def dim(self) -> int:
        return sum(multiplicity * irrep.dim for multiplicity, irrep in self)

    @property
    def dof(self) -> int:
        return sum(multiplicity * irrep.dof for multiplicity, irrep in self)

    @property
    def num_irreps(self) -> int:
        return sum(multiplicity for multiplicity, _ in self)

    @property
    def lmax(self) -> int:
        return max((irrep.l for _, irrep in self), default=-1)

    def expanded(self) -> Tuple[Irrep, ...]:
        return tuple(irrep for multiplicity, irrep in self for _ in range(multiplicity))

    def expanded_slices(self) -> Tuple[slice, ...]:
        slices = []
        start = 0
        for irrep in self.expanded():
            stop = start + irrep.dim
            slices.append(slice(start, stop))
            start = stop
        return tuple(slices)

    def simplify(self) -> "Irreps":
        if not self._groups:
            return self
        groups = []
        multiplicity, irrep = self._groups[0]
        for next_multiplicity, next_irrep in self._groups[1:]:
            if next_irrep == irrep:
                multiplicity += next_multiplicity
            else:
                groups.append((multiplicity, irrep))
                multiplicity, irrep = next_multiplicity, next_irrep
        groups.append((multiplicity, irrep))
        return Irreps(groups)

    def sort(self) -> "Irreps":
        return Irreps(sorted(self._groups, key=lambda item: (item[1].l, item[1].p)))

    def regroup(self) -> "Irreps":
        groups = {}
        for multiplicity, irrep in self:
            groups[irrep] = groups.get(irrep, 0) + multiplicity
        return Irreps(
            [
                (multiplicity, irrep)
                for irrep, multiplicity in sorted(
                    groups.items(), key=lambda item: (item[0].l, item[0].p)
                )
            ]
        )

    def count(self, irrep: IrrepLike) -> int:
        irrep = Irrep(irrep)
        return sum(multiplicity for multiplicity, item in self if item == irrep)

    def D_from_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return the direct-sum Cartesian representation matrix.

        Parameters
        ----------
        matrix : torch.Tensor
            Orthogonal matrices with shape ``(..., 3, 3)``.

        Returns
        -------
        torch.Tensor
            Block-diagonal matrices with shape ``(..., self.dim, self.dim)``.
        """
        matrix = torch.as_tensor(matrix)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError("matrix must have trailing shape (3, 3).")
        if not matrix.is_floating_point():
            matrix = matrix.to(torch.get_default_dtype())
        output = matrix.new_zeros(matrix.shape[:-2] + (self.dim, self.dim))
        offset = 0
        for irrep in self.expanded():
            block = irrep.D_from_matrix(matrix)
            output[
                ..., offset : offset + irrep.dim, offset : offset + irrep.dim
            ] = block
            offset += irrep.dim
        return output

    def __iter__(self) -> Iterator[Tuple[int, Irrep]]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Irreps(self._groups[index])
        return self._groups[index]

    def __add__(self, other: IrrepsLike) -> "Irreps":
        return Irreps(self._groups + tuple(Irreps(other)))

    def __mul__(self, multiplicity: int) -> "Irreps":
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            return NotImplemented
        if multiplicity < 1:
            raise ValueError("Irreps can only be multiplied by a positive integer.")
        return Irreps([(multiplicity * count, irrep) for count, irrep in self])

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        try:
            return self._groups == tuple(Irreps(other))
        except (TypeError, ValueError):
            return False

    def __hash__(self) -> int:
        return hash(self._groups)

    def __str__(self) -> str:
        return "+".join(
            f"{'' if multiplicity == 1 else f'{multiplicity}x'}{irrep}"
            for multiplicity, irrep in self
        )

    def __repr__(self) -> str:
        return f"Irreps({str(self)!r})"

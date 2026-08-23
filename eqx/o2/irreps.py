################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence, Tuple, Union

import torch
from e3nn import o3

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
    """A real irreducible representation of O(2).

    ``0e`` and ``0o`` are the one-dimensional reflection-even and
    reflection-odd irreps. Every positive order is a two-dimensional real
    irrep denoted by ``m``, for example ``1m`` or ``3m``.
    """

    m: int
    p: int

    def __init__(
        self,
        m: Union[int, str, Tuple[int, Parity], "Irrep"],
        p: Optional[Parity] = None,
    ) -> None:
        if isinstance(m, Irrep):
            if p is not None:
                raise TypeError("Do not provide parity when copying an Irrep.")
            order, parity = m.m, m.p
        elif isinstance(m, str):
            if p is not None:
                raise TypeError("Do not provide parity with an O(2) irrep string.")
            name = m.strip()
            try:
                order = int(name[:-1])
                parity = {"e": 1, "o": -1, "m": 0}[name[-1]]
            except (IndexError, KeyError, ValueError):
                raise ValueError(
                    "O(2) irreps must be written as 0e, 0o, or <positive>m."
                ) from None
        elif isinstance(m, tuple):
            if p is not None or len(m) != 2:
                raise TypeError("An Irrep tuple must be (m, p).")
            order, parity = m
        else:
            if p is None:
                raise TypeError("Parity is required when constructing Irrep(m, p).")
            order = m
            parity = p

        if isinstance(parity, str):
            try:
                parity = {"e": 1, "o": -1, "m": 0}[parity]
            except KeyError:
                raise ValueError(
                    "O(2) parity must be -1/'o', 0/'m', or +1/'e'."
                ) from None
        if (
            not isinstance(parity, int)
            or isinstance(parity, bool)
            or parity not in (-1, 0, 1)
        ):
            raise ValueError("O(2) parity must be -1/'o', 0/'m', or +1/'e'.")

        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise ValueError("O(2) order m must be a non-negative integer.")
        if order == 0 and parity not in (-1, 1):
            raise ValueError("Order zero must be either 0e or 0o.")
        if order > 0 and parity != 0:
            raise ValueError("Every positive O(2) order must use the m irrep.")

        object.__setattr__(self, "m", order)
        object.__setattr__(self, "p", parity)

    @property
    def dim(self) -> int:
        """Representation dimension."""
        return 1 if self.m == 0 else 2

    def __iter__(self) -> Iterator[int]:
        yield self.m
        yield self.p

    def __str__(self) -> str:
        suffix = "e" if self.p == 1 else "o" if self.p == -1 else "m"
        return f"{self.m}{suffix}"

    def __repr__(self) -> str:
        return str(self)

    def __mul__(self, other):
        try:
            other = Irrep(other)
        except (TypeError, ValueError):
            return NotImplemented
        if self.m == 0 and other.m == 0:
            return (Irrep(0, self.p * other.p),)
        if self.m == 0:
            return (other,)
        if other.m == 0:
            return (self,)
        if self.m == other.m:
            return (Irrep("0e"), Irrep("0o"), Irrep(2 * self.m, 0))
        return (
            Irrep(abs(self.m - other.m), 0),
            Irrep(self.m + other.m, 0),
        )

    def D_from_angle(
        self,
        angle,
        reflected: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the real matrix for ``S**reflected @ R(angle)``."""
        if not isinstance(reflected, bool):
            raise TypeError("reflected must be a bool.")
        angle = torch.as_tensor(angle, dtype=dtype, device=device)
        if not angle.is_floating_point():
            angle = angle.to(dtype=torch.get_default_dtype())
        if self.m == 0:
            sign = -1.0 if reflected and self.p == -1 else 1.0
            return angle.new_ones(angle.shape + (1, 1)) * sign

        cosine = torch.cos(self.m * angle)
        sine = torch.sin(self.m * angle)
        matrix = torch.stack(
            (
                torch.stack((cosine, -sine), dim=-1),
                torch.stack((sine, cosine), dim=-1),
            ),
            dim=-2,
        )
        if reflected:
            matrix = matrix.clone()
            matrix[..., 1, :] *= -1
        return matrix


class Irreps:
    """An immutable direct sum of complete real O(2) irreps.

    Iteration yields ``(multiplicity, irrep)`` pairs. Use :meth:`expanded`
    when one entry per irrep copy is required.
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
                            f"Invalid O(2) irreps term: {term!r}."
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

    @staticmethod
    def common_multiplicity(irreps: o3.Irreps) -> int:
        """Return the multiplicity shared by all O(3) irrep groups."""
        assert isinstance(irreps, o3.Irreps)
        multiplicities = {multiplicity for multiplicity, _ in irreps}
        assert len(multiplicities) == 1
        return next(iter(multiplicities))

    @property
    def dim(self) -> int:
        return sum(multiplicity * irrep.dim for multiplicity, irrep in self)

    @property
    def num_irreps(self) -> int:
        return sum(multiplicity for multiplicity, _ in self)

    @property
    def m_max(self) -> int:
        return max((irrep.m for _, irrep in self), default=-1)

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
        """Combine adjacent groups carrying the same irrep."""
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
        parity_order = {1: 0, -1: 1, 0: 2}
        return Irreps(
            sorted(
                self._groups,
                key=lambda group: (group[1].m, parity_order[group[1].p]),
            )
        )

    def regroup(self) -> "Irreps":
        """Collect equal irreps and return them in canonical order."""
        counts = {}
        for multiplicity, irrep in self:
            counts[irrep] = counts.get(irrep, 0) + multiplicity
        return Irreps(
            [(multiplicity, irrep) for irrep, multiplicity in counts.items()]
        ).sort()

    def randn(
        self,
        *size: int,
        normalization: str = "component",
        requires_grad: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return a random tensor with ``-1`` replaced by ``self.dim``.

        ``normalization="component"`` samples independent standard-normal
        components. ``normalization="norm"`` normalizes every irrep copy to
        unit norm along the representation axis. Additional dimensions, such
        as the explicit channel dimension used by the O(2) layers, are kept
        independent.
        """
        if size.count(-1) != 1:
            raise ValueError("size must contain exactly one -1.")
        if normalization not in ("component", "norm"):
            raise ValueError("normalization must be 'component' or 'norm'.")

        representation_axis = size.index(-1)
        shape = (
            *size[:representation_axis],
            self.dim,
            *size[representation_axis + 1 :],
        )
        if normalization == "component":
            return torch.randn(
                shape,
                dtype=dtype,
                device=device,
                requires_grad=requires_grad,
            )

        output = torch.empty(
            shape,
            dtype=dtype,
            device=device,
            requires_grad=requires_grad,
        )
        with torch.no_grad():
            for irrep, block_slice in zip(self.expanded(), self.expanded_slices()):
                block_shape = list(shape)
                block_shape[representation_axis] = irrep.dim
                block = torch.randn(block_shape, dtype=dtype, device=device)
                block /= block.norm(
                    dim=representation_axis,
                    keepdim=True,
                )
                output.narrow(
                    representation_axis,
                    block_slice.start,
                    irrep.dim,
                ).copy_(block)
        return output

    def count(self, irrep: IrrepLike) -> int:
        irrep = Irrep(irrep)
        return sum(
            multiplicity for multiplicity, candidate in self if candidate == irrep
        )

    def D_from_angle(
        self,
        angle,
        reflected: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the block-diagonal representation matrix."""
        if not isinstance(reflected, bool):
            raise TypeError("reflected must be a bool.")
        angle = torch.as_tensor(angle, dtype=dtype, device=device)
        if not angle.is_floating_point():
            angle = angle.to(dtype=torch.get_default_dtype())
        output = angle.new_zeros(angle.shape + (self.dim, self.dim))
        for irrep, block_slice in zip(self.expanded(), self.expanded_slices()):
            output[..., block_slice, block_slice] = irrep.D_from_angle(
                angle,
                reflected,
            )
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
        return Irreps([(multiplicity * count, irrep) for count, irrep in self._groups])

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

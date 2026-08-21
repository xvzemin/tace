################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple, Union

import torch

Parity = Union[int, str]
IrrepLike = Union["Irrep", str, Tuple[int, Parity]]
IrrepsLike = Union[
    "Irreps",
    "Irrep",
    str,
    Sequence[Union[IrrepLike, Tuple[int, IrrepLike]]],
]


def _normalize_o2_parity(parity: Parity) -> int:
    if isinstance(parity, str):
        aliases = {"e": 1, "o": -1, "m": 0}
        if parity in aliases:
            return aliases[parity]
    if isinstance(parity, int) and not isinstance(parity, bool):
        if parity in (-1, 0, 1):
            return parity
    raise ValueError("O(2) parity must be -1/'o', 0/'m', or +1/'e'.")


def _normalize_o3_parity(parity: Parity) -> int:
    if isinstance(parity, str):
        aliases = {"e": 1, "o": -1}
        if parity in aliases:
            return aliases[parity]
    if isinstance(parity, int) and not isinstance(parity, bool):
        if parity in (-1, 1):
            return parity
    raise ValueError("O(3) parity must be -1/'o' or +1/'e'.")


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
            match = re.fullmatch(r"(\d+)([eom])", m.strip())
            if match is None:
                raise ValueError(
                    "O(2) irreps must be written as 0e, 0o, or <positive>m."
                )
            order = int(match.group(1))
            parity = _normalize_o2_parity(match.group(2))
        elif isinstance(m, tuple):
            if p is not None or len(m) != 2:
                raise TypeError("An Irrep tuple must be (m, p).")
            order, parity = m
            parity = _normalize_o2_parity(parity)
        else:
            if p is None:
                raise TypeError("Parity is required when constructing Irrep(m, p).")
            order = m
            parity = _normalize_o2_parity(p)

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

    @property
    def is_scalar(self) -> bool:
        return self.m == 0

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
            return tensor_product_irreps(self, other)
        except (TypeError, ValueError):
            return NotImplemented


def check_o2_irrep(irrep: IrrepLike) -> Irrep:
    return irrep if isinstance(irrep, Irrep) else Irrep(irrep)


def tensor_product_irreps(
    irrep1: IrrepLike,
    irrep2: IrrepLike,
) -> Tuple[Irrep, ...]:
    """Return the complete real O(2) irreps in ``irrep1 x irrep2``."""
    irrep1 = check_o2_irrep(irrep1)
    irrep2 = check_o2_irrep(irrep2)
    if irrep1.m == 0 and irrep2.m == 0:
        return (Irrep(0, irrep1.p * irrep2.p),)
    if irrep1.m == 0:
        return (irrep2,)
    if irrep2.m == 0:
        return (irrep1,)
    if irrep1.m == irrep2.m:
        return (Irrep("0e"), Irrep("0o"), Irrep(2 * irrep1.m, 0))
    return (
        Irrep(abs(irrep1.m - irrep2.m), 0),
        Irrep(irrep1.m + irrep2.m, 0),
    )


def _parse_irreps_string(irreps: str) -> List[Tuple[int, Irrep]]:
    compact = irreps.replace(" ", "")
    if not compact:
        return []

    groups = []
    for term in compact.split("+"):
        match = re.fullmatch(r"(?:(\d+)x)?(\d+[eom])", term)
        if match is None:
            raise ValueError(f"Invalid O(2) irreps term: {term!r}.")
        multiplicity = int(match.group(1) or 1)
        if multiplicity < 1:
            raise ValueError("O(2) multiplicities must be positive.")
        groups.append((multiplicity, Irrep(match.group(2))))
    return groups


def _coerce_group(item) -> Tuple[int, Irrep]:
    if isinstance(item, (Irrep, str)):
        return 1, check_o2_irrep(item)
    if isinstance(item, tuple) and len(item) == 2:
        multiplicity, irrep = item
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            raise TypeError("An Irreps group must be (multiplicity, irrep).")
        if multiplicity < 1:
            raise ValueError("O(2) multiplicities must be positive.")
        return multiplicity, check_o2_irrep(irrep)
    raise TypeError("Irreps entries must be irreps or (multiplicity, irrep).")


class Irreps:
    """An immutable direct sum of complete real O(2) irreps.

    Iteration yields ``(multiplicity, irrep)`` pairs. Use :meth:`expanded`
    when one entry per irrep copy is required.
    """

    __slots__ = ("_groups",)

    def __init__(self, irreps: IrrepsLike = "") -> None:
        if isinstance(irreps, Irreps):
            groups = irreps.groups
        elif isinstance(irreps, Irrep):
            groups = ((1, irreps),)
        elif isinstance(irreps, str):
            groups = tuple(_parse_irreps_string(irreps))
        elif (
            isinstance(irreps, tuple)
            and len(irreps) == 2
            and isinstance(irreps[0], int)
        ):
            groups = (_coerce_group(irreps),)
        elif isinstance(irreps, Sequence):
            groups = tuple(_coerce_group(item) for item in irreps)
        else:
            raise TypeError("Unsupported Irreps input.")
        object.__setattr__(self, "_groups", tuple(groups))

    def __setattr__(self, name, value) -> None:
        raise AttributeError("Irreps metadata is immutable.")

    @property
    def groups(self) -> Tuple[Tuple[int, Irrep], ...]:
        return self._groups

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

    def group_slices(self) -> Tuple[slice, ...]:
        slices = []
        start = 0
        for multiplicity, irrep in self:
            stop = start + multiplicity * irrep.dim
            slices.append(slice(start, stop))
            start = stop
        return tuple(slices)

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
        parity_order = {1: 0, -1: 1, 0: 2}
        return Irreps(
            sorted(
                self._groups,
                key=lambda group: (group[1].m, parity_order[group[1].p]),
            )
        )

    def regroup(self) -> "Irreps":
        counts = {}
        for multiplicity, irrep in self:
            counts[irrep] = counts.get(irrep, 0) + multiplicity
        return Irreps(
            [(multiplicity, irrep) for irrep, multiplicity in counts.items()]
        ).sort()

    def count(self, irrep: IrrepLike) -> int:
        irrep = check_o2_irrep(irrep)
        return sum(
            multiplicity for multiplicity, candidate in self if candidate == irrep
        )

    def __iter__(self) -> Iterator[Tuple[int, Irrep]]:
        return iter(self._groups)

    def __len__(self) -> int:
        return len(self._groups)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Irreps(self._groups[index])
        return self._groups[index]

    def __add__(self, other: IrrepsLike) -> "Irreps":
        return Irreps(self._groups + Irreps(other).groups)

    def __mul__(self, multiplicity: int) -> "Irreps":
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            return NotImplemented
        if multiplicity < 1:
            raise ValueError("Irreps can only be multiplied by a positive integer.")
        return Irreps([(multiplicity * count, irrep) for count, irrep in self._groups])

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        try:
            return self._groups == Irreps(other).groups
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


def check_o2_irreps(irreps: IrrepsLike) -> Irreps:
    return irreps if isinstance(irreps, Irreps) else Irreps(irreps)


def restrict_o3_irrep(
    l: int,
    p: Parity,
    multiplicity: int = 1,
) -> Irreps:
    """Restrict one real O(3) irrep ``(l, p)`` to the local O(2) subgroup."""
    if not isinstance(l, int) or isinstance(l, bool) or l < 0:
        raise ValueError("O(3) degree l must be a non-negative integer.")
    if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
        raise TypeError("Multiplicity must be an integer.")
    if multiplicity < 1:
        raise ValueError("Multiplicity must be positive.")
    parity = _normalize_o3_parity(p)
    zero_parity = parity * ((-1) ** l)
    groups = [(multiplicity, Irrep(0, zero_parity))]
    groups.extend((multiplicity, Irrep(m, 0)) for m in range(1, l + 1))
    return Irreps(groups)


def restrict_o3_irreps(
    irreps: Iterable[Tuple[int, int, Parity]],
) -> Irreps:
    """Restrict a sequence of ``(multiplicity, l, parity)`` entries."""
    groups = []
    for multiplicity, l, parity in irreps:
        groups.extend(restrict_o3_irrep(l, parity, multiplicity).groups)
    return Irreps(groups)


def _as_floating_angle(
    angle,
    dtype: Optional[torch.dtype],
    device: Optional[torch.device],
) -> torch.Tensor:
    angle = torch.as_tensor(angle, dtype=dtype, device=device)
    if not angle.is_floating_point():
        angle = angle.to(dtype=torch.get_default_dtype())
    return angle


def o2_representation(
    irrep: IrrepLike,
    angle,
    reflected: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the real O(2) matrix for ``S**reflected @ R(angle)``."""
    if not isinstance(reflected, bool):
        raise TypeError("reflected must be a bool.")
    irrep = check_o2_irrep(irrep)
    angle = _as_floating_angle(angle, dtype, device)
    if irrep.m == 0:
        sign = -1.0 if reflected and irrep.p == -1 else 1.0
        return (
            torch.ones(
                angle.shape + (1, 1),
                dtype=angle.dtype,
                device=angle.device,
            )
            * sign
        )

    cosine = torch.cos(irrep.m * angle)
    sine = torch.sin(irrep.m * angle)
    first_row = torch.stack((cosine, -sine), dim=-1)
    second_row = torch.stack((sine, cosine), dim=-1)
    matrix = torch.stack((first_row, second_row), dim=-2)
    if reflected:
        matrix = matrix.clone()
        matrix[..., 1, :] *= -1
    return matrix


def o2_irreps_representation(
    irreps: IrrepsLike,
    angle,
    reflected: bool = False,
    *,
    dtype: Optional[torch.dtype] = None,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return the block-diagonal representation matrix for an Irreps layout."""
    irreps = check_o2_irreps(irreps)
    angle = _as_floating_angle(angle, dtype, device)
    output = torch.zeros(
        angle.shape + (irreps.dim, irreps.dim),
        dtype=angle.dtype,
        device=angle.device,
    )
    for irrep, block_slice in zip(irreps.expanded(), irreps.expanded_slices()):
        output[..., block_slice, block_slice] = o2_representation(
            irrep,
            angle,
            reflected,
        )
    return output

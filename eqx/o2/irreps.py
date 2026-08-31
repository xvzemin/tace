################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass
from typing import Iterator, NamedTuple, Optional, Sequence, Tuple, Union

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

    def is_even_scalar(self) -> bool:
        """Return whether this is the reflection-even scalar irrep."""
        return self.m == 0 and self.p == 1

    def is_odd_scalar(self) -> bool:
        """Return whether this is the reflection-odd scalar irrep."""
        return self.m == 0 and self.p == -1

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


class _MulIr(NamedTuple):
    mul: int
    ir: Irrep

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim


class Irreps:
    """A direct sum of real O(2) irreps.

    Iteration yields ``(multiplicity, irrep)`` pairs. Use :meth:`expanded`
    when one entry per irrep copy is required.
    """

    __slots__ = ("_irreps",)

    def __init__(self, irreps: IrrepsLike = "") -> None:
        if isinstance(irreps, Irreps):
            irrep_list = tuple(irreps)
        elif isinstance(irreps, Irrep):
            irrep_list = ((1, irreps),)
        elif isinstance(irreps, str):
            irrep_list = []
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
                    irrep_list.append((multiplicity, irrep))
            irrep_list = tuple(irrep_list)
        elif (
            isinstance(irreps, tuple)
            and len(irreps) == 2
            and isinstance(irreps[0], int)
        ):
            irrep_list = (self._from_mul_ir(irreps),)
        elif isinstance(irreps, Sequence):
            irrep_list = tuple(self._from_mul_ir(item) for item in irreps)
        else:
            raise TypeError("Unsupported Irreps input.")
        object.__setattr__(
            self,
            "_irreps",
            tuple(self._from_mul_ir(item) for item in irrep_list),
        )

    @staticmethod
    def _from_mul_ir(item) -> _MulIr:
        if isinstance(item, (Irrep, str)):
            return _MulIr(1, Irrep(item))
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "Each Irreps entry must be an irrep or a (multiplicity, irrep) pair."
            )
        multiplicity, irrep = item
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            raise TypeError("Irrep multiplicity must be an integer.")
        if multiplicity < 1:
            raise ValueError("Irrep multiplicities must be positive.")
        return _MulIr(multiplicity, Irrep(irrep))

    def __setattr__(self, name, value) -> None:
        raise AttributeError("Irreps metadata is immutable.")

    @staticmethod
    def common_multiplicity(
        irreps: Union[o3.Irreps, "Irreps"],
    ) -> int:
        """Return the multiplicity shared by all irrep entries."""
        multiplicities = {mul for mul, ir in irreps}
        if not multiplicities:
            raise ValueError("Irreps must contain at least one entry.")
        if len(multiplicities) != 1:
            raise ValueError("Irreps must use one common multiplicity.")
        return next(iter(multiplicities))

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, ir in self)

    @property
    def mmax(self) -> int:
        return max((ir.m for mul, ir in self), default=-1)

    def expanded(self) -> Tuple[Irrep, ...]:
        return tuple(ir for mul, ir in self for _ in range(mul))

    def expanded_slices(self) -> Tuple[slice, ...]:
        slices = []
        start = 0
        for ir in self.expanded():
            stop = start + ir.dim
            slices.append(slice(start, stop))
            start = stop
        return tuple(slices)

    def slices(self) -> Tuple[slice, ...]:
        """Return one flattened representation slice per ``(mul, ir)``."""
        slices = []
        start = 0
        for mul_ir in self:
            stop = start + mul_ir.dim
            slices.append(slice(start, stop))
            start = stop
        return tuple(slices)

    def filter(
        self,
        keep=None,
        *,
        drop=None,
        mmax: Optional[int] = None,
    ) -> "Irreps":
        """Filter irreps by type, predicate, or maximum O(2) order."""
        specified = sum(value is not None for value in (keep, drop, mmax))
        if specified == 0:
            return self
        if specified > 1:
            raise ValueError("Specify only one of keep, drop, or mmax.")

        if mmax is not None:
            if not isinstance(mmax, int) or isinstance(mmax, bool):
                raise TypeError("mmax must be an integer.")
            if mmax < 0:
                raise ValueError("mmax must be non-negative.")
            return Irreps([(mul, ir) for mul, ir in self if ir.m <= mmax])

        selection = keep if keep is not None else drop
        if callable(selection):
            predicate = selection
        else:
            if isinstance(selection, str):
                selection = Irreps(selection)
            elif isinstance(selection, (Irrep, _MulIr)):
                selection = [selection]
            irrep_set = {
                item.ir if isinstance(item, _MulIr) else Irrep(item)
                for item in selection
            }
            predicate = lambda mul_ir: mul_ir.ir in irrep_set

        if keep is not None:
            return Irreps([mul_ir for mul_ir in self if predicate(mul_ir)])
        return Irreps([mul_ir for mul_ir in self if not predicate(mul_ir)])

    def simplify(self) -> "Irreps":
        """Combine adjacent entries carrying the same irrep."""
        if not self._irreps:
            return self
        irrep_list = []
        mul, ir = self._irreps[0]
        for next_mul, next_ir in self._irreps[1:]:
            if next_ir == ir:
                mul += next_mul
            else:
                irrep_list.append((mul, ir))
                mul, ir = next_mul, next_ir
        irrep_list.append((mul, ir))
        return Irreps(irrep_list)

    def sort(self) -> "Irreps":
        parity_order = {1: 0, -1: 1, 0: 2}
        return Irreps(
            sorted(
                self._irreps,
                key=lambda mul_ir: (
                    mul_ir.ir.m,
                    parity_order[mul_ir.ir.p],
                ),
            )
        )

    def regroup(self) -> "Irreps":
        """Collect equal irreps and return them in canonical order."""
        counts = {}
        for mul, ir in self:
            counts[ir] = counts.get(ir, 0) + mul
        return Irreps(
            [(mul, ir) for ir, mul in counts.items()]
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
            for ir, block_slice in zip(self.expanded(), self.expanded_slices()):
                block_shape = list(shape)
                block_shape[representation_axis] = ir.dim
                block = torch.randn(block_shape, dtype=dtype, device=device)
                block /= block.norm(
                    dim=representation_axis,
                    keepdim=True,
                )
                output.narrow(
                    representation_axis,
                    block_slice.start,
                    ir.dim,
                ).copy_(block)
        return output

    def count(self, irrep: IrrepLike) -> int:
        irrep = Irrep(irrep)
        return sum(
            mul for mul, ir in self if ir == irrep
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
        for ir, block_slice in zip(self.expanded(), self.expanded_slices()):
            output[..., block_slice, block_slice] = ir.D_from_angle(
                angle,
                reflected,
            )
        return output

    def __iter__(self) -> Iterator[_MulIr]:
        return iter(self._irreps)

    def __len__(self) -> int:
        return len(self._irreps)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return Irreps(self._irreps[index])
        return self._irreps[index]

    def __add__(self, other: IrrepsLike) -> "Irreps":
        return Irreps(self._irreps + tuple(Irreps(other)))

    def __mul__(self, multiplicity: int) -> "Irreps":
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            return NotImplemented
        if multiplicity < 1:
            raise ValueError("Irreps can only be multiplied by a positive integer.")
        return Irreps([(multiplicity * mul, ir) for mul, ir in self._irreps])

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        try:
            return self._irreps == tuple(Irreps(other))
        except (TypeError, ValueError):
            return False

    def __hash__(self) -> int:
        return hash(self._irreps)

    def __str__(self) -> str:
        return "+".join(
            f"{'' if mul == 1 else f'{mul}x'}{ir}"
            for mul, ir in self
        )

    def __repr__(self) -> str:
        return f"Irreps({str(self)!r})"

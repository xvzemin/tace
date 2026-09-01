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
    Sequence[Union[IrrepLike, Tuple[IrrepLike, int]]],
]


@dataclass(frozen=True, init=False)
class Irrep:
    """A real irreducible representation of O(2).

    Parameters
    ----------
    m : int, str, tuple, or Irrep
        Non-negative order, canonical name such as ``"0e"`` or ``"2m"``,
        ``(order, parity)`` pair, or an existing instance.
    p : {-1, 0, 1, "o", "m", "e"}, optional
        Reflection parity when ``m`` is supplied as an integer. Order zero
        uses ``+1`` or ``-1``; positive orders use ``0``.

    Notes
    -----
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
        """Return the real representation matrix of an O(2) transformation.

        Parameters
        ----------
        angle : array-like
            Rotation angle in radians.
        reflected : bool, optional
            Compose the rotation with the canonical reflection.
        dtype : torch.dtype, optional
            Requested floating-point dtype.
        device : torch.device, optional
            Requested output device.

        Returns
        -------
        torch.Tensor
            Matrix with shape ``angle.shape + (dim, dim)`` representing
            ``S**reflected @ R(angle)``.
        """
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


class _IrMul(NamedTuple):
    ir: Irrep
    mul: int

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim


class Irreps:
    """A direct sum of real O(2) irreps.

    Parameters
    ----------
    irreps : IrrepsLike, optional
        Representation specification. Accepted forms include canonical strings,
        individual irreps, ``(irrep, multiplicity)`` pairs, and sequences of
        those pairs.

    Notes
    -----
    Iteration yields ``(irrep, multiplicity)`` pairs. Use :meth:`expanded`
    when one entry per irrep copy is required. Feature slices use flattened
    ``ir_mul`` order.
    """

    __slots__ = ("_irreps",)

    def __init__(self, irreps: IrrepsLike = "") -> None:
        if isinstance(irreps, Irreps):
            irrep_list = tuple(irreps)
        elif isinstance(irreps, Irrep):
            irrep_list = ((irreps, 1),)
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
                    irrep_list.append((irrep, multiplicity))
            irrep_list = tuple(irrep_list)
        elif (
            isinstance(irreps, tuple)
            and len(irreps) == 2
            and isinstance(irreps[1], int)
        ):
            irrep_list = (self._from_ir_mul(irreps),)
        elif isinstance(irreps, Sequence):
            irrep_list = tuple(self._from_ir_mul(item) for item in irreps)
        else:
            raise TypeError("Unsupported Irreps input.")
        object.__setattr__(
            self,
            "_irreps",
            tuple(self._from_ir_mul(item) for item in irrep_list),
        )

    @staticmethod
    def _from_ir_mul(item) -> _IrMul:
        if isinstance(item, (Irrep, str)):
            return _IrMul(Irrep(item), 1)
        if not isinstance(item, tuple) or len(item) != 2:
            raise TypeError(
                "Each Irreps entry must be an irrep or an (irrep, multiplicity) pair."
            )
        irrep, multiplicity = item
        if not isinstance(multiplicity, int) or isinstance(multiplicity, bool):
            raise TypeError("Irrep multiplicity must be an integer.")
        if multiplicity < 1:
            raise ValueError("Irrep multiplicities must be positive.")
        return _IrMul(Irrep(irrep), multiplicity)

    def __setattr__(self, name, value) -> None:
        raise AttributeError("Irreps metadata is immutable.")

    @staticmethod
    def common_multiplicity(
        irreps: Union[o3.Irreps, "Irreps"],
    ) -> int:
        """Return the multiplicity shared by all irrep entries."""
        if isinstance(irreps, o3.Irreps):
            multiplicities = {entry.mul for entry in irreps}
        else:
            multiplicities = {mul for _, mul in irreps}
        if not multiplicities:
            raise ValueError("Irreps must contain at least one entry.")
        if len(multiplicities) != 1:
            raise ValueError("Irreps must use one common multiplicity.")
        return next(iter(multiplicities))

    @property
    def dim(self) -> int:
        return sum(ir.dim * mul for ir, mul in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for _, mul in self)

    @property
    def mmax(self) -> int:
        return max((ir.m for ir, _ in self), default=-1)

    def expanded(self) -> Tuple[Irrep, ...]:
        return tuple(ir for ir, mul in self for _ in range(mul))

    def slices(self) -> Tuple[slice, ...]:
        """Return one ``ir_mul`` feature slice per ``(ir, mul)`` entry."""
        slices = []
        start = 0
        for ir_mul in self:
            stop = start + ir_mul.dim
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
        """Filter entries by type, predicate, or maximum order.

        Parameters
        ----------
        keep : optional
            Irreps, iterable of irrep types, or predicate selecting entries to
            retain.
        drop : optional
            Irreps, iterable of irrep types, or predicate selecting entries to
            remove.
        mmax : int, optional
            Retain entries whose order is at most this value.

        Returns
        -------
        Irreps
            Filtered representation. Exactly one selection argument may be
            supplied.
        """
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
            return Irreps([(ir, mul) for ir, mul in self if ir.m <= mmax])

        selection = keep if keep is not None else drop
        if callable(selection):
            predicate = selection
        else:
            if isinstance(selection, str):
                selection = Irreps(selection)
            elif isinstance(selection, (Irrep, _IrMul)):
                selection = [selection]
            irrep_set = {
                item.ir if isinstance(item, _IrMul) else Irrep(item)
                for item in selection
            }

            def predicate(ir_mul):
                return ir_mul.ir in irrep_set

        if keep is not None:
            return Irreps([ir_mul for ir_mul in self if predicate(ir_mul)])
        return Irreps([ir_mul for ir_mul in self if not predicate(ir_mul)])

    def simplify(self) -> "Irreps":
        """Combine adjacent entries carrying the same irrep."""
        if not self._irreps:
            return self
        irrep_list = []
        ir, mul = self._irreps[0]
        for next_ir, next_mul in self._irreps[1:]:
            if next_ir == ir:
                mul += next_mul
            else:
                irrep_list.append((ir, mul))
                ir, mul = next_ir, next_mul
        irrep_list.append((ir, mul))
        return Irreps(irrep_list)

    def sort(self) -> "Irreps":
        parity_order = {1: 0, -1: 1, 0: 2}
        return Irreps(
            sorted(
                self._irreps,
                key=lambda ir_mul: (
                    ir_mul.ir.m,
                    parity_order[ir_mul.ir.p],
                ),
            )
        )

    def regroup(self) -> "Irreps":
        """Collect equal irreps and return them in canonical order."""
        counts = {}
        for ir, mul in self:
            counts[ir] = counts.get(ir, 0) + mul
        return Irreps(list(counts.items())).sort()

    def randn(
        self,
        *size: int,
        normalization: str = "component",
        requires_grad: bool = False,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return a random tensor with ``-1`` replaced by ``self.dim``.

        Parameters
        ----------
        *size : int
            Requested shape containing exactly one ``-1`` representation axis.
        normalization : {"component", "norm"}, optional
            ``"component"`` samples independent standard-normal components.
            ``"norm"`` normalizes every irrep copy to unit norm.
        requires_grad : bool, optional
            Enable gradient tracking on the returned tensor.
        dtype : torch.dtype, optional
            Requested dtype.
        device : torch.device, optional
            Requested device.

        Returns
        -------
        torch.Tensor
            Random features with ``-1`` replaced by :attr:`dim`. Within each
            entry, features use flattened ``ir_mul`` order.
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

        output = torch.empty(shape, dtype=dtype, device=device)
        output_last = output.movedim(representation_axis, -1)
        with torch.no_grad():
            for (ir, mul), ir_slice in zip(self, self.slices()):
                values = torch.randn(
                    *output_last.shape[:-1],
                    ir.dim,
                    mul,
                    dtype=dtype,
                    device=device,
                )
                values = values / values.norm(dim=-2, keepdim=True)
                output_last[..., ir_slice].copy_(
                    values.reshape(*output_last.shape[:-1], mul * ir.dim)
                )
        return output.requires_grad_(requires_grad)

    def count(self, irrep: IrrepLike) -> int:
        irrep = Irrep(irrep)
        return sum(mul for ir, mul in self if ir == irrep)

    def D_from_angle(
        self,
        angle,
        reflected: bool = False,
        *,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Return the direct-sum representation matrix.

        Parameters
        ----------
        angle : array-like
            Rotation angle in radians.
        reflected : bool, optional
            Compose the rotation with the canonical reflection.
        dtype : torch.dtype, optional
            Requested floating-point dtype.
        device : torch.device, optional
            Requested output device.

        Returns
        -------
        torch.Tensor
            Block-diagonal matrix with shape
            ``angle.shape + (self.dim, self.dim)``.
        """
        if not isinstance(reflected, bool):
            raise TypeError("reflected must be a bool.")
        angle = torch.as_tensor(angle, dtype=dtype, device=device)
        if not angle.is_floating_point():
            angle = angle.to(dtype=torch.get_default_dtype())
        output = angle.new_zeros(angle.shape + (self.dim, self.dim))
        for (ir, mul), ir_slice in zip(self, self.slices()):
            matrix = ir.D_from_angle(angle, reflected)
            identity = torch.eye(mul, dtype=matrix.dtype, device=matrix.device)
            matrix = (
                matrix[..., :, None, :, None] * identity[..., None, :, None, :]
            ).reshape(*angle.shape, mul * ir.dim, mul * ir.dim)
            output[..., ir_slice, ir_slice] = matrix
        return output

    def __iter__(self) -> Iterator[_IrMul]:
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
        return Irreps([(ir, multiplicity * mul) for ir, mul in self._irreps])

    __rmul__ = __mul__

    def __eq__(self, other) -> bool:
        try:
            return self._irreps == tuple(Irreps(other))
        except (TypeError, ValueError):
            return False

    def __hash__(self) -> int:
        return hash(self._irreps)

    def __str__(self) -> str:
        return "+".join(f"{'' if mul == 1 else f'{mul}x'}{ir}" for ir, mul in self)

    def __repr__(self) -> str:
        return f"Irreps({str(self)!r})"

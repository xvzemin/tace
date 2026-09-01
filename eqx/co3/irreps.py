################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass
from typing import Iterator, NamedTuple, Optional, Sequence, Tuple, Union

import torch

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
    Degree ``l`` is stored in its ambient rank-``l`` Cartesian space of
    dimension ``3**l``. Its symmetric traceless subspace contains
    ``2*l + 1`` independent components.
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
                    "O(3) irreps must be written as <degree>e or <degree>o."
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
        """Number of independent symmetric traceless components."""
        return 2 * self.l + 1

    @property
    def pseudo(self) -> bool:
        """Whether the Cartesian realization carries a determinant factor."""
        return self.p != (-1) ** self.l

    def is_even_scalar(self) -> bool:
        """Return whether this is the inversion-even scalar irrep."""
        return self.l == 0 and self.p == 1

    def is_odd_scalar(self) -> bool:
        """Return whether this is the inversion-odd scalar irrep."""
        return self.l == 0 and self.p == -1

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
            Representation matrices with shape ``(..., dim, dim)``.
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


class _IrMul(NamedTuple):
    ir: Irrep
    mul: int

    @property
    def dim(self) -> int:
        return self.ir.dim * self.mul


class Irreps:
    """An immutable direct sum of Cartesian O(3) irreps.

    Parameters
    ----------
    irreps : IrrepsLike, optional
        Canonical string, individual irrep, ``(irrep, multiplicity)`` pair,
        or a sequence of those pairs.

    Notes
    -----
    Iteration yields ``(ir, mul)`` entries. Every feature entry uses flattened
    ``ir_mul`` order and can be viewed as ``(..., ir.dim, mul)``.
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
                            f"Invalid O(3) irreps term: {term!r}."
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

    @property
    def dim(self) -> int:
        return sum(ir.dim * mul for ir, mul in self)

    @property
    def dof(self) -> int:
        return sum(ir.dof * mul for ir, mul in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for _, mul in self)

    @property
    def lmax(self) -> int:
        return max((ir.l for ir, _ in self), default=-1)

    def expanded(self) -> Tuple[Irrep, ...]:
        return tuple(ir for ir, mul in self for _ in range(mul))

    def slices(self) -> Tuple[slice, ...]:
        """Return one flattened ``ir_mul`` slice per entry."""
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
        lmax: Optional[int] = None,
    ) -> "Irreps":
        """Filter entries by type, predicate, or maximum degree.

        Parameters
        ----------
        keep : optional
            Irreps, iterable of irrep types, or predicate selecting entries to
            retain.
        drop : optional
            Irreps, iterable of irrep types, or predicate selecting entries to
            remove.
        lmax : int, optional
            Retain entries whose tensor degree is at most this value.

        Returns
        -------
        Irreps
            Filtered representation. At most one selection argument may be
            supplied.
        """
        specified = sum(value is not None for value in (keep, drop, lmax))
        if specified == 0:
            return self
        if specified > 1:
            raise ValueError("Specify only one of keep, drop, or lmax.")

        if lmax is not None:
            if not isinstance(lmax, int) or isinstance(lmax, bool):
                raise TypeError("lmax must be an integer.")
            if lmax < 0:
                raise ValueError("lmax must be non-negative.")
            return Irreps([(ir, mul) for ir, mul in self if ir.l <= lmax])

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
        parity_order = {1: 0, -1: 1}
        return Irreps(
            sorted(
                self._irreps,
                key=lambda ir_mul: (ir_mul.ir.l, parity_order[ir_mul.ir.p]),
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
        """Return random symmetric traceless features.

        Parameters
        ----------
        *size : int
            Requested shape containing exactly one ``-1`` representation axis.
        normalization : {"component", "norm"}, optional
            ``"component"`` gives independent irreducible coordinates unit
            variance. ``"norm"`` gives every irrep copy unit norm.
        requires_grad : bool, optional
            Enable gradient tracking on the returned tensor.
        dtype : torch.dtype, optional
            Requested dtype.
        device : torch.device, optional
            Requested device.

        Returns
        -------
        torch.Tensor
            Tensor with ``-1`` replaced by :attr:`dim`, using flattened
            ``ir_mul`` layout.
        """
        if size.count(-1) != 1:
            raise ValueError("size must contain exactly one -1.")
        if normalization not in ("component", "norm"):
            raise ValueError("normalization must be 'component' or 'norm'.")

        from .projector import project

        representation_axis = size.index(-1)
        shape = (
            *size[:representation_axis],
            self.dim,
            *size[representation_axis + 1 :],
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
                values = project(values, ir.l)
                if normalization == "norm":
                    values = values / values.norm(dim=-2, keepdim=True)
                output_last[..., ir_slice].copy_(
                    values.reshape(*output_last.shape[:-1], ir.dim * mul)
                )
        return output.requires_grad_(requires_grad)

    def count(self, irrep: IrrepLike) -> int:
        irrep = Irrep(irrep)
        return sum(mul for ir, mul in self if ir == irrep)

    def D_from_matrix(self, matrix: torch.Tensor) -> torch.Tensor:
        """Return the direct-sum matrix in flattened ``ir_mul`` order."""
        matrix = torch.as_tensor(matrix)
        if matrix.shape[-2:] != (3, 3):
            raise ValueError("matrix must have trailing shape (3, 3).")
        if not matrix.is_floating_point():
            matrix = matrix.to(torch.get_default_dtype())
        output = matrix.new_zeros(matrix.shape[:-2] + (self.dim, self.dim))
        for (ir, mul), ir_slice in zip(self, self.slices()):
            block = ir.D_from_matrix(matrix)
            identity = torch.eye(mul, dtype=block.dtype, device=block.device)
            block = (
                block[..., :, None, :, None] * identity[..., None, :, None, :]
            ).reshape(*matrix.shape[:-2], ir.dim * mul, ir.dim * mul)
            output[..., ir_slice, ir_slice] = block
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
        return Irreps([(ir, multiplicity * mul) for ir, mul in self])

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

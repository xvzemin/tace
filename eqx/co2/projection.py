"""Plane projection and analytic detracing in Cartesian storage."""

from fractions import Fraction
from functools import lru_cache

import torch

from ..co3.symmetric import symmetric_basis


def plane_projector(direction):
    """Return the orthogonal projector perpendicular to a unit direction.

    Parameters
    ----------
    direction : torch.Tensor
        Unit vectors of shape (..., 3). Normalization is not performed.

    Returns
    -------
    torch.Tensor
        Matrices I - n n.T of shape (..., 3, 3).
    """
    if direction.shape[-1] != 3:
        raise ValueError("Directions must have trailing size 3.")
    identity = torch.eye(3, dtype=direction.dtype, device=direction.device)
    return identity - direction.unsqueeze(-1) * direction.unsqueeze(-2)


@lru_cache(maxsize=None)
def detracing_coefficients(rank):
    """Return coefficients of the averaged planar STF projection."""
    if not isinstance(rank, int) or rank < 0:
        raise ValueError("rank must be a non-negative integer.")
    value = Fraction(1)
    result = [1.0]
    for k in range(1, rank // 2 + 1):
        value *= -Fraction((rank - 2 * k + 2) * (rank - 2 * k + 1), 4 * k * (rank - k))
        result.append(float(value))
    return tuple(result)


@lru_cache(maxsize=None)
def detracing_basis(rank):
    """Return CPU float64 embedding and single-trace coefficients."""
    bases = []
    for degree in range(rank, -1, -2):
        data = symmetric_basis(degree)
        basis = torch.eye(data["norm"].numel(), dtype=torch.float64, device="cpu")
        bases.append(basis[data["indices"]] / data["norm"])
    data = {"basis": bases[0]}
    for k, (upper, lower) in enumerate(zip(bases[:-1], bases[1:])):
        degree = rank - 2 * k
        data[f"trace_{k}"] = torch.einsum(
            "abic,ij->abjc", upper.reshape(3, 3, 3 ** (degree - 2), -1), lower
        )
    return data


class PlaneProjector(torch.nn.Module):
    """Project every Cartesian index into a plane.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank. Tensors need not be symmetric.
    """

    def __init__(self, rank):
        super().__init__()
        if not isinstance(rank, int) or rank < 0:
            raise ValueError("rank must be a non-negative integer.")
        self.rank = rank
        self.dim = 3**rank

    def forward(self, tensor, plane):
        """Project (..., 3**rank) tensors using (..., 3, 3) plane matrices.

        Leading dimensions broadcast. For (edge, channel, 3**rank) tensors,
        pass plane matrices of shape (edge, 1, 3, 3).
        """
        if tensor.shape[-1] != self.dim or plane.shape[-2:] != (3, 3):
            raise ValueError("Tensor or plane has an incorrect trailing shape.")
        if self.rank == 0:
            shape = torch.broadcast_shapes(tensor.shape[:-1], plane.shape[:-2])
            return tensor.expand(*shape, 1)
        for axis in range(self.rank):
            tensor = torch.einsum(
                "...ij,...ajb->...aib",
                plane,
                tensor.unflatten(-1, (3**axis, 3, 3 ** (self.rank - axis - 1))),
            ).flatten(-3)
        return tensor

    def extra_repr(self):
        return f"rank={self.rank}"


class PlanarDetracer(torch.nn.Module):
    """Remove traces of symmetric tensors already projected into a plane.

    Parameters
    ----------
    rank : int
        Non-negative Cartesian tensor rank.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to torch.get_default_dtype().
    device : torch.device or str, optional
        Buffer device. Defaults to CPU.

    Notes
    -----
    Inputs must be symmetric and transverse to the plane normal. This
    operator does not project indices or symmetrize arbitrary inputs.
    The explicit matrix uses 3**(2*rank) entries per plane. Its coefficients
    are those of the two-dimensional symmetric trace-free projection [#planar-stf]_.

    References
    ----------
    .. [#planar-stf] V. T. Toth and S. G. Turyshev, "Efficient trace-free decomposition
       of symmetric tensors of arbitrary rank", 2022, Eq. (38).
       https://arxiv.org/abs/2109.11743
    """

    def __init__(self, rank, *, dtype=None, device=None):
        super().__init__()
        self.coefficients = detracing_coefficients(rank)
        if dtype is not None and not dtype.is_floating_point:
            raise TypeError("dtype must be a real floating-point dtype.")
        self.rank = rank
        self.dim = 3**rank
        device = "cpu" if device is None else device
        self.register_buffer(
            "identity",
            torch.eye(
                self.dim, dtype=dtype or torch.get_default_dtype(), device=device
            ),
            persistent=False,
        )
        if rank >= 2:
            for name, value in detracing_basis(rank).items():
                self.register_buffer(
                    name, value.to(self.identity).clone(), persistent=False
                )
            self.register_buffer(
                "trace_identity",
                torch.eye(
                    self.basis.shape[1], dtype=self.identity.dtype, device=device
                ),
                persistent=False,
            )

    def matrix(self, plane):
        """Return (..., 3**rank, 3**rank) detracing matrices for (..., 3, 3) planes."""
        if plane.shape[-2:] != (3, 3):
            raise ValueError("Plane matrices must have trailing shape (3, 3).")
        if self.rank < 2:
            return self.identity.expand(*plane.shape[:-2], self.dim, self.dim)
        trace = self.trace_identity
        correction = torch.zeros_like(self.trace_identity)
        for k, coefficient in enumerate(self.coefficients[1:]):
            contraction = torch.einsum(
                "...ab,abij->...ij", plane, getattr(self, f"trace_{k}")
            )
            trace = contraction @ trace
            correction = correction + coefficient * (trace.transpose(-1, -2) @ trace)
        return self.identity + self.basis @ correction @ self.basis.T

    def forward(self, tensor, plane):
        """Detrace (..., 3**rank) tensors using broadcastable (..., 3, 3) planes."""
        if tensor.shape[-1] != self.dim:
            raise ValueError("Tensor has an incorrect trailing size.")
        return (self.matrix(plane) @ tensor.unsqueeze(-1)).squeeze(-1)

    def extra_repr(self):
        return f"rank={self.rank}"

    def _apply(self, fn, recurse=True):
        dtype = self.identity.dtype
        super()._apply(fn, recurse)
        if (
            self.rank >= 2
            and torch.finfo(self.identity.dtype).eps < torch.finfo(dtype).eps
        ):
            for name, value in detracing_basis(self.rank).items():
                getattr(self, name).copy_(value.to(self.identity))
        return self

"""Two-dimensional Cartesian harmonic polynomials."""

import math

import torch

from .. import o2
from .basis import ChangeOfBasis
from .irreps import Irrep, Irreps


class CartesianHarmonics(torch.nn.Module):
    """Evaluate STF harmonics of a two-dimensional vector.

    Parameters
    ----------
    m : int or sequence of int
        Requested non-negative orders, in output order.
    normalize : bool, optional
        Normalize the input vector. False evaluates homogeneous polynomials.
    normalization : {"component", "norm", "integral"}, optional
        Normalization in orthonormal circular coordinates.
    time_reversal : bool, optional
        Assign odd time parity to the input vector.
    """

    def __init__(
        self, m, normalize=False, normalization="component", time_reversal=False
    ):
        super().__init__()
        orders = [m] if isinstance(m, int) else list(m)
        if any(not isinstance(order, int) or order < 0 for order in orders):
            raise ValueError("Harmonic orders must be non-negative integers.")
        if normalization not in ("component", "norm", "integral"):
            raise ValueError("Unknown harmonic normalization.")
        self.orders = tuple(orders)
        self.normalize, self.normalization = normalize, normalization
        self.irreps_in = Irreps([(1, Irrep(1, 0, -1 if time_reversal else 1))])
        self.irreps_out = Irreps(
            [
                (
                    1,
                    Irrep(
                        order, 0 if order else 1, (-1 if time_reversal else 1) ** order
                    ),
                )
                for order in orders
            ]
        )
        self.to_cartesian = ChangeOfBasis(self.irreps_out)
        self.harmonics = o2.CircularHarmonics(
            max(orders, default=0), normalize=normalize, time_reversal=time_reversal
        )

    def forward(self, vectors):
        """Evaluate vectors with shape (..., 2)."""
        values = self.harmonics(vectors)
        result = []
        for m in self.orders:
            value = values[..., :1] if m == 0 else values[..., 2 * m - 1 : 2 * m + 1]
            scale = 1.0 if self.normalization == "norm" or m == 0 else math.sqrt(2)
            if self.normalization == "integral":
                scale /= math.sqrt(2 * math.pi)
            result.append(value * scale)
        return self.to_cartesian(torch.cat(result, -1) if result else values[..., :0])

    def extra_repr(self):
        return f"{self.irreps_out}, normalize={self.normalize}, normalization={self.normalization}"

"""Two-dimensional Cartesian harmonic polynomials."""

import torch

from equivariantx import o2

from ..ops.precision import validate_module_precision
from ..ops.torch.co2 import cartesian_harmonics as torch_kernel
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

    backend : {"torch", "cuda"}, optional
        Execution backend. CPU tensors use Torch.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> layer = co2.CartesianHarmonics(2, normalize=True)
    >>> layer(torch.randn(4, 2)).shape
    torch.Size([4, 4])
    """

    def __init__(
        self,
        m,
        normalize=False,
        normalization="component",
        time_reversal=False,
        *,
        backend="torch",
    ):
        super().__init__()
        if backend not in ("torch", "cuda"):
            raise ValueError("backend must be torch or cuda.")
        self.backend = backend
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

        indices, scales = torch_kernel.layout(self.orders, normalization)
        self.register_buffer(
            "harmonic_indices",
            torch.tensor(indices, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "harmonic_scales",
            torch.tensor(scales, dtype=torch.get_default_dtype()),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        _, scales = torch_kernel.layout(self.orders, self.normalization)
        if hasattr(self, "harmonic_scales"):
            self.harmonic_scales = self.harmonic_scales.new_tensor(scales)
        return self

    def forward(self, vectors):
        """Evaluate vectors with shape (..., 2)."""
        validate_module_precision(self, (vectors,))
        if vectors.ndim < 1 or vectors.shape[-1] != 2:
            raise ValueError("Vectors must have trailing dimension two.")
        if getattr(self, "backend", "torch") == "cuda" and vectors.is_cuda:
            from ..ops.cuda.co2 import cartesian_harmonics as cuda_kernel

            return cuda_kernel.evaluate(self, vectors)
        return torch_kernel.evaluate(self, vectors)

    def extra_repr(self):
        return f"{self.irreps_out}, normalize={self.normalize}, normalization={self.normalization}"

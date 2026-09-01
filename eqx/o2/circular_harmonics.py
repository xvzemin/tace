################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch

from .irreps import Irrep, Irreps


def circular_harmonics(
    input: torch.Tensor,
    mmax: int,
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Evaluate real circular harmonics through order ``mmax``.

    Args:
        input: Two-dimensional vectors with shape ``(..., 2)``.
        mmax: Largest circular order to return.
        normalize: Normalize each nonzero input vector before constructing the
            harmonics. If ``False``, order ``m`` is homogeneous of degree
            ``m`` in the input vector.

    Returns:
        A tensor with shape ``(..., 1 + 2 * mmax)`` in contiguous order from
        ``0e`` through ``mmax``. Every positive-order block uses
        the real ``(cos(m theta), sin(m theta))`` component order.
    """
    if not isinstance(mmax, int):
        raise TypeError("mmax must be an integer.")
    if mmax < 0:
        raise ValueError("mmax must be non-negative.")
    if input.ndim < 1 or input.shape[-1] != 2:
        raise ValueError(
            "circular_harmonics input must have trailing dimension 2; "
            f"got {tuple(input.shape)}."
        )

    if normalize:
        input = torch.nn.functional.normalize(input, dim=-1)
    real = input[..., 0]
    imaginary = input[..., 1]
    outputs = [torch.ones_like(real).unsqueeze(-1)]

    order_real = torch.ones_like(real)
    order_imaginary = torch.zeros_like(imaginary)
    for _ in range(1, mmax + 1):
        next_real = order_real * real - order_imaginary * imaginary
        next_imaginary = order_real * imaginary + order_imaginary * real
        order_real = next_real
        order_imaginary = next_imaginary
        outputs.append(torch.stack((order_real, order_imaginary), dim=-1))
    return torch.cat(outputs, dim=-1)


class CircularHarmonics(torch.nn.Module):
    """Construct real O(2) circular harmonics from two-dimensional vectors.

    The output follows contiguous order from ``0e`` through ``mmax``. With
    ``normalize=True`` it depends only on direction. With ``normalize=False``
    it returns the corresponding homogeneous circular solid harmonics.
    """

    def __init__(self, mmax: int, *, normalize: bool = True) -> None:
        super().__init__()
        if not isinstance(mmax, int):
            raise TypeError("mmax must be an integer.")
        if mmax < 0:
            raise ValueError("mmax must be non-negative.")
        self.mmax = mmax
        self.normalize = bool(normalize)
        self.irreps_in = Irreps("1m")
        self.irreps_out = Irreps(
            [(1, Irrep("0e"))] + [(1, Irrep(order, 0)) for order in range(1, mmax + 1)]
        )

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return circular_harmonics(
            input,
            self.mmax,
            normalize=self.normalize,
        )

    def extra_repr(self) -> str:
        return f"mmax={self.mmax}, normalize={self.normalize}"

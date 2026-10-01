"""Fourier transforms on the circle."""

import math

import torch

__all__ = ["S1Grid"]


class S1Grid(torch.nn.Module):
    """Transform real circular coefficients to a uniform angular grid.

    Parameters
    ----------
    mmax : int
        Maximum circular order. Coefficients are ordered as
        ``0, +1, -1, ..., +mmax, -mmax``.
    resolution : int, optional
        Number of angular samples, at least ``2 * mmax + 1``.
        Defaults to ``4 * mmax + 1``.
    normalization : {"component", "norm", "integral"}, optional
        ``"component"`` gives equal grid variance per order for unit-variance
        coefficients. ``"norm"`` assumes unit expected squared norm per
        order. ``"integral"`` uses harmonics orthonormal under ``d alpha``.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to the default floating-point dtype.
    device : torch.device or str, optional
        Buffer device. Defaults to the default device.

    Attributes
    ----------
    grid : torch.Tensor
        Angles of shape ``(resolution,)``, in radians.
    weights : torch.Tensor
        Quadrature weights of shape ``(resolution,)``, summing to one.

    Notes
    -----
    Input shape is ``(..., 2 * mmax + 1)`` and grid shape is
    ``(..., resolution)``. Channels occupy a leading dimension. The grid
    weights sum to one. Constants are constructed in float64.

    The linear round trip is exact up to roundoff. A degree-d polynomial
    followed by projection to ``mmax`` is exact when
    ``resolution > (d + 1) * mmax``. General nonlinearities require a
    convergence check as the resolution increases.
    """

    def __init__(
        self,
        mmax,
        resolution=None,
        *,
        normalization="component",
        dtype=None,
        device=None,
    ):
        super().__init__()
        if not isinstance(mmax, int) or mmax < 0:
            raise ValueError("mmax must be a non-negative integer.")
        resolution = 4 * mmax + 1 if resolution is None else resolution
        if not isinstance(resolution, int) or resolution < 2 * mmax + 1:
            raise ValueError("resolution must be an integer at least 2 * mmax + 1.")
        if normalization not in ("component", "norm", "integral"):
            raise ValueError("normalization must be component, norm, or integral.")
        self.mmax = mmax
        self.resolution = resolution
        self.normalization = normalization
        alpha = torch.arange(resolution, dtype=torch.float64, device="cpu")
        alpha = alpha * (2 * math.pi / resolution)
        orders = torch.arange(1, mmax + 1, dtype=torch.float64, device="cpu")
        angles = alpha[:, None] * orders
        basis = torch.cat(
            (
                torch.ones_like(alpha[:, None]),
                math.sqrt(2)
                * torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(1),
            ),
            dim=-1,
        )
        scale = torch.ones(2 * mmax + 1, dtype=torch.float64, device="cpu")
        if normalization == "component":
            scale[1:] /= math.sqrt(2)
            scale /= math.sqrt(mmax + 1)
        elif normalization == "norm":
            scale /= math.sqrt(mmax + 1)
        else:
            scale /= math.sqrt(2 * math.pi)
        weights = torch.full_like(alpha, 1 / resolution)
        self.dim = 2 * mmax + 1
        self.grid_shape = (resolution,)
        self._constants = dict(
            grid=alpha,
            weights=weights,
            synthesis=basis * scale,
            analysis=(basis * (weights[:, None] / scale)).T,
        )
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.get_default_device() if device is None else device
        for name, value in self._constants.items():
            self.register_buffer(
                name, value.to(dtype=dtype, device=device, copy=True), persistent=False
            )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, value in self._constants.items():
            self._buffers[name] = value.to(self._buffers[name], copy=True)
        return self

    def forward(self, features):
        """Evaluate circular coefficients on the grid.

        Parameters
        ----------
        features : torch.Tensor
            Coefficients of shape ``(..., 2 * mmax + 1)``.

        Returns
        -------
        torch.Tensor
            Grid values of shape ``(..., resolution)``.
        """
        return self.to_grid(features)

    def to_grid(self, features):
        """Evaluate a band-limited signal on the circle.

        Parameters
        ----------
        features : torch.Tensor
            Coefficients of shape ``(..., 2 * mmax + 1)``. Leading dimensions
            may include independent batch and channel axes.

        Returns
        -------
        torch.Tensor
            Grid values of shape ``(..., resolution)``.
        """
        if features.ndim < 1 or features.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} coefficients on the last axis.")
        return features @ self.synthesis.T

    def from_grid(self, features):
        """Project grid values onto circular coefficients.

        Parameters
        ----------
        features : torch.Tensor
            Grid values of shape ``(..., resolution)``.

        Returns
        -------
        torch.Tensor
            Coefficients of shape ``(..., 2 * mmax + 1)``.
        """
        if features.ndim < 1 or features.shape[-1] != self.resolution:
            raise ValueError(
                f"Expected {self.resolution} grid values on the last axis."
            )
        return features @ self.analysis.T

    def extra_repr(self):
        return (
            f"mmax={self.mmax}, resolution={self.resolution}, "
            f"normalization={self.normalization}"
        )

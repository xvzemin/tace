"""Polynomial approximations of scalar activation functions."""

import math

import torch
from torch.nn import functional


class PolynomialActivation(torch.nn.Module):
    """Approximate an activation by a fixed Chebyshev polynomial.

    Parameters
    ----------
    act : {"silu", "gelu", "relu", "tanh", "sigmoid", "softplus"}, optional
        Activation to approximate. Defaults to ``"silu"``.
    degree : int, optional
        Maximum polynomial degree. Defaults to eight.
    bound : float, optional
        Positive endpoint of the approximation interval ``[-bound, bound]``.
        Defaults to three. Inputs are not clipped to this interval.
    dtype : torch.dtype, optional
        Coefficient dtype. Defaults to the default floating-point dtype.
    device : torch.device or str, optional
        Coefficient device. Defaults to the default device.

    Notes
    -----
    Coefficients are projected onto Chebyshev polynomials in float64 and
    evaluated with Clenshaw recurrence. ``tanh`` preserves odd parity;
    ``sigmoid`` preserves its constant-plus-odd decomposition. The remaining
    activations preserve their linear-plus-even decomposition.

    On a circle with maximum order M, projection of this activation back to M
    requires more than ``(degree + 1) * M`` samples. On a sphere with maximum
    degree L, quadrature must be exact through ``(degree + 1) * L``. These
    conditions remove aliasing, not approximation error relative to ``act``.
    Apply the activation to grid values, not to individual irrep coefficients.
    Outside the approximation interval the polynomial can grow rapidly.
    """

    def __init__(self, act="silu", degree=8, bound=3.0, *, dtype=None, device=None):
        super().__init__()
        activations = {
            "silu": functional.silu,
            "gelu": functional.gelu,
            "relu": functional.relu,
            "tanh": torch.tanh,
            "sigmoid": torch.sigmoid,
            "softplus": functional.softplus,
        }
        if act not in activations:
            raise ValueError(f"act must be one of {tuple(activations)}.")
        if not isinstance(degree, int) or degree < 1:
            raise ValueError("degree must be a positive integer.")
        bound = float(bound)
        if not math.isfinite(bound) or bound <= 0:
            raise ValueError("bound must be finite and positive.")
        self.act = act
        self.degree = degree
        self.bound = bound
        resolution = max(256, 4 * (degree + 1))
        angles = torch.arange(resolution, dtype=torch.float64, device="cpu")
        angles = (angles + 0.5) * (math.pi / resolution)
        orders = torch.arange(degree + 1, dtype=torch.float64, device="cpu")
        values = activations[act](bound * angles.cos())
        coefficients = (orders[:, None] * angles).cos() @ values * (2 / resolution)
        coefficients[0] *= 0.5
        if act in ("tanh", "sigmoid"):
            coefficients[::2] = 0
            if act == "sigmoid":
                coefficients[0] = 0.5
        else:
            coefficients[1::2] = 0
            coefficients[1] = bound / 2
        self._coefficients = coefficients
        dtype = torch.get_default_dtype() if dtype is None else dtype
        device = torch.get_default_device() if device is None else device
        self.register_buffer(
            "coefficients",
            coefficients.to(device=device, dtype=dtype, copy=True),
            persistent=False,
        )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        self.coefficients = self._coefficients.to(self.coefficients, copy=True)
        return self

    def forward(self, features):
        """Evaluate the polynomial elementwise, preserving the input shape."""
        features = features / self.bound
        previous = torch.zeros_like(features)
        current = torch.zeros_like(features)
        for i in range(self.degree, 0, -1):
            previous, current = (
                current,
                (self.coefficients[i] + 2 * features * current - previous),
            )
        return self.coefficients[0] + features * current - previous

    def extra_repr(self):
        return f"act={self.act}, degree={self.degree}, bound={self.bound}"

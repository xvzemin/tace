"""Fourier transforms for real O(2) representations."""

import math
from collections import Counter

import torch

from .irreps import Irreps

__all__ = ["O2Grid"]


class O2Grid(torch.nn.Module):
    """Transform O(2) features to one or two circular grids per channel.

    Parameters
    ----------
    irreps : Irreps, str, or sequence
        Input and reconstructed representation in flattened ``ir_mul`` order.
        Only time-even irreps are supported. Repeated entries and unequal
        multiplicities are allowed.
    resolution : int, optional
        Angular samples per grid, at least ``2 * mmax + 1``. Defaults to
        ``4 * mmax + 1``, where ``mmax`` is the largest input order.
    normalization : {"component", "norm", "integral"}, optional
        ``"component"`` gives equal grid variance per order for unit-variance
        coefficients. ``"norm"`` assumes unit expected squared norm per
        order. ``"integral"`` uses harmonics orthonormal under ``d alpha``.
        Two sets are combined by an orthogonal sum-and-difference transform.
        For ``"integral"``, the measure is ``d alpha`` summed over the grids.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to the default floating-point dtype.
    device : torch.device or str, optional
        Buffer device. Defaults to the default device.

    Attributes
    ----------
    num_sheets : int
        Two when ``irreps`` contains a reflection-odd scalar, otherwise one.
    num_channels : int
        Number of grid channels. Scalars occupy their respective coefficient
        sets; positive-order copies fill the first set before the second.
    grid_shape : tuple of int
        ``(num_channels, num_sheets, resolution)``.
    grid : torch.Tensor
        Angles of shape ``(resolution,)``, in radians.
    weights : torch.Tensor
        Quadrature weights of shape ``(resolution,)``, summing to one.

    Notes
    -----
    Without ``0o``, each channel carries an ordinary scalar field on the
    circle. With ``0o``, the two coefficient sets carry a scalar and a
    pseudoscalar field. Their normalized sum and difference are exchanged
    by reflection. The second set uses the inverse of the fixed O(2)
    basis change on positive orders. Unused coefficients are zero-filled;
    reconstruction projects onto the supplied irreps in their original order.

    Apply the same pointwise activation to both sheets. Projection of a
    degree-d polynomial is exactly O(2)-equivariant up to roundoff when
    ``resolution > (d + 1) * mmax``. General nonlinearities require a
    resolution-convergence check.
    """

    def __init__(
        self,
        irreps,
        resolution=None,
        *,
        normalization="component",
        dtype=None,
        device=None,
    ):
        super().__init__()
        self.irreps = Irreps(irreps)
        if any(ir.t != 1 for ir, _ in self.irreps):
            raise ValueError("O2Grid only supports time-even irreps.")
        self.dim = self.irreps.dim
        self.mmax = max(0, self.irreps.mmax)
        resolution = 4 * self.mmax + 1 if resolution is None else resolution
        if not isinstance(resolution, int) or resolution < 2 * self.mmax + 1:
            raise ValueError("resolution must be an integer at least 2 * mmax + 1.")
        if normalization not in ("component", "norm", "integral"):
            raise ValueError("normalization must be component, norm, or integral.")
        self.resolution = resolution
        self.normalization = normalization
        self.num_sheets = 2 if any(ir.is_odd_scalar() for ir, _ in self.irreps) else 1
        self.num_channels = max(
            (
                mul if ir.m == 0 else (mul + self.num_sheets - 1) // self.num_sheets
                for ir, mul in self.irreps.regroup()
            ),
            default=0,
        )
        self.grid_shape = (self.num_channels, self.num_sheets, self.resolution)
        dim = 2 * self.mmax + 1
        self._coefficient_shape = (self.num_channels, self.num_sheets, dim)

        alpha = torch.arange(resolution, dtype=torch.float64, device="cpu")
        alpha = alpha * (2 * math.pi / resolution)
        orders = torch.arange(1, self.mmax + 1, dtype=torch.float64, device="cpu")
        angles = alpha[:, None] * orders
        basis = torch.cat(
            (
                torch.ones_like(alpha[:, None]),
                math.sqrt(2)
                * torch.stack((angles.cos(), angles.sin()), dim=-1).flatten(1),
            ),
            dim=-1,
        )
        scale = torch.ones(dim, dtype=torch.float64, device="cpu")
        if normalization == "component":
            scale[1:] /= math.sqrt(2)
            scale /= math.sqrt(self.mmax + 1)
        elif normalization == "norm":
            scale /= math.sqrt(self.mmax + 1)
        else:
            scale /= math.sqrt(2 * math.pi)
        weights = torch.full_like(alpha, 1 / resolution)
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

        locations, signs = [], []
        offsets = Counter()
        for ir, mul in self.irreps:
            for component in range(ir.dim):
                for channel in range(offsets[ir], offsets[ir] + mul):
                    if ir.m == 0:
                        sheet, order, sign = int(ir.p == -1), 0, 1
                    else:
                        sheet = channel // self.num_channels
                        order = 2 * ir.m - 1 + component
                        sign = 1
                        if sheet:
                            # J^{-1}(h_+, h_-) = (h_-, -h_+).
                            order = 2 * ir.m - component
                            sign = -1 if component == 0 else 1
                    position = (
                        (channel % self.num_channels) * self.num_sheets + sheet
                    ) * dim + order
                    locations.append(position)
                    signs.append(sign)
            offsets[ir] += mul

        output_index = torch.tensor(locations, dtype=torch.long, device="cpu")
        output_sign = torch.tensor(signs, dtype=torch.int8, device="cpu")
        size = math.prod(self._coefficient_shape)
        input_index = torch.zeros(size, dtype=torch.long, device="cpu")
        input_sign = torch.zeros(size, dtype=torch.int8, device="cpu")
        input_index[output_index] = torch.arange(self.dim, device="cpu")
        input_sign[output_index] = output_sign
        for name, value in (
            ("input_index", input_index),
            ("input_sign", input_sign),
            ("output_index", output_index),
            ("output_sign", output_sign),
        ):
            self.register_buffer(
                name, value.to(device=self.grid.device), persistent=False
            )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, value in self._constants.items():
            self._buffers[name] = value.to(self._buffers[name], copy=True)
        return self

    def forward(self, features):
        """Evaluate O(2) features on the grid.

        Parameters
        ----------
        features : torch.Tensor
            Features of shape ``(..., irreps.dim)`` in flattened ``ir_mul`` order.

        Returns
        -------
        torch.Tensor
            Values of shape ``(..., num_channels, num_sheets, resolution)``.
        """
        return self.to_grid(features)

    def to_grid(self, features):
        """Evaluate circular coefficient sets and combine reflection sheets.

        Parameters
        ----------
        features : torch.Tensor
            Features of shape ``(..., irreps.dim)`` in flattened ``ir_mul`` order.

        Returns
        -------
        torch.Tensor
            Values of shape ``(..., num_channels, num_sheets, resolution)``.
        """
        if features.ndim < 1 or features.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} features on the last axis.")
        coefficients = features.index_select(-1, self.input_index) * self.input_sign
        values = coefficients.unflatten(-1, self._coefficient_shape) @ self.synthesis.T
        if self.num_sheets == 2:
            even, odd = values.unbind(-2)
            values = torch.stack((even + odd, even - odd), dim=-2) / math.sqrt(2)
        return values

    def from_grid(self, features):
        """Project grid values onto the original O(2) irreps.

        Parameters
        ----------
        features : torch.Tensor
            Values of shape ``(..., num_channels, num_sheets, resolution)``.

        Returns
        -------
        torch.Tensor
            Features of shape ``(..., irreps.dim)`` in flattened ``ir_mul`` order.
        """
        if features.ndim < 3 or features.shape[-3:] != self.grid_shape:
            raise ValueError(f"Expected trailing grid dimensions {self.grid_shape}.")
        if self.num_sheets == 2:
            positive, negative = features.unbind(-2)
            features = torch.stack(
                (positive + negative, positive - negative), dim=-2
            ) / math.sqrt(2)
        coefficients = (features @ self.analysis.T).flatten(-3)
        return coefficients.index_select(-1, self.output_index) * self.output_sign

    def extra_repr(self):
        return (
            f"irreps={self.irreps}, num_channels={self.num_channels}, "
            f"num_sheets={self.num_sheets}, resolution={self.resolution}, "
            f"normalization={self.normalization}"
        )

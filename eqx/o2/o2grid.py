"""Fourier transforms for real O(2) representations."""

import math
from collections import Counter

import torch

from .irreps import Irrep, Irreps

__all__ = ["O2Grid"]


class O2Grid(torch.nn.Module):
    """Transform O(2) features to two circular grids per channel.

    Parameters
    ----------
    irreps : Irreps, str, or sequence
        Input and reconstructed representation in flattened ``ir_mul`` order.
        Requires ``C x 0e + C x 0o`` and ``2C x m`` at every order from one
        to ``mmax``, with ``C > 0``. Multiplicities are summed over repeated
        entries, so ``2C x m`` may also be supplied as ``C x m + C x m``.
        Only time-even irreps are supported.
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
        Number of reflection sheets, always two.
    num_channels : int
        Number of grid channels, equal to ``C``.
    irreps_in1, irreps_in2 : Irreps
        Representations for separate inputs. Each has ``C`` copies per order
        in increasing order, starting with ``0e`` and ``0o``, respectively.
    grid_shape : tuple of int
        ``(num_channels, num_sheets, resolution)``.
    grid : torch.Tensor
        Angles of shape ``(resolution,)``, in radians.
    weights : torch.Tensor
        Quadrature weights of shape ``(resolution,)``, summing to one.

    Notes
    -----
    The first and second sets contain ``0e`` and ``0o``, respectively, and
    ``C`` copies of each positive-order irrep. For a combined input, the
    first ``C`` copies at each positive order enter the first set. The second
    set uses the inverse of the fixed O(2) basis change on positive orders.
    Their normalized sum and difference are exchanged by reflection.

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
        self.num_channels = self.irreps.count("0e")
        if self.num_channels == 0:
            raise ValueError("O2Grid requires C copies of 0e, with C > 0.")
        irrep_list = [(Irrep(m, 0), self.num_channels) for m in range(1, self.mmax + 1)]
        self.irreps_in1 = Irreps([(Irrep(0, 1), self.num_channels)] + irrep_list)
        self.irreps_in2 = Irreps([(Irrep(0, -1), self.num_channels)] + irrep_list)
        if self.irreps.regroup() != (self.irreps_in1 + self.irreps_in2).regroup():
            raise ValueError(
                "O2Grid requires C copies of 0e and 0o and 2C copies of each "
                "order from 1 to mmax, with C > 0."
            )
        resolution = 4 * self.mmax + 1 if resolution is None else resolution
        if not isinstance(resolution, int) or resolution < 2 * self.mmax + 1:
            raise ValueError("resolution must be an integer at least 2 * mmax + 1.")
        if normalization not in ("component", "norm", "integral"):
            raise ValueError("normalization must be component, norm, or integral.")
        self.resolution = resolution
        self.normalization = normalization
        self.num_sheets = 2
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
        input_index = output_index.argsort()
        input_sign = output_sign[input_index]
        second_index = torch.arange(dim, device="cpu")
        second_index[1:] = (
            second_index[1:].unflatten(0, (self.mmax, 2)).flip(-1).flatten()
        )
        second_sign = torch.ones(dim, dtype=torch.int8, device="cpu")
        second_sign[2::2] = -1
        for name, value in (
            ("input_index", input_index),
            ("input_sign", input_sign),
            ("output_index", output_index),
            ("output_sign", output_sign),
            ("second_index", second_index),
            ("second_sign", second_sign),
        ):
            self.register_buffer(
                name, value.to(device=self.grid.device), persistent=False
            )

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse=recurse)
        for name, value in self._constants.items():
            self._buffers[name] = value.to(self._buffers[name], copy=True)
        return self

    def forward(self, features, features2=None):
        """Evaluate O(2) features on the grid.

        Parameters
        ----------
        features : torch.Tensor
            Combined features of shape ``(..., irreps.dim)``, or the first
            coefficient set of shape ``(..., irreps_in1.dim)`` when
            ``features2`` is given. Both use flattened ``ir_mul`` order.
        features2 : torch.Tensor, optional
            Second coefficient set of shape ``(..., irreps_in2.dim)``, with
            the same leading dimensions as ``features``.

        Returns
        -------
        torch.Tensor
            Values of shape ``(..., num_channels, num_sheets, resolution)``.
        """
        return self.to_grid(features, features2)

    def to_grid(self, features, features2=None):
        """Evaluate circular coefficient sets and combine reflection sheets.

        Parameters
        ----------
        features : torch.Tensor
            Combined features of shape ``(..., irreps.dim)``, or the first
            coefficient set of shape ``(..., irreps_in1.dim)`` when
            ``features2`` is given. Both use flattened ``ir_mul`` order.
        features2 : torch.Tensor, optional
            Second coefficient set of shape ``(..., irreps_in2.dim)``, with
            the same leading dimensions as ``features``.

        Returns
        -------
        torch.Tensor
            Values of shape ``(..., num_channels, num_sheets, resolution)``.
        """
        if features2 is None:
            if features.ndim < 1 or features.shape[-1] != self.dim:
                raise ValueError(f"Expected {self.dim} features on the last axis.")
            coefficients = features.index_select(-1, self.input_index) * self.input_sign
            coefficients = coefficients.unflatten(-1, self._coefficient_shape)
        else:
            if features.ndim < 1 or features.shape[-1] != self.irreps_in1.dim:
                raise ValueError(
                    f"Expected {self.irreps_in1.dim} features in each coefficient set."
                )
            if features2.shape != features.shape:
                raise ValueError("Both coefficient sets must have the same shape.")
            shape = (2 * self.mmax + 1, self.num_channels)
            first = features.unflatten(-1, shape).transpose(-1, -2)
            second = features2.unflatten(-1, shape).transpose(-1, -2)
            second = second.index_select(-1, self.second_index) * self.second_sign
            coefficients = torch.stack((first, second), dim=-2)
        values = coefficients @ self.synthesis.T
        even, odd = values.unbind(-2)
        return torch.stack((even + odd, even - odd), dim=-2) / math.sqrt(2)

    def from_grid(self, features, *, split=False):
        """Project grid values onto the original O(2) irreps.

        Parameters
        ----------
        features : torch.Tensor
            Values of shape ``(..., num_channels, num_sheets, resolution)``.
        split : bool, optional
            Return the two coefficient sets separately. Defaults to ``False``.

        Returns
        -------
        torch.Tensor or tuple of torch.Tensor
            Combined features of shape ``(..., irreps.dim)``, or two tensors
            with trailing dimensions ``irreps_in1.dim`` and ``irreps_in2.dim``
            when ``split=True``. All use flattened ``ir_mul`` order.
        """
        if features.ndim < 3 or features.shape[-3:] != self.grid_shape:
            raise ValueError(f"Expected trailing grid dimensions {self.grid_shape}.")
        positive, negative = features.unbind(-2)
        features = torch.stack(
            (positive + negative, positive - negative), dim=-2
        ) / math.sqrt(2)
        coefficients = features @ self.analysis.T
        if split:
            first, second = coefficients.unbind(-2)
            second = (second * self.second_sign).index_select(-1, self.second_index)
            return first.transpose(-1, -2).flatten(-2), second.transpose(
                -1, -2
            ).flatten(-2)
        coefficients = coefficients.flatten(-3)
        return coefficients.index_select(-1, self.output_index) * self.output_sign

    def extra_repr(self):
        return (
            f"irreps={self.irreps}, num_channels={self.num_channels}, "
            f"num_sheets={self.num_sheets}, resolution={self.resolution}, "
            f"normalization={self.normalization}"
        )

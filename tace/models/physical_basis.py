"""Degree-resolved physical radial basis."""

import math

import numpy as np
import torch
from ase.data import covalent_radii
from scipy.interpolate import CubicHermiteSpline

from equivariantx.nn import fourier_radial, uniform_cubic_spline
from tace.utils.env import acceleration_enabled


def spline_coefficients(frequencies, coefficients, num_cos):
    def evaluate(x):
        values, slopes = [], []
        for i, rows in enumerate((coefficients[:num_cos], coefficients[num_cos:])):
            if not len(rows):
                continue
            w = frequencies[i]
            phase = x[:, None] * w
            cosine, sine = np.cos(phase), np.sin(phase)
            values.append((sine if i else cosine) @ rows.T)
            slopes.append(((cosine if i else -sine) * w) @ rows.T)
        return np.concatenate(values, axis=1), np.concatenate(slopes, axis=1)

    intervals = 10
    while intervals <= 50000:
        grid = np.linspace(0, 10, intervals + 1)
        spline = CubicHermiteSpline(grid, *evaluate(grid))
        mid = (grid[:-1] + grid[1:]) * 0.5
        actual, _ = evaluate(mid)
        error = np.abs(spline(mid) - actual)
        if (
            error.mean() < 1e-8
            or (error / np.maximum(np.abs(actual), 1e-300)).mean() < 1e-8
        ):
            return 10 / intervals, np.ascontiguousarray(spline.c.transpose(1, 0, 2))
        intervals *= 2
    raise ValueError("Physical basis spline failed to converge.")


class PhysicalRadialBasis(torch.nn.Module):
    """Evaluate fixed physical eigenfunctions at element-scaled distances.

    Parameters
    ----------
    cutoff : float
        Cutoff radius in angstroms for volume normalization.
    lmax : int
        Maximum angular degree.
    num_basis : list of int, optional
        Number of radial modes per degree, instead of an eigenvalue threshold.
    max_eigenvalue : float, optional
        Inclusive threshold in the tabulated dimensionless spectrum. Defaults
        to the eighth degree-zero eigenvalue.
    use_interpolation : bool, optional
        Evaluate cubic Hermite splines instead of the full trigonometric expansion.

    Examples
    --------
    >>> PhysicalRadialBasis(6.0, 3).widths
    (8, 7, 6, 6)
    """

    def __init__(
        self,
        cutoff: float,
        lmax: int,
        num_basis: list[int] | tuple[int, ...] | None = None,
        max_eigenvalue: float | None = None,
        use_interpolation: bool = False,
    ) -> None:
        super().__init__()
        from physical_basis import PhysicalBasis

        from .radial import DEFAULT_RADIAL_MODES, check_radial_selection

        check_radial_selection(num_basis, max_eigenvalue)
        data = PhysicalBasis()
        if not isinstance(lmax, int) or not 0 <= lmax <= data.l_max:
            raise ValueError(f"physical_basis requires 0 <= lmax <= {data.l_max}.")
        if not isinstance(use_interpolation, bool):
            raise ValueError("use_interpolation must be a boolean.")
        if num_basis is None:
            limit = (
                data.E_ln[0, DEFAULT_RADIAL_MODES - 1]
                if max_eigenvalue is None
                else max_eigenvalue
            )
            minimum = float(data.E_ln[: lmax + 1, 0].max())
            if limit < minimum:
                raise ValueError(
                    f"max_eigenvalue={limit:g} leaves an empty degree through "
                    f"lmax={lmax}; use at least {minimum:.17g}."
                )
            num_basis = (data.E_ln[: lmax + 1] <= limit).sum(axis=1).tolist()
        if (
            not isinstance(num_basis, (list, tuple))
            or len(num_basis) != lmax + 1
            or any(
                type(n) is not int or not 1 <= n <= data.n_max_big for n in num_basis
            )
        ):
            raise ValueError(
                f"num_radial_basis must specify {lmax + 1} counts from 1 to {data.n_max_big}."
            )
        self.widths = tuple(num_basis)
        self.use_interpolation = use_interpolation
        q = np.arange(data.n_max_big, dtype=np.float64)
        frequencies = np.stack((q + 0.5, q + 1.0)) * (np.pi / 10)
        coefficients = np.concatenate(
            [data.eigenvectors[l, :, :n].T for l, n in enumerate(self.widths)]
        ) * math.sqrt(cutoff**3 / 3)
        self.register_buffer("lengthscale", torch.from_numpy(0.7 * covalent_radii))
        self.register_buffer("frequencies", torch.from_numpy(frequencies))
        self.register_buffer(
            "coefficients", torch.from_numpy(np.ascontiguousarray(coefficients))
        )
        self.register_buffer("spline_table", None, persistent=False)
        if self.use_interpolation:
            self.build_spline()

    def build_spline(self):
        """Build the Hermite table from the stored Fourier coefficients."""
        self.step, table = spline_coefficients(
            self.frequencies.detach().to(device="cpu", dtype=torch.float64).numpy(),
            self.coefficients.detach().to(device="cpu", dtype=torch.float64).numpy(),
            self.widths[0],
        )
        self.spline_table = torch.from_numpy(table).to(self.coefficients)

    def _load_from_state_dict(self, *args, **kwargs):
        super()._load_from_state_dict(*args, **kwargs)
        if self.use_interpolation:
            self.build_spline()

    def forward(
        self,
        distance: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return all selected modes, ordered by degree, for each edge."""
        radii = self.lengthscale[atomic_numbers]
        x = distance.reshape(-1) / (0.1 + (radii[edge_index[0]] + radii[edge_index[1]]))
        backend = "cuda" if acceleration_enabled("eqx", kernel="conv") else "torch"
        if self.use_interpolation:
            return uniform_cubic_spline(
                x, self.spline_table, self.step, backend=backend
            )
        return fourier_radial(
            x,
            self.frequencies,
            self.coefficients,
            self.widths[0],
            10.0,
            backend=backend,
        )

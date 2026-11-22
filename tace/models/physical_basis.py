################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from collections.abc import Sequence
from functools import lru_cache

import numpy as np
import torch
from ase.data import covalent_radii
from scipy.interpolate import CubicHermiteSpline
from scipy.linalg import qr, solve_triangular, svd
from scipy.special import roots_legendre

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


def space_spectrum():
    """Load the tabulated dimensionless SPACE spectrum."""
    try:
        from physical_basis import PhysicalBasis
    except ModuleNotFoundError as error:
        if error.name != "physical_basis":
            raise
        raise ImportError(
            "The spb radial basis requires pip install 'tace[spb]'."
        ) from error
    return PhysicalBasis()


def scaled_pair_distance(distance, types, edge_index, lengthscale):
    """Return the dimensionless distance for each edge."""
    radii = lengthscale[types]
    return distance.reshape(-1) / (0.1 + (radii[edge_index[0]] + radii[edge_index[1]]))


class SpacePhysicalBasis(torch.nn.Module):
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
    >>> SpacePhysicalBasis(6.0, 3).widths
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
        from .radial import DEFAULT_RADIAL_MODES, check_radial_selection

        check_radial_selection(num_basis, max_eigenvalue)
        data = space_spectrum()
        if not isinstance(lmax, int) or not 0 <= lmax <= data.l_max:
            raise ValueError(f"spb requires 0 <= lmax <= {data.l_max}.")
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

    def _load_from_state_dict(self, state_dict, prefix, *args, **kwargs):
        key = prefix + "coefficients"
        table = state_dict.get(key)
        legacy_spline = (
            table is not None
            and table.ndim == 3
            and table.shape[1:] == (4, sum(self.widths))
            and prefix + "frequencies" not in state_dict
        )
        if legacy_spline:
            # Early spline checkpoints stored cubic polynomials, not Fourier rows.
            state_dict[key] = self.coefficients
            state_dict[prefix + "frequencies"] = self.frequencies
        super()._load_from_state_dict(state_dict, prefix, *args, **kwargs)
        if self.use_interpolation:
            if legacy_spline:
                self.spline_table = table.to(self.coefficients)
                self.step = 10.0 / table.shape[0]
            else:
                self.build_spline()

    def forward(
        self,
        distance: torch.Tensor,
        atomic_numbers: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> torch.Tensor:
        """Return all selected modes, ordered by degree, for each edge."""
        x = scaled_pair_distance(distance, atomic_numbers, edge_index, self.lengthscale)
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


FOURIER_TERMS = 200


@lru_cache(maxsize=2)
def fourier_trial_matrices(cosine: bool):
    """Return weighted trial values, derivatives and the mass factor on the unit interval."""
    t, w = roots_legendre(512)
    t, sw = (t + 1) * 0.5, (w * 0.5) ** 0.5
    q = np.pi * (np.arange(FOURIER_TERMS) + (0.5 if cosine else 1.0))
    phase = t[:, None] * q
    value = np.cos(phase) if cosine else np.sin(phase)
    slope = (-np.sin(phase) if cosine else np.cos(phase)) * q
    radial_weight = (sw * t)[:, None]
    mass = qr(radial_weight * value, mode="r")[0][:FOURIER_TERMS]
    return t, radial_weight * slope, sw[:, None] * value, mass


@lru_cache(maxsize=32)
def radial_spectrum(extent: float, degree: int) -> tuple[np.ndarray, np.ndarray]:
    """Return dimensionless eigenvalues and mass-normalized Fourier rows on [0, extent]."""
    t, derivative, angular, mass = fourier_trial_matrices(degree == 0)
    weight = np.exp(0.5 * extent * t)[:, None]
    trial = weight * derivative
    if degree:
        trial = np.vstack((trial, math.sqrt(degree * (degree + 1)) * weight * angular))
    stiffness = qr(trial, mode="r")[0][:FOURIER_TERMS]
    # Factor the weighted trial matrices before squaring. Diagonalizing the
    # stiffness Gram matrix directly loses the low modes on long intervals.
    inverse = solve_triangular(stiffness.T, mass.T, lower=True).T
    u, singular, _ = svd(inverse, full_matrices=False)
    rows = solve_triangular(mass, u).T / extent**1.5
    rows *= np.sign(rows[np.arange(len(rows)), np.abs(rows).argmax(axis=1)])[:, None]
    return 1 / (extent * singular) ** 2, rows


class ExpWeightedjnSphericalBesselBasis(torch.nn.Module):
    """Evaluate shared, exponentially weighted eigenfunctions at element-scaled distances.

    Parameters
    ----------
    cutoff : float
        Graph cutoff in angstroms. Its numerical value sets the Dirichlet
        boundary in the dimensionless coordinate ``x = r / r0``.
    lmax : int
        Maximum angular degree.
    atomic_numbers : sequence of int, optional
        Atomic numbers indexed by node type. If omitted, inputs are atomic numbers.
    num_basis : list of int, optional
        Number of modes for each degree.
    max_eigenvalue : float, optional
        Inclusive threshold in the shared dimensionless spectrum. Defaults to
        the eighth degree-zero eigenvalue.

    Examples
    --------
    >>> ExpWeightedjnSphericalBesselBasis(6.0, 3, max_eigenvalue=284).widths
    (10, 9, 8, 8)
    """

    def __init__(
        self,
        cutoff: float,
        lmax: int,
        atomic_numbers: Sequence[int] | None = None,
        num_basis: list[int] | tuple[int, ...] | None = None,
        max_eigenvalue: float | None = None,
    ) -> None:
        super().__init__()
        from .radial import DEFAULT_RADIAL_MODES, check_radial_selection

        check_radial_selection(num_basis, max_eigenvalue)
        if (
            not math.isfinite(cutoff)
            or cutoff <= 0
            or type(lmax) is not int
            or lmax < 0
        ):
            raise ValueError(
                "ejn requires positive finite cutoff and nonnegative integer lmax."
            )
        if num_basis is not None and (
            not isinstance(num_basis, (tuple, list))
            or len(num_basis) != lmax + 1
            or any(type(n) is not int or not 1 <= n <= FOURIER_TERMS for n in num_basis)
        ):
            raise ValueError(
                f"num_radial_basis must specify {lmax + 1} counts from 1 to {FOURIER_TERMS}."
            )
        radii = covalent_radii
        if atomic_numbers is not None:
            numbers = np.asarray(atomic_numbers)
            if (
                numbers.ndim != 1
                or not numbers.size
                or numbers.dtype.kind not in "iu"
                or numbers.min() < 1
                or numbers.max() >= len(radii)
            ):
                raise ValueError(
                    "ejn requires valid atomic numbers with known covalent radii."
                )
            radii = radii[numbers]
        self.cutoff = float(cutoff)
        eigenvalues, coefficients = [], []
        limit = max_eigenvalue
        for l in range(lmax + 1):
            values, rows = radial_spectrum(self.cutoff, l)
            if num_basis is None:
                if l == 0 and limit is None:
                    limit = values[DEFAULT_RADIAL_MODES - 1]
                n = int(np.searchsorted(values, limit, side="right"))
                if not n:
                    raise ValueError(
                        f"max_eigenvalue={limit:g} leaves an empty degree l={l} at cutoff={cutoff:g}."
                    )
            else:
                n = num_basis[l]
            eigenvalues.append(values[:n])
            coefficients.append(rows[:n])
        self.widths = tuple(len(v) for v in eigenvalues)
        q = np.arange(FOURIER_TERMS, dtype=np.float64)
        freq = np.stack((q + 0.5, q + 1.0)) * (np.pi / self.cutoff)
        self.register_buffer("lengthscale", torch.from_numpy(0.7 * radii))
        self.register_buffer("frequencies", torch.from_numpy(freq))
        self.register_buffer(
            "eigenvalues", torch.from_numpy(np.concatenate(eigenvalues))
        )
        self.register_buffer(
            "coefficients",
            torch.from_numpy(
                np.concatenate(coefficients) * math.sqrt(self.cutoff**3 / 3)
            ),
        )

    def forward(
        self, distance: torch.Tensor, node_type: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate the degree-ordered modes on the given edges."""
        x = scaled_pair_distance(distance, node_type, edge_index, self.lengthscale)
        backend = "cuda" if acceleration_enabled("eqx", kernel="conv") else "torch"
        return fourier_radial(
            x,
            self.frequencies,
            self.coefficients,
            self.widths[0],
            self.cutoff,
            backend=backend,
        )

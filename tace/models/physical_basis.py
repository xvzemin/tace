"""Degree-resolved radial eigenfunctions for spherical harmonic convolutions."""

import math

import numpy as np
import torch
from ase.data import covalent_radii
from scipy.interpolate import CubicHermiteSpline

from equivariantx.nn import uniform_cubic_spline
from tace.utils.env import acceleration_enabled


def scaled_distance(distance, atomic_numbers, edge_index, lengthscale):
    source, target = edge_index
    radius = lengthscale[atomic_numbers]
    return distance.reshape(-1) / (0.1 + (radius[source] + radius[target]))


def mode_threshold(value, default, minimum, lmax):
    if value is None:
        return max(default, minimum)
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError("max_eigenvalue must be finite.")
    if value < minimum:
        raise ValueError(
            f"max_eigenvalue={value} does not cover lmax={lmax}; "
            f"use at least {minimum} or specify num_radial_basis."
        )
    return value


def check_mode_counts(counts, lmax):
    if (
        not isinstance(counts, list)
        or len(counts) != lmax + 1
        or not all(type(n) is int and n >= 1 for n in counts)
    ):
        raise ValueError(
            f"num_radial_basis must be a list of {lmax + 1} positive counts."
        )


class PhysicalRadialBasis(torch.nn.Module):
    r"""Evaluate the tabulated physical basis at element-scaled distances.

    Parameters
    ----------
    cutoff : float
        Cutoff radius used for the volume normalization, in angstroms.
    lmax : int
        Highest spherical harmonic degree in the convolution.
    num_basis : list of int or None
        Number of modes for each degree. Overrides ``max_eigenvalue``.
    max_eigenvalue : float or None
        Maximum dimensionless eigenvalue. By default use 36 or the first
        eigenvalue of the highest degree, whichever is larger.

    Examples
    --------
    >>> basis = PhysicalRadialBasis(6.0, 2, max_eigenvalue=25)
    >>> basis.widths
    (2, 1, 1)
    """

    def __init__(self, cutoff, lmax, num_basis=None, max_eigenvalue=None):
        super().__init__()
        try:
            from physical_basis import PhysicalBasis
        except ModuleNotFoundError as exc:
            if exc.name != "physical_basis":
                raise
            raise ImportError(
                "Install the physical radial basis with pip install '.[physical]'."
            ) from exc

        data = PhysicalBasis()
        if not 0 <= lmax <= data.l_max:
            raise ValueError(f"physical requires 0 <= lmax <= {data.l_max}.")
        if num_basis is None:
            max_eigenvalue = mode_threshold(
                max_eigenvalue, 36.0, data.E_ln[lmax, 0], lmax
            )
            num_basis = (data.E_ln[: lmax + 1] <= max_eigenvalue).sum(axis=1).tolist()
        check_mode_counts(num_basis, lmax)
        if max(num_basis) > data.n_max_big:
            raise ValueError(
                f"physical supports at most {data.n_max_big} modes per degree."
            )
        self.widths = tuple(num_basis)
        # The upstream solver expands each mode in the same 200 trial functions.
        # Evaluate all selected modes together, only at initialization.
        coeff = [data.eigenvectors[l, :, :n] for l, n in enumerate(num_basis)]
        coeff_cos = coeff[0] * math.sqrt(cutoff**3 / 3)
        coeff_sin = (
            np.concatenate(coeff[1:], axis=1) * math.sqrt(cutoff**3 / 3)
            if lmax
            else None
        )
        q = np.arange(data.n_max_big)
        wc, ws = (q + 0.5) * (np.pi / 10), (q + 1) * (np.pi / 10)

        def evaluate(x):
            xc = x[:, None] * wc
            v, d = [np.cos(xc) @ coeff_cos], [(-wc * np.sin(xc)) @ coeff_cos]
            if coeff_sin is not None:
                xs = x[:, None] * ws
                v.append(np.sin(xs) @ coeff_sin)
                d.append((ws * np.cos(xs)) @ coeff_sin)
            return np.concatenate(v, axis=1), np.concatenate(d, axis=1)

        intervals = 10
        while True:
            grid = np.linspace(0, 10, intervals + 1)
            values, slopes = evaluate(grid)
            spline = CubicHermiteSpline(grid, values, slopes)
            mid = (grid[1:] + grid[:-1]) * 0.5
            actual, _ = evaluate(mid)
            error = np.abs(spline(mid) - actual)
            if (
                error.mean() < 1e-8
                or (error / np.maximum(np.abs(actual), 1e-300)).mean() < 1e-8
            ):
                break
            intervals *= 2
            if intervals > 50000:
                raise ValueError("Physical basis spline failed to converge.")
        self.step = 10 / intervals
        self.register_buffer(
            "coefficients", torch.from_numpy(spline.c.transpose(1, 0, 2).copy())
        )
        self.register_buffer("lengthscale", torch.from_numpy(0.7 * covalent_radii))

    def forward(self, distance, atomic_numbers, edge_index):
        x = scaled_distance(distance, atomic_numbers, edge_index, self.lengthscale)
        return uniform_cubic_spline(
            x,
            self.coefficients,
            self.step,
            backend="cuda" if acceleration_enabled("eqx", kernel="conv") else "torch",
        )


class PtPhysicalRadialBasis(torch.nn.Module):
    r"""Pöschl–Teller radial modes evaluated by a finite Jacobi recurrence.

    Parameters
    ----------
    cutoff : float
        Cutoff radius in angstroms for volume normalization.
    lmax : int
        Highest spherical harmonic degree.
    num_basis : list of int or None
        Number of radial modes per degree. Overrides ``max_eigenvalue``.
    max_eigenvalue : float or None
        Maximum dimensionless eigenvalue. By default use 77 or the first
        eigenvalue of the highest degree, whichever is larger.

    Examples
    --------
    >>> PtPhysicalRadialBasis(5.0, 3).widths
    (3, 2, 1, 1)
    """

    def __init__(self, cutoff, lmax, num_basis=None, max_eigenvalue=None):
        super().__init__()
        if not isinstance(lmax, int) or lmax < 0:
            raise ValueError("pt_physical requires a non-negative lmax.")
        b = [(1 + math.sqrt(4 * l * (l + 1) + 25)) / 2 for l in range(lmax + 1)]
        if num_basis is None:
            edge = mode_threshold(
                max_eigenvalue, 77.0, (lmax + 1 + b[-1]) ** 2 - 1, lmax
            )
            num_basis = []
            for l, bl in enumerate(b):
                n = 1
                while (2 * n + l + 1 + bl) ** 2 - 1 <= edge:
                    n += 1
                num_basis.append(n)
        check_mode_counts(num_basis, lmax)
        self.widths = tuple(num_basis)
        self.register_buffer("lengthscale", torch.from_numpy(0.7 * covalent_radii))
        self.modes = []
        for l, (bl, count) in enumerate(zip(b, num_basis)):
            alpha, beta = l + 0.5, bl - 0.5
            s = alpha + beta
            # Unit norm in x^2 dx before applying the TACE volume factor.
            norm = math.sqrt(2 * cutoff**3 / 3) * math.exp(
                (math.lgamma(s + 2) - math.lgamma(alpha + 1) - math.lgamma(beta + 1))
                / 2
            )
            steps = []
            prev = 0.0
            for n in range(count - 1):
                k = 2 * n + s
                diag = (beta**2 - alpha**2) / (k * (k + 2))
                off = (
                    2
                    / (k + 2)
                    * math.sqrt(
                        (n + 1)
                        * (n + alpha + 1)
                        * (n + beta + 1)
                        * (n + s + 1)
                        / ((k + 1) * (k + 3))
                    )
                )
                steps.append((diag, 1 / off, prev))
                prev = off
            self.modes.append((bl + 2, norm, tuple(steps)))

    def forward(self, distance, atomic_numbers, edge_index):
        x = scaled_distance(distance, atomic_numbers, edge_index, self.lengthscale)
        cos_t = torch.rsqrt(1 + x.square())
        sin_t = x * cos_t
        z = 2 * cos_t.square() - 1
        values = []
        for l, (power, norm, steps) in enumerate(self.modes):
            current = norm * cos_t.pow(power)
            if l:
                current = current * sin_t.pow(l)
            values.append(current)
            previous = 0.0
            for diag, inv_off, prev_off in steps:
                previous, current = (
                    current,
                    ((z - diag) * current - prev_off * previous) * inv_off,
                )
                values.append(current)
        return torch.stack(values, dim=-1)


PHYSICAL_BASES = {"physical": PhysicalRadialBasis, "pt_physical": PtPhysicalRadialBasis}

"""Spherical-harmonic quadrature transforms."""

import math

import torch
from e3nn import o3
from scipy.integrate import lebedev_rule

from .._grid import Grid


def sphere_grid(resolution, quadrature):
    """Return CPU float64 sphere points, normalized weights, and exactness degree."""
    if quadrature == "lebedev":
        orders = (*range(3, 32, 2), *range(35, 132, 6))
        order = next((order for order in orders if order >= resolution), None)
        if order is None:
            raise ValueError(
                "Lebedev exactness is limited to 131; use gauss_legendre for higher orders."
            )
        points, weights = lebedev_rule(order)
        return (
            torch.tensor(points.T.copy(), dtype=torch.float64, device="cpu"),
            torch.tensor(weights.copy(), dtype=torch.float64, device="cpu")
            / (4 * math.pi),
            order,
        )
    if quadrature != "gauss_legendre":
        raise ValueError("quadrature must be lebedev, gauss_legendre, or equiangular.")
    res_beta = (resolution + 2) // 2
    k = torch.arange(1, res_beta, dtype=torch.float64, device="cpu")
    diagonal = k / torch.sqrt(4 * k.square() - 1)
    matrix = torch.diag(diagonal, diagonal=1) + torch.diag(diagonal, diagonal=-1)
    if res_beta == 1:
        matrix = torch.zeros(1, 1, dtype=torch.float64, device="cpu")
    nodes, vectors = torch.linalg.eigh(matrix)
    weights = vectors[0].square()
    alpha = torch.arange(resolution + 1, dtype=torch.float64, device="cpu")
    alpha = alpha * (2 * math.pi / (resolution + 1))
    beta, alpha = torch.meshgrid(nodes.acos(), alpha, indexing="ij")
    points = o3.angles_to_xyz(alpha, beta).reshape(-1, 3)
    weights = weights[:, None].expand(-1, resolution + 1).reshape(-1) / (resolution + 1)
    return points, weights, resolution


def wigner_basis(lmax, alpha, beta):
    """Evaluate real Wigner matrices without changing the default dtype."""
    from ..co2.spherical import generators

    for l in range(lmax + 1):
        generator = generators(l)
        yield torch.matrix_exp(alpha[:, None, None] * generator[1]) @ torch.matrix_exp(
            beta[:, None, None] * generator[0]
        )


class S2Grid(Grid):
    """Transform spherical-harmonic coefficients to a sphere grid.

    Parameters
    ----------
    lmax : int
        Maximum angular degree. The coefficient dimension is ``(lmax + 1)**2``.
    resolution : int, optional
        Requested quadrature exactness degree, at least ``2 * lmax``.
        Defaults to ``max(1, 3 * lmax)``. Lebedev selects the next available rule.
    normalization : {"component", "norm", "integral"}, optional
        Spherical grid normalization. ``"component"`` and ``"norm"`` give
        equal variance per degree for unit-variance and unit-norm inputs,
        respectively. ``"integral"`` uses surface-orthonormal harmonics.
    quadrature : {"lebedev", "gauss_legendre", "equiangular"}, optional
        Sphere integration rule.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to the default floating-point dtype.
    device : torch.device or str, optional
        Buffer device. Defaults to the default device.

    Notes
    -----
    Coefficients use the real spherical basis in degree order. Input shape
    is ``(..., (lmax + 1)**2)`` and grid shape is ``(..., n_points)``.
    Channels occupy a leading dimension. Weights integrate the normalized
    sphere measure and sum to one. A degree-d pointwise polynomial is
    projected exactly when the quadrature degree is at least ``(d + 1) * lmax``.
    """

    def __init__(
        self,
        lmax,
        resolution=None,
        *,
        normalization="component",
        quadrature="lebedev",
        dtype=None,
        device=None,
    ):
        if not isinstance(lmax, int) or lmax < 0:
            raise ValueError("lmax must be a non-negative integer.")
        resolution = max(1, 3 * lmax) if resolution is None else resolution
        if not isinstance(resolution, int) or resolution < max(1, 2 * lmax):
            raise ValueError("resolution must be an integer at least max(1, 2 * lmax).")
        if normalization not in ("component", "norm", "integral"):
            raise ValueError("normalization must be component, norm, or integral.")
        self.lmax = lmax
        self.resolution = resolution
        self.normalization = normalization
        self.quadrature = quadrature
        degrees = torch.tensor(
            [2 * l + 1 for l in range(lmax + 1) for _ in range(2 * l + 1)],
            dtype=torch.float64,
            device="cpu",
        )
        if normalization == "component":
            scale = (degrees * (lmax + 1)).rsqrt()
        elif normalization == "norm":
            scale = torch.full_like(degrees, 1 / math.sqrt(lmax + 1))
        else:
            scale = torch.full_like(degrees, 1 / math.sqrt(4 * math.pi))
        if quadrature == "equiangular":
            res_beta = max(2 * (lmax + 1), 2 * ((resolution + 2) // 2))
            res_alpha = resolution + 1
            to_grid = o3.ToS2Grid(
                lmax,
                (res_beta, res_alpha),
                normalization=normalization,
                dtype=torch.float64,
                device="cpu",
            )
            points = to_grid.grid.reshape(-1, 3)
            synthesis = torch.einsum("am,mbi->bai", to_grid.sha, to_grid.shb).reshape(
                -1, (lmax + 1) ** 2
            )
            # Midpoint sphere quadrature, evaluated entirely in float64.
            orders = torch.arange(1, res_beta, 2, dtype=torch.float64, device="cpu")
            beta = to_grid.betas
            weights = (beta[:, None] * orders).sin().div(orders).sum(-1)
            weights = weights * beta.sin() * (2 / res_beta)
            weights = weights[:, None].expand(-1, res_alpha).reshape(-1) / res_alpha
            analysis = (synthesis * weights[:, None] / scale.square()).T
            self.degree = min(res_beta - 1, res_alpha - 1)
        else:
            points, weights, self.degree = sphere_grid(resolution, quadrature)
            if lmax <= 12:
                basis = o3.spherical_harmonics(
                    list(range(lmax + 1)),
                    points,
                    normalize=False,
                    normalization="component",
                )
            else:
                alpha, beta = o3.xyz_to_angles(points)
                basis = torch.cat(
                    [
                        matrix[:, :, l] * math.sqrt(2 * l + 1)
                        for l, matrix in enumerate(wigner_basis(lmax, alpha, beta))
                    ],
                    dim=-1,
                )
            synthesis = basis * scale
            analysis = (basis * (weights[:, None] / scale)).T
        super().__init__(
            dict(grid=points, weights=weights, synthesis=synthesis, analysis=analysis),
            dtype=dtype,
            device=device,
        )

    def extra_repr(self):
        return (
            f"lmax={self.lmax}, resolution={self.resolution}, "
            f"normalization={self.normalization}, quadrature={self.quadrature}"
        )

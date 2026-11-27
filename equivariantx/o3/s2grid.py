"""Unit-sphere sampling and spherical-harmonic quadrature transforms."""

import math

import torch
from e3nn import o3

from ..ops import resolve_backend
from ..ops.precision import SUPPORTED_DTYPES, validate_module_precision
from ..ops.torch.o3 import s2grid as torch_kernel
from .rotation import so3_generators

__all__ = ["S2Grid", "s2_points"]


def _healpix_rings(num_points, device):
    nside = math.isqrt(num_points // 12)
    if 12 * nside**2 != num_points:
        raise ValueError("HEALPix requires num_points = 12 * nside**2.")
    ring = torch.arange(1, 4 * nside, device=device)
    polar_ring = torch.minimum(ring, 4 * nside - ring)
    width = 4 * polar_ring.clamp(max=nside)
    row = torch.repeat_interleave(width, output_size=num_points)
    offset = torch.arange(num_points, device=device) - (width.cumsum(0) - width)[row]

    polar = polar_ring < nside
    r = polar_ring.double() / nside
    # h = 1 - |z|, computed directly to preserve transverse precision at the poles.
    h = torch.where(polar, r.square() / 3, (2 * r - 1) / 3)
    z = (2 * nside - ring).sign() * (1 - h)
    half_step = polar | ((ring + nside) % 2 == 0)
    phi = ((offset + 0.5 * half_step[row].double()) / width[row]) * math.tau
    return z[row], phi, (h * (2 - h)).sqrt()[row]


def _icosphere_points(num_points, device):
    levels = 0
    while 10 * 4**levels + 2 < num_points:
        levels += 1
    if 10 * 4**levels + 2 != num_points:
        raise ValueError("Icosphere requires num_points = 10 * 4**level + 2.")
    phi = (1 + math.sqrt(5)) / 2
    base = torch.tensor(
        [(0, 1, phi), (0, -1, phi), (0, 1, -phi), (0, -1, -phi)],
        dtype=torch.float64,
        device=device,
    )
    points = torch.nn.functional.normalize(
        torch.cat((base, base.roll(-1, 1), base[[0, 2, 1, 3]].roll(1, 1))), dim=1
    )
    if levels == 0:
        return points
    # Two distinct icosahedron vertices are adjacent iff their dot product is positive.
    adjacent = (points @ points.T) > 0
    faces = torch.combinations(torch.arange(12, device=device), r=3)
    faces = faces[adjacent[faces, faces.roll(-1, 1)].all(1)]
    for level in range(levels):
        size = points.shape[0]
        other = faces.roll(-1, 1)
        keys = torch.minimum(faces, other) * size + torch.maximum(faces, other)
        if level + 1 == levels:
            edges = torch.unique(keys)
        else:
            edges, inverse = torch.unique(keys, return_inverse=True)
            a, b, c = faces.unbind(1)
            ab, bc, ca = (inverse + size).unbind(1)
            faces = torch.stack(
                (a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca), dim=1
            ).reshape(-1, 3)
        midpoints = points[edges // size] + points[edges % size]
        points = torch.cat((points, torch.nn.functional.normalize(midpoints, dim=1)))
    return points


def s2_points(
    num_points: int,
    method: str = "fibonacci",
    *,
    dtype: torch.dtype | None = None,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return Cartesian unit directions on the two-sphere.

    Parameters
    ----------
    num_points : int
        Number of directions. HEALPix requires ``12 * nside**2`` for positive
        integer ``nside``. Icosphere requires ``10 * 4**level + 2`` for
        non-negative integer ``level``.
    method : {"fibonacci", "healpix", "icosphere", "random"}, optional
        Fibonacci spiral including both poles for two or more points,
        HEALPix pixel centers in RING order, subdivided icosahedron vertices,
        or independent directions uniform in surface area.
    dtype : torch.dtype, optional
        Output dtype, float32 or float64. Defaults to the default floating dtype.
    device : torch.device or str, optional
        Output device. Defaults to the PyTorch default device.
    generator : torch.Generator, optional
        Random generator on the output device for ``method="random"``.

    Returns
    -------
    torch.Tensor
        Unit vectors of shape ``(num_points, 3)`` in ``(x, y, z)`` order.
        For quadrature weights and harmonic transforms, see :class:`S2Grid`.

    Examples
    --------
    >>> from equivariantx import o3
    >>> o3.s2_points(48, method="healpix").shape
    torch.Size([48, 3])
    """
    if type(num_points) is not int or num_points < 1:
        raise ValueError("num_points must be a positive integer.")
    dtype = torch.get_default_dtype() if dtype is None else dtype
    if dtype not in SUPPORTED_DTYPES:
        raise TypeError("s2_points requires float32 or float64.")
    if generator is not None and method != "random":
        raise ValueError("generator is only supported with method='random'.")
    device = torch.get_default_device() if device is None else torch.device(device)
    # Deterministic grids are constructed in float64, including long spiral phases.
    work_device = "cpu" if device.type == "mps" else device
    radius = None
    if method == "fibonacci":
        index = torch.arange(num_points, dtype=torch.float64, device=work_device)
        z = (
            1 - 2 * index / (num_points - 1)
            if num_points > 1
            else torch.zeros_like(index)
        )
        phi = index * (math.pi * (3 - math.sqrt(5)))
    elif method == "healpix":
        z, phi, radius = _healpix_rings(num_points, work_device)
    elif method == "icosphere":
        return _icosphere_points(num_points, work_device).to(device=device, dtype=dtype)
    elif method == "random":
        u, v = torch.rand(
            2, num_points, dtype=dtype, device=device, generator=generator
        )
        z, phi = 2 * v - 1, math.tau * u
    else:
        raise ValueError("method must be fibonacci, healpix, icosphere, or random.")
    if radius is None:
        radius = ((1 - z) * (1 + z)).sqrt()
    return torch.stack((radius * phi.cos(), radius * phi.sin(), z), dim=1).to(
        device=device, dtype=dtype
    )


def execution(module, features):
    if getattr(module, "backend", "torch") == "cuda" and features.is_cuda:
        from ..ops.cuda.o3 import s2grid

        return s2grid
    return torch_kernel


def sphere_grid(resolution, quadrature):
    """Return CPU float64 sphere points, normalized weights, and exactness degree."""
    if quadrature == "lebedev":
        from scipy.integrate import lebedev_rule

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
    """Yield real Wigner D matrices without changing the default dtype."""
    for l in range(lmax + 1):
        generator = so3_generators(l)
        yield torch.matrix_exp(alpha[:, None, None] * generator[1]) @ torch.matrix_exp(
            beta[:, None, None] * generator[0]
        )


class S2Grid(torch.nn.Module):
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
        Sphere integration rule. Defaults to ``"lebedev"``.
    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CPU tensors use Torch.
    dtype : torch.dtype, optional
        Buffer dtype. Defaults to the default floating-point dtype.
    device : torch.device or str, optional
        Buffer device. Defaults to the default device.

    Attributes
    ----------
    grid : torch.Tensor
        Unit-sphere positions of shape ``(n_points, 3)``.
    weights : torch.Tensor
        Quadrature weights of shape ``(n_points,)``, summing to one.
    degree : int
        Polynomial exactness degree of the selected quadrature.

    Notes
    -----
    Coefficients use the real spherical basis in degree order. Input shape
    is ``(..., (lmax + 1)**2)`` and grid shape is ``(..., n_points)``.
    Channels occupy a leading dimension. Weights integrate the normalized
    sphere measure and sum to one. A degree-d pointwise polynomial is
    projected exactly when the quadrature degree is at least ``(d + 1) * lmax``.
    General nonlinearities require a resolution-convergence check. The grid
    represents scalar functions with natural spatial parity; it does not
    encode arbitrary spatial or time-parity assignments.

    Examples
    --------
    >>> from equivariantx import o3
    >>> grid = o3.S2Grid(lmax=2, resolution=8)
    >>> x = torch.randn(5, 9)
    >>> values = grid.to_grid(x)
    >>> torch.testing.assert_close(grid.from_grid(values), x)
    >>> grid.from_grid(values.pow(3)).shape
    torch.Size([5, 9])
    """

    def __init__(
        self,
        lmax,
        resolution=None,
        *,
        normalization="component",
        quadrature="lebedev",
        backend=None,
        dtype=None,
        device=None,
    ):
        backend = resolve_backend(backend, device)
        super().__init__()
        self.backend = backend
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
        self.dim = (lmax + 1) ** 2
        self.grid_shape = tuple(weights.shape)
        self._constants = dict(
            grid=points, weights=weights, synthesis=synthesis, analysis=analysis
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
        """Evaluate spherical-harmonic coefficients on the grid.

        Parameters
        ----------
        features : torch.Tensor
            Coefficients of shape ``(..., (lmax + 1)**2)``.

        Returns
        -------
        torch.Tensor
            Grid values of shape ``(..., n_points)``.
        """
        return self.to_grid(features)

    def to_grid(self, features):
        """Evaluate a band-limited signal on the sphere.

        Parameters
        ----------
        features : torch.Tensor
            Coefficients of shape ``(..., (lmax + 1)**2)``. Leading dimensions
            may include independent batch and channel axes.

        Returns
        -------
        torch.Tensor
            Grid values of shape ``(..., n_points)``.
        """
        validate_module_precision(self, (features,))
        if features.ndim < 1 or features.shape[-1] != self.dim:
            raise ValueError(f"Expected {self.dim} coefficients on the last axis.")
        return execution(self, features).to_grid(self, features)

    def from_grid(self, features):
        """Project grid values onto spherical-harmonic coefficients.

        Parameters
        ----------
        features : torch.Tensor
            Grid values of shape ``(..., n_points)``.

        Returns
        -------
        torch.Tensor
            Coefficients of shape ``(..., (lmax + 1)**2)``.
        """
        validate_module_precision(self, (features,))
        if features.ndim < 1 or features.shape[-1] != self.weights.numel():
            raise ValueError(
                f"Expected {self.weights.numel()} grid values on the last axis."
            )
        return execution(self, features).from_grid(self, features)

    def extra_repr(self):
        return (
            f"lmax={self.lmax}, resolution={self.resolution}, "
            f"normalization={self.normalization}, quadrature={self.quadrature}"
        )

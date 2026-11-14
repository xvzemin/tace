"""Transverse couplings acting directly on spherical coefficients."""

import math
from functools import lru_cache

import torch
from e3nn import o3

from ..o3.rotation import so3_generators
from ..ops import resolve_backend
from ..ops.common.sparse import bilinear_rows
from ..ops.precision import validate_module_precision
from ..ops.torch.co2 import spherical as torch_kernel


def execution(module, features):
    if getattr(module, "backend", "torch") == "cuda" and features.is_cuda:
        from ..ops.cuda.co2 import spherical as cuda_kernel

        return cuda_kernel
    return torch_kernel


@lru_cache(maxsize=256)
def coupling_polynomial(l1, l2, l3, normalization="component"):
    """Return the Chebyshev coefficients of a transverse coupling.

    Parameters
    ----------
    l1, l2, l3 : int
        Input, harmonic and output degrees satisfying the triangle rule.
    normalization : {"component", "norm", "integral"}, optional
        Harmonic normalization. Tensor-product path weights are not included.

    Returns
    -------
    tuple of float
        Coefficients in ascending polynomial order.

    Notes
    -----
    Coefficients are computed from reference-axis CG entries in float64.
    Odd couplings apply the rotation generator once in addition to the
    polynomial of its square.
    """
    if not abs(l1 - l3) <= l2 <= l1 + l3:
        raise ValueError("The degrees must satisfy the triangle rule.")
    if normalization not in ("component", "norm", "integral"):
        raise ValueError("normalization must be component, norm, or integral.")
    delta = abs(l1 - l3)
    odd = (l1 + l2 + l3) % 2
    order = (l2 - delta - odd) // 2
    bridge = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")[
        :, delta, :
    ].T
    degree = min(l1, l3)
    generator = so3_generators(degree)[1] / math.sqrt(max(1, degree * (degree + 1)))

    def apply_generator(value):
        return value @ generator if l1 < l3 else generator @ value

    if odd:
        bridge = apply_generator(bridge)
    # Eigenvalues lie in [-1, 1]; this avoids a monomial Vandermonde system.
    spectral_scale = degree * (degree + 1) / max(1, degree**2)

    def operator(value):
        return -2 * spectral_scale * apply_generator(apply_generator(value)) - value

    values = [bridge]
    if order:
        values.append(operator(bridge))
    for _ in range(2, order + 1):
        values.append(2 * operator(values[-1]) - values[-2])
    target = o3.wigner_3j(l1, l2, l3, dtype=torch.float64, device="cpu")[:, l2, :].T
    scale = 1.0 if normalization == "norm" else math.sqrt(2 * l2 + 1)
    if normalization == "integral":
        scale /= math.sqrt(4 * math.pi)
    matrix = torch.stack([value.flatten() for value in values], dim=1)
    coefficients = torch.linalg.lstsq(
        matrix, target.flatten() * scale, driver="gelsd"
    ).solution
    residual = matrix @ coefficients - target.flatten() * scale
    if residual.abs().max() > 2e-12 * max(1.0, scale):
        raise RuntimeError("The transverse polynomial does not resolve the coupling.")
    return tuple(coefficients.tolist())


@lru_cache(maxsize=256)
def coupling_recurrence(l1, l3, lmax=None):
    """Return adjacent-degree coefficients of normalized CG operators.

    Parameters
    ----------
    l1, l3 : int
        Input and output angular degrees.
    lmax : int, optional
        Largest harmonic degree needed. Defaults to ``l1 + l3``.

    Returns
    -------
    tuple of float
        Coefficients connecting successive degrees from ``abs(l1 - l3)``
        through ``lmax``, with signs following the installed CG convention.

    Notes
    -----
    The generator is divided by ``sqrt(d * (d + 1))``, where
    ``d = min(l1, l3)``. Reference-axis operators have unit Frobenius norm.
    Coefficients are independent of direction and feature multiplicity.
    """
    degree, delta = min(l1, l3), abs(l1 - l3)
    lmax = l1 + l3 if lmax is None else lmax
    if not delta <= lmax <= l1 + l3:
        raise ValueError("lmax must satisfy the triangle rule.")
    if lmax == delta:
        return ()
    generator = so3_generators(degree)[1]
    previous = o3.wigner_3j(l1, delta, l3, dtype=torch.float64, device="cpu")[
        :, delta, :
    ].T
    coefficients = []
    for l in range(delta + 1, lmax + 1):
        current = o3.wigner_3j(l1, l, l3, dtype=torch.float64, device="cpu")[:, l, :].T
        acted = previous @ generator if l1 < l3 else generator @ previous
        magnitude = math.sqrt(
            ((l1 + l3 + 1) ** 2 - l**2)
            * (l**2 - delta**2)
            / (4 * (4 * l**2 - 1) * degree * (degree + 1))
        )
        coefficients.append(math.copysign(magnitude, (acted * current).sum().item()))
        previous = current
    return tuple(coefficients)


class SphericalCoupling(torch.nn.Module):
    """Apply a transverse coupling without selecting a local frame.

    Parameters
    ----------
    l1, l2, l3 : int
        Input, harmonic and output degrees.
    normalization : {"component", "norm", "integral"}, optional
        Harmonic normalization.
    method : {"chebyshev", "recurrence"}, optional
        Evaluate a Chebyshev expansion or a CG recurrence at fixed parity.

    backend : {None, "torch", "cuda"}, optional
        ``None`` selects CUDA when available, otherwise optimized PyTorch.
        Execution backend. CPU tensors use Torch.

    Notes
    -----
    Features remain spherical. A minimum-degree coupling and a polynomial
    of the rotation generator implement the transverse restriction and lift.

    Examples
    --------
    >>> import torch
    >>> from equivariantx import co2
    >>> layer = co2.SphericalCoupling(1, 1, 0)
    >>> layer(torch.randn(4, 3), torch.randn(4, 3)).shape
    torch.Size([4, 1])
    """

    def __init__(
        self,
        l1,
        l2,
        l3,
        normalization="component",
        method="chebyshev",
        *,
        backend=None,
    ):
        backend = resolve_backend(backend)
        super().__init__()
        if method not in ("chebyshev", "recurrence"):
            raise ValueError("method must be chebyshev or recurrence.")
        self.backend = backend
        self.method = method
        self.l1, self.l2, self.l3 = l1, l2, l3
        self.normalization = normalization
        self.delta = abs(l1 - l3)
        self.degree = min(l1, l3)
        self.odd = (l1 + l2 + l3) % 2
        self.spectral_scale = self.degree * (self.degree + 1) / max(1, self.degree**2)
        self.coefficients = coupling_polynomial(l1, l2, l3, normalization)
        self.recurrence = coupling_recurrence(l1, l3, l2)
        self.harmonics = o3.SphericalHarmonics(
            self.delta, normalize=False, normalization="norm"
        )
        self.register_buffer(
            "cg",
            torch.empty(2 * l1 + 1, 2 * self.delta + 1, 2 * l3 + 1),
            persistent=False,
        )
        self.register_buffer(
            "generator",
            torch.empty(3, 2 * self.degree + 1, 2 * self.degree + 1),
            persistent=False,
        )
        self._apply(lambda value: value)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        cg = o3.wigner_3j(
            self.l1, self.delta, self.l3, dtype=torch.float64, device="cpu"
        )
        generator = so3_generators(self.degree) / math.sqrt(
            max(1, self.degree * (self.degree + 1))
        )
        self.cg = cg.to(self.cg)
        self.generator = generator.to(self.generator)
        for name, coefficients in (
            ("bridge", cg),
            ("generator", generator.permute(2, 0, 1)),
        ):
            for suffix, value in zip(
                ("indices", "columns", "coefficients"), bilinear_rows(coefficients)
            ):
                key = name + "_" + suffix
                value = value.to(
                    device=self.cg.device,
                    dtype=self.cg.dtype if value.is_floating_point() else torch.long,
                )
                if key in self._buffers:
                    setattr(self, key, value)
                else:
                    self.register_buffer(key, value, persistent=False)
        return self

    def forward(self, features, vectors, method=None):
        """Couple (..., 2*l1+1) features to harmonics of (..., 3) vectors."""
        validate_module_precision(self, (features, vectors, method))
        method = self.method if method is None else method
        if method not in ("chebyshev", "recurrence"):
            raise ValueError("method must be chebyshev or recurrence.")
        return execution(self, features).forward(self, features, vectors, method)

    def extra_repr(self):
        return (
            f"{self.l1} x Y({self.l2}) -> {self.l3}, "
            f"normalization={self.normalization!r}, method={self.method!r}"
        )

"""Transverse couplings acting directly on spherical coefficients."""

import math
from functools import lru_cache

import torch
from e3nn import o3


@lru_cache(maxsize=64)
def generators(l):
    """Return real rotation generators in the spherical basis, in CPU float64."""
    if l == 0:
        return torch.zeros(3, 1, 1, dtype=torch.float64, device="cpu")
    value = -math.sqrt(l * (l + 1) * (2 * l + 1)) * o3.wigner_3j(
        l, 1, l, dtype=torch.float64, device="cpu"
    ).permute(1, 2, 0)
    return value * value[1, l - 1, l + 1].sign()


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
    The minimum-degree coupling maps degree l1 to l3. Each transverse order
    is then weighted by a polynomial of the squared rotation generator.
    Odd couplings additionally apply that generator once. Coefficients are
    resolved from exact reference-axis CG entries, not sampled directions.
    No Cartesian tensors or alignment rotations are constructed.
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
    generator = generators(degree)[1] / math.sqrt(max(1, degree * (degree + 1)))

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
    generator = generators(degree)[1]
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

    Notes
    -----
    Features remain spherical. A minimum-degree coupling and a polynomial
    of the rotation generator implement the transverse restriction and lift.
    """

    def __init__(self, l1, l2, l3, normalization="component", method="chebyshev"):
        super().__init__()
        if method not in ("chebyshev", "recurrence"):
            raise ValueError("method must be chebyshev or recurrence.")
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
        self.cg = o3.wigner_3j(
            self.l1, self.delta, self.l3, dtype=torch.float64, device="cpu"
        ).to(self.cg)
        self.generator = (
            generators(self.degree) / math.sqrt(max(1, self.degree * (self.degree + 1)))
        ).to(self.generator)
        return self

    def forward(self, features, vectors, method=None):
        """Couple (..., 2*l1+1) features to harmonics of (..., 3) vectors."""
        method = self.method if method is None else method
        if method not in ("chebyshev", "recurrence"):
            raise ValueError("method must be chebyshev or recurrence.")
        direction = vectors / vectors.norm(dim=-1, keepdim=True)
        harmonic = self.harmonics(direction)

        def bridge(value):
            projected = (value @ self.cg.flatten(1)).unflatten(
                -1, (2 * self.delta + 1, 2 * self.l3 + 1)
            )
            return (projected * harmonic.unsqueeze(-1)).sum(-2)

        value = features if self.l1 < self.l3 else bridge(features)

        def apply_generator(value):
            projected = (value @ self.generator.flatten(0, 1).T).unflatten(
                -1, (3, 2 * self.degree + 1)
            )
            return (projected * direction.unsqueeze(-1)).sum(-2)

        if method == "recurrence":
            value = math.sqrt(2 * self.delta + 1) * value
            if self.odd:
                value = apply_generator(value) / self.recurrence[0]
            previous = None
            for k in range(self.odd, self.l2 - self.delta, 2):
                diagonal = self.recurrence[k] ** 2
                if k:
                    diagonal += self.recurrence[k - 1] ** 2
                following = apply_generator(apply_generator(value)) + diagonal * value
                if k >= 2:
                    following = following - (
                        self.recurrence[k - 1] * self.recurrence[k - 2] * previous
                    )
                previous, value = (
                    value,
                    following / (self.recurrence[k] * self.recurrence[k + 1]),
                )
            if self.normalization == "norm":
                value = value / math.sqrt(2 * self.l2 + 1)
            elif self.normalization == "integral":
                value = value / math.sqrt(4 * math.pi)
            return bridge(value) if self.l1 < self.l3 else value

        if self.odd:
            value = apply_generator(value)
        previous = value
        result = self.coefficients[0] * value
        for k, coefficient in enumerate(self.coefficients[1:], start=1):
            following = (
                -2 * self.spectral_scale * apply_generator(apply_generator(value))
                - value
            )
            if k > 1:
                following = 2 * following - previous
            previous, value = value, following
            result = result + coefficient * value
        return bridge(result) if self.l1 < self.l3 else result

    def extra_repr(self):
        return (
            f"{self.l1} x Y({self.l2}) -> {self.l3}, "
            f"normalization={self.normalization!r}, method={self.method!r}"
        )

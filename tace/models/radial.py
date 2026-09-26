################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Union

import ase.data
import numpy as np
import torch
from scipy.optimize import brentq
from scipy.special import spherical_jn

from tace.utils.torch_scatter import scatter_sum


def compute_jn_zeros(n: int, k: int) -> np.ndarray:
    """Return the first ``k`` positive zeros of the spherical Bessel function."""
    if not isinstance(n, int) or n < 0:
        raise ValueError("n must be a nonnegative integer.")
    if not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")
    if n == 0:
        return np.arange(1, k + 1, dtype=np.float64) * math.pi

    # The first zero exceeds n, and consecutive positive zeros are more than pi apart.
    zeros = []
    left = float(n)
    value = spherical_jn(n, left)
    while len(zeros) < k:
        right = left + math.pi
        next_value = spherical_jn(n, right)
        if value * next_value < 0 or next_value == 0:
            zeros.append(
                brentq(lambda x: spherical_jn(n, x), left, right, xtol=1e-14)
            )
        left, value = right, next_value
    return np.array(zeros, dtype=np.float64)


def spherical_bessel_jn(n: int, x: torch.Tensor) -> torch.Tensor:
    """Evaluate a spherical Bessel function, including derivatives at zero.

    Parameters
    ----------
    n : int
        Nonnegative order.
    x : torch.Tensor
        Real arguments in double precision.
    """
    limit = math.sqrt(8 * n + 12) if n else 1.0
    small = x.abs() <= limit
    argument = torch.where(small, x, 0.0)
    squared = argument.square()
    coefficients = [1.0]
    for k in range(1, 25 if n else 13):
        coefficients.append(-coefficients[-1] / (2 * k * (2 * n + 2 * k + 1)))
    series = torch.full_like(x, coefficients[-1])
    for coefficient in reversed(coefficients[:-1]):
        series = series * squared + coefficient
    if n:
        # Scaling before taking the power avoids overflow in x**n and (2*n+1)!!.
        scale = math.exp(math.log(math.prod(range(1, 2 * n + 2, 2))) / n)
        series = (argument / scale).pow(n) * series

    # Ascending recurrence is stable above the order. Inactive branches must
    # also remain finite, otherwise torch.where can propagate NaN derivatives.
    argument = torch.where(x.abs() > max(n, limit), x, max(n, limit))
    previous = argument.sin() / argument
    if n == 0:
        return torch.where(small, series, previous)
    current = (previous - argument.cos()) / argument
    for k in range(1, n):
        previous, current = current, (2 * k + 1) / argument * current - previous

    if n > limit:
        middle = (~small) & (x.abs() <= n)
        argument = torch.where(middle, x, limit)
        previous = torch.zeros_like(argument)
        value = torch.ones_like(argument)
        target = torch.zeros_like(argument)
        # Start above the turning region, whose width grows as n**(1/3).
        for k in range(n + 32 + math.ceil(8 * n ** (1 / 3)), 0, -1):
            previous, value = value, (2 * k + 1) / argument * value - previous
            if k - 1 == n:
                target = value
            if k % 16 == 0 or k == 1:
                scale = torch.maximum(value.abs(), previous.abs()).clamp_min(1.0)
                value, previous, target = value / scale, previous / scale, target / scale
        j0 = argument.sin() / argument
        j1 = (j0 - argument.cos()) / argument
        # j0 and j1 have no common nonzero root; use both to normalize Miller's recurrence.
        descending = target * (j0 * value + j1 * previous) / (
            value.square() + previous.square()
        )
        current = torch.where(middle, descending, current)
    return torch.where(small, series, current)


class j0SphericalBesselBasis(torch.nn.Module):
    """The Bessel Basis is proposed in the DimeNet: https://www.cs.cit.tum.de/daml/dimenet/"""

    def __init__(
        self, cutoff: float = 6.0, num_basis: int = 8, trainable: bool = False
    ) -> None:
        super().__init__()
        self.num_basis = num_basis
        bessel_roots = (
            math.pi
            / cutoff
            * torch.linspace(
                start=1.0,
                end=num_basis,
                steps=num_basis,
                dtype=torch.get_default_dtype(),
            )
        )
        if trainable:
            self.bessel_weights = torch.nn.Parameter(bessel_roots)
        else:
            self.register_buffer("bessel_weights", bessel_roots)

        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "prefactor",
            torch.tensor(math.sqrt(2.0 / cutoff), dtype=torch.get_default_dtype()),
        )

    def forward(
        self, r: torch.Tensor, node_attrs: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:  # [..., 1]
        basis = self.bessel_weights * torch.sinc((self.bessel_weights / math.pi) * r)
        return self.prefactor * basis
    
    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, num_basis={self.num_basis}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


class jnSphericalBesselBasis(torch.nn.Module):
    """Spherical Bessel radial basis, normalized with the measure ``r**2 dr``.

    Parameters
    ----------
    cutoff : float
        Positive radial cutoff.
    order : int or list of int
        Nonnegative orders, in output order.
    num_basis : int or list of int
        Number of positive zeros for each order.
    trainable : bool
        Whether to optimize the zeros. Normalization retains its initial value.
    """

    def __init__(
        self,
        cutoff: float = 6.0,
        order: Union[int, list[int]] = 0,
        num_basis: Union[int, list[int]] = 8,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if not math.isfinite(cutoff) or cutoff <= 0:
            raise ValueError("cutoff must be positive and finite.")
        order = [order] if isinstance(order, int) else order
        num_basis = [num_basis] if isinstance(num_basis, int) else num_basis
        if not isinstance(order, list) or not order or not all(
            isinstance(n, int) and n >= 0 for n in order
        ):
            raise ValueError("order must be a nonnegative integer or a nonempty list of them.")
        if not isinstance(num_basis, list) or not num_basis or not all(
            isinstance(k, int) and k > 0 for k in num_basis
        ):
            raise ValueError("num_basis must be a positive integer or a nonempty list of them.")
        if len(order) != len(num_basis):
            raise ValueError("order and num_basis must have the same length.")

        zeros = [compute_jn_zeros(n, k) for n, k in zip(order, num_basis)]
        normalizer = np.concatenate(
            [
                math.sqrt(2.0 / cutoff**3) / np.abs(spherical_jn(n + 1, z))
                for n, z in zip(order, zeros)
            ]
        )
        self.register_buffer(
            "normalizer",
            torch.tensor(normalizer, dtype=torch.get_default_dtype()).unsqueeze(0),
        )
        zeros = torch.tensor(
            np.concatenate(zeros), dtype=torch.get_default_dtype()
        ).unsqueeze(0)
        if trainable:
            self.zeros = torch.nn.Parameter(zeros)
        else:
            self.register_buffer("zeros", zeros)
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.order = list(order)
        self.num_zero = list(num_basis)

    def forward(
        self, r: torch.Tensor, node_attrs: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:
        argument = self.zeros.to(torch.float64) * (
            r.to(torch.float64) / self.cutoff.to(torch.float64)
        )
        basis = torch.cat(
            [
                spherical_bessel_jn(n, x)
                for n, x in zip(self.order, argument.split(self.num_zero, dim=-1))
            ],
            dim=-1,
        )
        return (basis * self.normalizer.to(torch.float64)).to(r.dtype)

    def extra_repr(self):
        return (
            f"cutoff={self.cutoff.item()}, order={self.order}, num_basis={self.num_zero}, "
            f"trainable={self.zeros.requires_grad}"
        )


class GaussianBasis(torch.nn.Module):
    def __init__(
        self,
        cutoff: float = 6.0,
        num_basis: int = 64,
        width: float = 2.0,
    ) -> None:
        super().__init__()
        self.cutoff = cutoff
        self.num_basis = num_basis
        self.width = width
        offset = torch.linspace(0.0, cutoff, num_basis)
        self.coeff = -0.5 / (width * (offset[1] - offset[0])).item() ** 2
        self.register_buffer("offset", offset, persistent=False)

    def forward(
        self, r: torch.Tensor, node_attrs: torch.Tensor, edge_index: torch.Tensor
    ) -> torch.Tensor:  # [..., 1]
        r = r - self.offset.view(1, -1)
        return torch.exp(self.coeff * torch.pow(r, 2))

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff}, width={self.width})"


class MagneticChebyshevBasis(torch.nn.Module):
    """
    Chebyshev basis assuming inputs are already normalized to `[-1, 1]`.
    num_basis should generally remain small (typically below 10), 
    as numerical errors may accumulate at higher orders (recurrence relation).
    """

    def __init__(
        self,
        num_basis: int = 8,
        include_constant: bool = False,
    ) -> None:
        super().__init__()

        if not isinstance(include_constant, bool):
            raise TypeError("include_constant must be a bool")
        self.num_basis = num_basis
        self.include_constant = include_constant

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t0 = torch.ones_like(x)
        t1 = x
        basis = [t0, t1]
        start = 0 if self.include_constant else 1
        stop = start + self.num_basis
        for _ in range(2, stop):
            t0, t1 = t1, 2.0 * x * t1 - t0
            basis.append(t1)
        return torch.cat(basis[start:stop], dim=-1)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(num_basis={self.num_basis}, "
            f"include_constant={self.include_constant})"
        )


class CosineCutoff(torch.nn.Module):
    """
    The fourth derivative and above are discontinuous
    """

    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.register_buffer(
            "cutoff",
            torch.tensor(cutoff, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, x: torch.Tensor, cutoff: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if cutoff is None:
            cutoff = self.cutoff
        return self.calculate_envelope(x, cutoff)

    @staticmethod
    def calculate_envelope(r_ij: torch.Tensor, cutoff: torch.Tensor) -> torch.Tensor:
        x = r_ij / cutoff
        envelope = (0.5 * torch.cos(torch.pi * x) + 0.5).unsqueeze(-1)
        return envelope * (r_ij < cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"


class MollifierCutoff(torch.nn.Module):
    """
    Derivatives of any order are continuous
    """

    def __init__(self, cutoff: float, eps: float = 1e-9):
        super().__init__()
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.eps = eps

    def forward(
        self, x: torch.Tensor, cutoff: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if cutoff is None:
            cutoff = self.cutoff
        return self.calculate_envelope(x, cutoff, self.eps)

    @staticmethod
    def calculate_envelope(
        r_ij: torch.Tensor,
        cutoff: torch.Tensor,
        eps: float = 1e-9,
    ) -> torch.Tensor:
        x = r_ij / cutoff
        envelope = torch.exp(1 - 1 / (1 - x**2 + eps))
        return envelope * (r_ij < cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(cutoff={self.cutoff})"


class C2PolynomialCutoff(torch.nn.Module):
    """
    f(1) = f'(1) = f''(1) = 0
    Envelope function(PolynomialCutoff funciton) is proposed
    in the DimeNet: https://www.cs.cit.tum.de/daml/dimenet/
    """

    def __init__(self, cutoff: float, p: int = 5):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )

    def forward(
        self, x: torch.Tensor, cutoff: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if cutoff is None:
            cutoff = self.cutoff
        return self.calculate_envelope(x, cutoff, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        r_ij: torch.Tensor, cutoff: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        x = r_ij / cutoff  # [edge, 1]
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(x, p)
            + p * (p + 2.0) * torch.pow(x, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(x, p + 2)
        )
        return envelope * (r_ij < cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, cutoff={self.cutoff})"


class C3PolynomialCutoff(torch.nn.Module):
    """
    Considering matbench has f'''(x), the undetermined coefficient can yield c3
    f(1) = f'(1) = f''(1) = f'''(1) = 0
    """

    def __init__(self, cutoff: float, p: int = 6):
        super().__init__()
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )

    def forward(
        self, x: torch.Tensor, cutoff: Union[torch.Tensor, None] = None
    ) -> torch.Tensor:
        if cutoff is None:
            cutoff = self.cutoff
        return self.calculate_envelope(x, cutoff, self.p.to(torch.int))

    @staticmethod
    def calculate_envelope(
        r_ij: torch.Tensor, cutoff: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        x = r_ij / cutoff

        # coefficients
        a = -((p + 1.0) * (p + 2.0) * (p + 3.0) / 6.0)
        b = p * (p + 2.0) * (p + 3.0) / 2.0
        c = -(p * (p + 1.0) * (p + 3.0) / 2.0)
        d = p * (p + 1.0) * (p + 2.0) / 6.0

        envelope = (
            1.0
            + a * torch.pow(x, p)
            + b * torch.pow(x, p + 1)
            + c * torch.pow(x, p + 2)
            + d * torch.pow(x, p + 3)
        )

        return envelope * (r_ij < cutoff)

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, cutoff={self.cutoff})"


class SmoothDynamicCutoff(torch.nn.Module):
    """See https://arxiv.org/abs/2601.21147"""

    def __init__(
        self,
        r_max: float,
        cutoff_fn: torch.nn.Module,
        mu: float,
        sigma: float = 4,
        alpha: float = 10.0,
        p: int = 50,
        eps: float = 1e-4,
    ):
        super().__init__()
        self.r_max = r_max
        self.cutoff_fn = cutoff_fn
        self.mu = mu
        self.sigma = sigma
        self.alpha = alpha
        self.eps = eps
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.coef = 1.0 / (self.sigma * math.sqrt(2 * math.pi))

    def __repr__(self):
        return f"{self.__class__.__name__}(mu={self.mu}, sigma={self.sigma}, alpha={self.alpha}, p={self.p.item()})"

    def _group_distances(self, target, edge_length, num_nodes):
        """
        (edge,) -> (node, kmax​)
            r_grouped  # (N, k_max)
            mask       # (N, k_max)  actual negitbor
        """
        edge_length = edge_length.squeeze(-1)
        perm = torch.argsort(target)
        target = target[perm]
        edge_length = edge_length[perm]

        ones = torch.ones_like(target)
        num_neighbors = scatter_sum(ones, target, dim=0, dim_size=num_nodes)
        k_max = num_neighbors.max().item()

        cum = torch.cumsum(num_neighbors, dim=0)
        start = cum - num_neighbors  # each node's edge start at which idx

        edge_pos = torch.arange(len(target), device=target.device)  # global edge_idx
        local_idx = edge_pos - start[target]

        node_length = torch.zeros(
            num_nodes,
            k_max,
            device=edge_length.device,
            dtype=edge_length.dtype,
        )

        node_length[target, local_idx] = edge_length

        mask = node_length > 0
        return node_length, mask, target, perm

    def _soft_rank(
        self, node_length: torch.Tensor, mask: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        k = node_length.size(1)
        dr = node_length.unsqueeze(-1) - node_length.unsqueeze(
            -2
        )  # diff[v, u, t] = r_uv - r_tv
        S = torch.sigmoid(self.alpha * dr)
        eye = torch.eye(k, device=S.device).bool()
        S = S * (~eye.unsqueeze(0))
        p = p.unsqueeze(-2)
        R = (S * p).sum(dim=-1)
        return R

    def _gaussian_weight(self, R):
        return self.coef * torch.exp(-0.5 * ((R - self.mu) / self.sigma) ** 2)

    def _compute_node_cutoff(self, node_length, mask):
        p = (
            self.cutoff_fn.calculate_envelope(
                node_length, cutoff=self.cutoff_fn.cutoff, p=self.p
            )
            * mask
        )
        R = self._soft_rank(node_length, mask, p)
        w = self._gaussian_weight(R) * mask
        wp = w * p
        numerator = (wp * node_length).sum(dim=-1) + self.r_max * self.eps
        denominator = wp.sum(dim=-1) + self.eps
        c_node = numerator / denominator  # (N,)

        return c_node

    def forward(
        self,
        edge_length: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:

        node_length, mask, sorted_target, perm = self._group_distances(
            edge_index[1],
            edge_length,
            num_nodes,
        )
        dcutoff = self._compute_node_cutoff(node_length, mask)
        dcutoff = dcutoff[sorted_target]
        inv_perm = torch.argsort(perm)
        dcutoff = dcutoff[inv_perm]

        return dcutoff.unsqueeze(-1)


class AgnesiTransform(torch.nn.Module):
    """
    See https://doi.org/10.1063/5.0158783
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable=False,
    ):
        super().__init__()
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(a, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(q, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(p, requires_grad=True))
        else:
            self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
            self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
            self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        if node_type is None:
            node_type = node_attrs.argmax(dim=-1)
        node_atomic_numbers = atomic_numbers[node_type].unsqueeze(-1)
        Z_u = node_atomic_numbers[source]
        Z_v = node_atomic_numbers[target]
        r_0: torch.Tensor = 0.5 * (self.covalent_radii[Z_u] + self.covalent_radii[Z_v])
        r_over_r_0 = x / r_0
        return (
            1
            + (
                self.a
                * torch.pow(r_over_r_0, self.q)
                / (1 + torch.pow(r_over_r_0, self.q - self.p))
            )
        ).reciprocal_()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(a={self.a:.4f}, q={self.q:.4f}, p={self.p:.4f})"
        )


class SoftTransform(torch.nn.Module):
    """
    See https://doi.org/10.1063/5.0158783
    """

    def __init__(self, a: float = 0.2, b: float = 3.0, trainable=False):
        super().__init__()
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(a, requires_grad=True))
            self.b = torch.nn.Parameter(torch.tensor(b, requires_grad=True))
        else:
            self.register_buffer("a", torch.tensor(a))
            self.register_buffer("b", torch.tensor(b))

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        if node_type is None:
            node_type = node_attrs.argmax(dim=-1)
        node_atomic_numbers = atomic_numbers[node_type].unsqueeze(-1)
        Z_u = node_atomic_numbers[source]
        Z_v = node_atomic_numbers[target]
        r_0 = (self.covalent_radii[Z_u] + self.covalent_radii[Z_v]) / 4
        r_over_r_0 = x / r_0
        y = (
            x
            + (1 / 2) * torch.tanh(-r_over_r_0 - self.a * torch.pow(r_over_r_0, self.b))
            + 1 / 2
        )
        return y

    def __repr__(self):
        return f"{self.__class__.__name__}(a={self.a.item()}, b={self.b.item()})"


class ZBLBasis(torch.nn.Module):
    """
    Metal units
    From LAMMPS pair_zbl_const.h
    Code from MACE
    """

    p: torch.Tensor

    def __init__(self, cutoff_fn: str, trainable: bool = False, p: int = 5):
        super().__init__()
        # Pre-calculate the p coefficients for the ZBL potential
        self.register_buffer(
            "c",
            torch.tensor(
                [0.1818, 0.5099, 0.2802, 0.02817], dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer("p", torch.tensor(p, dtype=torch.int))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a_exp = torch.nn.Parameter(torch.tensor(0.300, requires_grad=True))
            self.a_prefactor = torch.nn.Parameter(
                torch.tensor(0.4543, requires_grad=True)
            )
        else:
            self.register_buffer("a_exp", torch.tensor(0.300))
            self.register_buffer("a_prefactor", torch.tensor(0.4543))

        self.is_polynomial_cutoff = False
        if cutoff_fn == "mollifier":
            self.cutoff_fn = MollifierCutoff.calculate_envelope
        elif cutoff_fn == "cosine":
            self.cutoff_fn = CosineCutoff.calculate_envelope
        elif cutoff_fn == "c3poly":
            self.is_polynomial_cutoff = True
            self.cutoff_fn = C3PolynomialCutoff.calculate_envelope
        else:
            self.is_polynomial_cutoff = True
            self.cutoff_fn = C2PolynomialCutoff.calculate_envelope

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        if node_type is None:
            node_type = node_attrs.argmax(dim=-1)
        node_atomic_numbers = atomic_numbers[node_type].unsqueeze(-1)
        Z_u = node_atomic_numbers[source]
        Z_v = node_atomic_numbers[target]
        a = (
            self.a_prefactor
            * 0.529
            / (torch.pow(Z_u, self.a_exp) + torch.pow(Z_v, self.a_exp))
        )
        r_over_a = x / a
        phi = (
            self.c[0] * torch.exp(-3.2 * r_over_a)
            + self.c[1] * torch.exp(-0.9423 * r_over_a)
            + self.c[2] * torch.exp(-0.4028 * r_over_a)
            + self.c[3] * torch.exp(-0.2016 * r_over_a)
        )
        v_edges = (14.3996 * Z_u * Z_v) / x * phi
        r_max = self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        if self.is_polynomial_cutoff:
            envelope = self.cutoff_fn(x, r_max, self.p)
        else:
            envelope = self.cutoff_fn(x, r_max)
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter_sum(v_edges, target, dim=0, dim_size=node_attrs.size(0))
        return V_ZBL.squeeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(c={[float(f'{c:.4f}') for c in self.c.tolist()]})"


class RadialBasis(torch.nn.Module):
    def __init__(
        self,
        cutoff: float = 6.0,
        num_basis: int = 8,
        polynomial_cutoff: int = 5,
        radial_basis: str = "j0",
        distance_transform=None,
        order: Union[int, list[int]] = [0],
        trainable: bool = False,
        apply_cutoff: bool = True,
        cutoff_fn: str = "mollifier",  # ['cosine', 'mollifier', 'polynomial']
        gaussian_width: float = 2.0,
    ):
        super().__init__()

        assert isinstance(trainable, bool)
        assert isinstance(apply_cutoff, bool)

        if cutoff_fn == "mollifier":
            self.cutoff_fn = MollifierCutoff(cutoff=cutoff)
        elif cutoff_fn == "cosine":
            self.cutoff_fn = CosineCutoff(cutoff=cutoff)
        elif cutoff_fn == "c3poly":
            self.cutoff_fn = C3PolynomialCutoff(cutoff=cutoff, p=polynomial_cutoff)
        else:
            self.cutoff_fn = C2PolynomialCutoff(cutoff=cutoff, p=polynomial_cutoff)

        if radial_basis == "bessel" or radial_basis == "j0":
            self.radial_fn = j0SphericalBesselBasis(
                cutoff=cutoff,
                num_basis=num_basis,
                trainable=trainable,
            )
        elif radial_basis == "jn":
            self.radial_fn = jnSphericalBesselBasis(
                cutoff=cutoff,
                order=order,
                num_basis=num_basis,
                trainable=trainable,
            )
        # elif radial_basis == "chebychev":
        #     self.radial_fn = ChebyshevTBasis(
        #         cutoff=cutoff,
        #         num_basis=num_basis,
        #     )
        elif radial_basis == "gaussian":
            self.radial_fn = GaussianBasis(
                cutoff=cutoff,
                num_basis=num_basis,
                width=gaussian_width,
            )
        else:
            raise ValueError(f"Unknown radial_basis: {radial_basis}")

        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
            self.use_distance_transform = True
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
            self.use_distance_transform = True
        else:
            self.use_distance_transform = False

        if not isinstance(num_basis, int):
            num_basis = sum(num_basis)
            self.out_dim = num_basis
            self.num_basis = num_basis
        else:
            self.out_dim = num_basis
            self.num_basis = num_basis
        # self.num_basis = self.radial_fn.num_basis
        self.apply_cutoff = apply_cutoff

    def forward(
        self,
        edge_length: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
        node_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        cutoff = self.cutoff_fn(edge_length)
        if self.use_distance_transform:
            edge_length = self.distance_transform(
                edge_length, node_attrs, edge_index, atomic_numbers, node_type
            )

        radial = self.radial_fn(edge_length, node_attrs, edge_index)

        if self.apply_cutoff:
            return radial * cutoff, None
        else:
            return radial, cutoff

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import List


import ase.data
import numpy as np
import torch
from torch import Tensor
from omegaconf import ListConfig
from scipy.optimize import brentq
from scipy.special import jv


from ..util.torch_scatter import scatter_sum


def jn_taylor(n, x, terms=5):
    x = x.to(torch.float64)
    result = torch.zeros_like(x)
    for k in range(terms):
        coeff = ((-1) ** k) / (math.factorial(k) * math.factorial(2 * k + n + 1))
        term = coeff * x ** (2 * k + n)
        result += term
    result /= 2**n
    return result


def compute_jn_zeros(n: int, k: int) -> np.ndarray:
    def spherical_bessel_jn(r, order):
        return np.sqrt(np.pi / (2 * r)) * jv(
            order + 0.5, r
        )  # from first bessel to first spherical bessel

    zeros = []
    guess_points = np.arange(1, k + 20) * np.pi

    found = 0
    i = 0
    while found < k and i < len(guess_points) - 1:
        a, b = guess_points[i], guess_points[i + 1]
        try:
            root = brentq(
                spherical_bessel_jn, a, b, args=(n)
            )  # search roots for spherical_bessel_jn in [a, b]
            zeros.append(root)
            found += 1
        except ValueError:
            pass
        i += 1
    return np.array(zeros, dtype=np.float64)


class j0_SphericalBesselBasis(torch.nn.Module):
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [..., 1]
        numerator = torch.sin(self.bessel_weights * x)  # [..., num_basis]
        return self.prefactor * (numerator / x)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, num_basis={len(self.bessel_weights)}, "
            f"trainable={self.bessel_weights.requires_grad})"
        )


class jn_SphericalBesselBasis(torch.nn.Module):
    "arbitrary order n >= 0"

    def __init__(
        self,
        cutoff: float = 6.0,
        order: int | List[int] = 0,
        num_basis: int | List[int] = 8,
        trainable: bool = False,
    ) -> None:
        super().__init__()

        # assert torch.get_default_dtype() == torch.float64, f"This class requires float64 precision, but got {torch.get_default_dtype()}"

        num_zero = num_basis
        if isinstance(order, int):
            if order < 0:
                raise ValueError("order must be a nonnegative integer")
            order = [order]
        else:
            if not isinstance(order, (List, ListConfig)):
                raise TypeError("order must be a list of nonnegative integer")
            if not all(isinstance(x, int) and x >= 0 for x in order):
                raise ValueError("All elements of order must be nonnegative integer")

        if isinstance(num_zero, int):
            if num_zero <= 0:
                raise ValueError("num_zero must be a positive integer")
            num_zero = [num_zero]
        else:
            if not isinstance(order, (List, ListConfig)):
                raise TypeError("num_zero must be a list of positive integers")
            if not all(isinstance(x, int) and x > 0 for x in num_zero):
                raise ValueError("All elements of num_zero must be positive integers")

        if len(order) != len(num_zero):
            raise ValueError(
                f"order and num_zero must have the same length, "
                f"but got {len(order)} and {len(num_zero)}"
            )

        zeros = []
        for o, n in zip(order, num_zero):
            zeros.append(compute_jn_zeros(o, n))

        normalizer = self._compute_normalizer(cutoff, order, zeros)

        self.register_buffer(
            "normalizer",
            torch.tensor(
                [y for x in normalizer for y in x], dtype=torch.get_default_dtype()
            ).unsqueeze(0),
        )  # (1, sum(order*zeros))
        if trainable:
            self.zeros = torch.nn.Parameter(
                torch.tensor(
                    [y for x in zeros for y in x], dtype=torch.get_default_dtype()
                ).unsqueeze(0)
            )
        else:
            self.register_buffer(
                "zeros",
                torch.tensor(
                    [y for x in zeros for y in x], dtype=torch.get_default_dtype()
                ).unsqueeze(0),
            )
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )
        self.order = order
        self.num_zero = num_zero

    def torch_jn(self, order, x):
        if order == 0:
            return torch.sin(x) / x
        elif order == 1:
            return torch.sin(x) / x**2 - torch.cos(x) / x
        elif order == 2:
            return (3 / x**3 - 1 / x) * torch.sin(x) - (3 * torch.cos(x) / x**2)
        elif order == 3:
            return (15 / x**4 - 6 / x**2) * torch.sin(x) - (
                15 / x**3 - 1 / x
            ) * torch.cos(x)
        elif order == 4:
            return (105 / x**5 - 45 / x**3 + 1 / x) * torch.sin(x) - (
                105 / x**4 - 10 / x**2
            ) * torch.cos(x)
        elif order == 5:
            return (945 / x**6 - 420 / x**4 + 15 / x**2) * torch.sin(x) - (
                945 / x**5 - 105 / x**3 + 1 / x
            ) * torch.cos(x)
        elif order == 6:
            return (10395 / x**7 - 4725 / x**5 + 210 / x**3 - 1 / x) * torch.sin(
                x
            ) - (10395 / x**6 - 1260 / x**4 + 21 / x**2) * torch.cos(x)
        elif order == 7:
            return (
                135135 / x**8 - 62370 / x**6 + 3150 / x**4 - 28 / x**2
            ) * torch.sin(x) - (
                135135 / x**7 - 17325 / x**5 + 378 / x**3 - 1 / x
            ) * torch.cos(
                x
            )
        elif order == 8:
            return (
                2027025 / x**9 - 945945 / x**7 + 51975 / x**5 - 630 / x**3 + 1 / x
            ) * torch.sin(x) - (
                2027025 / x**8 - 270270 / x**6 + 6930 / x**4 - 36 / x**2
            ) * torch.cos(
                x
            )
        elif order == 9:
            return (
                34459425 / x**10
                - 16216200 / x**8
                + 945945 / x**6
                - 13860 / x**4
                + 45 / x**2
            ) * torch.sin(x) - (
                34459425 / x**9 - 4729725 / x**7 + 135135 / x**5 - 990 / x**3 + 1 / x
            ) * torch.cos(
                x
            )
        else:

            N = (
                order + 100
            )  # Starting point for backward recursion (Miller's algorithm)
            device, dtype = x.device, x.dtype
            j = torch.zeros(N + 1, *x.shape, dtype=dtype, device=device)

            j[N] = 1e-40
            j[N - 1] = 0.0

            for n in range(N - 1, 0, -1):
                j[n - 1] = (2 * n + 1) / x * j[n] - j[n + 1]

            # Normalize using accurately computed j0
            j0_ground_truth = torch.sin(x) / x
            scale_factor = j0_ground_truth / j[0]  # shape: (...,)
            j_normalized = j * scale_factor.unsqueeze(0)

            return j_normalized[order]

    def forward(self, r: Tensor):  # [..., 1]
        orig_dtype = r.dtype
        r = r.to(torch.float64)
        cutoff = self.cutoff.to(torch.float64)
        zeros = self.zeros.to(torch.float64)
        normalizer = self.normalizer.to(torch.float64)

        r = zeros * (r / cutoff)

        basis = []
        idx = 0
        for o, n in zip(self.order, self.num_zero):
            r_order = r[..., idx : idx + n]
            jn = self.torch_jn(o, r_order)
            basis.append(jn)
            idx += n
        basis = torch.cat(basis, dim=-1)

        out = basis * normalizer

        return out.to(orig_dtype)

    def _compute_normalizer(self, cutoff, order, zeros):
        normalizer = []
        for o, zero in zip(order, zeros):
            o_normalizer = []
            for i in range(len(zero)):
                z = zero[i]
                j_next = np.sqrt(np.pi / (2 * z)) * jv(o + 1 + 0.5, z)
                norm = np.sqrt(2 / (cutoff**3 * j_next**2))
                o_normalizer.append(norm)
            normalizer.append(o_normalizer)
        return normalizer

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cutoff={self.cutoff}, order={self.order},  num_zero={self.num_zero}, "
            f"trainable={self.zeros.requires_grad})"  # zeros={self.zeros.tolist()}
        )


class Normalized_j0_SphericalBesselBasis(torch.nn.Module):
    """
    From Allergo https://www.nature.com/articles/s41467-023-36329-y
    """

    def __init__(
        self,
        r_max: float,
        r_min: float = 0.0,
        original_basis=j0_SphericalBesselBasis,
        original_basis_kwargs: dict = {},
        n: int = 4000,  # number of integral grid points
        norm_basis_mean_shift: bool = True,  # calculate the mean value and offset it; otherwise, only standardize it
    ):
        super().__init__()
        self.basis = original_basis(**original_basis_kwargs)
        self.r_min = r_min
        self.r_max = r_max
        assert self.r_min >= 0.0
        assert self.r_max > r_min
        self.n = n
        self.num_basis = self.basis.num_basis

        # Uniform distribution on [r_min, r_max)
        with torch.no_grad():
            # don't take 0 in case of weirdness like bessel at 0 # weirdness 不可思议的 离奇的 singular point?
            rs = torch.linspace(r_min, r_max, n + 1)[1:]
            rs.unsqueeze_(-1)
            bs = self.basis(rs)  # TODO 修改后适配了自己的 j0_SphericalBesselBasis
            assert bs.ndim == 2 and len(bs) == n
            if norm_basis_mean_shift:
                basis_std, basis_mean = torch.std_mean(bs, dim=0)
            else:
                basis_std = bs.square().mean().sqrt()  # rms
                basis_mean = torch.as_tensor(
                    0.0, device=basis_std.device, dtype=basis_std.dtype
                )

        self.register_buffer("_mean", basis_mean)
        self.register_buffer("_inv_std", torch.reciprocal(basis_std))

    def forward(self, x: torch.Tensor) -> Tensor:
        return (self.basis(x) - self._mean) * self._inv_std


class PolynomialCutoff(torch.nn.Module):
    """Envelope function(PolynomialCutoff funciton) is proposed in the DimeNet: https://www.cs.cit.tum.de/daml/dimenet/"""

    def __init__(self, cutoff: float, p: float = 6.0, start: float = 0.0) -> None:
        super().__init__()
        assert p > 0.1, f"In PolynomialCutoff function p ({p}) should greater than 0.1"
        assert (
            start < cutoff
        ), f"In cutoff function start ({start}) should less than cutoff ({cutoff})"
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "start", torch.tensor(start, dtype=torch.get_default_dtype())
        )
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=torch.get_default_dtype())
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        x = (x.clamp(self.start, self.cutoff) - self.start) / (self.cutoff - self.start)
        envelope = (
            1.0
            - ((self.p + 1.0) * (self.p + 2.0) / 2.0) * torch.pow(x, self.p)
            + self.p * (self.p + 2.0) * torch.pow(x, self.p + 1)
            - (self.p * (self.p + 1.0) / 2) * torch.pow(x, self.p + 2)
        )

        return envelope

    def __repr__(self):
        return f"{self.__class__.__name__}(p={self.p}, r_min={self.start}, r_max(cutoff)={self.cutoff})"

    @staticmethod
    def calculate_envelope(
        x: torch.Tensor, r_max: torch.Tensor, p: torch.Tensor
    ) -> torch.Tensor:
        r_over_r_max = x / r_max
        envelope = (
            1.0
            - ((p + 1.0) * (p + 2.0) / 2.0) * torch.pow(r_over_r_max, p)
            + p * (p + 2.0) * torch.pow(r_over_r_max, p + 1)
            - (p * (p + 1.0) / 2) * torch.pow(r_over_r_max, p + 2)
        )
        return envelope * (x < r_max)


class AgnesiTransform(torch.nn.Module):
    """
    From MACE https://github.com/ACEsuit/mace https://doi.org/10.1063/5.0158783
    """

    def __init__(
        self,
        q: float = 0.9183,
        p: float = 4.5791,
        a: float = 1.0805,
        trainable=False,
    ):
        super().__init__()
        self.register_buffer("q", torch.tensor(q, dtype=torch.get_default_dtype()))
        self.register_buffer("p", torch.tensor(p, dtype=torch.get_default_dtype()))
        self.register_buffer("a", torch.tensor(a, dtype=torch.get_default_dtype()))
        self.register_buffer(
            "covalent_radii",
            torch.tensor(
                ase.data.covalent_radii,
                dtype=torch.get_default_dtype(),
            ),
        )
        if trainable:
            self.a = torch.nn.Parameter(torch.tensor(1.0805, requires_grad=True))
            self.q = torch.nn.Parameter(torch.tensor(0.9183, requires_grad=True))
            self.p = torch.nn.Parameter(torch.tensor(4.5791, requires_grad=True))

    def forward(
        self,
        x: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        atomic_numbers: Tensor,
    ) -> Tensor:
        source = edge_index[0]
        target = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        ) 
        Z_u = node_atomic_numbers[source]
        Z_v = node_atomic_numbers[target]
        r_0: torch.Tensor = 0.5 * (
            self.covalent_radii[Z_u] + self.covalent_radii[Z_v]
        ) 
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
    From MACE https://github.com/ACEsuit/mace https://doi.org/10.1063/5.0158783
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
        x: Tensor,
        node_attrs: Tensor,
        edge_index: Tensor,
        atomic_numbers: Tensor,
    ) -> Tensor:
        source = edge_index[0]
        target = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
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


class RadialBasis(torch.nn.Module):
    def __init__(
        self,
        cutoff: float = 6.0,
        num_basis: int = 8,
        polynomial_cutoff: int = 5.0,
        radial_basis: str = "bessel",
        distance_transform=None,
        order: int | List[int] = [0],
        trainable: bool = False,
        apply_cutoff: bool = True,
    ):
        super().__init__()
        if radial_basis == "bessel" or radial_basis == "j0":
            self.radial_fn = j0_SphericalBesselBasis(
                cutoff=cutoff, num_basis=num_basis, trainable=trainable
            )
        elif radial_basis == "jn":
            self.radial_fn = jn_SphericalBesselBasis(
                cutoff=cutoff,
                order=order,
                num_basis=num_basis,
                trainable=trainable,
            )
        elif radial_basis == "normalized_bessel" or radial_basis == "n_j0":
            self.radial_fn = Normalized_j0_SphericalBesselBasis(
                r_max=cutoff,
                original_basis_kwargs={"cutoff": cutoff, "num_basis": num_basis},
            )

        elif radial_basis == "chebychev":
            self.radial_fn = ChebychevBasis(
                r_max=cutoff,
                num_basis=num_basis,
            )
        elif radial_basis == "gaussian":
            self.radial_fn = GaussianBasis(
                r_max=cutoff,
                num_basis=num_basis,
                trainable=trainable,
            )
        else:
            raise ValueError(f"Unknown radial_basis: {radial_basis}")
        if distance_transform == "Agnesi":
            self.distance_transform = AgnesiTransform()
        elif distance_transform == "Soft":
            self.distance_transform = SoftTransform()
        self.cutoff_fn = PolynomialCutoff(cutoff=cutoff, p=polynomial_cutoff)

        if not isinstance(num_basis, int):
            num_basis = sum(num_basis)
            self.out_dim = num_basis
            self.num_basis = num_basis
        else:
            self.out_dim = num_basis
            self.num_basis = num_basis

        self.apply_cutoff = apply_cutoff
        self.for_copy = {
            "cutoff": cutoff,
            "num_basis": num_basis,
            "polynomial_cutoff": polynomial_cutoff,
            "radial_basis": radial_basis,
            "distance_transform": distance_transform,
            "order": order,
            "trainable": trainable,
            "apply_cutoff": apply_cutoff,
        }

    def forward(
        self,
        edge_length: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        cutoff = self.cutoff_fn(edge_length)
        if hasattr(self, "distance_transform"):
            edge_length = self.distance_transform(
                edge_length, node_attrs, edge_index, atomic_numbers
            )
        radial = self.radial_fn(edge_length)
        if self.apply_cutoff:
            return radial * cutoff, None
        else:
            return radial, cutoff


    # j1 j2 j3 ...
    def copy(self, max_rank: int = -1):
        return self.replicate()

    def replicate(self):
        for_copy = self.for_copy
        return self.__class__(**for_copy)


# # @compile_mode("script")
class CosineCutoff(torch.nn.Module):
    def __init__(self, cutoff: float) -> None:
        super().__init__()
        self.register_buffer(
            "cutoff",
            torch.tensor(cutoff, dtype=torch.get_default_dtype()),
        )

    def forward(self, length: torch.Tensor) -> torch.Tensor:
        return torch.square(
            0.5 * torch.cos(length * (torch.pi / self.cutoff)) + 0.5
        ).unsqueeze(-1)


class ChebychevBasis(torch.nn.Module):
    def __init__(self, r_max: float, num_basis=8):
        super().__init__()

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(r_max={self.r_max}, num_basis={self.num_basis},"
        )


class GaussianBasis(torch.nn.Module):
    def __init__(self, r_max: float, num_basis=128, trainable=False):
        super().__init__()

    def __repr__(self):
        return f"{self.__class__.__name__}(r_max={self.cutoff}, num_basis={self.num_basis},"


class ZBLBasis(torch.nn.Module):
    """
    From MACE
    Implementation of the Ziegler-Biersack-Littmark (ZBL) potential
    with a polynomial cutoff envelope.
    """

    p: torch.Tensor

    def __init__(self, p=6, trainable=False, **kwargs):
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

    def forward(
        self,
        x: torch.Tensor,
        node_attrs: torch.Tensor,
        edge_index: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        source = edge_index[0]
        target = edge_index[1]
        node_atomic_numbers = atomic_numbers[torch.argmax(node_attrs, dim=1)].unsqueeze(
            -1
        )
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
        envelope = PolynomialCutoff.calculate_envelope(x, r_max, self.p)
        v_edges = 0.5 * v_edges * envelope
        V_ZBL = scatter_sum(v_edges, target, dim=0, dim_size=node_attrs.size(0))
        return V_ZBL.squeeze(-1)

    def __repr__(self):
        return f"{self.__class__.__name__}(c={[float(f'{c:.4f}') for c in self.c.tolist()]})"


# # draw
# import torch
# import matplotlib.pyplot as plt
# import numpy as np

# torch.set_default_dtype(torch.float64)
# cutoff = 6.0
# num_basis = 8
# num_basis_list = [15] # [8, 9, 10]

# j0 = j0_SphericalBesselBasis(cutoff=cutoff, num_basis=num_basis)
# jn = jn_SphericalBesselBasis(cutoff=cutoff, order=8, num_basis=num_basis_list)

# x = torch.linspace(0, cutoff, 200).unsqueeze(-1)

# a = j0(x)  # shape: [200, 8]
# b = jn(x)  # shape: [200, 8]


# x_np = x.squeeze().numpy()        # shape: [200]
# a = a.detach().numpy()         # shape: [200, 8]
# b = b.detach().numpy()         # shape: [200, 8]


# plt.figure(figsize=(10, 5))

# for i in range(a.shape[1]):
#     # plt.plot(x_np, a[:, i], label=f'j0_basis_{i}', linestyle='-')
#     plt.plot(x_np, b[:, i], label=f'jn_basis_{i}', linestyle='--')

# plt.xlabel("x")
# plt.ylabel("Basis value")
# plt.title("Comparison of j0 vs jn spherical Bessel basis")
# plt.legend(fontsize='small', ncol=2)
# plt.grid(True)
# plt.tight_layout()
# plt.show()
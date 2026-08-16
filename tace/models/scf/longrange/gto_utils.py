from typing import List, Sequence

import numpy as np
import scipy.integrate
import scipy.special
import torch
from e3nn import o3
from scipy.constants import pi

from .utils import FIELD_CONSTANT


class RadialIntegralDirect(torch.nn.Module):
    """Direct evaluation of f_{nl}(k) for max_l <= 1."""

    def __init__(self, sigmas: Sequence[float], max_l: int, k_space_cutoff: float):
        super().__init__()
        if max_l > 1:
            raise NotImplementedError("RadialIntegralDirect only supports max_l <= 1.")
        self.num_sigma = len(sigmas)
        self.max_l = max_l

        sigmas_t = torch.as_tensor(sigmas, dtype=torch.get_default_dtype())
        pref_const = (4 * pi) * np.sqrt(pi / 2.0)

        self.register_buffer("sigma2", sigmas_t * sigmas_t)
        self.register_buffer("pref0", pref_const * sigmas_t**3)
        if max_l == 1:
            self.register_buffer("pref1", pref_const * sigmas_t**5)

    def forward(self, k_mods: torch.Tensor) -> torch.Tensor:
        k2 = k_mods * k_mods
        exp_term = torch.exp(-0.5 * k2.unsqueeze(-1) * self.sigma2)

        if self.max_l == 0:
            out = self.pref0 * exp_term
            return out.unsqueeze(-1)

        out = torch.empty(
            (*k_mods.shape, self.num_sigma, 2),
            dtype=k_mods.dtype,
            device=k_mods.device,
        )
        torch.mul(self.pref0, exp_term, out=out[..., 0])
        torch.mul(self.pref1, k_mods.unsqueeze(-1) * exp_term, out=out[..., 1])
        return out


def _normalization_denominator(
    sigmas: torch.Tensor, ls: torch.Tensor, normalize: str
) -> torch.Tensor:
    ls = ls.to(dtype=sigmas.dtype)
    if normalize == "multipoles":
        l_dep_part = (
            (4 * pi / (2 * ls + 1)) ** 0.5
            * 2 ** ((2 * ls + 1) / 2)
            * torch.exp(torch.lgamma((2 * ls + 3) / 2))
        )
        return l_dep_part * sigmas.unsqueeze(-1) ** (2 * ls + 3)
    if normalize == "receiver":
        l_dep_part = 2 ** ((ls + 1) / 2) * torch.exp(
            torch.lgamma((ls + 3) / 2)
        )
        return l_dep_part * sigmas.unsqueeze(-1) ** (ls + 3)
    if normalize == "none":
        return torch.ones((len(sigmas), len(ls)), dtype=torch.get_default_dtype())
    raise ValueError("normalize must be one of 'multipoles', 'none', or 'receiver'")


class GTOBasis(torch.nn.Module):
    """Fourier coefficients for a GTO basis parameterized by sigmas and max_l."""

    def __init__(
        self,
        max_l: int,
        sigmas: Sequence[float],
        kspace_cutoff: float,
        normalize: str,
        use_spline: bool = False,
    ):
        super().__init__()
        if normalize not in {"none", "multipoles", "receiver"}:
            raise ValueError(
                "normalize must be one of 'multipoles', 'none', or 'receiver'"
            )
        if len(sigmas) == 0:
            raise ValueError("sigmas must contain at least one value")

        self.max_l = int(max_l)
        self.sigmas = [float(sigma) for sigma in sigmas]
        self.num_sigma = len(self.sigmas)
        self.normalize = normalize
        self.kspace_cutoff = float(kspace_cutoff)
        self.lm_dim = (self.max_l + 1) ** 2
        self.use_spline = bool(use_spline)

        self._initialize_basis_state()

    def _set_or_register_buffer(self, name: str, tensor: torch.Tensor):
        if name in self._buffers:
            setattr(self, name, tensor)
        else:
            self.register_buffer(name, tensor)

    def _initialize_basis_state(self) -> None:
        if self.use_spline:
            raise NotImplementedError("splining of radial integral not supported")
        else:
            self.radial_spline = RadialIntegralDirect(
                self.sigmas, self.max_l, self.kspace_cutoff
            )
        self.spherical_harmonics = o3.SphericalHarmonics(
            o3.Irreps.spherical_harmonics(self.max_l),
            normalize=True,
            normalization="integral",
        )

        sigmas_t = torch.as_tensor(self.sigmas, dtype=torch.get_default_dtype())
        ls = torch.arange(self.max_l + 1, dtype=torch.int64)
        cl_inverse = _normalization_denominator(sigmas_t, ls, self.normalize)
        self._set_or_register_buffer("cl_scale", 1.0 / cl_inverse)

        expanded_l = torch.repeat_interleave(ls, 2 * ls + 1)
        self._set_or_register_buffer("expanded_l_indices", expanded_l)

        long_ones = torch.ones(expanded_l.shape, dtype=torch.get_default_dtype())
        even_l_phases = torch.pow(-long_ones, expanded_l / 2)
        odd_l_phases = -torch.pow(-long_ones, (expanded_l - 1.0) / 2)

        real_phase_factors = torch.where(
            (expanded_l % 2 == 0), even_l_phases, torch.zeros_like(long_ones)
        )
        imag_phase_factors = torch.where(
            (expanded_l % 2 == 0), torch.zeros_like(long_ones), odd_l_phases
        )
        self._set_or_register_buffer("real_phase_factors", real_phase_factors)
        self._set_or_register_buffer("imag_phase_factors", imag_phase_factors)
        self._set_or_register_buffer(
            "permute_indices", torch.tensor([1, 2, 0], dtype=torch.int64)
        )

    def set_kspace_cutoff(self, kspace_cutoff: float) -> None:
        self.kspace_cutoff = float(kspace_cutoff)
        self._initialize_basis_state()

    def _prepare_k_moduli(
        self, k_norm2: torch.Tensor, k0_mask: torch.Tensor
    ) -> torch.Tensor:
        k_moduli = torch.sqrt(torch.clamp_min(k_norm2, 0.0))
        k_moduli.masked_fill_(k0_mask > 0.0, 0.0)
        return k_moduli

    def _compute_ylmk(self, k_vectors: torch.Tensor) -> torch.Tensor:
        k_vectors = torch.index_select(k_vectors, -1, self.permute_indices)
        return self.spherical_harmonics(k_vectors)

    def _evaluate_fourier_basis(
        self, k_moduli: torch.Tensor, yklm: torch.Tensor
    ) -> torch.Tensor:
        fnlk = self.radial_spline(k_moduli) * self.cl_scale
        expanded_fnlk = torch.index_select(fnlk, -1, self.expanded_l_indices)
        xnlk = expanded_fnlk * yklm.unsqueeze(-2)

        fourier_coefficients = torch.empty(
            (*xnlk.size(), 2), dtype=xnlk.dtype, layout=xnlk.layout, device=xnlk.device
        )
        fourier_coefficients[..., 0] = xnlk * self.real_phase_factors
        fourier_coefficients[..., 1] = xnlk * self.imag_phase_factors
        return fourier_coefficients

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k0_mask: torch.Tensor,
    ) -> torch.Tensor:
        k_moduli = self._prepare_k_moduli(k_norm2, k0_mask)
        yklm = self._compute_ylmk(k_vectors)
        return self._evaluate_fourier_basis(k_moduli, yklm)


def gto_basis_kspace_cutoff(sigmas: Sequence[float], max_l: int) -> float:
    # Heuristic with the expected sigma and angular scaling.
    return 0.75 * (1.0 / min(sigmas)) * (max_l + 1) ** 0.3 * 3.0


def integral_F1(r, l, sigma):
    r_part = np.power(r, -(l + 1))
    gammas = scipy.special.gammainc(
        (2 * l + 3) / 2, 0.5 * np.power(r, 2) / sigma**2
    ) * scipy.special.gamma((2 * l + 3) / 2) - scipy.special.gammainc(
        l + 1.5, 0
    ) * scipy.special.gamma(l + 0.5)
    return 2 ** (l + 0.5) * sigma ** (2 * l + 3) * gammas * r_part


def integral_F2(r, l, sigma):
    return sigma**2 * np.power(r, l) * np.exp(-0.5 * np.power(r, 2) / sigma**2)


def get_Cl_sigma(l, sigma, normalize):
    assert normalize in ["multipoles", "none", "receiver"]
    if normalize == "multipoles":
        l_dep_part = (
            (4 * pi / (2 * l + 1)) ** 0.5
            * 2 ** ((2 * l + 1) / 2)
            * scipy.special.gamma((2 * l + 3) / 2)
        )
        Cls_inverse = l_dep_part * sigma ** (2 * l + 3)  # [1, source_l+1]
    elif normalize == "receiver":
        l_dep_part = 2 ** ((l + 1) / 2) * scipy.special.gamma((l + 3) / 2)
        Cls_inverse = l_dep_part * sigma ** (l + 3)
    elif normalize == "none":
        Cls_inverse = 1.0
    else:
        raise ValueError("normalize must be one of 'multipoles', 'none', or 'receiver'")
    return 1 / Cls_inverse


class GTOSelfInteractionBlock(torch.nn.Module):
    def __init__(
        self,
        l_source: int,
        sigma_source: float,
        l_receive: int,
        sigmas_receive: List[float],
        normalize_source: str,
        normalize_receive: str,
    ):
        super().__init__()

        sh_irreps = o3.Irreps.spherical_harmonics(l_receive)
        self.features_irreps = (sh_irreps * len(sigmas_receive)).sort()[0].simplify()

        overlap_constants = np.zeros(
            (len(sigmas_receive) * (min(l_receive, l_source) + 1) ** 2)
        )
        self.num_radial = len(sigmas_receive)

        for ll in range(min(l_source, l_receive) + 1):
            for i_s, sigma_2 in enumerate(sigmas_receive):
                grid = np.linspace(0.0001, 10 * max(sigma_2, sigma_source), 10000)
                F_total_s2_l = integral_F1(grid, ll, sigma_2) + integral_F2(
                    grid, ll, sigma_2
                )
                integrand = (
                    np.power(grid, ll + 2)
                    * np.exp(-0.5 * np.power(grid, 2) / sigma_source**2)
                    * F_total_s2_l
                )
                value = scipy.integrate.trapezoid(integrand, x=grid)
                # print(value)

                prefactor = FIELD_CONSTANT * 1 / (2 * ll + 1)

                # Cl from source
                Cl_source = get_Cl_sigma(ll, sigma_source, normalize=normalize_source)
                Cl_receive = get_Cl_sigma(ll, sigma_2, normalize=normalize_receive)

                for m in range(2 * ll + 1):
                    overlap_constants[
                        len(sigmas_receive) * ll**2 + m + i_s * (2 * ll + 1)
                    ] = Cl_source * Cl_receive * prefactor * value

        overlap_constants = torch.from_numpy(overlap_constants).to(
            torch.get_default_dtype()
        )
        self.register_buffer(
            "overlap_constants",
            overlap_constants,  # [n_sigma_receive * (l_receive+1)^2]
        )

        indices = []
        for ll in range(min(l_source, l_receive) + 1):
            for s_i in range(len(sigmas_receive)):
                for m in range(2 * ll + 1):
                    indices.append((ll**2) + m)

        self.register_buffer("select_indices", torch.tensor(indices, dtype=torch.int64))
        self.non_zero_terms = len(sigmas_receive) * (min(l_source, l_receive) + 1) ** 2

    def forward(
        self,
        charge_density,  # [n_node, (l_source+1)^2]
    ) -> (
        torch.Tensor
    ):  # [n_node, n_sigma * (l_receive+1)^2], the values to subtract from the features

        qs_expanded = torch.index_select(
            charge_density, dim=-1, index=self.select_indices
        )

        features = torch.zeros(
            (charge_density.shape[0], self.features_irreps.dim),
            device=charge_density.device,
            dtype=charge_density.dtype,
        )
        features[..., : self.non_zero_terms] = torch.einsum(
            "d,nd->nd", self.overlap_constants, qs_expanded
        )
        return features


class DisplacedGTOExternalFieldBlock(torch.nn.Module):
    def __init__(
        self, l_receive: int, sigmas_receive: List[float], normalize_receive: str
    ):
        super().__init__()
        self.projections_dim = (l_receive + 1) ** 2 * len(sigmas_receive)

        matrix = torch.zeros((self.projections_dim, 4), dtype=torch.get_default_dtype())

        # potential only affects l=0
        for i_s, sigma in enumerate(sigmas_receive):
            matrix[i_s, 0] = (
                get_Cl_sigma(0, sigma, normalize_receive)
                * (8 * pi) ** 0.5
                * scipy.special.gamma(1.5)
                * sigma**3
            )

        # field only affects l=1
        if l_receive >= 1:
            for i_s, sigma in enumerate(sigmas_receive):
                magnitude = (
                    get_Cl_sigma(1, sigma, normalize_receive)
                    * (1.5) ** 0.5
                    * sigma**5
                    * 2
                    * pi
                )
                for m in range(3):
                    matrix[len(sigmas_receive) + i_s * 3 + m, 1 + m] = magnitude

        permutation_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.get_default_dtype(),
        )
        matrix = torch.matmul(matrix, permutation_matrix)
        self.register_buffer("matrix", matrix)  # [field_feats_dim, 4]

    def forward(
        self,
        batch,  # [n_node]
        positions,
        field,  # [n_graph, 4]
    ):
        # field contains [V, E_x, E_y, E_z]
        assert field.dim() == 2

        node_fields = torch.index_select(field, 0, batch)  # [n_nodes, 4]
        potential = node_fields[:, 0].clone()

        potential_from_displacement = torch.einsum(
            "bi,bi->b", positions, node_fields[:, 1:].clone()
        )
        node_fields[:, 0] += potential_from_displacement

        node_fields = node_fields[:, [0, 3, 1, 2]]
        projections = torch.einsum("pf,nf->np", self.matrix, node_fields)

        return projections  # [n_node, n_feat]


class GTOInternalFieldtoFeaturesBlock(torch.nn.Module):
    def __init__(
        self, l_receive: int, sigmas_receive: List[float], normalize_receive: str
    ):
        super().__init__()
        self.projections_dim = (l_receive + 1) ** 2 * len(sigmas_receive)

        matrix = torch.zeros((self.projections_dim, 4), dtype=torch.get_default_dtype())

        # potential only affects l=0
        for i_s, sigma in enumerate(sigmas_receive):
            matrix[i_s, 0] = (
                get_Cl_sigma(0, sigma, normalize_receive)
                * (8 * pi) ** 0.5
                * scipy.special.gamma(1.5)
                * sigma**3
            )

        # field only affects l=1
        if l_receive >= 1:
            for i_s, sigma in enumerate(sigmas_receive):
                magnitude = (
                    get_Cl_sigma(1, sigma, normalize_receive)
                    * (1.5) ** 0.5
                    * sigma**5
                    * 2
                    * pi
                )
                for m in range(3):
                    matrix[len(sigmas_receive) + i_s * 3 + m, 1 + m] = magnitude

        permutation_matrix = torch.tensor(
            [[1, 0, 0, 0], [0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0]],
            dtype=torch.get_default_dtype(),
        )
        matrix = torch.matmul(matrix, permutation_matrix)
        self.register_buffer("matrix", matrix)  # [field_feats_dim, 4]

    def forward(
        self,
        batch,  # [n_node]
        positions,
        node_fields,  # [n_node, 4]
    ):
        # field contains [V, E_x, E_y, E_z]
        assert node_fields.dim() == 2
        node_fields = node_fields[:, [0, 3, 1, 2]]
        projections = torch.einsum("pf,nf->np", self.matrix, node_fields)
        return projections  # [n_node, n_feat]

###########################################################################################
# Functions for computing global electrostatic features from atomic multipoles
# and gaussian type orbitals
# Authors: Will Baldwin
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from typing import List, Optional

import torch
from scipy.constants import pi

from .gto_utils import (
    GTOBasis,
    GTOInternalFieldtoFeaturesBlock,
    GTOSelfInteractionBlock,
)
from .realspace_electrostatics import RealSpaceFiniteDifferenceElectrostaticFeatures
from .slabs import (
    CorrectivePotentialBlock,
    _get_total_dipole_z,
    slab_dipole_correction_node_fields,
)
from .utils import FIELD_CONSTANT

CORRECTION_MODE_PBC = 0
CORRECTION_MODE_MOLECULE = 1
CORRECTION_MODE_SLAB = 2
CORRECTION_MODE_MIXED = 3


def assemble_fourier_series_batch(
    source_feats: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    density_basis_fs: torch.Tensor,
    volume_per_k: torch.Tensor,
) -> torch.Tensor:
    """Assemble rho(k) for flattened batch.

    Args:
        source_feats: [n_nodes, 1, m_dim] or [n_nodes, m_dim]
        cosines: [n_k_total, n_nodes]
        sines: [n_k_total, n_nodes]
        density_basis_fs: [n_k_total, n_sigma(=1), m_dim, 2]
        volume_per_k: [n_k_total]
    Returns:
        density: [n_k_total, 2]
    """
    n_nodes = source_feats.size(0)
    n_sigma = density_basis_fs.size(1)
    m_dim = density_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    coeff_2d = source_feats.reshape(n_nodes, sm_dim)
    coeff_cos = torch.matmul(cosines, coeff_2d)
    coeff_sin = torch.matmul(sines, coeff_2d)

    density_basis_r = density_basis_fs[..., 0].reshape(density_basis_fs.size(0), sm_dim)
    density_basis_i = density_basis_fs[..., 1].reshape(density_basis_fs.size(0), sm_dim)

    rho_real = (density_basis_r * coeff_cos).sum(dim=-1) + (
        density_basis_i * coeff_sin
    ).sum(dim=-1)
    rho_imag = (density_basis_i * coeff_cos).sum(dim=-1) - (
        density_basis_r * coeff_sin
    ).sum(dim=-1)

    density = torch.stack([rho_real, rho_imag], dim=-1)
    density = (2 * pi) ** 3 * density / volume_per_k.unsqueeze(-1)
    return density


def apply_coulomb_kernel_batch(
    k_norm2: torch.Tensor,
    density: torch.Tensor,
    k_factor_coulomb: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Apply Coulomb kernel in k-space for flattened batch."""
    if k_factor_coulomb is None:
        k_factor_coulomb = torch.empty_like(k_norm2)
        k_factor_coulomb[k_norm2 == 0] = 0.0
        k_factor_coulomb[k_norm2 != 0] = 1.0 / k_norm2[k_norm2 != 0]
    factor = k_factor_coulomb.reshape(-1, *([1] * (density.dim() - 1)))
    potential = density * factor
    potential.mul_(FIELD_CONSTANT)
    return potential


def project_to_features_batch(
    potential: torch.Tensor,
    feature_basis_fs: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project potential to local GTO features for flattened batch."""
    n_k = feature_basis_fs.size(0)
    n_sigma = feature_basis_fs.size(1)
    m_dim = feature_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    proj_basis_r = feature_basis_fs[..., 0].reshape(n_k, sm_dim)
    proj_basis_i = feature_basis_fs[..., 1].reshape(n_k, sm_dim)

    A = potential[:, 0].unsqueeze(-1) * proj_basis_r + (
        potential[:, 1].unsqueeze(-1) * proj_basis_i
    )
    B = potential[:, 0].unsqueeze(-1) * proj_basis_i - (
        potential[:, 1].unsqueeze(-1) * proj_basis_r
    )

    if k_factor_proj is not None:
        A = A * k_factor_proj.unsqueeze(-1)
        B = B * k_factor_proj.unsqueeze(-1)

    proj_cos = torch.matmul(A.t(), cosines)
    proj_sin = torch.matmul(B.t(), sines)
    proj_total = 2.0 * (proj_cos + proj_sin)
    projections = proj_total.t().reshape(cosines.size(1), n_sigma, m_dim)
    return projections / (2 * pi) ** 3


def reconstruct_esps_batch(
    potential: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k0_mask: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct electrostatic potential at node positions."""
    summand = (
        2.0 * potential[:, 0].unsqueeze(-1) * cosines
        - 2.0 * potential[:, 1].unsqueeze(-1) * sines
    )
    k0_factor = torch.ones_like(k0_mask)
    k0_factor[k0_mask > 0.0] = 0.5
    summand = summand * k0_factor.unsqueeze(-1)
    return torch.sum(summand, dim=0) / (2 * pi) ** 3


def assemble_fourier_series_batch_channels(
    source_feats: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    density_basis_fs: torch.Tensor,
    volume_per_k: torch.Tensor,
) -> torch.Tensor:
    """Assemble rho(k) for flattened batch with independent channels.

    Args:
        source_feats: [n_nodes, n_channels, m_dim] or [n_nodes, m_dim]
        cosines: [n_k_total, n_nodes]
        sines: [n_k_total, n_nodes]
        density_basis_fs: [n_k_total, n_sigma(=1), m_dim, 2]
        volume_per_k: [n_k_total]
    Returns:
        density: [n_k_total, n_channels, 2]
    """
    if source_feats.dim() == 2:
        source_feats = source_feats.unsqueeze(1)
    n_nodes = source_feats.size(0)
    n_channels = source_feats.size(1)
    n_sigma = density_basis_fs.size(1)
    m_dim = density_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    coeff = source_feats.reshape(n_nodes, n_channels * sm_dim)
    coeff_cos = torch.matmul(cosines, coeff).view(cosines.size(0), n_channels, sm_dim)
    coeff_sin = torch.matmul(sines, coeff).view(sines.size(0), n_channels, sm_dim)

    density_basis_r = density_basis_fs[..., 0].reshape(density_basis_fs.size(0), sm_dim)
    density_basis_i = density_basis_fs[..., 1].reshape(density_basis_fs.size(0), sm_dim)

    rho_real = (density_basis_r.unsqueeze(1) * coeff_cos).sum(dim=-1) + (
        density_basis_i.unsqueeze(1) * coeff_sin
    ).sum(dim=-1)
    rho_imag = (density_basis_i.unsqueeze(1) * coeff_cos).sum(dim=-1) - (
        density_basis_r.unsqueeze(1) * coeff_sin
    ).sum(dim=-1)

    density = torch.stack([rho_real, rho_imag], dim=-1)
    density = (2 * pi) ** 3 * density / volume_per_k[:, None, None]
    return density


def project_to_features_batch_channels(
    potential: torch.Tensor,
    feature_basis_fs: torch.Tensor,
    cosines: torch.Tensor,
    sines: torch.Tensor,
    k_factor_proj: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Project potential to local GTO features for flattened batch with channels."""
    n_k = feature_basis_fs.size(0)
    n_sigma = feature_basis_fs.size(1)
    m_dim = feature_basis_fs.size(2)
    sm_dim = n_sigma * m_dim

    proj_basis_r = feature_basis_fs[..., 0].reshape(n_k, sm_dim)
    proj_basis_i = feature_basis_fs[..., 1].reshape(n_k, sm_dim)

    potential_r = potential[..., 0].unsqueeze(-1)
    potential_i = potential[..., 1].unsqueeze(-1)
    A = potential_r * proj_basis_r.unsqueeze(1) + potential_i * proj_basis_i.unsqueeze(
        1
    )
    B = potential_r * proj_basis_i.unsqueeze(1) - potential_i * proj_basis_r.unsqueeze(
        1
    )

    if k_factor_proj is not None:
        factor = k_factor_proj[:, None, None]
        A = A * factor
        B = B * factor

    A_flat = A.reshape(n_k, -1)
    B_flat = B.reshape(n_k, -1)

    proj_cos = torch.matmul(A_flat.t(), cosines)
    proj_sin = torch.matmul(B_flat.t(), sines)
    proj_total = 2.0 * (proj_cos + proj_sin)
    projections = proj_total.t().reshape(cosines.size(1), -1, n_sigma, m_dim)
    return projections / (2 * pi) ** 3


class NonPeriodicFeatureCorrections(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        projection_max_l: int,
        projection_smearing_widths: List[float],
        integral_normalization: str = "receiver",
        quadrupole_feature_corrections: bool = False,
    ):
        super().__init__()
        self.self_field = CorrectivePotentialBlock(
            density_max_l=density_max_l,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )
        self.displaced_interactions = GTOInternalFieldtoFeaturesBlock(
            l_receive=projection_max_l,
            sigmas_receive=projection_smearing_widths,
            normalize_receive=integral_normalization,
        )

    def forward(
        self,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volumes: torch.Tensor,
        pbc: torch.Tensor,
        correction_mode: Optional[int] = None,
        correction_node_masks: Optional[dict] = None,
    ) -> torch.Tensor:
        source_feats_lm = (
            source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        )
        if correction_mode is None:
            pbc_bool = pbc.to(dtype=torch.bool)
            is_pbc_graph = pbc_bool.all(dim=1)
            is_molecule_graph = (~pbc_bool).all(dim=1)
            is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
            if is_pbc_graph.all():
                correction_mode = CORRECTION_MODE_PBC
            elif is_molecule_graph.all():
                correction_mode = CORRECTION_MODE_MOLECULE
            elif is_slab_graph.all():
                correction_mode = CORRECTION_MODE_SLAB
            else:
                correction_mode = CORRECTION_MODE_MIXED
                correction_node_masks = {
                    "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
                    "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
                }

        if correction_mode == CORRECTION_MODE_PBC:
            n_nodes = node_positions.size(0)
            return node_positions.new_zeros(
                (n_nodes, self.displaced_interactions.projections_dim)
            )

        if correction_mode == CORRECTION_MODE_MOLECULE:
            node_fields = self.self_field(
                charge_coefficients=source_feats_lm,
                positions=node_positions,
                volumes=volumes,
                batch=batch,
            )
            return self.displaced_interactions(
                batch=batch,
                positions=node_positions,
                node_fields=node_fields,
            )

        if correction_mode == CORRECTION_MODE_SLAB:
            node_fields = slab_dipole_correction_node_fields(
                source_feats=source_feats_lm,
                node_positions=node_positions,
                volumes=volumes,
                batch=batch,
            )
            return self.displaced_interactions(
                batch=batch,
                positions=node_positions,
                node_fields=node_fields,
            )

        if correction_node_masks is None:
            pbc_bool = pbc.to(dtype=torch.bool)
            is_molecule_graph = (~pbc_bool).all(dim=1)
            is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
            correction_node_masks = {
                "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
                "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
            }

        node_fields_molecule = self.self_field(
            charge_coefficients=source_feats_lm,
            positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        node_fields_slab = slab_dipole_correction_node_fields(
            source_feats=source_feats_lm,
            node_positions=node_positions,
            volumes=volumes,
            batch=batch,
        )
        is_molecule = correction_node_masks["is_molecule_node"]
        is_slab = correction_node_masks["is_slab_node"]
        node_fields = torch.zeros_like(node_fields_molecule)
        node_fields[is_molecule] = node_fields_molecule[is_molecule]
        node_fields[is_slab] = node_fields_slab[is_slab]

        return self.displaced_interactions(
            batch=batch,
            positions=node_positions,
            node_fields=node_fields,
        )


class GTOElectrostaticFeatures(torch.nn.Module):
    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: List[float],
        include_self_interaction: bool,
        kspace_cutoff: float,
        quadrupole_feature_corrections: bool = False,
        integral_normalization: str = "receiver",
    ):
        super().__init__()
        self.density_basis = GTOBasis(
            max_l=density_max_l,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.feature_basis = GTOBasis(
            max_l=feature_max_l,
            sigmas=feature_smearing_widths,
            kspace_cutoff=kspace_cutoff,
            normalize=integral_normalization,
        )
        self.kspace_cutoff = kspace_cutoff
        self.include_self_interaction = include_self_interaction

        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=feature_max_l,
            sigmas_receive=feature_smearing_widths,
            normalize_source="multipoles",
            normalize_receive=integral_normalization,
        )
        self.realspace_features = RealSpaceFiniteDifferenceElectrostaticFeatures(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=feature_max_l,
            projection_smearing_widths=feature_smearing_widths,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
        )
        self.non_periodic_correction_terms = NonPeriodicFeatureCorrections(
            density_max_l=density_max_l,
            projection_max_l=feature_max_l,
            projection_smearing_widths=feature_smearing_widths,
            integral_normalization=integral_normalization,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )
        self.register_buffer(
            "output_permutation",
            self._build_output_permutation(
                max_l=feature_max_l,
                n_radial=len(feature_smearing_widths),
            ),
        )

        self.static_quantities = None

    @staticmethod
    def _as_volume_scalar(volume: torch.Tensor) -> torch.Tensor:
        return volume.item()

    @staticmethod
    def _build_output_permutation(max_l: int, n_radial: int) -> torch.Tensor:
        indices = []
        block = (max_l + 1) ** 2
        for l in range(max_l + 1):
            for c in range(n_radial):
                offset = c * block
                indices += range(l**2 + offset, (l + 1) ** 2 + offset)
        return torch.tensor(indices, dtype=torch.long)

    def _permute_output_channels(self, features_flat: torch.Tensor) -> torch.Tensor:
        return torch.index_select(features_flat, dim=-1, index=self.output_permutation)

    @staticmethod
    def _build_correction_cache(pbc: torch.Tensor, batch: torch.Tensor) -> dict:
        pbc_bool = pbc.to(dtype=torch.bool)
        is_pbc_graph = pbc_bool.all(dim=1)
        is_molecule_graph = (~pbc_bool).all(dim=1)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])

        if is_pbc_graph.all():
            return {"correction_mode": CORRECTION_MODE_PBC}
        if is_molecule_graph.all():
            return {"correction_mode": CORRECTION_MODE_MOLECULE}
        if is_slab_graph.all():
            return {"correction_mode": CORRECTION_MODE_SLAB}

        return {
            "correction_mode": CORRECTION_MODE_MIXED,
            "correction_node_masks": {
                "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
                "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
            },
        }

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        force_pbc_evaluator: bool = False,
    ) -> torch.Tensor:
        self.static_quantities = self.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
            force_pbc_evaluator=force_pbc_evaluator,
        )
        return self.forward_dynamic(
            cache=self.static_quantities,
            source_feats=source_feats,
            pbc=pbc,
        )

    def precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        force_pbc_evaluator: bool = False,
    ) -> dict:
        if torch.any(pbc) or force_pbc_evaluator:
            return self._pbc_precompute_geometry(
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
                node_positions=node_positions,
                batch=batch,
                volume=volume,
                pbc=pbc,
            )
        return self._realspace_precompute_geometry(
            node_positions=node_positions,
            batch=batch,
        )

    def forward_dynamic(
        self, cache: dict, source_feats: torch.Tensor, pbc: torch.Tensor
    ) -> torch.Tensor:
        if cache.get("mode") == "realspace":
            return self._realspace_forward_dynamic(
                source_feats=source_feats,
                cache=cache,
            )
        return self._pbc_forward_dynamic(
            source_feats=source_feats,
            cache=cache,
        )

    def compute_esps(
        self, cache: dict, source_feats: torch.Tensor, pbc: torch.Tensor
    ) -> torch.Tensor:
        if cache.get("mode") == "realspace":
            raise NotImplementedError(
                "Real-space ESP evaluation is not implemented yet."
            )
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            k_norm2=cache["k_norm2"],
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        esps = reconstruct_esps_batch(
            potential=potential,
            cosines=cache["cosines"],
            sines=cache["sines"],
            k0_mask=cache["k0_mask"],
        )
        pbc_bool = cache["pbc"].to(dtype=torch.bool)
        is_pbc_graph = pbc_bool.all(dim=1)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])
        if not torch.all(is_pbc_graph | is_slab_graph):
            raise ValueError("ESP corrections only support TTT or TTF geometries.")
        if is_slab_graph.any():
            source_feats_lm = (
                source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
            )
            total_dipole_z = _get_total_dipole_z(
                source_feats_lm, cache["node_positions"], cache["batch"]
            )
            field_z = FIELD_CONSTANT * total_dipole_z / cache["volumes"]
            spread_field_z = torch.index_select(field_z, 0, cache["batch"])
            delta_esps = spread_field_z * cache["node_positions"][:, 2]
            slab_node_mask = torch.index_select(is_slab_graph, 0, cache["batch"])
            esps = esps + delta_esps * slab_node_mask.to(dtype=delta_esps.dtype)
        return esps

    def _realspace_precompute_geometry(
        self,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict:
        return {
            "mode": "realspace",
            "node_positions": node_positions,
            "batch": batch,
        }

    def _realspace_forward_dynamic(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        features, _, _ = self.realspace_features(
            source_feats=source_feats,
            node_positions=cache["node_positions"],
            batch=cache["batch"],
        )
        return features

    def _pbc_precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        inner_products = torch.matmul(
            k_vectors, node_positions.t()
        )  # [n_k_total, n_nodes]
        mask = k_vector_batch[:, None] == batch[None, :]
        mask_f = mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        feature_basis_fs = self.feature_basis(k_vectors, k_norm2, k0_mask)

        volume_per_k = volume.reshape(-1)[k_vector_batch]
        k0_mask_bool = k0_mask > 0.0
        k_factor_coulomb = torch.zeros_like(k_norm2)
        k_factor_coulomb[~k0_mask_bool] = 1.0 / k_norm2[~k0_mask_bool]
        k_factor_proj = torch.ones_like(k_norm2)
        k_factor_proj[k0_mask_bool] = 0.5

        correction_cache = self._build_correction_cache(pbc=pbc, batch=batch)

        return {
            "mode": "pbc",
            "k_vectors": k_vectors,
            "k_norm2": k_norm2,
            "k_vector_batch": k_vector_batch,
            "k0_mask": k0_mask,
            "volume_per_k": volume_per_k,
            "k_factor_coulomb": k_factor_coulomb,
            "k_factor_proj": k_factor_proj,
            "volumes": volume.reshape(-1),
            "batch": batch,
            "node_positions": node_positions,
            "pbc": pbc,
            "cosines": cosines,
            "sines": sines,
            "density_basis_fs": density_basis_fs,
            "feature_basis_fs": feature_basis_fs,
            **correction_cache,
        }

    def _pbc_forward_dynamic(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        density = assemble_fourier_series_batch(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            k_norm2=cache["k_norm2"],
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        features_si = project_to_features_batch(
            potential=potential,
            feature_basis_fs=cache["feature_basis_fs"],
            cosines=cache["cosines"],
            sines=cache["sines"],
            k_factor_proj=cache["k_factor_proj"],
        )
        features_flat = features_si.reshape(features_si.size(0), -1)
        features_flat = self._permute_output_channels(features_flat)

        correction_mode = cache.get("correction_mode", CORRECTION_MODE_MIXED)
        if correction_mode != CORRECTION_MODE_PBC:
            correction_terms = self.non_periodic_correction_terms(
                source_feats=source_feats,
                node_positions=cache["node_positions"],
                batch=cache["batch"],
                volumes=cache["volumes"],
                pbc=cache["pbc"],
                correction_mode=correction_mode,
                correction_node_masks=cache.get("correction_node_masks"),
            )

        source_feats_lm = (
            source_feats.squeeze(-2) if source_feats.dim() == 3 else source_feats
        )
        si_terms = self.self_interaction_terms(source_feats_lm)
        if not self.include_self_interaction:
            features_flat = features_flat - si_terms
        if correction_mode != CORRECTION_MODE_PBC:
            features_flat = features_flat + correction_terms
        return features_flat


class GTOElectrostaticFeaturesMultiChannel(torch.nn.Module):
    """Compute electrostatic features for independent scalar channels."""

    def __init__(
        self,
        density_max_l: int,
        density_smearing_width: float,
        feature_max_l: int,
        feature_smearing_widths: List[float],
        kspace_cutoff: float,
        include_self_interaction: bool = False,
        integral_normalization: str = "receiver",
        quadrupole_feature_corrections: bool = False,
    ):
        super().__init__()
        self.density_basis = GTOBasis(
            max_l=density_max_l,
            sigmas=[density_smearing_width],
            kspace_cutoff=kspace_cutoff,
            normalize="multipoles",
        )
        self.feature_basis = GTOBasis(
            max_l=feature_max_l,
            sigmas=feature_smearing_widths,
            kspace_cutoff=kspace_cutoff,
            normalize=integral_normalization,
        )
        self.kspace_cutoff = kspace_cutoff
        self.include_self_interaction = include_self_interaction

        self.self_interaction_terms = GTOSelfInteractionBlock(
            l_source=density_max_l,
            sigma_source=density_smearing_width,
            l_receive=feature_max_l,
            sigmas_receive=feature_smearing_widths,
            normalize_source="multipoles",
            normalize_receive=integral_normalization,
        )
        self.realspace_features = RealSpaceFiniteDifferenceElectrostaticFeatures(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            projection_max_l=feature_max_l,
            projection_smearing_widths=feature_smearing_widths,
            include_self_interaction=include_self_interaction,
            integral_normalization=integral_normalization,
        )
        self.non_periodic_correction_terms = NonPeriodicFeatureCorrections(
            density_max_l=density_max_l,
            projection_max_l=feature_max_l,
            projection_smearing_widths=feature_smearing_widths,
            integral_normalization=integral_normalization,
            quadrupole_feature_corrections=quadrupole_feature_corrections,
        )
        self.register_buffer(
            "output_permutation",
            self._build_output_permutation(
                max_l=feature_max_l,
                n_radial=len(feature_smearing_widths),
            ),
        )

        self.static_quantities = None

    @staticmethod
    def _build_output_permutation(max_l: int, n_radial: int) -> torch.Tensor:
        indices = []
        block = (max_l + 1) ** 2
        for l in range(max_l + 1):
            for c in range(n_radial):
                offset = c * block
                indices += range(l**2 + offset, (l + 1) ** 2 + offset)
        return torch.tensor(indices, dtype=torch.long)

    def _permute_output_channels(self, features_flat: torch.Tensor) -> torch.Tensor:
        return torch.index_select(features_flat, dim=-1, index=self.output_permutation)

    @staticmethod
    def _build_correction_cache(pbc: torch.Tensor, batch: torch.Tensor) -> dict:
        pbc_bool = pbc.to(dtype=torch.bool)
        is_pbc_graph = pbc_bool.all(dim=1)
        is_molecule_graph = (~pbc_bool).all(dim=1)
        is_slab_graph = pbc_bool[:, 0] & pbc_bool[:, 1] & (~pbc_bool[:, 2])

        if is_pbc_graph.all():
            return {"correction_mode": CORRECTION_MODE_PBC}
        if is_molecule_graph.all():
            return {"correction_mode": CORRECTION_MODE_MOLECULE}
        if is_slab_graph.all():
            return {"correction_mode": CORRECTION_MODE_SLAB}

        return {
            "correction_mode": CORRECTION_MODE_MIXED,
            "correction_node_masks": {
                "is_molecule_node": torch.index_select(is_molecule_graph, 0, batch),
                "is_slab_node": torch.index_select(is_slab_graph, 0, batch),
            },
        }

    @staticmethod
    def _ensure_channel_dim(source_feats: torch.Tensor) -> torch.Tensor:
        if source_feats.dim() == 2:
            return source_feats.unsqueeze(1)
        if source_feats.dim() == 3:
            return source_feats
        raise ValueError(
            "source_feats must have shape [n_nodes, m_dim] or [n_nodes, n_channels, m_dim]"
        )

    @staticmethod
    def _repeat_nodes_for_channels(
        node_positions: torch.Tensor, batch: torch.Tensor, n_channels: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if n_channels == 1:
            return node_positions, batch
        return (
            node_positions.repeat_interleave(n_channels, dim=0),
            batch.repeat_interleave(n_channels, dim=0),
        )

    def forward(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        source_feats: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        force_pbc_evaluator: bool = False,
    ) -> torch.Tensor:
        self.static_quantities = self.precompute_geometry(
            k_vectors=k_vectors,
            k_norm2=k_norm2,
            k_vector_batch=k_vector_batch,
            k0_mask=k0_mask,
            node_positions=node_positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
            force_pbc_evaluator=force_pbc_evaluator,
        )
        return self.forward_dynamic(
            cache=self.static_quantities,
            source_feats=source_feats,
            pbc=pbc,
        )

    def precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
        force_pbc_evaluator: bool = False,
    ) -> dict:
        if torch.any(pbc) or force_pbc_evaluator:
            return self._pbc_precompute_geometry(
                k_vectors=k_vectors,
                k_norm2=k_norm2,
                k_vector_batch=k_vector_batch,
                k0_mask=k0_mask,
                node_positions=node_positions,
                batch=batch,
                volume=volume,
                pbc=pbc,
            )
        return self._realspace_precompute_geometry(
            node_positions=node_positions,
            batch=batch,
        )

    def forward_dynamic(
        self, cache: dict, source_feats: torch.Tensor, pbc: torch.Tensor
    ) -> torch.Tensor:
        if cache.get("mode") == "realspace":
            return self._realspace_forward_dynamic(
                source_feats=source_feats,
                cache=cache,
            )
        return self._pbc_forward_dynamic(
            source_feats=source_feats,
            cache=cache,
        )

    def _realspace_precompute_geometry(
        self,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
    ) -> dict:
        return {
            "mode": "realspace",
            "node_positions": node_positions,
            "batch": batch,
        }

    def _realspace_forward_dynamic(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        source_feats = self._ensure_channel_dim(source_feats)
        n_nodes, n_channels, m_dim = source_feats.shape
        source_flat = source_feats.reshape(n_nodes * n_channels, m_dim)
        node_positions, batch = self._repeat_nodes_for_channels(
            cache["node_positions"], cache["batch"], n_channels
        )
        features, _, _ = self.realspace_features(
            source_feats=source_flat,
            node_positions=node_positions,
            batch=batch,
        )
        features = features.reshape(n_nodes, n_channels, -1)
        return features

    def _pbc_precompute_geometry(
        self,
        k_vectors: torch.Tensor,
        k_norm2: torch.Tensor,
        k_vector_batch: torch.Tensor,
        k0_mask: torch.Tensor,
        node_positions: torch.Tensor,
        batch: torch.Tensor,
        volume: torch.Tensor,
        pbc: torch.Tensor,
    ) -> dict:
        inner_products = torch.matmul(
            k_vectors, node_positions.t()
        )  # [n_k_total, n_nodes]
        mask = k_vector_batch[:, None] == batch[None, :]
        mask_f = mask.to(dtype=inner_products.dtype)
        cosines = torch.cos(inner_products) * mask_f
        sines = torch.sin(inner_products) * mask_f

        density_basis_fs = self.density_basis(k_vectors, k_norm2, k0_mask)
        feature_basis_fs = self.feature_basis(k_vectors, k_norm2, k0_mask)

        volume_per_k = volume.reshape(-1)[k_vector_batch]
        k0_mask_bool = k0_mask > 0.0
        k_factor_coulomb = torch.zeros_like(k_norm2)
        k_factor_coulomb[~k0_mask_bool] = 1.0 / k_norm2[~k0_mask_bool]
        k_factor_proj = torch.ones_like(k_norm2)
        k_factor_proj[k0_mask_bool] = 0.5

        correction_cache = self._build_correction_cache(pbc=pbc, batch=batch)

        return {
            "mode": "pbc",
            "k_vectors": k_vectors,
            "k_norm2": k_norm2,
            "k_vector_batch": k_vector_batch,
            "k0_mask": k0_mask,
            "volume_per_k": volume_per_k,
            "k_factor_coulomb": k_factor_coulomb,
            "k_factor_proj": k_factor_proj,
            "volumes": volume.reshape(-1),
            "batch": batch,
            "node_positions": node_positions,
            "pbc": pbc,
            "cosines": cosines,
            "sines": sines,
            "density_basis_fs": density_basis_fs,
            "feature_basis_fs": feature_basis_fs,
            **correction_cache,
        }

    def _pbc_forward_dynamic(
        self, source_feats: torch.Tensor, cache: dict
    ) -> torch.Tensor:
        source_feats = self._ensure_channel_dim(source_feats)
        n_nodes, n_channels, m_dim = source_feats.shape

        density = assemble_fourier_series_batch_channels(
            source_feats=source_feats,
            cosines=cache["cosines"],
            sines=cache["sines"],
            density_basis_fs=cache["density_basis_fs"],
            volume_per_k=cache["volume_per_k"],
        )
        potential = apply_coulomb_kernel_batch(
            k_norm2=cache["k_norm2"],
            density=density,
            k_factor_coulomb=cache["k_factor_coulomb"],
        )
        features_si = project_to_features_batch_channels(
            potential=potential,
            feature_basis_fs=cache["feature_basis_fs"],
            cosines=cache["cosines"],
            sines=cache["sines"],
            k_factor_proj=cache["k_factor_proj"],
        )
        features_flat = features_si.reshape(n_nodes, n_channels, -1)
        features_flat = self._permute_output_channels(features_flat)

        correction_mode = cache.get("correction_mode", CORRECTION_MODE_MIXED)
        if correction_mode != CORRECTION_MODE_PBC:
            node_positions, batch = self._repeat_nodes_for_channels(
                cache["node_positions"], cache["batch"], n_channels
            )
            source_flat = source_feats.reshape(n_nodes * n_channels, m_dim)
            correction_node_masks = cache.get("correction_node_masks")
            if correction_node_masks is not None:
                correction_node_masks = {
                    "is_molecule_node": correction_node_masks[
                        "is_molecule_node"
                    ].repeat_interleave(n_channels),
                    "is_slab_node": correction_node_masks[
                        "is_slab_node"
                    ].repeat_interleave(n_channels),
                }
            correction_terms = self.non_periodic_correction_terms(
                source_feats=source_flat,
                node_positions=node_positions,
                batch=batch,
                volumes=cache["volumes"],
                pbc=cache["pbc"],
                correction_mode=correction_mode,
                correction_node_masks=correction_node_masks,
            )
            correction_terms = correction_terms.reshape(n_nodes, n_channels, -1)
        else:
            correction_terms = None

        source_flat = source_feats.reshape(n_nodes * n_channels, m_dim)
        si_terms = self.self_interaction_terms(source_flat).reshape(
            n_nodes, n_channels, -1
        )

        if not self.include_self_interaction:
            features_flat = features_flat - si_terms
        if correction_terms is not None:
            features_flat = features_flat + correction_terms
        return features_flat

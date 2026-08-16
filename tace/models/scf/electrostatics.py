################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import math
from typing import Dict, List, Optional

import torch

from tace.utils.torch_scatter import scatter_sum

from .longrange.energy import GTOElectrostaticEnergy
from .longrange.features import GTOElectrostaticFeatures
from .longrange.gto_utils import gto_basis_kspace_cutoff
from .longrange.kspace import compute_k_vectors_flat
from .utils import num_graphs


class PairwiseGaussianElectrostatics(torch.nn.Module):
    """Differentiable non-periodic Gaussian charge electrostatics.

    This backend is useful for small molecules and tests. The full periodic and
    multipolar implementation is provided by ``GraphLongRangeElectrostatics``.
    """

    def __init__(
        self,
        smearing_width: float = 1.0,
        include_self_interaction: bool = False,
    ) -> None:
        super().__init__()
        if smearing_width <= 0.0:
            raise ValueError("smearing_width must be positive")
        self.register_buffer("smearing_width", torch.tensor(float(smearing_width)))
        self.include_self_interaction = bool(include_self_interaction)
        self.coulomb_constant = 14.399645478425668

    def _validate(self, density_coefficients: torch.Tensor) -> None:
        if density_coefficients.shape[-1] != 1:
            raise ValueError(
                "The pairwise backend supports monopoles only; use backend="
                "'graph_longrange' for multipoles"
            )

    def potential(
        self,
        density_coefficients: torch.Tensor,
        data: Dict[str, torch.Tensor],
        graph=None,
    ) -> torch.Tensor:
        self._validate(density_coefficients)
        charges = density_coefficients[:, 0]
        positions = data["positions"]
        batch = data["batch"]
        potential = torch.zeros_like(charges)
        sigma = self.smearing_width.to(charges)
        for graph_index in range(num_graphs(data)):
            indices = torch.where(batch == graph_index)[0]
            xyz = positions.index_select(0, indices)
            q = charges.index_select(0, indices)
            distances = torch.cdist(xyz, xyz)
            mask = ~torch.eye(
                indices.numel(), dtype=torch.bool, device=positions.device
            )
            safe_distances = distances.masked_fill(~mask, 1.0)
            kernel = torch.erf(safe_distances / (2.0 * sigma)) / safe_distances
            self_term = (
                1.0 / (math.sqrt(math.pi) * sigma)
                if self.include_self_interaction
                else sigma.new_zeros(())
            )
            kernel = kernel.masked_fill(~mask, self_term)
            potential.index_copy_(
                0, indices, self.coulomb_constant * torch.matmul(kernel, q)
            )
        return potential.unsqueeze(-1)

    def energy(
        self,
        density_coefficients: torch.Tensor,
        data: Dict[str, torch.Tensor],
        graph=None,
    ) -> torch.Tensor:
        potential = self.potential(density_coefficients, data, graph)[:, 0]
        node_energy = 0.5 * density_coefficients[:, 0] * potential
        return scatter_sum(
            node_energy,
            data["batch"],
            dim=0,
            dim_size=num_graphs(data),
        )

    def features(
        self,
        density_coefficients: torch.Tensor,
        data: Dict[str, torch.Tensor],
        graph=None,
    ) -> torch.Tensor:
        return self.potential(density_coefficients, data, graph)


class GraphLongRangeElectrostatics(torch.nn.Module):
    """Gaussian multipole electrostatics backed by graph_longrange."""

    def __init__(
        self,
        density_max_l: int = 0,
        density_smearing_width: float = 1.0,
        feature_max_l: int = 0,
        feature_smearing_widths: Optional[List[float]] = None,
        kspace_cutoff_factor: float = 1.5,
        include_self_interaction: bool = False,
    ) -> None:
        super().__init__()
        if feature_smearing_widths is None:
            feature_smearing_widths = [density_smearing_width]
        widths = [density_smearing_width, *feature_smearing_widths]
        kspace_cutoff = kspace_cutoff_factor * gto_basis_kspace_cutoff(
            widths, max(density_max_l, feature_max_l)
        )
        self.register_buffer("kspace_cutoff", torch.tensor(kspace_cutoff))
        self.energy_operator = GTOElectrostaticEnergy(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            kspace_cutoff=kspace_cutoff,
            include_self_interaction=include_self_interaction,
        )
        self.feature_operator = GTOElectrostaticFeatures(
            density_max_l=density_max_l,
            density_smearing_width=density_smearing_width,
            feature_max_l=feature_max_l,
            feature_smearing_widths=feature_smearing_widths,
            include_self_interaction=include_self_interaction,
            kspace_cutoff=kspace_cutoff,
        )
        self.compute_k_vectors_flat = compute_k_vectors_flat
        self.feature_max_l = int(feature_max_l)
        self.num_feature_widths = len(feature_smearing_widths)

    def _geometry(self, data: Dict[str, torch.Tensor], graph=None):
        positions = data["positions"] if graph is None else graph.positions
        batch = data["batch"]
        n_graphs = num_graphs(data)
        cell = data.get("lattice", data.get("cell"))
        if cell is None:
            cell = positions.new_zeros((n_graphs, 3, 3))
        cell = cell.reshape(n_graphs, 3, 3)
        pbc = data.get("pbc")
        if pbc is None:
            pbc = torch.zeros((n_graphs, 3), dtype=torch.bool, device=positions.device)
        pbc = pbc.reshape(n_graphs, 3).to(dtype=torch.bool)
        volume = torch.linalg.det(cell).abs()
        if torch.any(pbc):
            if torch.any(volume <= 0.0):
                raise ValueError("Periodic SCF structures require a non-singular cell")
            reciprocal = 2.0 * math.pi * torch.linalg.inv(cell).transpose(-1, -2)
            k_vectors = self.compute_k_vectors_flat(
                float(self.kspace_cutoff.item()), cell, reciprocal
            )
        else:
            k_vectors = (
                positions.new_zeros((1, 3)),
                positions.new_zeros(1),
                torch.zeros(1, dtype=torch.long, device=positions.device),
                positions.new_ones(1),
            )
            volume = torch.ones_like(volume)
        return positions, batch, volume, pbc, k_vectors

    def energy(self, density_coefficients, data, graph=None):
        positions, batch, volume, pbc, k = self._geometry(data, graph)
        return self.energy_operator(
            k_vectors=k[0],
            k_norm2=k[1],
            k_vector_batch=k[2],
            k0_mask=k[3],
            source_feats=density_coefficients,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )

    def features(self, density_coefficients, data, graph=None):
        positions, batch, volume, pbc, k = self._geometry(data, graph)
        return self.feature_operator(
            k_vectors=k[0],
            k_norm2=k[1],
            k_vector_batch=k[2],
            k0_mask=k[3],
            source_feats=density_coefficients,
            node_positions=positions,
            batch=batch,
            volume=volume,
            pbc=pbc,
        )


def build_electrostatics(config: Optional[Dict] = None) -> torch.nn.Module:
    config = {} if config is None else dict(config)
    backend = config.pop("backend", "graph_longrange")
    if backend == "graph_longrange":
        return GraphLongRangeElectrostatics(**config)
    if backend == "pairwise":
        allowed = {"smearing_width", "include_self_interaction"}
        unexpected = sorted(set(config).difference(allowed))
        if unexpected:
            raise ValueError(f"Unsupported pairwise electrostatics keys: {unexpected}")
        return PairwiseGaussianElectrostatics(**config)
    raise ValueError("electrostatics backend must be 'graph_longrange' or 'pairwise'")

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import math
from typing import Dict, List, Optional, Sequence

import ase
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..utils.torch_scatter import scatter
from ..utils.utils import log_statistics_to_yaml
from .element import TorchElement
from .quantity import KeySpecification


def _canonical_atomic_energy(
    atomic_energy: Dict[int, float], atomic_numbers: Sequence[int]
) -> Dict[int, float]:
    values = {int(z): float(value) for z, value in atomic_energy.items()}
    return {int(z): values.get(int(z), 0.0) for z in atomic_numbers}


class OneHotToAtomicEnergy(torch.nn.Module):
    def __init__(
        self,
        atomic_energies: List[Dict[int, float]],
        atomic_numbers: Sequence[int],
    ) -> None:
        super().__init__()
        if atomic_energies is None:
            raise ValueError("atomic_energies must be provided")
        atomic_energy_list = [
            list(_canonical_atomic_energy(energy, atomic_numbers).values())
            for energy in atomic_energies
        ]
        self.register_buffer(
            "atomic_energy",
            torch.tensor(atomic_energy_list, dtype=torch.get_default_dtype()),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.atomic_energy.T)

    def __repr__(self):
        values = [f"{x:.4f}" for x in self.atomic_energy.reshape(-1).tolist()]
        return f"{self.__class__.__name__}(atomic_energies={values})"


class _RunningMoments:
    """Constant-memory population moments merged one batch at a time."""

    def __init__(self) -> None:
        self.count = 0
        self.mean: Optional[torch.Tensor] = None
        self.m2: Optional[torch.Tensor] = None

    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(dtype=torch.float64)
        if values.ndim == 0:
            values = values.reshape(1)
        if values.shape[0] == 0:
            return

        batch_count = int(values.shape[0])
        batch_mean = values.mean(dim=0).cpu()
        batch_m2 = ((values - batch_mean.to(values.device)) ** 2).sum(dim=0).cpu()
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return

        delta = batch_mean - self.mean
        new_count = self.count + batch_count
        self.mean = self.mean + delta * (batch_count / new_count)
        self.m2 = (
            self.m2 + batch_m2 + delta.square() * (self.count * batch_count / new_count)
        )
        self.count = new_count

    def mean_or_zeros(self, shape=()):
        if self.count == 0:
            return torch.zeros(shape, dtype=torch.float64)
        return self.mean

    def std_or_zeros(self, shape=(), unbiased: bool = False):
        if self.count == 0 or (unbiased and self.count == 1):
            return torch.zeros(shape, dtype=torch.float64)
        denominator = self.count - 1 if unbiased else self.count
        return torch.sqrt(torch.clamp(self.m2 / denominator, min=0.0))

    def rms_or_zeros(self, shape=()):
        if self.count == 0:
            return torch.zeros(shape, dtype=torch.float64)
        mean_square = self.m2 / self.count + self.mean.square()
        return torch.sqrt(torch.clamp(mean_square, min=0.0))


def _compute_atomic_energy(
    points: Sequence[ase.Atoms],
    element: TorchElement,
    keyspec: KeySpecification,
) -> Dict[int, float]:
    energy_key = keyspec.info_keys["energy"]
    points = [atoms for atoms in points if energy_key in atoms.info]
    if not points:
        return {z: 0.0 for z in element.zs}

    matrix = np.zeros((len(points), len(element)), dtype=np.float64)
    energies = np.zeros(len(points), dtype=np.float64)
    for row, atoms in enumerate(points):
        energies[row] = float(atoms.info[energy_key])
        atomic_numbers = atoms.get_atomic_numbers()
        for column, z in enumerate(element.zs):
            matrix[row, column] = np.count_nonzero(atomic_numbers == z)

    try:
        solution, _, rank, _ = np.linalg.lstsq(matrix, energies, rcond=None)
    except np.linalg.LinAlgError:
        logging.warning(
            "Failed to fit isolated atomic energies; using 0.0 for every element"
        )
        return {z: 0.0 for z in element.zs}

    if rank < min(matrix.shape):
        logging.warning(
            "Atomic-energy composition matrix is rank deficient (%s < %s); "
            "unconstrained values use the least-squares minimum-norm solution",
            rank,
            min(matrix.shape),
        )
    return {z: float(solution[idx]) for idx, z in enumerate(element.zs)}


def compute_atomic_energy(
    atomsList: Sequence[ase.Atoms],
    element: TorchElement,
    keyspec: KeySpecification,
    fidelity_idx: int,
) -> Dict[int, float]:
    fidelity_key = keyspec.info_keys["fidelity_idx"]
    this_atoms_list = [
        atoms
        for atoms in atomsList
        if int(atoms.info.get(fidelity_key, 0)) == fidelity_idx
    ]
    if this_atoms_list:
        return _compute_atomic_energy(this_atoms_list, element, keyspec)

    logging.warning(
        "No training structures found for fidelity %s; using atomic energy 0.0",
        fidelity_idx,
    )
    return {z: 0.0 for z in element.zs}


def _finite_scale(value: float) -> float:
    if not math.isfinite(value):
        return 1.0
    return value


def _compute_statistics(
    dataloader_train: DataLoader,
    atomic_numbers: List[int],
    atomic_energies: Optional[List[Dict[int, float]]],
    target_property: List[str],
    device: str = "cpu",
    num_fidelities: int = 1,
) -> List[Dict]:
    if num_fidelities < 1:
        raise ValueError("At least one fidelity must be configured")

    graph_counts = torch.zeros(num_fidelities, dtype=torch.int64)
    atom_counts = torch.zeros(num_fidelities, dtype=torch.int64)
    neighbor_moments = _RunningMoments()
    neighbor_by_element = [_RunningMoments() for _ in atomic_numbers]

    energy_moments = [_RunningMoments() for _ in range(num_fidelities)]
    energy_per_atom_moments = [_RunningMoments() for _ in range(num_fidelities)]
    delta_per_atom_moments = [_RunningMoments() for _ in range(num_fidelities)]
    force_moments = [_RunningMoments() for _ in range(num_fidelities)]
    force_norm_moments = [_RunningMoments() for _ in range(num_fidelities)]
    force_by_element = [
        [_RunningMoments() for _ in atomic_numbers] for _ in range(num_fidelities)
    ]
    force_norm_by_element = [
        [_RunningMoments() for _ in atomic_numbers] for _ in range(num_fidelities)
    ]
    magmoms_norm_by_element = torch.zeros(len(atomic_numbers), dtype=torch.float64)
    has_noncollinear_magmoms = False

    atomic_energy_fn = None
    if "energy" in target_property:
        if atomic_energies is None:
            raise ValueError("atomic_energies are required for energy statistics")
        atomic_energy_fn = OneHotToAtomicEnergy(atomic_energies, atomic_numbers).to(
            device
        )

    with torch.no_grad():
        for data in dataloader_train:
            data = data.to(device)
            num_graphs = int(data.ptr.numel() - 1)
            num_nodes = data.ptr[1:] - data.ptr[:-1]
            element_idx = data["node_attrs"].argmax(dim=-1)

            fidelity_idx = data.get("fidelity_idx")
            if fidelity_idx is None:
                fidelity_idx = torch.zeros(
                    num_graphs, dtype=torch.int64, device=data.ptr.device
                )
            else:
                fidelity_idx = fidelity_idx.reshape(-1).to(dtype=torch.int64)
            if fidelity_idx.numel() != num_graphs:
                raise ValueError(
                    "fidelity_idx must contain exactly one value per structure"
                )
            if fidelity_idx.numel() and (
                int(fidelity_idx.min()) < 0 or int(fidelity_idx.max()) >= num_fidelities
            ):
                raise ValueError(
                    "fidelity_idx values must be in "
                    f"[0, {num_fidelities - 1}], got "
                    f"[{int(fidelity_idx.min())}, {int(fidelity_idx.max())}]"
                )

            node_fidelity = fidelity_idx[data["batch"]]
            graph_counts += torch.bincount(fidelity_idx.cpu(), minlength=num_fidelities)
            atom_counts += torch.bincount(node_fidelity.cpu(), minlength=num_fidelities)

            source = data.edge_index[0]
            neighbor_counts = torch.bincount(source, minlength=data.batch.size(0)).to(
                torch.float64
            )
            neighbor_moments.update(neighbor_counts)
            for element_id in element_idx.unique().tolist():
                neighbor_by_element[element_id].update(
                    neighbor_counts[element_idx == element_id]
                )

            initial_noncollinear_magmoms = data.get("initial_noncollinear_magmoms")
            if initial_noncollinear_magmoms is not None:
                has_noncollinear_magmoms = True
                magnetic_norm = torch.linalg.vector_norm(
                    initial_noncollinear_magmoms,
                    dim=-1,
                )
                for element_id in element_idx.unique().tolist():
                    element_mask = element_idx == element_id
                    magmoms_norm_by_element[element_id] = torch.maximum(
                        magmoms_norm_by_element[element_id],
                        magnetic_norm[element_mask].max().cpu(),
                    )

            if "energy" in target_property:
                num_atoms_arange = torch.arange(
                    data["node_attrs"].size(0),
                    device=data["node_attrs"].device,
                    dtype=torch.int64,
                )
                e0_node_energy = atomic_energy_fn(data["node_attrs"])[
                    num_atoms_arange, node_fidelity
                ]
                e0 = scatter(
                    e0_node_energy,
                    data["batch"],
                    dim=0,
                    dim_size=num_graphs,
                    reduce="sum",
                )
                energy = data["energy"].reshape(-1)
                energy_per_atom = energy / num_nodes
                delta_per_atom = (energy - e0) / num_nodes
                energy_weight = data.get("energy_weight")
                if energy_weight is None:
                    valid_energy = torch.ones_like(fidelity_idx, dtype=torch.bool)
                else:
                    valid_energy = energy_weight.reshape(-1) > 0
                for level in fidelity_idx.unique().tolist():
                    mask = (fidelity_idx == level) & valid_energy
                    energy_moments[level].update(energy[mask])
                    energy_per_atom_moments[level].update(energy_per_atom[mask])
                    delta_per_atom_moments[level].update(delta_per_atom[mask])

            if "forces" in target_property or "direct_forces" in target_property:
                forces = data.get("forces")
                if forces is None and "direct_forces" in target_property:
                    forces = data.get("direct_forces")
                if forces is None:
                    raise KeyError(
                        "Force statistics requested but no forces were found"
                    )
                force_norm = torch.linalg.vector_norm(forces, dim=1)
                force_weight = data.get("forces_weight")
                if force_weight is None and "direct_forces" in target_property:
                    force_weight = data.get("direct_forces_weight")
                if force_weight is None:
                    valid_force = torch.ones_like(node_fidelity, dtype=torch.bool)
                else:
                    valid_force = force_weight.reshape(-1)[data["batch"]] > 0
                for level in node_fidelity.unique().tolist():
                    level_mask = (node_fidelity == level) & valid_force
                    level_forces = forces[level_mask]
                    level_elements = element_idx[level_mask]
                    level_norms = force_norm[level_mask]
                    force_moments[level].update(level_forces)
                    force_norm_moments[level].update(level_norms)
                    for element_id in level_elements.unique().tolist():
                        element_mask = level_elements == element_id
                        force_by_element[level][element_id].update(
                            level_forces[element_mask]
                        )
                        force_norm_by_element[level][element_id].update(
                            level_norms[element_mask]
                        )

    if neighbor_moments.count == 0:
        raise ValueError("Cannot compute statistics from an empty training dataset")

    avg_num_neighbors = float(neighbor_moments.mean.item())
    avg_neighbors_by_element = {
        z: float(neighbor_by_element[idx].mean_or_zeros().item())
        for idx, z in enumerate(atomic_numbers)
    }
    if has_noncollinear_magmoms:
        logging.info(
            "Automatically computed magmoms_norm_by_element: %s",
            {
                z: float(magmoms_norm_by_element[idx])
                for idx, z in enumerate(atomic_numbers)
            },
        )

    per_level_stats = []
    for level in range(num_fidelities):
        has_data = int(graph_counts[level]) > 0
        stats = {
            "fidelity_idx": level,
            "available": has_data,
            "num_graphs": int(graph_counts[level]),
            "num_atoms": int(atom_counts[level]),
            "atomic_numbers": atomic_numbers,
            "avg_num_neighbors": avg_num_neighbors,
            "avg_neighbors_by_element": avg_neighbors_by_element,
        }
        if has_noncollinear_magmoms:
            stats["magmoms_norm_by_element"] = {
                z: float(magmoms_norm_by_element[idx])
                for idx, z in enumerate(atomic_numbers)
            }

        if "energy" in target_property:
            energy_mean = float(energy_moments[level].mean_or_zeros().item())
            energy_std = float(energy_moments[level].std_or_zeros(unbiased=True).item())
            energy_per_atom_mean = float(
                energy_per_atom_moments[level].mean_or_zeros().item()
            )
            delta_per_atom_mean = float(
                delta_per_atom_moments[level].mean_or_zeros().item()
            )
            canonical_energy = _canonical_atomic_energy(
                atomic_energies[level], atomic_numbers
            )
            stats.update(
                {
                    "__mean_energy": energy_mean,
                    "__std_energy": energy_std,
                    "__mean_energy_per_atom": energy_per_atom_mean,
                    "__mean_delta_energy_per_atom": delta_per_atom_mean,
                    "atomic_energy": canonical_energy,
                    "scalar_mean_energy_per_atom": energy_per_atom_mean,
                    "mean_energy": {z: energy_mean for z in atomic_numbers},
                    "mean_energy_by_element": {z: energy_mean for z in atomic_numbers},
                    "std_energy": {z: energy_std for z in atomic_numbers},
                    "std_energy_by_element": {z: energy_std for z in atomic_numbers},
                    "mean_energy_per_atom": {
                        z: energy_per_atom_mean for z in atomic_numbers
                    },
                    "mean_energy_per_atom_by_element": {
                        z: energy_per_atom_mean for z in atomic_numbers
                    },
                    "mean_delta_energy_per_atom": {
                        z: delta_per_atom_mean for z in atomic_numbers
                    },
                    "mean_delta_energy_per_atom_by_element": {
                        z: delta_per_atom_mean for z in atomic_numbers
                    },
                }
            )

        if "forces" in target_property or "direct_forces" in target_property:
            force_mean = force_moments[level].mean_or_zeros((3,))
            force_std = force_moments[level].std_or_zeros((3,), unbiased=True)
            force_rms = force_moments[level].rms_or_zeros((3,))
            norm_mean = float(force_norm_moments[level].mean_or_zeros().item())
            norm_std = float(
                force_norm_moments[level].std_or_zeros(unbiased=True).item()
            )
            norm_rms = float(force_norm_moments[level].rms_or_zeros().item())

            mean_3d_by_element = {}
            std_3d_by_element = {}
            rms_3d_by_element = {}
            mean_1d_by_element = {}
            std_1d_by_element = {}
            rms_1d_by_element = {}
            scale_rms_by_element = {}
            scale_std_by_element = {}
            for element_id, z in enumerate(atomic_numbers):
                element_force = force_by_element[level][element_id]
                element_norm = force_norm_by_element[level][element_id]
                mean_3d_by_element[z] = element_force.mean_or_zeros((3,)).tolist()
                std_3d_by_element[z] = element_force.std_or_zeros((3,)).tolist()
                rms_3d_by_element[z] = element_force.rms_or_zeros((3,)).tolist()
                mean_1d_by_element[z] = float(element_norm.mean_or_zeros().item())
                std_value = float(element_norm.std_or_zeros().item())
                rms_value = float(element_norm.rms_or_zeros().item())
                std_1d_by_element[z] = std_value
                rms_1d_by_element[z] = rms_value
                scale_std_by_element[z] = _finite_scale(std_value)
                scale_rms_by_element[z] = _finite_scale(rms_value)

            stats.update(
                {
                    "__mean_forces_3d": force_mean.tolist(),
                    "__std_forces_3d": force_std.tolist(),
                    "__rms_forces_3d": force_rms.tolist(),
                    "__mean_forces_1d": norm_mean,
                    "__std_forces_1d": norm_std,
                    "__rms_forces_1d": norm_rms,
                    "__mean_forces_3d_by_element": mean_3d_by_element,
                    "__std_forces_3d_by_element": std_3d_by_element,
                    "__rms_forces_3d_by_element": rms_3d_by_element,
                    "__mean_forces_1d_by_element": mean_1d_by_element,
                    "__std_forces_1d_by_element": std_1d_by_element,
                    "__rms_forces_1d_by_element": rms_1d_by_element,
                    "mean_forces_for_normalize": norm_mean,
                    "std_forces_for_normalize": norm_std,
                    "rms_forces": {z: _finite_scale(norm_rms) for z in atomic_numbers},
                    "std_forces": {z: _finite_scale(norm_std) for z in atomic_numbers},
                    "rms_forces_by_element": scale_rms_by_element,
                    "std_forces_by_element": scale_std_by_element,
                }
            )

        if not has_data:
            logging.warning(
                "Fidelity %s has no training structures; using neutral scale/shift "
                "statistics and the configured atomic-energy defaults",
                level,
            )
        per_level_stats.append(stats)

    log_statistics_to_yaml(per_level_stats)
    return per_level_stats

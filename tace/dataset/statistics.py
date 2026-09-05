################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict, List, Optional, Sequence

import ase
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from ..utils.utils import log_statistics_to_yaml
from .element import TorchElement
from .quantity import KeySpecification


def balanced_element_weights(
    counts: torch.Tensor,
    mean_losses: torch.Tensor,
    alpha: float = 0.5,
    minimum: float = 0.25,
    maximum: float = 4.0,
) -> torch.Tensor:
    """Recommend element weights from their loss contributions.

    The unweighted contribution of element ``z`` is estimated as
    ``counts[z] / counts.sum() * mean_losses[z]``. ``alpha=0`` leaves the loss
    unchanged, while ``alpha=1`` fully balances those contributions. The
    default ``alpha=0.5`` is a tempered compromise. Returned weights have an
    atom-count-weighted mean of one.
    """
    counts = torch.as_tensor(counts, dtype=torch.float64)
    mean_losses = torch.as_tensor(mean_losses, dtype=torch.float64)
    if counts.ndim != 1 or mean_losses.shape != counts.shape:
        raise ValueError("counts and mean_losses must be one-dimensional and aligned")
    if torch.any(counts < 0) or torch.any(mean_losses < 0):
        raise ValueError("counts and mean_losses must be non-negative")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if not 0.0 < minimum <= maximum:
        raise ValueError("weight bounds must satisfy 0 < minimum <= maximum")

    present = counts > 0
    weights = torch.ones_like(counts)
    if not torch.any(present):
        return weights

    frequencies = counts[present] / counts[present].sum()
    contributions = frequencies * mean_losses[present]
    largest = contributions.max()
    if not torch.isfinite(largest) or largest <= 0:
        return weights

    epsilon = torch.finfo(contributions.dtype).eps * largest
    raw = contributions.clamp_min(epsilon).pow(-alpha)

    lower_scale = 0.0
    upper_scale = 1.0
    while torch.sum(frequencies * (raw * upper_scale).clamp(minimum, maximum)) < 1:
        upper_scale *= 2.0
    for _ in range(64):
        scale = 0.5 * (lower_scale + upper_scale)
        weighted_mean = torch.sum(frequencies * (raw * scale).clamp(minimum, maximum))
        if weighted_mean < 1:
            lower_scale = scale
        else:
            upper_scale = scale
    weights[present] = (raw * upper_scale).clamp(minimum, maximum)
    return weights


def _canonical_atomic_energy(
    atomic_energy: Dict[int, float], atomic_numbers: Sequence[int]
) -> Dict[int, float]:
    values = {int(z): float(value) for z, value in atomic_energy.items()}
    return {int(z): values.get(int(z), 0.0) for z in atomic_numbers}


class _GroupedMoments:
    """Numerically stable streaming moments for indexed groups."""

    def __init__(
        self, num_groups: int, num_features: int, device: torch.device
    ) -> None:
        self.num_groups = num_groups
        self.num_features = num_features
        self.count = torch.zeros(num_groups, dtype=torch.int64, device=device)
        self.mean = torch.zeros(
            (num_groups, num_features), dtype=torch.float64, device=device
        )
        self.m2 = torch.zeros_like(self.mean)

    def update(self, values: torch.Tensor, groups: torch.Tensor) -> None:
        values = values.detach().reshape(-1, self.num_features).to(torch.float64)
        groups = groups.detach().reshape(-1).to(dtype=torch.int64)
        if values.shape[0] != groups.numel():
            raise ValueError("values and groups must have the same leading dimension")
        if groups.numel() == 0:
            return

        batch_count = torch.bincount(groups, minlength=self.num_groups)
        batch_sum = torch.zeros_like(self.mean).index_add_(0, groups, values)
        batch_mean = batch_sum / batch_count.clamp_min(1).unsqueeze(-1)
        batch_m2 = torch.zeros_like(self.m2).index_add_(
            0,
            groups,
            (values - batch_mean[groups]).square(),
        )

        count = self.count + batch_count
        delta = batch_mean - self.mean
        self.mean += delta * (batch_count / count.clamp_min(1)).unsqueeze(-1)
        self.m2 += batch_m2 + delta.square() * (
            self.count * batch_count / count.clamp_min(1)
        ).unsqueeze(-1)
        self.count = count

    def statistics(self, group_size: Optional[int] = None, unbiased: bool = False):
        count = self.count
        mean = self.mean
        m2 = self.m2
        if group_size is not None:
            if self.num_groups % group_size:
                raise ValueError("group_size must divide num_groups")
            count_by_group = count.reshape(-1, group_size)
            mean_by_group = mean.reshape(-1, group_size, self.num_features)
            m2_by_group = m2.reshape(-1, group_size, self.num_features)
            count = count_by_group.sum(dim=1)
            mean = (mean_by_group * count_by_group.unsqueeze(-1)).sum(
                dim=1
            ) / count.clamp_min(1).unsqueeze(-1)
            m2 = (
                m2_by_group
                + count_by_group.unsqueeze(-1)
                * (mean_by_group - mean.unsqueeze(1)).square()
            ).sum(dim=1)

        denominator = count - 1 if unbiased else count
        std = torch.where(
            (denominator > 0).unsqueeze(-1),
            torch.sqrt(torch.clamp(m2 / denominator.clamp_min(1).unsqueeze(-1), min=0)),
            0.0,
        )
        rms = torch.where(
            (count > 0).unsqueeze(-1),
            torch.sqrt(
                torch.clamp(
                    m2 / count.clamp_min(1).unsqueeze(-1) + mean.square(),
                    min=0,
                )
            ),
            0.0,
        )
        return count, mean, std, rms


def _by_element(values: torch.Tensor, atomic_numbers: Sequence[int]) -> Dict:
    return dict(zip(atomic_numbers, values.tolist()))


def compute_atomic_energies(
    atoms_list: Sequence[ase.Atoms],
    element: TorchElement,
    keyspec: KeySpecification,
    fidelity_indices: Sequence[int],
) -> Dict[int, Dict[int, float]]:
    """Fit isolated atomic energies for several fidelities in one data pass."""
    fidelity_indices = tuple(dict.fromkeys(int(idx) for idx in fidelity_indices))
    fidelity_key = keyspec.info_keys["fidelity_idx"]
    energy_key = keyspec.info_keys["energy"]
    element_indices = np.full(max(element.zs) + 1, -1, dtype=np.int64)
    element_indices[element.zs] = np.arange(len(element), dtype=np.int64)
    compositions = {idx: [] for idx in fidelity_indices}
    energies = {idx: [] for idx in fidelity_indices}

    for atoms in atoms_list:
        fidelity_idx = int(atoms.info.get(fidelity_key, 0))
        if fidelity_idx not in compositions or energy_key not in atoms.info:
            continue
        composition = np.bincount(
            element_indices[atoms.get_atomic_numbers()], minlength=len(element)
        ).astype(np.float64, copy=False)
        compositions[fidelity_idx].append(composition)
        energies[fidelity_idx].append(float(atoms.info[energy_key]))

    results = {}
    for fidelity_idx in fidelity_indices:
        if not compositions[fidelity_idx]:
            logging.warning(
                "No training structures found for fidelity %s; using atomic energy 0.0",
                fidelity_idx,
            )
            results[fidelity_idx] = {z: 0.0 for z in element.zs}
            continue

        matrix = np.stack(compositions[fidelity_idx])
        target = np.asarray(energies[fidelity_idx], dtype=np.float64)
        try:
            solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=None)
        except np.linalg.LinAlgError:
            logging.warning(
                "Failed to fit isolated atomic energies for fidelity %s; using "
                "0.0 for every element",
                fidelity_idx,
            )
            results[fidelity_idx] = {z: 0.0 for z in element.zs}
            continue

        if rank < min(matrix.shape):
            logging.warning(
                "Atomic-energy composition matrix for fidelity %s is rank deficient "
                "(%s < %s); unconstrained values use the least-squares "
                "minimum-norm solution",
                fidelity_idx,
                rank,
                min(matrix.shape),
            )
        results[fidelity_idx] = {
            z: float(solution[idx]) for idx, z in enumerate(element.zs)
        }
    return results


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

    needs_energy = "energy" in target_property
    needs_forces = "forces" in target_property or "direct_forces" in target_property
    if needs_energy and (
        atomic_energies is None or len(atomic_energies) != num_fidelities
    ):
        raise ValueError("One atomic-energy mapping is required for every fidelity")

    stats_device = torch.device(device)
    num_elements = len(atomic_numbers)
    num_node_groups = num_fidelities * num_elements
    element_lookup = torch.full(
        (max(atomic_numbers) + 1,), -1, dtype=torch.int64, device=stats_device
    )
    element_lookup[
        torch.tensor(atomic_numbers, dtype=torch.int64, device=stats_device)
    ] = torch.arange(num_elements, device=stats_device)
    graph_counts = torch.zeros(num_fidelities, dtype=torch.int64, device=stats_device)
    atom_counts = torch.zeros_like(graph_counts)
    min_fidelity = torch.full(
        (), num_fidelities, dtype=torch.int64, device=stats_device
    )
    max_fidelity = torch.full((), -1, dtype=torch.int64, device=stats_device)
    neighbor_moments = _GroupedMoments(num_elements, 1, stats_device)
    energy_moments = (
        _GroupedMoments(num_fidelities, 3, stats_device) if needs_energy else None
    )
    forces_moments = (
        _GroupedMoments(num_node_groups, 4, stats_device) if needs_forces else None
    )
    noncollinear_magmoms_moments = _GroupedMoments(num_node_groups, 4, stats_device)
    max_noncollinear_magmoms_norm = torch.zeros(
        num_node_groups, dtype=torch.float64, device=stats_device
    )
    has_noncollinear_magmoms = False

    atomic_energy_table = None
    if needs_energy:
        atomic_energy_table = torch.tensor(
            [
                list(_canonical_atomic_energy(energy, atomic_numbers).values())
                for energy in atomic_energies
            ],
            dtype=torch.float64,
            device=stats_device,
        )

    with torch.no_grad():
        for data in dataloader_train:
            num_graphs = data.ptr.numel() - 1
            batch = data["batch"].to(stats_device, non_blocking=True)
            num_nodes = torch.bincount(batch, minlength=num_graphs)
            node_atomic_numbers = data["atomic_numbers"].to(
                stats_device, non_blocking=True
            )
            element_idx = element_lookup[node_atomic_numbers]

            fidelity_idx = data.get("fidelity_idx")
            if fidelity_idx is None:
                fidelity_idx = torch.zeros(
                    num_graphs, dtype=torch.int64, device=stats_device
                )
            else:
                fidelity_idx = fidelity_idx.reshape(-1).to(
                    device=stats_device,
                    dtype=torch.int64,
                    non_blocking=True,
                )
            if fidelity_idx.numel() != num_graphs:
                raise ValueError(
                    "fidelity_idx must contain exactly one value per structure"
                )
            min_fidelity = torch.minimum(min_fidelity, fidelity_idx.min())
            max_fidelity = torch.maximum(max_fidelity, fidelity_idx.max())
            fidelity_idx = fidelity_idx.clamp(0, num_fidelities - 1)

            node_fidelity = fidelity_idx[batch]
            node_groups = node_fidelity * num_elements + element_idx
            graph_counts += torch.bincount(fidelity_idx, minlength=num_fidelities)
            atom_counts += torch.bincount(node_fidelity, minlength=num_fidelities)

            neighbor_counts = torch.bincount(
                data.edge_index[0].to(stats_device, non_blocking=True),
                minlength=batch.numel(),
            )
            neighbor_moments.update(neighbor_counts, element_idx)

            initial_noncollinear_magmoms = data.get("initial_noncollinear_magmoms")
            if initial_noncollinear_magmoms is not None:
                has_noncollinear_magmoms = True
                initial_noncollinear_magmoms = initial_noncollinear_magmoms.to(
                    stats_device, non_blocking=True
                )
                noncollinear_magmoms_norm = torch.linalg.vector_norm(
                    initial_noncollinear_magmoms, dim=-1
                )
                noncollinear_magmoms_moments.update(
                    torch.cat(
                        (
                            initial_noncollinear_magmoms,
                            noncollinear_magmoms_norm.unsqueeze(-1),
                        ),
                        dim=-1,
                    ),
                    node_groups,
                )
                max_noncollinear_magmoms_norm.scatter_reduce_(
                    0,
                    node_groups,
                    noncollinear_magmoms_norm.to(torch.float64),
                    reduce="amax",
                    include_self=True,
                )

            if needs_energy:
                e0_node_energy = atomic_energy_table[node_fidelity, element_idx]
                e0 = torch.zeros(
                    num_graphs,
                    dtype=e0_node_energy.dtype,
                    device=stats_device,
                ).index_add_(0, batch, e0_node_energy)
                batch_energy = (
                    data["energy"].reshape(-1).to(stats_device, non_blocking=True)
                )
                energy_values = torch.stack(
                    (
                        batch_energy,
                        batch_energy / num_nodes,
                        (batch_energy - e0) / num_nodes,
                    ),
                    dim=-1,
                )
                energy_weight = data.get("energy_weight")
                valid_energy = (
                    torch.ones_like(fidelity_idx, dtype=torch.bool)
                    if energy_weight is None
                    else energy_weight.reshape(-1).to(stats_device, non_blocking=True)
                    > 0
                )
                energy_moments.update(
                    energy_values[valid_energy], fidelity_idx[valid_energy]
                )

            if needs_forces:
                forces = data.get("forces")
                if forces is None and "direct_forces" in target_property:
                    forces = data.get("direct_forces")
                if forces is None:
                    raise KeyError(
                        "Force statistics requested but no forces were found"
                    )
                forces = forces.to(stats_device, non_blocking=True)
                forces_weight = data.get("forces_weight")
                if forces_weight is None and "direct_forces" in target_property:
                    forces_weight = data.get("direct_forces_weight")
                valid_forces = (
                    torch.ones_like(node_fidelity, dtype=torch.bool)
                    if forces_weight is None
                    else forces_weight.reshape(-1).to(stats_device, non_blocking=True)[
                        batch
                    ]
                    > 0
                )
                forces_moments.update(
                    torch.cat(
                        (
                            forces,
                            torch.linalg.vector_norm(forces, dim=-1, keepdim=True),
                        ),
                        dim=-1,
                    )[valid_forces],
                    node_groups[valid_forces],
                )

    if not torch.any(neighbor_moments.count):
        raise ValueError("Cannot compute statistics from an empty training dataset")

    min_fidelity = int(min_fidelity.cpu())
    max_fidelity = int(max_fidelity.cpu())
    if min_fidelity < 0 or max_fidelity >= num_fidelities:
        raise ValueError(
            "fidelity_idx values must be in "
            f"[0, {num_fidelities - 1}], got "
            f"[{min_fidelity}, {max_fidelity}]"
        )

    graph_counts = graph_counts.cpu()
    atom_counts = atom_counts.cpu()
    _, neighbor_mean_by_element, _, _ = neighbor_moments.statistics()
    _, neighbor_mean, _, _ = neighbor_moments.statistics(group_size=num_elements)
    avg_num_neighbors = float(neighbor_mean[0, 0].cpu())
    avg_neighbors_by_element = _by_element(
        neighbor_mean_by_element[:, 0].cpu(), atomic_numbers
    )

    energy_summary = None
    if needs_energy:
        energy_summary = tuple(
            value.cpu() for value in energy_moments.statistics(unbiased=True)
        )

    forces_by_element_summary = None
    forces_summary = None
    if needs_forces:
        forces_by_element_summary = tuple(
            value.reshape(num_fidelities, num_elements, -1).cpu()
            for value in forces_moments.statistics()
        )
        forces_summary = tuple(
            value.cpu() for value in forces_moments.statistics(group_size=num_elements)
        )

    noncollinear_magmoms_by_element_summary = None
    noncollinear_magmoms_summary = None
    if has_noncollinear_magmoms:
        noncollinear_magmoms_by_element_summary = tuple(
            value.reshape(num_fidelities, num_elements, -1).cpu()
            for value in noncollinear_magmoms_moments.statistics()
        )
        noncollinear_magmoms_summary = tuple(
            value.cpu()
            for value in noncollinear_magmoms_moments.statistics(
                group_size=num_elements
            )
        )
        max_noncollinear_magmoms_norm = max_noncollinear_magmoms_norm.reshape(
            num_fidelities, num_elements
        ).cpu()

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
            (
                mag_count_by_element,
                mag_mean_by_element,
                mag_std_by_element,
                mag_rms_by_element,
            ) = (value[level] for value in noncollinear_magmoms_by_element_summary)
            _, mag_mean, mag_std, mag_rms = (
                value[level] for value in noncollinear_magmoms_summary
            )
            stats.update(
                {
                    "mean_initial_noncollinear_magmoms_xyz": mag_mean[:3].tolist(),
                    "std_initial_noncollinear_magmoms_xyz": mag_std[:3].tolist(),
                    "rms_initial_noncollinear_magmoms_xyz": mag_rms[:3].tolist(),
                    "mean_initial_noncollinear_magmoms_norm": float(mag_mean[3]),
                    "std_initial_noncollinear_magmoms_norm": float(mag_std[3]),
                    "rms_initial_noncollinear_magmoms_norm": float(mag_rms[3]),
                    "max_initial_noncollinear_magmoms_norm": float(
                        max_noncollinear_magmoms_norm[level].max()
                    ),
                    "mean_initial_noncollinear_magmoms_xyz_by_element": _by_element(
                        mag_mean_by_element[:, :3], atomic_numbers
                    ),
                    "std_initial_noncollinear_magmoms_xyz_by_element": _by_element(
                        mag_std_by_element[:, :3], atomic_numbers
                    ),
                    "rms_initial_noncollinear_magmoms_xyz_by_element": _by_element(
                        mag_rms_by_element[:, :3], atomic_numbers
                    ),
                    "mean_initial_noncollinear_magmoms_norm_by_element": _by_element(
                        mag_mean_by_element[:, 3], atomic_numbers
                    ),
                    "std_initial_noncollinear_magmoms_norm_by_element": _by_element(
                        mag_std_by_element[:, 3], atomic_numbers
                    ),
                    "rms_initial_noncollinear_magmoms_norm_by_element": _by_element(
                        mag_rms_by_element[:, 3], atomic_numbers
                    ),
                    "max_initial_noncollinear_magmoms_norm_by_element": _by_element(
                        max_noncollinear_magmoms_norm[level], atomic_numbers
                    ),
                    "num_initial_noncollinear_magmoms_by_element": _by_element(
                        mag_count_by_element[:, 0], atomic_numbers
                    ),
                }
            )

        if needs_energy:
            _, energy_mean, energy_std, _ = (value[level] for value in energy_summary)
            mean_energy_per_atom = float(energy_mean[1])
            mean_delta_energy_per_atom = float(energy_mean[2])
            stats.update(
                {
                    "mean_energy": float(energy_mean[0]),
                    "std_energy": float(energy_std[0]),
                    "atomic_energy": _canonical_atomic_energy(
                        atomic_energies[level], atomic_numbers
                    ),
                    "mean_energy_per_atom": {
                        z: mean_energy_per_atom for z in atomic_numbers
                    },
                    "mean_delta_energy_per_atom": {
                        z: mean_delta_energy_per_atom for z in atomic_numbers
                    },
                }
            )

        if needs_forces:
            (
                forces_count_by_element,
                forces_mean_by_element,
                forces_std_by_element,
                forces_rms_by_element,
            ) = (value[level] for value in forces_by_element_summary)
            _, forces_mean, forces_std, forces_rms = (
                value[level] for value in forces_summary
            )
            rms_forces = torch.sqrt(forces_rms[:3].square().mean())
            safe_rms_forces = torch.where(
                torch.isfinite(rms_forces) & (rms_forces > 0),
                rms_forces,
                1.0,
            )
            mean_squared_forces_xyz = forces_rms_by_element[:, :3].square().mean(dim=-1)
            rms_forces_by_element = torch.sqrt(mean_squared_forces_xyz)
            safe_rms_forces_by_element = torch.where(
                torch.isfinite(rms_forces_by_element) & (rms_forces_by_element > 0),
                rms_forces_by_element,
                1.0,
            )
            rms_forces_norm = forces_rms[3]
            safe_rms_forces_norm = torch.where(
                torch.isfinite(rms_forces_norm) & (rms_forces_norm > 0),
                rms_forces_norm,
                1.0,
            )
            rms_forces_norm_by_element = forces_rms_by_element[:, 3]
            safe_rms_forces_norm_by_element = torch.where(
                torch.isfinite(rms_forces_norm_by_element)
                & (rms_forces_norm_by_element > 0),
                rms_forces_norm_by_element,
                1.0,
            )
            num_forces = forces_count_by_element[:, 0]
            alphas = (0, 0.25, 0.5, 0.75, 1.0)
            recommended_forces_element_weights = {
                f"alpha_{alpha:g}": balanced_element_weights(
                    num_forces, mean_squared_forces_xyz, alpha=alpha
                ).tolist()
                for alpha in alphas
            }
            logging.info(
                "Recommended forces element weights for fidelity %s "
                "(atomic-number order %s):\n%s",
                level,
                atomic_numbers,
                "\n".join(
                    f"- alpha={alpha:.2f}: ["
                    + ", ".join(
                        f"{weight:.4f}"
                        for weight in recommended_forces_element_weights[
                            f"alpha_{alpha:g}"
                        ]
                    )
                    + "]"
                    for alpha in alphas
                ),
            )
            stats.update(
                {
                    "mean_forces_xyz": forces_mean[:3].tolist(),
                    "std_forces_xyz": forces_std[:3].tolist(),
                    "rms_forces_xyz": forces_rms[:3].tolist(),
                    "mean_forces_norm": float(forces_mean[3]),
                    "std_forces_norm": float(forces_std[3]),
                    "mean_forces_xyz_by_element": _by_element(
                        forces_mean_by_element[:, :3], atomic_numbers
                    ),
                    "std_forces_xyz_by_element": _by_element(
                        forces_std_by_element[:, :3], atomic_numbers
                    ),
                    "rms_forces_xyz_by_element": _by_element(
                        forces_rms_by_element[:, :3], atomic_numbers
                    ),
                    "mean_forces_norm_by_element": _by_element(
                        forces_mean_by_element[:, 3], atomic_numbers
                    ),
                    "std_forces_norm_by_element": _by_element(
                        forces_std_by_element[:, 3], atomic_numbers
                    ),
                    "rms_forces": {z: float(safe_rms_forces) for z in atomic_numbers},
                    "rms_forces_by_element": _by_element(
                        safe_rms_forces_by_element, atomic_numbers
                    ),
                    "rms_forces_norm": {
                        z: float(safe_rms_forces_norm) for z in atomic_numbers
                    },
                    "rms_forces_norm_by_element": _by_element(
                        safe_rms_forces_norm_by_element, atomic_numbers
                    ),
                    "num_forces_by_element": _by_element(num_forces, atomic_numbers),
                    "mean_squared_forces_xyz_by_element": _by_element(
                        mean_squared_forces_xyz, atomic_numbers
                    ),
                    "recommended_forces_element_weights": (
                        recommended_forces_element_weights
                    ),
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

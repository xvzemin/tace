################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import multiprocessing
import os
import queue
import time
import traceback
from typing import Dict, List, Optional, Sequence

import ase
import numpy as np
import torch
import torch.distributed as dist
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from ..utils.utils import log_statistics_to_yaml
from .cache import GRAPH_SCHEMA
from .element import TorchElement
from .quantity import KeySpecification, get_need_property


def missing_model_statistics(
    statistics: Sequence[Dict],
    model_config: Dict,
    target_property: Sequence[str],
    embedding_property: Sequence[str],
) -> set[str]:
    """Return model-requested statistics missing from any fidelity."""

    required = {"atomic_energy"} if "energy" in target_property else set()
    scale_shift = model_config.get("scale_shift", {})
    if scale_shift.get("enable") and "energy" in target_property:
        required.update(
            value
            for key in ("scale_type", "shift_type")
            if (value := scale_shift.get(key)) is not None
        )

    needed_property = get_need_property(
        list(target_property),
        list(embedding_property),
        training=True,
    )
    if "initial_noncollinear_magmoms" in needed_property:
        required.add("max_noncollinear_magmoms_norm_by_element")

    return {
        name
        for name in required
        if any(name not in fidelity_statistics for fidelity_statistics in statistics)
    }


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
        denominator = count.clamp_min(1).to(torch.float64)
        self.mean += delta * (batch_count / denominator).unsqueeze(-1)
        self.m2 += batch_m2 + delta.square() * (
            self.count.to(torch.float64) * batch_count / denominator
        ).unsqueeze(-1)
        self.count = count

    def state(self):
        """Return owned CPU arrays, without keeping CUDA storage alive."""
        return {
            name: getattr(self, name).cpu().numpy().copy()
            for name in ("count", "mean", "m2")
        }

    def merge(self, state):
        """Combine independent grouped moments using the parallel Welford rule."""
        if isinstance(state, _GroupedMoments):
            state = state.state()
        other_count = torch.as_tensor(state["count"], device=self.count.device)
        other_mean = torch.as_tensor(state["mean"], device=self.mean.device)
        other_m2 = torch.as_tensor(state["m2"], device=self.m2.device)
        if (
            other_count.shape != self.count.shape
            or other_mean.shape != self.mean.shape
            or other_m2.shape != self.m2.shape
        ):
            raise ValueError("Cannot merge moments with different group shapes")
        count = self.count + other_count
        denominator = count.clamp_min(1).to(torch.float64)
        delta = other_mean - self.mean
        self.mean += delta * (other_count / denominator).unsqueeze(-1)
        self.m2 += other_m2 + delta.square() * (
            self.count.to(torch.float64) * other_count / denominator
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


class AtomicEnergyFit:
    """Bounded, FP64 least-squares fit using incremental QR factorization."""

    def __init__(self, atomic_numbers, fidelity_indices):
        self.atomic_numbers = list(atomic_numbers)
        self.states = {int(index): [None, None, 0] for index in fidelity_indices}

    def update(self, index, counts, energies):
        counts = np.asarray(counts, dtype=np.float64).reshape(
            -1, len(self.atomic_numbers)
        )
        energies = np.asarray(energies, dtype=np.float64).reshape(-1)
        if len(counts) != len(energies):
            raise ValueError("Composition and energy counts differ.")
        if not len(counts):
            return
        previous_r, previous_target, count = self.states[index]
        if previous_r is not None:
            counts = np.concatenate((previous_r, counts), axis=0)
            energies = np.concatenate((previous_target, energies))
        q, r = np.linalg.qr(counts, mode="reduced")
        self.states[index] = [
            r,
            q.T @ energies,
            count
            + len(energies)
            - (0 if previous_target is None else len(previous_target)),
        ]

    def result(self):
        results = {}
        for index, (matrix, target, count) in self.states.items():
            if matrix is None:
                logging.warning(
                    "No energy labels for fidelity %s; using atomic energy 0.0", index
                )
                results[index] = dict.fromkeys(self.atomic_numbers, 0.0)
                continue
            rcond = np.finfo(np.float64).eps * max(count, len(self.atomic_numbers))
            solution, _, rank, _ = np.linalg.lstsq(matrix, target, rcond=rcond)
            if rank < min(count, len(self.atomic_numbers)):
                logging.warning(
                    "Atomic-energy fit for fidelity %s is rank deficient; using the minimum-norm solution",
                    index,
                )
            results[index] = dict(zip(self.atomic_numbers, map(float, solution)))
        return results


def compute_atomic_energies(
    atoms_list: Sequence[ase.Atoms],
    element: TorchElement,
    keyspec: KeySpecification,
    fidelity_indices: Sequence[int],
) -> Dict[int, Dict[int, float]]:
    """Fit offsets using the same label resolution and masks as graph training."""
    from .records import record_from_atoms, resolve_property, validate_fidelity

    fit = AtomicEnergyFit(element.zs, fidelity_indices)
    element_indices = {z: index for index, z in enumerate(element.zs)}
    for start in range(0, len(atoms_list), 1024):
        indices = range(start, min(start + 1024, len(atoms_list)))
        records = (
            atoms_list.get_records(indices)
            if hasattr(atoms_list, "get_records")
            else [record_from_atoms(atoms_list[index]) for index in indices]
        )
        counts = {index: [] for index in fit.states}
        energies = {index: [] for index in fit.states}
        for record in records:
            value = resolve_property(record, "fidelity_idx", keyspec)
            index = validate_fidelity(0 if value is None else value)
            energy = resolve_property(record, "energy", keyspec)
            weight = record.info.get(
                "energy_weight", record.data.get("energy_weight", 1.0)
            )
            if index not in fit.states or energy is None or float(weight) <= 0:
                continue
            counts[index].append(
                np.bincount(
                    [element_indices[int(z)] for z in record.numbers],
                    minlength=len(element),
                )
            )
            energies[index].append(float(energy))
        for index in fit.states:
            fit.update(index, counts[index], energies[index])
    return fit.result()


def fit_atomic_energies(compositions, energies, atomic_numbers):
    """Fit elemental offsets without forming normal equations."""
    fit = AtomicEnergyFit(atomic_numbers, compositions)
    for index in compositions:
        for start in range(0, len(compositions[index]), 1024):
            fit.update(
                index,
                compositions[index][start : start + 1024],
                energies[index][start : start + 1024],
            )
    return fit.result()


def compute_atomic_energies_from_graphs(
    dataloader,
    element: TorchElement,
    fidelity_indices: Sequence[int],
) -> Dict[int, Dict[int, float]]:
    """Fit elemental offsets in one bounded-memory pass over graph labels."""
    fit = AtomicEnergyFit(element.zs, fidelity_indices)
    for data in dataloader:
        num_graphs = data.ptr.numel() - 1
        element_idx = element.z2idx(data["atomic_numbers"])
        counts = torch.bincount(
            data.batch * len(element) + element_idx, minlength=num_graphs * len(element)
        ).reshape(num_graphs, len(element))
        fidelity = data.get("fidelity_idx")
        if fidelity is None:
            fidelity = torch.zeros(num_graphs, dtype=torch.int64)
        fidelity = fidelity.reshape(-1)
        energy_weight = data.get("energy_weight")
        for index in fit.states:
            selected = fidelity == index
            if energy_weight is not None:
                selected &= energy_weight.reshape(-1) > 0
            fit.update(
                index,
                counts[selected].cpu().numpy(),
                data["energy"].reshape(-1)[selected].double().cpu().numpy(),
            )
    return fit.result()


class _StatisticsAccumulator:
    """Mergeable FP64 moments for complete graph-dataset statistics."""

    def __init__(
        self,
        atomic_numbers,
        atomic_energies,
        target_property,
        device="cpu",
        num_fidelities=1,
    ):
        self.atomic_numbers = list(atomic_numbers)
        self.atomic_energies = atomic_energies
        self.target_property = list(target_property)
        self.num_fidelities = num_fidelities
        if self.num_fidelities < 1:
            raise ValueError("At least one fidelity must be configured")

        self.needs_energy = "energy" in self.target_property
        self.needs_forces = (
            "forces" in self.target_property or "direct_forces" in self.target_property
        )
        if self.needs_energy and (
            self.atomic_energies is None
            or len(self.atomic_energies) != self.num_fidelities
        ):
            raise ValueError("One atomic-energy mapping is required for every fidelity")

        self.device = torch.device(device)
        if self.device.type == "mps":
            logging.info(
                "Computing float64 dataset statistics on CPU; MPS has no float64 support"
            )
            self.device = torch.device("cpu")
        self.num_elements = len(self.atomic_numbers)
        self.num_node_groups = self.num_fidelities * self.num_elements
        self.element_lookup = torch.full(
            (max(self.atomic_numbers) + 1,), -1, dtype=torch.int64, device=self.device
        )
        self.element_lookup[
            torch.tensor(self.atomic_numbers, dtype=torch.int64, device=self.device)
        ] = torch.arange(self.num_elements, device=self.device)
        self.graph_counts = torch.zeros(
            self.num_fidelities, dtype=torch.int64, device=self.device
        )
        self.atom_counts = torch.zeros_like(self.graph_counts)
        self.min_fidelity = torch.full(
            (), self.num_fidelities, dtype=torch.int64, device=self.device
        )
        self.max_fidelity = torch.full((), -1, dtype=torch.int64, device=self.device)
        self.neighbor_moments = _GroupedMoments(self.num_elements, 1, self.device)
        self.energy_moments = (
            _GroupedMoments(self.num_fidelities, 3, self.device)
            if self.needs_energy
            else None
        )
        self.forces_moments = (
            _GroupedMoments(self.num_node_groups, 4, self.device)
            if self.needs_forces
            else None
        )
        self.noncollinear_magmoms_moments = _GroupedMoments(
            self.num_node_groups, 4, self.device
        )
        self.max_noncollinear_magmoms_norm = torch.zeros(
            self.num_node_groups, dtype=torch.float64, device=self.device
        )
        self.has_noncollinear_magmoms = False

        self.atomic_energy_table = None
        if self.needs_energy:
            self.atomic_energy_table = torch.tensor(
                [
                    list(_canonical_atomic_energy(energy, self.atomic_numbers).values())
                    for energy in self.atomic_energies
                ],
                dtype=torch.float64,
                device=self.device,
            )

    @torch.no_grad()
    def update(self, data):
        num_graphs = data.ptr.numel() - 1
        batch = data["batch"].to(self.device, non_blocking=True)
        num_nodes = data.ptr.diff().to(self.device, non_blocking=True)
        node_atomic_numbers = data["atomic_numbers"].to(self.device, non_blocking=True)
        element_idx = self.element_lookup[node_atomic_numbers]

        fidelity_idx = data.get("fidelity_idx")
        if fidelity_idx is None:
            fidelity_idx = torch.zeros(
                num_graphs, dtype=torch.int64, device=self.device
            )
        else:
            fidelity_idx = fidelity_idx.reshape(-1).to(
                device=self.device,
                dtype=torch.int64,
                non_blocking=True,
            )
        if fidelity_idx.numel() != num_graphs:
            raise ValueError(
                "fidelity_idx must contain exactly one value per structure"
            )
        self.min_fidelity = torch.minimum(self.min_fidelity, fidelity_idx.min())
        self.max_fidelity = torch.maximum(self.max_fidelity, fidelity_idx.max())
        fidelity_idx = fidelity_idx.clamp(0, self.num_fidelities - 1)

        node_fidelity = fidelity_idx[batch]
        node_groups = node_fidelity * self.num_elements + element_idx
        self.graph_counts += torch.bincount(fidelity_idx, minlength=self.num_fidelities)
        self.atom_counts += torch.bincount(node_fidelity, minlength=self.num_fidelities)

        neighbor_counts = data.get("num_neighbors")
        if neighbor_counts is None:
            neighbor_counts = torch.bincount(
                data.edge_index[0].to(self.device, non_blocking=True),
                minlength=batch.numel(),
            )
        else:
            neighbor_counts = neighbor_counts.to(self.device, non_blocking=True)
        self.neighbor_moments.update(neighbor_counts, element_idx)

        initial_noncollinear_magmoms = data.get("initial_noncollinear_magmoms")
        if initial_noncollinear_magmoms is not None:
            self.has_noncollinear_magmoms = True
            initial_noncollinear_magmoms = initial_noncollinear_magmoms.to(
                device=self.device, dtype=torch.float64, non_blocking=True
            )
            noncollinear_magmoms_norm = torch.linalg.vector_norm(
                initial_noncollinear_magmoms, dim=-1
            )
            self.noncollinear_magmoms_moments.update(
                torch.cat(
                    (
                        initial_noncollinear_magmoms,
                        noncollinear_magmoms_norm.unsqueeze(-1),
                    ),
                    dim=-1,
                ),
                node_groups,
            )
            self.max_noncollinear_magmoms_norm.scatter_reduce_(
                0,
                node_groups,
                noncollinear_magmoms_norm.to(torch.float64),
                reduce="amax",
                include_self=True,
            )

        if self.needs_energy:
            e0_node_energy = self.atomic_energy_table[node_fidelity, element_idx]
            e0 = torch.zeros(
                num_graphs,
                dtype=e0_node_energy.dtype,
                device=self.device,
            ).index_add_(0, batch, e0_node_energy)
            batch_energy = (
                data["energy"]
                .reshape(-1)
                .to(device=self.device, dtype=torch.float64, non_blocking=True)
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
                else energy_weight.reshape(-1).to(self.device, non_blocking=True) > 0
            )
            self.energy_moments.update(
                energy_values[valid_energy], fidelity_idx[valid_energy]
            )

        if self.needs_forces:
            forces = data.get("forces")
            if forces is None and "direct_forces" in self.target_property:
                forces = data.get("direct_forces")
            if forces is None:
                raise KeyError("Force statistics requested but no forces were found")
            forces = forces.to(
                device=self.device, dtype=torch.float64, non_blocking=True
            )
            forces_weight = data.get("forces_weight")
            if forces_weight is None and "direct_forces" in self.target_property:
                forces_weight = data.get("direct_forces_weight")
            valid_forces = (
                torch.ones_like(node_fidelity, dtype=torch.bool)
                if forces_weight is None
                else forces_weight.reshape(-1).to(self.device, non_blocking=True)[batch]
                > 0
            )
            self.forces_moments.update(
                torch.cat(
                    (
                        forces,
                        torch.linalg.vector_norm(forces, dim=-1, keepdim=True),
                    ),
                    dim=-1,
                )[valid_forces],
                node_groups[valid_forces],
            )

    def state(self):
        """Return small, owned NumPy arrays suitable for spawned processes."""
        result = {
            "min_fidelity": int(self.min_fidelity.cpu()),
            "max_fidelity": int(self.max_fidelity.cpu()),
            "has_noncollinear_magmoms": self.has_noncollinear_magmoms,
        }
        for name in ("graph_counts", "atom_counts", "max_noncollinear_magmoms_norm"):
            result[name] = getattr(self, name).cpu().numpy().copy()
        for name in (
            "neighbor_moments",
            "energy_moments",
            "forces_moments",
            "noncollinear_magmoms_moments",
        ):
            moments = getattr(self, name)
            result[name] = None if moments is None else moments.state()
        return result

    @torch.no_grad()
    def merge(self, state):
        """Merge unnormalized moments without losing within-group variance."""
        for name in ("graph_counts", "atom_counts"):
            value = getattr(self, name)
            value.add_(torch.as_tensor(state[name], device=self.device))
        self.min_fidelity = torch.minimum(
            self.min_fidelity,
            torch.as_tensor(state["min_fidelity"], device=self.device),
        )
        self.max_fidelity = torch.maximum(
            self.max_fidelity,
            torch.as_tensor(state["max_fidelity"], device=self.device),
        )
        self.has_noncollinear_magmoms |= state["has_noncollinear_magmoms"]
        self.max_noncollinear_magmoms_norm = torch.maximum(
            self.max_noncollinear_magmoms_norm,
            torch.as_tensor(state["max_noncollinear_magmoms_norm"], device=self.device),
        )
        for name in (
            "neighbor_moments",
            "energy_moments",
            "forces_moments",
            "noncollinear_magmoms_moments",
        ):
            moments = getattr(self, name)
            if moments is not None:
                moments.merge(state[name])

    def result(self, data_fingerprint=None):
        """Finalize the existing per-fidelity statistics without writing files."""
        atomic_numbers = self.atomic_numbers
        atomic_energies = self.atomic_energies
        num_fidelities = self.num_fidelities
        num_elements = self.num_elements
        needs_energy = self.needs_energy
        needs_forces = self.needs_forces
        graph_counts = self.graph_counts
        atom_counts = self.atom_counts
        min_fidelity = self.min_fidelity
        max_fidelity = self.max_fidelity
        neighbor_moments = self.neighbor_moments
        energy_moments = self.energy_moments
        forces_moments = self.forces_moments
        noncollinear_magmoms_moments = self.noncollinear_magmoms_moments
        max_noncollinear_magmoms_norm = self.max_noncollinear_magmoms_norm
        has_noncollinear_magmoms = self.has_noncollinear_magmoms

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
                value.cpu()
                for value in forces_moments.statistics(group_size=num_elements)
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
                "data_schema": GRAPH_SCHEMA,
                "data_dtype": "float64",
                "data_fingerprint": data_fingerprint,
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
                rms_noncollinear_magmoms = torch.sqrt(mag_rms[:3].square().mean())
                safe_rms_noncollinear_magmoms = torch.where(
                    torch.isfinite(rms_noncollinear_magmoms)
                    & (rms_noncollinear_magmoms > 0),
                    rms_noncollinear_magmoms,
                    1.0,
                )
                rms_noncollinear_magmoms_by_element = torch.sqrt(
                    mag_rms_by_element[:, :3].square().mean(dim=-1)
                )
                safe_rms_noncollinear_magmoms_by_element = torch.where(
                    torch.isfinite(rms_noncollinear_magmoms_by_element)
                    & (rms_noncollinear_magmoms_by_element > 0),
                    rms_noncollinear_magmoms_by_element,
                    1.0,
                )
                rms_noncollinear_magmoms_norm = mag_rms[3]
                safe_rms_noncollinear_magmoms_norm = torch.where(
                    torch.isfinite(rms_noncollinear_magmoms_norm)
                    & (rms_noncollinear_magmoms_norm > 0),
                    rms_noncollinear_magmoms_norm,
                    1.0,
                )
                rms_noncollinear_magmoms_norm_by_element = mag_rms_by_element[:, 3]
                safe_rms_noncollinear_magmoms_norm_by_element = torch.where(
                    torch.isfinite(rms_noncollinear_magmoms_norm_by_element)
                    & (rms_noncollinear_magmoms_norm_by_element > 0),
                    rms_noncollinear_magmoms_norm_by_element,
                    1.0,
                )
                stats.update(
                    {
                        "mean_noncollinear_magmoms_xyz": mag_mean[:3].tolist(),
                        "std_noncollinear_magmoms_xyz": mag_std[:3].tolist(),
                        "rms_noncollinear_magmoms_xyz": mag_rms[:3].tolist(),
                        "mean_noncollinear_magmoms_norm": float(mag_mean[3]),
                        "std_noncollinear_magmoms_norm": float(mag_std[3]),
                        "max_noncollinear_magmoms_norm": float(
                            max_noncollinear_magmoms_norm[level].max()
                        ),
                        "mean_noncollinear_magmoms_xyz_by_element": _by_element(
                            mag_mean_by_element[:, :3], atomic_numbers
                        ),
                        "std_noncollinear_magmoms_xyz_by_element": _by_element(
                            mag_std_by_element[:, :3], atomic_numbers
                        ),
                        "rms_noncollinear_magmoms_xyz_by_element": _by_element(
                            mag_rms_by_element[:, :3], atomic_numbers
                        ),
                        "mean_noncollinear_magmoms_norm_by_element": _by_element(
                            mag_mean_by_element[:, 3], atomic_numbers
                        ),
                        "std_noncollinear_magmoms_norm_by_element": _by_element(
                            mag_std_by_element[:, 3], atomic_numbers
                        ),
                        "rms_noncollinear_magmoms": {
                            z: float(safe_rms_noncollinear_magmoms)
                            for z in atomic_numbers
                        },
                        "rms_noncollinear_magmoms_by_element": _by_element(
                            safe_rms_noncollinear_magmoms_by_element,
                            atomic_numbers,
                        ),
                        "rms_noncollinear_magmoms_norm": {
                            z: float(safe_rms_noncollinear_magmoms_norm)
                            for z in atomic_numbers
                        },
                        "rms_noncollinear_magmoms_norm_by_element": _by_element(
                            safe_rms_noncollinear_magmoms_norm_by_element,
                            atomic_numbers,
                        ),
                        "max_noncollinear_magmoms_norm_by_element": _by_element(
                            max_noncollinear_magmoms_norm[level], atomic_numbers
                        ),
                        "num_noncollinear_magmoms_by_element": _by_element(
                            mag_count_by_element[:, 0], atomic_numbers
                        ),
                    }
                )

            if needs_energy:
                _, energy_mean, energy_std, _ = (
                    value[level] for value in energy_summary
                )
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
                mean_squared_forces_xyz = (
                    forces_rms_by_element[:, :3].square().mean(dim=-1)
                )
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
                        "rms_forces": {
                            z: float(safe_rms_forces) for z in atomic_numbers
                        },
                        "rms_forces_by_element": _by_element(
                            safe_rms_forces_by_element, atomic_numbers
                        ),
                        "rms_forces_norm": {
                            z: float(safe_rms_forces_norm) for z in atomic_numbers
                        },
                        "rms_forces_norm_by_element": _by_element(
                            safe_rms_forces_norm_by_element, atomic_numbers
                        ),
                        "num_forces_by_element": _by_element(
                            num_forces, atomic_numbers
                        ),
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

        return per_level_stats


def _accumulate_statistics(dataloader, accumulator, progress=None):
    """Show local progress or send bounded-rate counts to the parent reader."""
    dataset = getattr(dataloader, "dataset", None)
    total = len(dataset) if dataset is not None else None
    processed = 0
    last_report = time.monotonic()
    reporter = None
    if progress is None:
        logging.info(
            "Computing dataset statistics on %s: %s graphs, batch_size=%s",
            accumulator.device,
            total if total is not None else "unknown number of",
            getattr(dataloader, "batch_size", "custom"),
        )
        reporter = tqdm(
            total=total,
            desc="statistics",
            unit="graph",
            disable=dist.is_available()
            and dist.is_initialized()
            and dist.get_rank() != 0,
        )
    try:
        for data in dataloader:
            accumulator.update(data)
            count = data.ptr.numel() - 1
            processed += count
            if reporter is not None:
                reporter.update(count)
            else:
                now = time.monotonic()
                if now - last_report >= 1:
                    progress(processed)
                    last_report = now
        return processed
    finally:
        if reporter is not None:
            reporter.close()
        else:
            progress(processed)


def _statistics_worker(task, progress=None):
    """Read assigned immutable shards and return only compact CPU moments."""
    from .dataloader import StatisticsDataLoader
    from .store import GraphDatasetLMDB

    torch.set_num_threads(task.get("threads", 1))
    device = torch.device(task["device"])
    if device.type == "cuda":
        torch.cuda.set_device(device)
    dataset = GraphDatasetLMDB(
        task["paths"],
        shard_lengths=task["lengths"],
        fields=task["fields"],
        cache_size=0,
        max_open_shards=task["max_open_shards"],
        readahead=task["readahead"],
    )
    dataset.metadata = task["metadata"]
    try:
        loader = StatisticsDataLoader(dataset, **task["loader"])
        accumulator = _StatisticsAccumulator(
            task["atomic_numbers"],
            task["atomic_energies"],
            task["target_property"],
            device=device,
            num_fidelities=task["num_fidelities"],
        )
        processed = _accumulate_statistics(loader, accumulator, progress)
        if processed != len(dataset):
            raise RuntimeError(
                f"Statistics worker processed {processed} of {len(dataset)} graphs"
            )
        return accumulator.state()
    finally:
        dataset.close()


def _statistics_process(index, task, messages):
    """Isolate CUDA and LMDB handles inside one spawned statistics reader."""
    try:
        state = _statistics_worker(
            task,
            progress=lambda count: messages.put(("progress", index, count)),
        )
        messages.put(("result", index, state))
    except BaseException:
        messages.put(("error", index, traceback.format_exc()))


def _statistics_tasks(
    dataloader,
    atomic_numbers,
    atomic_energies,
    target_property,
    num_fidelities,
    devices,
):
    """Balance whole shards by atom count, retaining sequential reads per shard."""
    from torch.utils.data import SequentialSampler

    from .store import GraphDatasetLMDB

    dataset = getattr(dataloader, "dataset", None)
    if not isinstance(dataset, GraphDatasetLMDB) or dataset.in_memory:
        raise ValueError("Multi-GPU statistics require an on-disk GraphDatasetLMDB")
    if (
        dataloader.batch_size is None
        or dataloader.drop_last
        or not isinstance(dataloader.sampler, SequentialSampler)
    ):
        raise ValueError("Multi-GPU statistics require complete sequential batches")
    if not dataset.lmdb_paths:
        raise ValueError("Cannot compute statistics from an empty training dataset")
    if any(length <= 0 for length in dataset.lengths):
        raise ValueError("Statistics shards must contain at least one graph")
    devices = devices[: min(len(devices), len(dataset.lmdb_paths))]
    partitions = [[] for _ in devices]
    loads = [0] * len(devices)
    metadata = dataset.metadata or {}
    sizes = metadata.get("shard_stats", {})
    weights = [
        int(sizes.get(path.name, {}).get("num_nodes", length))
        for path, length in zip(dataset.lmdb_paths, dataset.lengths)
    ]
    for shard in sorted(
        range(len(weights)), key=lambda index: (-weights[index], index)
    ):
        rank = min(range(len(loads)), key=loads.__getitem__)
        partitions[rank].append(shard)
        loads[rank] += weights[shard]

    collate = getattr(dataloader, "collate_fn", None)
    loader_options = {
        "batch_size": dataloader.batch_size,
        "shuffle": False,
        "drop_last": False,
        # The statistics loader resolves its own worker count. Preserve an
        # explicit choice; large prefetched batches also consume /dev/shm.
        "num_workers": dataloader.num_workers,
        "pin_memory": dataloader.pin_memory,
        "follow_batch": getattr(collate, "follow_batch", None),
        "exclude_keys": getattr(collate, "exclude_keys", None),
    }
    if dataloader.num_workers:
        loader_options["prefetch_factor"] = dataloader.prefetch_factor or 1
        # CUDA is initialized in this reader before its loader starts.
        loader_options["multiprocessing_context"] = "spawn"
    tasks = []
    for device, shards in zip(devices, partitions):
        shards.sort()
        tasks.append(
            {
                "device": str(device),
                "paths": [str(dataset.lmdb_paths[index]) for index in shards],
                "lengths": [dataset.lengths[index] for index in shards],
                "fields": dataset.fields,
                "metadata": {"fields": metadata.get("fields", [])},
                "max_open_shards": dataset.max_open_shards,
                "readahead": dataset.readahead,
                "loader": loader_options,
                "atomic_numbers": atomic_numbers,
                "atomic_energies": atomic_energies,
                "target_property": target_property,
                "num_fidelities": num_fidelities,
                "threads": max(1, min(4, (os.cpu_count() or 1) // len(devices))),
            }
        )
    return tasks


def _parallel_statistics(tasks, accumulator):
    """Collect worker moments in assignment order, with no graph IPC."""
    context = multiprocessing.get_context("spawn")
    messages = context.Queue()
    processes = []
    states = [None] * len(tasks)
    completed = set()
    counts = [0] * len(tasks)
    total = sum(sum(task["lengths"]) for task in tasks)
    logging.info(
        "Computing statistics for %d graphs on %s; batch_size=%d per device, "
        "independent shard readers",
        total,
        ", ".join(task["device"] for task in tasks),
        tasks[0]["loader"]["batch_size"],
    )
    progress = tqdm(
        total=total,
        desc="statistics",
        unit="graph",
        disable=dist.is_available() and dist.is_initialized() and dist.get_rank() != 0,
    )
    try:
        for index, task in enumerate(tasks):
            process = context.Process(
                target=_statistics_process,
                args=(index, task, messages),
                name=f"tace-statistics-{index}",
            )
            process.start()
            processes.append(process)
        while len(completed) < len(tasks):
            try:
                kind, index, payload = messages.get(timeout=1)
            except queue.Empty:
                for index, process in enumerate(processes):
                    if process.exitcode not in (None, 0):
                        raise RuntimeError(
                            f"Statistics worker on {tasks[index]['device']} exited "
                            f"with code {process.exitcode}"
                        )
                if all(process.exitcode is not None for process in processes):
                    raise RuntimeError(
                        "Statistics workers exited before returning all results"
                    )
            else:
                if kind == "error":
                    raise RuntimeError(
                        f"Statistics worker on {tasks[index]['device']} failed:\n{payload}"
                    )
                previous = counts[index]
                if kind == "progress":
                    counts[index] = payload
                elif kind == "result":
                    states[index] = payload
                    counts[index] = sum(tasks[index]["lengths"])
                    completed.add(index)
                progress.update(counts[index] - previous)
        for state in states:
            accumulator.merge(state)
    finally:
        progress.close()
        for process in processes:
            if len(completed) < len(tasks) and process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
        messages.close()
        messages.join_thread()


def _compute_statistics(
    dataloader_train: DataLoader,
    atomic_numbers: List[int],
    atomic_energies: Optional[List[Dict[int, float]]],
    target_property: List[str],
    device: str = "cpu",
    num_fidelities: int = 1,
    data_fingerprint: Optional[str] = None,
    devices: Optional[Sequence] = None,
) -> List[Dict]:
    """Compute full-dataset FP64 moments, optionally using independent GPUs.

    ``devices=None`` retains the existing single-device behavior. Multiple
    devices require a sequential loader backed by immutable LMDB graph shards.
    Batch size is per device; only the parent writes statistics YAML files.
    """
    if devices is None:
        selected = [torch.device(device)]
    else:
        if isinstance(devices, (str, torch.device)):
            devices = [devices]
        selected = [
            torch.device("cuda", value)
            if isinstance(value, int)
            else torch.device(value)
            for value in devices
        ]
        if not selected:
            raise ValueError("At least one statistics device must be selected")
    if len(selected) > 1:
        if any(item.type != "cuda" or item.index is None for item in selected):
            raise ValueError(
                "Multiple statistics devices must be explicit CUDA devices"
            )
        if len(set(selected)) != len(selected):
            raise ValueError("Statistics devices must be distinct")
        count = torch.cuda.device_count()
        if any(not 0 <= item.index < count for item in selected):
            raise ValueError("Statistics device is outside the visible CUDA devices")
        accumulator = _StatisticsAccumulator(
            atomic_numbers, atomic_energies, target_property, "cpu", num_fidelities
        )
        tasks = _statistics_tasks(
            dataloader_train,
            atomic_numbers,
            atomic_energies,
            target_property,
            num_fidelities,
            selected,
        )
        _parallel_statistics(tasks, accumulator)
    else:
        accumulator = _StatisticsAccumulator(
            atomic_numbers,
            atomic_energies,
            target_property,
            selected[0],
            num_fidelities,
        )
        _accumulate_statistics(dataloader_train, accumulator)
    statistics = accumulator.result(data_fingerprint)
    log_statistics_to_yaml(statistics)
    return statistics

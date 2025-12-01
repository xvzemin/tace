################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################


import logging
from typing import Dict, List

import numpy as np
import torch
from ase import Atoms
from torch_geometric.loader import DataLoader

from .element import Element
from .quantity import KeySpecification
from ..utils.utils import log_statistics_to_yaml
from ..utils.torch_scatter import scatter, scatter_add


class OneHotToAtomicEnergy(torch.nn.Module):
    def __init__(self, atomic_energies: List[Dict[int, float]]) -> None:
        super().__init__()
        assert atomic_energies is not None
        atomic_energy_list = []
        for atomic_energy in atomic_energies:
            atomic_energy_list.append(
                [float(v) for _, v in atomic_energy.items()]
            )
        self.register_buffer(
            "atomic_energy",
            torch.tensor(
                atomic_energy_list,
                dtype=torch.get_default_dtype(),
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return torch.matmul(x, self.atomic_energy.T)

    def __repr__(self):
        return f"{self.__class__.__name__}(atomic_eneries={[f'{x:.4f}' for x in self.atomic_energy.reshape(-1).tolist()]})"
    
class Statistics:
    """
    A container class for storing statistical information of physical quantities.

    This class dynamically accepts arbitrary keyword arguments and stores them as attributes.
    It is typically used to store mean, standard deviation, rms values, or other
    statistical metadata for normalization or scale-shift operations in the TACE framework.

    Attributes can be accessed directly using dot notation:

    Parameters
    ----------
    **kwargs : dict
        Arbitrary keyword arguments that will be stored as attributes of the instance.

    Examples
    --------
    >>> stats = Statistics(mean_energy=-5.2, std_force=0.03)
    >>> stats.mean_energy
    -5.2
    >>> 'std_force' in stats
    True
    """

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __contains__(self, key):
        return hasattr(self, key)

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Statistics({attrs})"

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

    def __delitem__(self, key):
        delattr(self, key)

    def __iter__(self):
        return iter(self.__dict__)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __repr__(self):
        attrs = {k: v for k, v in self.__dict__.items() if not k.startswith("_")}
        return f"Statistics({attrs})"


def _compute_statistics(
    dataloader_train: DataLoader,
    # dataloader_valid: DataLoader,
    atomic_numbers: List[str],
    atomic_energies: List[Dict[int, float]],
    target_property: List[str],
    device: str = "cpu",
    num_levels: int = 1,
) -> List[Statistics]:
    return [
            Statistics(**stat)
            for stat in analyze_dataset_statistics(
                dataloader_train,
                # dataloader_valid,
                atomic_numbers,
                atomic_energies,
                target_property,
                device,
                num_levels,
            )
        ]

def compute_atomic_energy_per_level(
    points: Atoms, element: Element, keyspec: KeySpecification,
) -> Dict[int, float]:
    len_train = len(points)
    A = np.zeros((len_train, len(element)))
    B = np.zeros(len_train)
    energyList = [
        points[i].info.get(keyspec.info_keys["energy"], None) for i in range(len_train)
    ]
    atomicNumbersList = [points[i].get_atomic_numbers() for i in range(len_train)]

    for i in range(len_train):
        B[i] = energyList[i]
        for j, z in enumerate(element.atomic_numbers):
            A[i, j] = np.count_nonzero(atomicNumbersList[i] == z)
    try:
        x = np.linalg.lstsq(A, B, rcond=None)[0]
        atomic_energy = {}
        for i, z in enumerate(element.atomic_numbers):
            atomic_energy[z] = x[i]
    except np.linalg.LinAlgError:
        logging.info(
            "Failed to compute Isolated Atomic Energies automatically , using Isolated Atomic Energies = 0.0 for all atoms"
        )
        atomic_energy = {}
        for i, z in enumerate(element.atomic_numbers):
            atomic_energy[z] = 0.0
    return atomic_energy

def compute_atomic_energy(
    points: Atoms, element: Element, keyspec: KeySpecification, num_levels: int,
) -> Dict[int, float]:
    atomic_energies = []
    len_train = len(points)
    for level_idx in range(num_levels):
        points_per_level = []
        for i in range(len_train):
            level = points[i].info.get(keyspec.info_keys['level'], 0)

            if level == level_idx:
                points_per_level.append(points[i])
        assert len(points_per_level) > 0
        atomic_energies.append(
            compute_atomic_energy_per_level(
                points_per_level, element, keyspec,
            )
        )
    return atomic_energies

def analyze_dataset_statistics(
    dataloader_train,
    # dataloader_valid,
    atomic_numbers: List[int],
    atomic_energies: List[Dict[int, float]],
    targe_property: List[str],
    device: str = "cpu",
    num_levels: int = 1,
) -> List[Dict]:

    energyList_per_level = [[] for _ in range(num_levels)]
    energyPerAtomList_per_level = [[] for _ in range(num_levels)]
    deltaEnergyPerAtomList_per_level = [[] for _ in range(num_levels)]
    forcesList_per_level = [[] for _ in range(num_levels)]
    elementIdxList_per_level = [[] for _ in range(num_levels)]

    neighborCountsList = []
    num_elements = len(atomic_numbers)
    neighbor_sum_per_element = torch.zeros(num_elements, device=device)
    count_per_element = torch.zeros(num_elements, dtype=torch.int64, device=device)

    compute_atomic_energy_fn = None
    if "energy" in targe_property:
        compute_atomic_energy_fn = OneHotToAtomicEnergy(atomic_energies).to(device)

    with torch.no_grad():
        for data in dataloader_train:
            data = data.to(device)
            num_graphs = len(data.ptr) - 1
            num_nodes = data.ptr[1:] - data.ptr[:-1]
            element_idx = data["node_attrs"].argmax(dim=-1)
            source = data.edge_index[0]

            # === average number of neighbors ===
            neighbor_counts = torch.bincount(source, minlength=data.batch.size(0)).float()
            neighborCountsList.append(neighbor_counts)
            neighbor_sum = scatter_add(
                neighbor_counts, element_idx, dim=0, dim_size=num_elements
            )
            element_count = scatter_add(
                torch.ones_like(neighbor_counts),
                element_idx,
                dim=0,
                dim_size=num_elements,
            )
            neighbor_sum_per_element += neighbor_sum
            count_per_element += element_count.long()

            if "level" in data:
                level = data["level"]
            else:
                level = torch.zeros(num_graphs, dtype=torch.int64, device=device)

            node_level = level[data['batch']]
            num_atoms_arange = torch.arange(
                data["node_attrs"].size(0), 
                device=data["node_attrs"].device,
                dtype=torch.int64,
            )
            # === energy ===
            if "energy" in targe_property:
                e0_node_energy = compute_atomic_energy_fn(
                    data["node_attrs"]
                )[num_atoms_arange, node_level]

                E0 = scatter(
                    e0_node_energy,
                    data['batch'],
                    dim=0,
                    dim_size=num_graphs,
                    reduce="sum",
                )
                energy = data['energy']
                energy_per_atom = energy / num_nodes
                delta_per_atom = (energy - E0) / num_nodes

                for lvl in range(num_levels):
                    mask = (level == lvl)
                    if mask.any():
                        energyList_per_level[lvl].append(energy[mask])
                        energyPerAtomList_per_level[lvl].append(energy_per_atom[mask])
                        deltaEnergyPerAtomList_per_level[lvl].append(delta_per_atom[mask])

            # === forces ===
            if "forces" in targe_property or "direct_forces" in targe_property:
                forces = data.get('forces', None)
                if forces is None and "direct_forces" in targe_property:
                    forces = data.get('direct_forces', None)
                for lvl in range(num_levels):
                    mask = (node_level == lvl)
                    if mask.any():
                        forcesList_per_level[lvl].append(forces[mask])
                        elementIdxList_per_level[lvl].append(element_idx[mask])

    # === Neighbor statistics ===
    neighborCounts = torch.cat(neighborCountsList)
    avg_num_neighbors = neighborCounts.mean().item()
    avg_neighbors_by_element = {
        atomic_numbers[i]: (
            (neighbor_sum_per_element[i] / count_per_element[i]).item()
            if count_per_element[i] > 0
            else 0.0
        )
        for i in range(num_elements)
    }

    def build_stats_from_lists(
        idx,
        energy_lists,
        energyPerAtom_lists,
        delta_lists,
        forces_lists,
        elemIdx_lists,
    ):
        stats = {
            "level": int(idx),
            "atomic_numbers": atomic_numbers,
            "avg_num_neighbors": avg_num_neighbors,
            "avg_neighbors_by_element": avg_neighbors_by_element,
        }

        # === Energy statistics === #
        if "energy" in targe_property:
            energy = torch.cat(energy_lists)
            energyPerAtom = torch.cat(energyPerAtom_lists)
            deltaEnergyPerAtom = torch.cat(delta_lists)
            mean_energy = energy.mean().item()
            std_energy = energy.std().item()
            mean_energy_per_atom = energyPerAtom.mean().item()
            mean_delta_energy_per_atom = deltaEnergyPerAtom.mean().item()
            stats.update(
                {
                    "__mean_energy": mean_energy,
                    "__std_energy": std_energy,
                    "__mean_energy_per_atom": mean_energy_per_atom,
                    "__mean_delta_energy_per_atom": mean_delta_energy_per_atom,
                    "atomic_energy": {k: float(v) for k, v in atomic_energies[idx].items()},
                    "scalar_mean_energy_per_atom": mean_energy_per_atom,
                    "mean_energy": {z: mean_energy for z in atomic_numbers},
                    "mean_energy_by_element": {z: mean_energy for z in atomic_numbers},
                    "std_energy": {z: std_energy for z in atomic_numbers},
                    "std_energy_by_element": {z: std_energy for z in atomic_numbers},
                    "mean_energy_per_atom": {
                        z: mean_energy_per_atom for z in atomic_numbers
                    },
                    "mean_energy_per_atom_by_element": {
                        z: mean_energy_per_atom for z in atomic_numbers
                    },
                    "mean_delta_energy_per_atom": {
                        z: mean_delta_energy_per_atom for z in atomic_numbers
                    },
                    "mean_delta_energy_per_atom_by_element": {
                        z: mean_delta_energy_per_atom for z in atomic_numbers
                    },
                }
            )

        # === Force statistics === #
        if "forces" in targe_property or 'direct_forces' in targe_property:
            forces = torch.cat(forces_lists, dim=0)  # [N, 3]
            elementIdx = torch.cat(elemIdx_lists, dim=0)  # [N]
            N = forces.size(0)

            # Global force statistics (3D)
            mean_forces_3d = forces.mean(dim=0).tolist()
            std_forces_3d = forces.std(dim=0).tolist()
            rms_forces_3d = torch.sqrt(torch.mean(forces**2, dim=0)).tolist()

            # Global force norm statistics (1D)
            forces_norm = torch.norm(forces, dim=1)  # [N]
            mean_forces_1d = forces_norm.mean().item()
            std_forces_1d = forces_norm.std().item()
            rms_forces_1d = torch.sqrt(torch.mean(forces_norm**2)).item()

            # Per-element 3D force statistics
            count = scatter_add(
                torch.ones((N, 1), device=device),
                elementIdx.unsqueeze(1),
                dim=0,
                dim_size=num_elements,
            ).clamp(min=1)
            mean_force = (
                scatter_add(forces, elementIdx, dim=0, dim_size=num_elements) / count
            )
            diff = forces - mean_force[elementIdx]
            std_force = torch.sqrt(
                scatter_add(diff**2, elementIdx, dim=0, dim_size=num_elements) / count
            )
            rms_force = torch.sqrt(
                scatter_add(forces**2, elementIdx, dim=0, dim_size=num_elements) / count
            )

            # Per-element 1D force norm statistics
            mean_force_1d_by_elem = (
                scatter_add(forces_norm, elementIdx, dim=0, dim_size=num_elements)
                / count.squeeze()
            )
            std_force_1d_by_elem = torch.sqrt(
                scatter_add(
                    (forces_norm - mean_force_1d_by_elem[elementIdx]) ** 2,
                    elementIdx,
                    dim=0,
                    dim_size=num_elements,
                )
                / count.squeeze()
            )
            rms_force_1d_by_elem = torch.sqrt(
                scatter_add(forces_norm**2, elementIdx, dim=0, dim_size=num_elements)
                / count.squeeze()
            )

            # Assemble results
            mean_forces_3d_by_element = {
                atomic_numbers[i]: mean_force[i].tolist() for i in range(num_elements)
            }
            std_forces_3d_by_element = {
                atomic_numbers[i]: std_force[i].tolist() for i in range(num_elements)
            }
            rms_forces_3d_by_element = {
                atomic_numbers[i]: rms_force[i].tolist() for i in range(num_elements)
            }
            mean_forces_1d_by_element = {
                atomic_numbers[i]: mean_force_1d_by_elem[i].item()
                for i in range(num_elements)
            }
            std_forces_1d_by_element = {
                atomic_numbers[i]: std_force_1d_by_elem[i].item()
                for i in range(num_elements)
            }
            rms_forces_1d_by_element = {
                atomic_numbers[i]: rms_force_1d_by_elem[i].item()
                for i in range(num_elements)
            }

            stats.update(
                {
                    # Global
                    "__mean_forces_3d": mean_forces_3d,
                    "__std_forces_3d": std_forces_3d,
                    "__rms_forces_3d": rms_forces_3d,
                    "__mean_forces_1d": mean_forces_1d,
                    "__std_forces_1d": std_forces_1d,
                    "__rms_forces_1d": rms_forces_1d,
                    # Global Per element
                    "__mean_forces_3d_by_element": mean_forces_3d_by_element,
                    "__std_forces_3d_by_element": std_forces_3d_by_element,
                    "__rms_forces_3d_by_element": rms_forces_3d_by_element,
                    "__mean_forces_1d_by_element": mean_forces_1d_by_element,
                    "__std_forces_1d_by_element": std_forces_1d_by_element,
                    "__rms_forces_1d_by_element": rms_forces_1d_by_element,
                    # for normalize:
                    "mean_forces_for_normalize": mean_forces_1d,
                    "std_forces_for_normalize": std_forces_1d,
                    # for scale
                    "rms_forces": {z: rms_forces_1d for z in atomic_numbers},
                    "std_forces": {z: std_forces_1d for z in atomic_numbers},
                    "rms_forces_by_element": rms_forces_1d_by_element,
                    "std_forces_by_element": std_forces_1d_by_element,
                }
            )

        return stats

    per_level_stats = []
    for lvl in range(num_levels):
        stats = build_stats_from_lists(
            lvl,
            energyList_per_level[lvl],
            energyPerAtomList_per_level[lvl],
            deltaEnergyPerAtomList_per_level[lvl],
            forcesList_per_level[lvl],
            elementIdxList_per_level[lvl],
        )
        per_level_stats.append(stats)

    log_statistics_to_yaml(per_level_stats)

    return per_level_stats


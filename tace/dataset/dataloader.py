################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import gc
import yaml
import logging
from typing import Dict, List
from pathlib import Path

import torch
from tqdm import tqdm
from hydra.utils import instantiate
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .element import build_element_lookup
from .read import _read
from .graph import from_atoms
from .statistics import compute_atomic_energy, _compute_statistics, Statistics
from .quantity import KeySpecification


@rank_zero_only
def create_graphs_for_main_rank(atomsList, element, for_dataset, stage):
    dataset = []
    for atoms in tqdm(atomsList, desc=f"Building graphs for {stage}"):
        dataset.append(from_atoms(element, atoms, **for_dataset))
    return dataset

@rank_zero_only
def build_atomsList(
    cfg: dict,
    target_property: List[str],
    keyspec: KeySpecification,
    embedding_property: List[str],
    num_levels: int = 1,
):
    threeAtomsList = _read(cfg, target_property, keyspec, embedding_property)

    # ==== read atomic_numbers and atomic_energy from dataset and cfg ===
    try:
        atomsList = (
            (threeAtomsList[0])
            if threeAtomsList[1] is None
            else threeAtomsList[0] + threeAtomsList[1]
        )
        atomic_numbers_from_dataset = set(
            int(atomic_number)
            for atoms in atomsList
            for atomic_number in atoms.get_atomic_numbers()
        )
    except Exception as e:
        raise RuntimeError(f"Failed to extract atomic numbers from dataset: {e}")

    atomic_numbers = cfg['model']['config'].get("atomic_numbers", None)
    if  atomic_numbers is not None:
        atomic_numbers_from_cfg = set(atomic_numbers)
        assert atomic_numbers_from_dataset.issubset(atomic_numbers_from_cfg), (
            f"cfg.model.config.atomic_numbers must include all atomic numbers present in the dataset, "
            f"but is missing: {atomic_numbers_from_dataset - atomic_numbers_from_cfg}"
        )
        atomic_numbers_from_dataset = atomic_numbers_from_cfg

    element = build_element_lookup(atomic_numbers_from_dataset)

    # === multi-level atomic_energy ===
    atomic_energies= []
    if "energy" in target_property:
        mixed_precision = cfg['model']['config'].get("mixed_precision", {})
        level_names = mixed_precision.get('level_names', ['default']) or ['default']
        atomic_energies_cfg = mixed_precision.get('atomic_energies', None) or None

        assert isinstance(level_names, List)
        assert atomic_energies_cfg is None or isinstance(atomic_energies_cfg, List)

        if num_levels > 1 and atomic_energies_cfg is not None:
            assert len(level_names) == len(atomic_energies_cfg)
            for v in atomic_energies_cfg:
                assert isinstance(v, Dict) or v is None, "If you want to use mixed_precision training, "
                "you must provide each level's atomic energy or set all level to null"

        if atomic_energies_cfg is None:
            logging.info("Computing Isolated Atomic Energies (IAE) automatically for each level")
            atomic_energies = compute_atomic_energy(
                threeAtomsList[0], 
                element, 
                keyspec,
                num_levels
            ) 
        else:
            for idx, energy_cfg in enumerate(atomic_energies_cfg):
                atomic_energy = {int(k): float(v) for k, v in energy_cfg.items()}
                atomic_energy_keys = set(atomic_energy.keys())
                assert atomic_energy_keys.issubset(atomic_numbers_from_dataset), (
                    f"Level {idx}: Keys in atomic_energy must be subset of dataset atomic numbers. "
                    f"Unexpected: {atomic_energy_keys - atomic_numbers_from_dataset}"
                )
                for z in atomic_numbers_from_dataset:
                    if z not in atomic_energy:
                        atomic_energy[z] = 0.0
                        logging.warning(
                            f"Level {idx}: No isolated atomic energy provided for Z={z}, using 0.0 as default."
                        )
                atomic_energies.append(atomic_energy)

        logging.info("Isolated Atomic Energies per level:")
        for idx, energy in enumerate(atomic_energies):
            logging.info(f"  Level {idx}: {energy} {(level_names[idx])}")

    return element, threeAtomsList, atomic_energies

@rank_zero_only
def compute_statistics(
    cfg: Dict,
    target_property: List[str],
    keyspec: KeySpecification,
    embedding_property: List[str],
    num_levels: int = 1,
):

    element, threeAtomsList, atomic_energies = build_atomsList(
        cfg, target_property, keyspec, embedding_property, num_levels
    )

    # dataloader_valid = None
    # if threeAtomsList[1] is not None:
    #     dataset_valid = create_graphs_for_main_rank(threeAtomsList[1], element, for_dataset, 'valid')
    #     # if dataset_valid is None:
    #     #     if read_graphs:
    #     #         dataset_valid = read_graphs_from_pt('valid')
    #     # if save_graphs:
    #     #     save_graphs_to_pt(dataset_valid, 'valid')
    #     dataloader_valid = instantiate(
    #         cfg["dataset"]["valid_dataloader"],
    #         dataset=dataset_valid
    #     )
    #     logging.info(f"Number of configs in valid: {len(threeAtomsList[1])}")

    # dataloader_tests = None
    # if threeAtomsList[2] is not None:
    #     dataloader_tests = []
    #     for idx, test_atoms_list in enumerate(threeAtomsList[2]):
    #         dataset_test = create_graphs_for_main_rank(test_atoms_list, element, for_dataset, f'test{idx}')
    #         # if dataset_test is None:
    #         #     if read_graphs:
    #         #         dataset_test = read_graphs_from_pt(f'test{idx}')
    #         # if save_graphs:
    #         #     save_graphs_to_pt(dataset_test, f'test{idx}')
    #         dataloader_test = instantiate(
    #             cfg["dataset"]["test_dataloader"],
    #             dataset=dataset_test
    #         )
    #         dataloader_tests.append(dataloader_test)
    #     for idx, test_atoms_list in enumerate(threeAtomsList[2]):
    #         logging.info(f"Number of configs in test{idx}: {len(test_atoms_list)}")

    # === compute statistics ===
    statistics_yaml = [Path('.') / f'statistics_{i}.yaml' for i in range(num_levels)]
    all_exist = all(p.exists() for p in statistics_yaml)
    if all_exist:
        with open(statistics_yaml[0], "r") as f:
            statistics_data = yaml.safe_load(f)
        if element.atomic_numbers == statistics_data.get('atomic_numbers', []):
            recompute = False
        else:
            recompute = True
    else:
        recompute = True

    if recompute:
        for_dataset = {
            "cutoff": float(cfg.get("misc", {}).get("cutoff", 5.0)),
            "max_neighbors": cfg.get("misc", {}).get("max_neighbors", None),
            "keyspec": keyspec,
            "target_property": target_property,
            "embedding_property": embedding_property,
            "universal_embedding": cfg.get("model", {})
            .get("config", {})
            .get("universal_embedding", None),
        }

        dataloader_train = None
        dataset_train = create_graphs_for_main_rank(threeAtomsList[0], element, for_dataset, 'train')
        dataloader_train = instantiate(
            cfg["dataset"]["train_dataloader"],
            dataset=dataset_train
        )
        logging.info(f"Number of configs in train: {len(threeAtomsList[0])}")

        statistics = _compute_statistics(
            dataloader_train,
            # dataloader_valid,
            sorted(element.atomic_numbers),
            atomic_energies,
            target_property=target_property,
            device=cfg.get("misc", {}).get("device", "cpu"),
            num_levels=num_levels,
        )
    else:
        # To ensure complete reproducibility, since the statistics were originally 
        # computed by going through the training dataloader once, we need to go through 
        # it again here to ensure consistency.
        with torch.no_grad():
            for data in dataloader_train:
                batch = data['batch']
        statistics = []
        for yaml_file in statistics_yaml:
            with open(yaml_file, "r") as f:
                statistics_data = yaml.safe_load(f)
            statistics.append(Statistics(**statistics_data))

    del dataloader_train
    gc.collect()
    
    return statistics

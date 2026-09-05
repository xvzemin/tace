################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict, List, Sequence, Union

from hydra.utils import instantiate
from lightning.pytorch.utilities.rank_zero import rank_zero_only

from .element import TorchElement, build_element_lookup
from .graph import from_atoms
from .quantity import KeySpecification
from .read import tace_read_all_files
from .statistics import _compute_statistics, compute_atomic_energies


def build_statistics_dataloader(
    cfg: Dict,
    dataset,
    target_property: Sequence[str] = (),
):
    """Build a large, deterministic loader containing only statistics inputs."""
    config = {
        key: value
        for key, value in cfg["dataset"]["train_dataloader"].items()
        if key != "extra"
    }
    config.update(cfg["dataset"].get("statistics_dataloader", {}))
    config.pop("extra", None)
    config["shuffle"] = False
    config["drop_last"] = False
    config.setdefault(
        "pin_memory",
        str(cfg.get("misc", {}).get("device", "cpu")).startswith("cuda"),
    )

    required_keys = {
        "atomic_numbers",
        "edge_index",
        "fidelity_idx",
        "initial_noncollinear_magmoms",
        "num_nodes",
    }
    if "energy" in target_property:
        required_keys.update(("energy", "energy_weight"))
    if "forces" in target_property:
        required_keys.update(("forces", "forces_weight"))
    if "direct_forces" in target_property:
        required_keys.update(("direct_forces", "direct_forces_weight"))
    if len(dataset):
        excluded_keys = set(config.get("exclude_keys") or ())
        excluded_keys.update(set(dataset[0].keys()) - required_keys)
        config["exclude_keys"] = sorted(excluded_keys - required_keys)
    return instantiate(config, dataset=dataset)


@rank_zero_only
def create_graphs_for_main_rank(atomsList, element, for_dataset, stage):
    dataset = []
    for atoms in atomsList:
        dataset.append(from_atoms(element, atoms, **for_dataset))
    return dataset


@rank_zero_only
def build_atomsList(
    cfg: Dict,
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeySpecification,
):
    threeAtomsList = tace_read_all_files(
        cfg, target_property, embedding_property, keyspec
    )
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

    atomic_numbers = cfg["model"]["config"].get("atomic_numbers", None)
    if atomic_numbers is not None:
        atomic_numbers_from_cfg = set(atomic_numbers)
        assert atomic_numbers_from_dataset.issubset(atomic_numbers_from_cfg), (
            f"cfg.model.config.atomic_numbers must include all atomic numbers present in the dataset, "
            f"but is missing: {atomic_numbers_from_dataset - atomic_numbers_from_cfg}"
        )
        atomic_numbers_from_dataset = atomic_numbers_from_cfg

    # === multi-fidelity atomic_energy ===
    fidelity = cfg["model"]["config"]["fidelity"]
    num_fidelities = len(fidelity)
    if "energy" in target_property:
        atomic_energies_cfg: List[Union[Dict[int, float], None]] = []
        for idx in range(num_fidelities):
            this_atomic_energy = fidelity[idx].get("atomic_energy", None)
            assert this_atomic_energy is None or isinstance(this_atomic_energy, dict), (
                "If you want to use multi-fidelity or multi-head training, "
                "you must provide each fidelity's atomic_energy or set to null"
            )
            if this_atomic_energy is not None:
                this_atomic_energy = {
                    int(z): float(value) for z, value in this_atomic_energy.items()
                }
            atomic_energies_cfg.append(this_atomic_energy)
        atomic_numbers_from_energy = set()
        for this_atomic_energy in atomic_energies_cfg:
            if this_atomic_energy is None:
                pass
            else:
                atomic_numbers_from_energy.update(this_atomic_energy.keys())
        atomic_numbers_from_dataset = (
            atomic_numbers_from_dataset | atomic_numbers_from_energy
        )
        element = build_element_lookup(atomic_numbers_from_dataset)

        missing_energy = [
            idx
            for idx, atomic_energy in enumerate(atomic_energies_cfg)
            if atomic_energy is None
        ]
        for idx in missing_energy:
            logging.info(
                "Computing isolated atomic energy automatically for fidelity %s",
                fidelity[idx]["name"],
            )
        computed_atomic_energies = (
            compute_atomic_energies(threeAtomsList[0], element, keyspec, missing_energy)
            if missing_energy
            else {}
        )

        atomic_energies = []
        for idx, atomic_energy_cfg in enumerate(atomic_energies_cfg):
            if atomic_energy_cfg is None:
                atomic_energies.append(computed_atomic_energies[idx])
            else:
                atomic_energy = {
                    z: float(atomic_energy_cfg.get(z, 0.0)) for z in element.zs
                }
                for z in element.zs:
                    if z not in atomic_energy_cfg:
                        logging.warning(
                            f"Fidelity {fidelity[idx]['name']}: No isolated atomic energy "
                            f"provided for Z={z}, using 0.0 as default."
                        )
                atomic_energies.append(atomic_energy)

        logging.info("Isolated atomic energy per computational fidelity:")
        for idx, energy in enumerate(atomic_energies):
            logging.info(f"  {fidelity[idx]['name']}: {energy}")
    else:
        element = build_element_lookup(atomic_numbers_from_dataset)
        atomic_energies = None
    return element, threeAtomsList, atomic_energies


@rank_zero_only
def compute_statistics(
    cfg: Dict,
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeySpecification,
    element: TorchElement,
    threeAtomsList,
    fidelity,
    atomic_energies: List[Dict[int, float]],
    dataloader_train=None,
):

    if dataloader_train is None:
        for_dataset = {
            "cutoff": float(cfg["model"]["config"]["cutoff"]),
            "max_neighbors": cfg["model"]["config"].get("max_neighbors", None),
            "keyspec": keyspec,
            "target_property": list(target_property),
            "embedding_property": list(embedding_property),
            "neighborlist_backend": cfg["dataset"].get(
                "neighborlist_backend", "matscipy"
            ),
        }
        dataset_train = create_graphs_for_main_rank(
            threeAtomsList[0], element, for_dataset, "train"
        )
        dataloader_train = build_statistics_dataloader(
            cfg, dataset_train, target_property
        )

    statistics = _compute_statistics(
        dataloader_train,
        element.zs,
        atomic_energies,
        target_property=target_property,
        device=cfg.get("misc", {}).get("device", "cpu"),
        num_fidelities=len(fidelity),
    )

    return statistics

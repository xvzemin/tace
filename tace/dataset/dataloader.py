################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from typing import Dict, List, Sequence, Union

import torch
from hydra.utils import instantiate

from .cache import statistics_cache_key
from .element import TorchElement, build_element_lookup
from .quantity import KeySpecification
from .statistics import (
    _compute_statistics,
    compute_atomic_energies_from_graphs,
)


def build_graph_dataloader(config, dataset):
    """Build a loader without bypassing batched dataset reads."""
    config = {key: value for key, value in config.items() if key != "extra"}
    source = getattr(dataset, "dataset", dataset)
    device = getattr(source, "device", None)
    if (
        device is not None
        and torch.device(device).type == "cuda"
        and config.get("num_workers", 0)
    ):
        raise ValueError("CUDA online graph construction requires num_workers=0.")
    return instantiate(config, dataset=dataset)


def build_statistics_dataloader(
    cfg: Dict,
    dataset,
    target_property: Sequence[str] = (),
    neighbors: bool = True,
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
        "fidelity_idx",
        "initial_noncollinear_magmoms",
        "num_nodes",
    }
    if neighbors:
        fields = (getattr(dataset, "metadata", None) or {}).get("fields", [])
        required_keys.add(
            "num_neighbors" if "num_neighbors" in fields else "edge_index"
        )
    if "energy" in target_property:
        required_keys.update(("energy", "energy_weight"))
    if "forces" in target_property:
        required_keys.update(("forces", "forces_weight"))
    if "direct_forces" in target_property:
        required_keys.update(("direct_forces", "direct_forces_weight"))
    if hasattr(dataset, "project"):
        dataset = dataset.project(required_keys)
        config.pop("exclude_keys", None)
    elif len(dataset):
        excluded_keys = set(config.get("exclude_keys") or ())
        excluded_keys.update(set(dataset[0].keys()) - required_keys)
        config["exclude_keys"] = sorted(excluded_keys - required_keys)
    return build_graph_dataloader(config, dataset)


def build_atomsList(
    cfg: Dict,
    target_property: List[str],
    embedding_property: List[str],
    keyspec: KeySpecification,
    datamodule=None,
):
    threeAtomsList = None
    if datamodule is None:
        from .datamodule import build_datamodule

        datamodule = build_datamodule(
            cfg, None, target_property, embedding_property, keyspec
        )
        datamodule.setup("fit")
    atomic_numbers_from_dataset = set(datamodule.element.zs)

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
        if datamodule is not None:
            datamodule.element = element

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
        computed_atomic_energies = {}
        if missing_energy:
            computed_atomic_energies = compute_atomic_energies_from_graphs(
                build_statistics_dataloader(
                    cfg, datamodule.train_dataset, ["energy"], neighbors=False
                ),
                element,
                missing_energy,
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
        from .datamodule import build_datamodule

        datamodule = build_datamodule(
            cfg,
            element.zs,
            target_property,
            embedding_property,
            keyspec,
            threeAtomsList,
        )
        datamodule.setup("fit")
        dataloader_train = datamodule.statistics_dataloader()

    statistics = _compute_statistics(
        dataloader_train,
        element.zs,
        atomic_energies,
        target_property=target_property,
        device=cfg.get("misc", {}).get("device", "cpu"),
        num_fidelities=len(fidelity),
        data_fingerprint=statistics_cache_key(
            cfg, target_property, embedding_property, keyspec
        ),
    )

    return statistics

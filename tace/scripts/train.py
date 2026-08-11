################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import copy
import logging
import warnings
from pathlib import Path

import hydra
import yaml
from omegaconf import DictConfig, OmegaConf

from tace.dataset.dataloader import build_atomsList, compute_statistics
from tace.dataset.datamodule import build_datamodule
from tace.dataset.quantity import (
    KEYS,
    KeySpecification,
    get_embedding_property,
    get_need_property,
    get_target_property,
    update_keyspec_from_kwargs,
)
from tace.lightning.lit_model import finetune, load_tace
from tace.lightning.torch_model import create_model
from tace.lightning.trainer import train
from tace.utils.env import set_env
from tace.utils.hydra_resolver import register_resolvers
from tace.utils.logger import set_logger
from tace.utils.utils import (
    deep_convert,
    save_full_cfg,
    set_global_seed,
    set_precision,
)

register_resolvers()

TRANSFORMER_TACE_TARGET = "tace.models._transformer.TransformerTACE"
SINGLE_LEVEL_DATA_CONFIG = [{"name": "default", "atomic_energy": None}]


def initialize(cfg):
    cfg = deep_convert(cfg)
    set_logger(cfg["misc"].get("log_level", "info"))
    if cfg["misc"].get("ignore_warning", True):
        try:
            warnings.simplefilter("ignore", FutureWarning)
            warnings.filterwarnings(
                "ignore", module="pydantic._internal._generate_schema"
            )
        except Exception:
            pass
    save_full_cfg(cfg)
    set_env(cfg)
    set_global_seed(cfg)
    set_precision(cfg)
    return cfg


def _model_config(cfg):
    return cfg["model"]["config"]


def _is_transformer_tace(cfg) -> bool:
    return _model_config(cfg).get("_target_") == TRANSFORMER_TACE_TARGET


def _prepare_data_fidelity(cfg):
    """Adapt the shared data/statistics pipeline to a single-task TransformerTACE."""
    model_config = _model_config(cfg)
    if _is_transformer_tace(cfg):
        if "fidelity" in model_config:
            raise ValueError(
                "TransformerTACE is single-task; remove `fidelity` from its model config"
            )
        model_config["fidelity"] = copy.deepcopy(SINGLE_LEVEL_DATA_CONFIG)
    return model_config["fidelity"]


def _create_model(cfg, statistics, target_property, embedding_property):
    """Instantiate a model while hiding data-pipeline-only Transformer fields."""
    model_config = _model_config(cfg)
    if not _is_transformer_tace(cfg):
        return create_model(cfg, statistics, target_property, embedding_property)

    data_fidelity = model_config.pop("fidelity")
    try:
        return create_model(cfg, statistics, target_property, embedding_property)
    finally:
        model_config["fidelity"] = data_fidelity


def _remove_data_fidelity(cfg):
    """Remove the temporary shared-pipeline adapter before training/checkpointing."""
    if _is_transformer_tace(cfg):
        _model_config(cfg).pop("fidelity", None)


def _restore_loaded_model_config(cfg, model):
    """Preserve the legacy checkpoint flow without requiring Transformer model_config."""
    loaded_config = getattr(model.readout_fn, "model_config", None)
    if loaded_config is not None:
        cfg["model"]["config"] = loaded_config
    if _is_transformer_tace(cfg):
        _model_config(cfg)["fidelity"] = copy.deepcopy(SINGLE_LEVEL_DATA_CONFIG)


def build(cfg: DictConfig):
    cfg = initialize(
        OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict")
    )
    target_property = get_target_property(cfg)
    embedding_property = get_embedding_property(cfg)
    needs_noncollinear_magmoms = "initial_noncollinear_magmoms" in get_need_property(
        target_property,
        embedding_property,
        training=True,
    )
    userKeys = copy.deepcopy(KEYS)
    userKeys.update(cfg["dataset"].get("keys", {}))
    keyspec = KeySpecification()
    update_keyspec_from_kwargs(keyspec, userKeys)
    fidelity = _prepare_data_fidelity(cfg)

    # train from scratch, calculate statistics
    statistics = None
    threeAtomsList = None
    datamodule = None
    if not (cfg.get("finetune_from_model", None) or cfg.get("resume_from_model", None)):
        statistics_yaml = [
            Path(".") / f"statistics_{idx}.yaml" for idx in range(len(fidelity))
        ]
        if all(yaml_file.exists() for yaml_file in statistics_yaml):
            statistics = []
            for yaml_file in statistics_yaml:
                with open(yaml_file, "r") as f:
                    statistics_data = yaml.safe_load(f)
                    statistics.append(statistics_data)
            if needs_noncollinear_magmoms and any(
                "magmoms_norm_by_element" not in stats for stats in statistics
            ):
                logging.info(
                    "Cached statistics do not contain magmoms_norm_by_element; "
                    "recomputing statistics for the noncollinear magnetic input"
                )
                statistics = None
            else:
                for idx, yaml_file in enumerate(statistics_yaml):
                    logging.info(
                        "Using '%s' for fidelity %s",
                        yaml_file,
                        fidelity[idx]["name"],
                    )
                atomic_numbers = statistics[0]["atomic_numbers"]
        if statistics is None:
            logging.info("Computing statistics information from scratch")
            element, threeAtomsList, atomic_energies = build_atomsList(
                cfg, target_property, embedding_property, keyspec
            )
            atomic_numbers = element.zs
            datamodule = build_datamodule(
                cfg,
                atomic_numbers,
                target_property,
                embedding_property,
                keyspec,
                threeAtomsList,
            )
            datamodule.setup("fit")
            statistics = compute_statistics(
                cfg,
                target_property,
                embedding_property,
                keyspec,
                element,
                threeAtomsList,
                fidelity,
                atomic_energies,
                dataloader_train=datamodule.statistics_dataloader(),
            )
    # finetune or reusme, statistics is no need to recalculate
    if cfg.get("finetune_from_model", None):
        model = finetune(cfg)
        statistics = model.readout_fn.statistics
        atomic_numbers = statistics[0]["atomic_numbers"]
        _restore_loaded_model_config(cfg, model)
        finetune_cfg = cfg.get("finetune", {})
        if finetune_cfg:
            logging.info("Using finetune_config from your main train config.")
        else:
            yaml_path = Path("finetune_config.yaml")
            if yaml_path.exists():
                logging.info(f"Using finetune_config from {yaml_path}.")
                try:
                    finetune_cfg = yaml.safe_load(yaml_path.read_text())
                except Exception as e:
                    logging.error(f"Failed to read {yaml_path}: {e}")
            else:
                logging.warning(
                    f"{yaml_path} not found, skipping lora and parameter freezing."
                )
        cfg["finetune"] = finetune_cfg
    elif cfg.get("resume_from_model", None):
        model = load_tace(
            cfg["resume_from_model"],
            device="cpu",
            strict=True,
            use_ema=True,
            dtype=cfg["trainer"]["precision"],
        )
        statistics = model.readout_fn.statistics
        atomic_numbers = statistics[0]["atomic_numbers"]
        _restore_loaded_model_config(cfg, model)
        cfg["finetune"] = cfg.get("finetune", {})
    else:  # From scratch
        model = _create_model(cfg, statistics, target_property, embedding_property)

    if datamodule is None:
        datamodule = build_datamodule(
            cfg,
            atomic_numbers,
            target_property,
            embedding_property,
            keyspec,
            threeAtomsList,
        )
    _remove_data_fidelity(cfg)

    return cfg, statistics, target_property, embedding_property, model, datamodule


@hydra.main(version_base="1.3", config_path=str(Path.cwd()), config_name="tace")
def main(cfg: DictConfig):
    cfg, statistics, target_property, embedding_property, model, datamodule = build(cfg)
    train_arguments = {
        "cfg": cfg,
        "statistics": statistics,
        "target_property": target_property,
        "embedding_property": embedding_property,
        "model": model,
        "datamodule": datamodule,
    }
    train(**train_arguments)


if __name__ == "__main__":
    main()

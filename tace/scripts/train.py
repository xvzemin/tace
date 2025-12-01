################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import yaml
import logging
import warnings
from pathlib import Path


import hydra
from omegaconf import DictConfig, OmegaConf


from ..dataset.statistics import Statistics
from ..lightning.trainer import train
from ..lightning.lit_model import finetune
from ..lightning.select_model import select_model
from ..dataset.dataloader import compute_statistics
from ..dataset.datamodule import build_datamodule
from ..utils.hydra_resolver import register_resolvers
from ..utils.logger import set_logger
from ..utils.utils import (
    log_parameters,
    set_global_seed,
    set_precision,
    save_full_cfg,
    deep_convert,
)
from ..utils.env import set_env
from ..dataset.quantity import (
    KEYS,
    KeySpecification,
    update_keyspec_from_kwargs,
    get_target_property,
)
from ..dataset.quantity import get_embedding_property


register_resolvers()


def initialize(cfg):
    cfg = deep_convert(cfg)
    set_logger()
    if cfg['misc'].get('ignore_warning', True): 
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


@hydra.main(version_base="1.3", config_path=str(Path.cwd()), config_name="tace")
def main(cfg: DictConfig):

    cfg = OmegaConf.to_container(cfg, resolve=True, structured_config_mode="dict")
    cfg = initialize(cfg)

    target_property = get_target_property(cfg)
    embedding_property = get_embedding_property(cfg)

    keySpecification = KeySpecification()
    update_keyspec_from_kwargs(
        keySpecification, cfg.get('dataset', {}).get('keys', KEYS)
    )

    num_levels = len(cfg['model']['config'].get("mixed_precision", {}).get("level_names", ['default']))

    statistics_yaml = [Path('.') / f'statistics_{i}.yaml' for i in range(num_levels)]
    if all(yaml_file.exists() for yaml_file in statistics_yaml):
        statistics = []
        for yaml_file in statistics_yaml:
            with open(yaml_file, "r") as f:
                statistics_data = yaml.safe_load(f)
                statistics.append(Statistics(**statistics_data))
        for idx, yaml_file in enumerate(statistics_yaml):
            logging.info(f"Using statistics_yaml from '{str(yaml_file)}' for level {idx}")
    else:
        logging.info(f"Computing statistics information from scratch")
        statistics = compute_statistics(cfg, target_property, keySpecification, embedding_property, num_levels)

    datamodule = build_datamodule(cfg, statistics, target_property, keySpecification, embedding_property, num_levels)

    # Model trained from scratch or from fine-tuning
    if cfg.get("finetune_from_model", None) is not None:
        model = finetune(cfg)
    else:
        model = select_model(cfg, statistics, target_property, embedding_property)

    if cfg.get('resume_from_model', None) is None:
        log_parameters(model)

    train_arguments = {
        "cfg": cfg,
        "statistics": statistics,
        "target_property": target_property,
        "model": model,
        "datamodule": datamodule,
        # "trainer": trainer,
    }
    train(**train_arguments)


if __name__ == "__main__":
    main()

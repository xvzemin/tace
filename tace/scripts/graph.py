################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import yaml
import logging
from pathlib import Path

import hydra
from omegaconf import DictConfig, OmegaConf


from ..dataset.statistics import Statistics
from ..dataset.dataloader import compute_statistics
from ..dataset.datamodule import build_datamodule
from ..utils.hydra_resolver import register_resolvers
from ..utils.logger import set_logger
from ..utils.utils import (
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
    # cfg = update_cfg(cfg)
    save_full_cfg(cfg)
    set_env(cfg)
    set_global_seed(cfg)
    set_precision(cfg)
    return cfg


@hydra.main(version_base="1.3", config_path=str(Path.cwd()), config_name="tace")
def main(cfg: DictConfig) -> None:

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
    datamodule.prepare_data()
    datamodule.setup('fit')
    logging.info(f"Finished building lmdb dataset")

if __name__ == "__main__":
    main()

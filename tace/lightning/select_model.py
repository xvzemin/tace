################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Any, Dict, Optional, List


import torch
from hydra.utils import instantiate


from ..dataset.statistics import Statistics
from ..models.wrapper import WrapModel
from ..utils.utils import deep_convert


def _select_key(target, options, name):
    keys = [k for k in options if k in target]
    if len(keys) > 1:
        raise ValueError(
            f"Only one of {options} is allowed in target_property for {name}."
        )
    return keys[0] if keys else ""


def select_heads(cfg: Dict) -> Any:
    return WrapModel


def select_model(
    cfg: Dict,
    statistics: Optional[Statistics],
    target_property: List[str],
    embedding_property: List[str],
) -> torch.nn.Module:
    """
    Constructs the model using the appropriate head and configuration.

    Parameters:
        cfg (Dict): Hydra configuration object.
        statistics (Statistics): Data statistics passed into the model (e.g. mean/std/rms).

    Returns:
        torch.nn.Module: Instantiated PyTorch model.
    """

    MODEL_HEAD = select_heads(cfg)

    try:
        MODEL_CONFIG = deep_convert(cfg['model']['config'])
        __TARGET = (
            cfg.get('model', {})
            .get('config', {})
            .get('_target_', 'tace.models.tace.TACE')
        )
        if "TACE" in __TARGET:
            from ..models.tace import TACE

            MODEL_CLS = TACE
        elif "TMP" in __TARGET:
            from tmp.tmp import TMP

            MODEL_CLS = TMP
        elif "MACE" in __TARGET:
            from ..models.mace.mace import MACE

            MODEL_CLS = MACE
        elif "CACE" in __TARGET:
            from ..models.cace.cace import CACE

            MODEL_CLS = CACE
        elif "CAMP" in __TARGET:
            from ..models.camp.camp import CAMP

            MODEL_CLS = CAMP
        elif "TensorNet" in __TARGET:
            from ..models.tensornet.tensornet import TensorNet

            MODEL_CLS = TensorNet
        elif "REANN" in __TARGET:
            from ..models.reann.reann import REANN

            MODEL_CLS = REANN
        elif "Allegro" in __TARGET:
            from ..models.allegro.allegro import Allegro

            MODEL_CLS = Allegro
        elif "ICTP" in __TARGET:
            from ..models.ictp.model.ictp import ICTP

            MODEL_CLS = ICTP
        elif "eSCN" in __TARGET:
            from ..models.escn.escn import eSCN

            MODEL_CLS = eSCN
        else:
            raise
        MODEL_CONFIG = {k: v for k, v in MODEL_CONFIG.items() if k != "_target_"}

    except AttributeError as e:
        raise RuntimeError(
            "Missing required config: 'cfg.model.config' is not defined."
        ) from e
    try:
        # MODEL = MODEL_HEAD(
        #     instantiate(
        #         MODEL_CONFIG,
        #         statistics=statistics,
        #         cutoff=cfg["misc"]["cutoff"],
        #         target_property=loss_property,
        #         embedding_property=embedding_property,
        #     )
        # )
        MODEL = MODEL_HEAD(
            MODEL_CLS(
                **MODEL_CONFIG,
                statistics=statistics,
                cutoff=cfg['misc']['cutoff'],
                target_property=target_property,
                embedding_property=embedding_property,
            )
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to instantiate the model using the provided configuration.\n"
            f"MODEL_HEAD: {MODEL_HEAD.__name__ if hasattr(MODEL_HEAD, '__name__') else MODEL_HEAD}\n"
            f"MODEL_CONFIG: {MODEL_CONFIG}"
        ) from e

    return MODEL

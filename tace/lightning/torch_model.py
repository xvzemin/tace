################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# TODO, refactor code


import importlib
import logging
from typing import Any, Dict, List

import torch

from tace.utils.env import acceleration_enabled
from tace.utils.utils import deep_convert


def _should_warn_without_aoti(target_property: List[str]) -> bool:
    aoti_properties = {
        "energy",
        "forces",
        "stress",
        "virials",
        "direct_forces",
        "direct_stress",
        "direct_virials",
        "noncollinear_magnetic_forces",
    }
    return any(prop in aoti_properties for prop in target_property)


def select_wrapper(model_config: Dict, wrapper_path: str = None) -> Any:
    if wrapper_path is None:
        wrapper_path = model_config.get("wrapper", {}).get(
            "_target_", "tace.models.TensorModel"
        )
    module_name, class_name = wrapper_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    wrap_cls = getattr(module, class_name)
    return wrap_cls


def create_model(
    cfg: Dict,
    statistics: Dict,
    target_property: List[str],
    embedding_property: List[str],
    **kwargs,
) -> torch.nn.Module:

    if "model" in cfg:
        model_config = cfg["model"]["config"]
    else:
        model_config = cfg

    # === model cls ===
    if "kwargs" in model_config:
        model_path = model_config["kwargs"].get("_target_", "tace.models.e3nnTACE")
    else:
        model_path = model_config.get("_target_", "tace.models.e3nnTACE")

    wrapper_path = model_config.get("wrapper", {}).get(
        "_target_", "tace.models.TensorModel"
    )

    use_aoti = acceleration_enabled("compile") and model_path in {
        "tace.models.e3nnTACE",
        "tace.models._e3nn.e3nnTACE",
    }
    if use_aoti:
        model_path = "tace.models.compile.e3nnTACE"
        if wrapper_path == "tace.models.TensorModel":
            wrapper_path = "tace.models.CompileTensorModel"
    elif _should_warn_without_aoti(
        model_config.get("target_property") or target_property
    ):
        logging.warning(
            "AOTI is not enabled for energy, forces, stress, virials, or "
            "noncollinear magnetic forces outputs. "
            "For acceleration options, see "
            "https://tace.readthedocs.io/en/latest/guide/acceleration.html"
        )

    # === wrapper cls ===
    WRAPPER_CLS = select_wrapper(model_config, wrapper_path)

    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    MODEL_CLS = getattr(module, class_name)
    model_config = deep_convert(model_config)
    if "statistics" not in model_config:
        model_config["statistics"] = statistics
    if "target_property" not in model_config:
        model_config["target_property"] = target_property
    if "embedding_property" not in model_config:
        model_config["embedding_property"] = embedding_property
    # === instantiate ===
    try:
        MODEL = WRAPPER_CLS(
            MODEL_CLS(
                **model_config,
            )
        )
    except Exception as e:
        raise RuntimeError(
            "Failed to instantiate the model using the provided configuration.\n"
            # f"Model config: {model_config}"
        ) from e

    return MODEL

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# TODO, refactor code


import importlib
import logging
from copy import deepcopy
from typing import Any, Dict, List

import torch

from tace.utils.env import acceleration_enabled
from tace.utils.utils import deep_convert


_E3NN_MODEL_PATHS = {
    "tace.models.e3nnTACE",
    "tace.models._e3nn.e3nnTACE",
}


def _prune_removed_basis_keys(model_config: Dict) -> Dict:
    """Remove obsolete e3nn architecture keys from a saved model config."""
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG

    model_config = deepcopy(model_config)
    if "kwargs" in model_config:
        architecture_config = model_config["kwargs"]
    else:
        architecture_config = model_config
    for section in ("atomic_basis", "product_basis"):
        saved_section = architecture_config.get(section)
        if not isinstance(saved_section, dict):
            continue
        current_keys = DEFAULT_MODEL_CONFIG[section]
        for key in tuple(saved_section):
            if key not in current_keys:
                saved_section.pop(key)
    return model_config


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
    prune_removed_keys: bool = False,
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
    is_e3nn_model = model_path in _E3NN_MODEL_PATHS

    wrapper_path = model_config.get("wrapper", {}).get(
        "_target_", "tace.models.TensorModel"
    )

    use_aoti = acceleration_enabled("compile") and is_e3nn_model
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
    if prune_removed_keys and is_e3nn_model:
        model_config = _prune_removed_basis_keys(model_config)
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

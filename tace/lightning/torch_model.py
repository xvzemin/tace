################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

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
        "charges",
    }
    return bool(target_property) and set(target_property).issubset(aoti_properties)


def _prune_removed_keys(model_config: Dict) -> Dict:
    """Remove legacy architecture keys from a saved model config."""
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG

    pruned_config = model_config.copy()
    for section in ("atomic_basis", "product_basis"):
        saved_section = model_config.get(section)
        if isinstance(saved_section, dict):
            allowed_keys = DEFAULT_MODEL_CONFIG[section].keys()
            pruned_config[section] = {
                key: value
                for key, value in saved_section.items()
                if key in allowed_keys
            }
    return pruned_config


def select_wrapper(wrapper_path: str) -> Any:
    """Import a model wrapper from its fully qualified class path."""
    module_name, class_name = wrapper_path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), class_name)


def create_model(
    model_config: Dict,
    statistics: Dict,
    target_property: List[str],
    embedding_property: List[str],
    prune_removed_keys: bool = False,
) -> torch.nn.Module:

    # === model cls ===
    model_path = model_config.get("_target_", "tace.models.e3nnTACE")
    is_e3nn_model = model_path in {
        "tace.models.e3nnTACE",
        "tace.models._e3nn.e3nnTACE",
    }
    wrapper_config = model_config.get("wrapper") or {}
    wrapper_path = wrapper_config.get("_target_", "tace.models.TensorModel")

    use_aoti = acceleration_enabled("compile") and is_e3nn_model
    if use_aoti:
        model_path = "tace.models.compile.e3nnTACE"
        if wrapper_path == "tace.models.TensorModel":
            wrapper_path = "tace.models.CompileTensorModel"
    elif _should_warn_without_aoti(
        model_config.get("target_property") or target_property
    ):
        logging.warning(
            "AOTI is not enabled for energy, forces, stress, virials, charges "
            "or noncollinear magnetic forces outputs. "
            "For acceleration options, see "
            "https://tace.readthedocs.io/en/latest/guide/acceleration.html"
        )

    # === wrapper cls ===
    WRAPPER_CLS = select_wrapper(wrapper_path)

    module_name, class_name = model_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    MODEL_CLS = getattr(module, class_name)
    model_config = deep_convert(model_config)
    if prune_removed_keys and is_e3nn_model:
        model_config = _prune_removed_keys(model_config)
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

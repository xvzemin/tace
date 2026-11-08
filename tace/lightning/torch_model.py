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
        "direct_hessian",
    }
    return bool(target_property) and set(target_property).issubset(aoti_properties)


def _prune_removed_keys(model_config: Dict) -> Dict:
    """Remove legacy architecture keys from a saved model config."""
    from tace.models.spherical.default import (
        DEFAULT_MODEL_CONFIG,
        normalize_atomic_basis,
    )

    pruned_config = model_config.copy()
    for section in ("atomic_basis", "product_basis"):
        saved_section = model_config.get(section)
        if isinstance(saved_section, dict):
            if section == "atomic_basis":
                saved_section = normalize_atomic_basis(saved_section)
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
    if module_name == "tace.models" or module_name.startswith("tace.models."):
        for old, new in (("_e3nn", "spherical"), ("_cart", "cartesian")):
            prefix = f"tace.models.{old}"
            if module_name == prefix or module_name.startswith(prefix + "."):
                module_name = f"tace.models.{new}" + module_name[len(prefix) :]
                break
        class_name = {"e3nnTACE": "SphericalTACE", "cartTACE": "CartesianTACE"}.get(
            class_name, class_name
        )
    return getattr(importlib.import_module(module_name), class_name)


def create_model(
    model_config: Dict,
    statistics: Dict,
    target_property: List[str],
    embedding_property: List[str],
    prune_removed_keys: bool = False,
) -> torch.nn.Module:
    from tace.models import CartesianTACE, SphericalTACE

    # === model cls ===
    model_path = model_config.get("_target_", "tace.models.SphericalTACE")
    MODEL_CLS = select_wrapper(model_path)
    is_spherical_model = MODEL_CLS is SphericalTACE
    is_cartesian_model = MODEL_CLS is CartesianTACE
    wrapper_config = model_config.get("wrapper") or {}
    wrapper_path = wrapper_config.get("_target_", "tace.models.TensorModel")

    use_aoti = acceleration_enabled("compile") and (
        is_spherical_model or is_cartesian_model
    )
    if use_aoti:
        model_path = (
            "tace.models.compile.CartesianTACE"
            if is_cartesian_model
            else "tace.models.compile.SphericalTACE"
        )
        MODEL_CLS = select_wrapper(model_path)
        if wrapper_path == "tace.models.TensorModel":
            wrapper_path = "tace.models.CompileTensorModel"
    elif _should_warn_without_aoti(
        model_config.get("target_property") or target_property
    ):
        logging.warning(
            "AOTI is not enabled for energy, forces, stress, virials, charges, direct_hessian "
            "or noncollinear magnetic forces outputs. \n"
            "For acceleration options, see "
            "https://tace.readthedocs.io/en/latest/guide/acceleration.html"
        )

    # === wrapper cls ===
    WRAPPER_CLS = select_wrapper(wrapper_path)

    model_config = deep_convert(model_config)
    if prune_removed_keys and (is_spherical_model or is_cartesian_model):
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

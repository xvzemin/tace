################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
import math
from collections.abc import Mapping
from copy import deepcopy
from numbers import Real
from typing import Any

from ...dataset.quantity import PROPERTIES
from ..mlp import MLP
from ..radial import check_radial_selection

# Most defaults are suitable for general use. Some are no longer recommended,
# but must remain unchanged for compatibility with existing models and configurations.
# After merging defaults, raise on conflicting options instead of overriding them.
DEFAULT_MODEL_CONFIG = {
    "mmax": 2,
    "Lmax": 2,
    "lmax": 3,
    "parity": None,
    "num_channel": 64,
    "num_layers": 2,
    "max_neighbors": None,
    "mlp_type": "mlp_v1",
    "target_property": ["energy", "forces"],
    "embedding_property": [],
    "fidelity": {
        "name": "PBE",
        "atomic_energy": None,
        "magnetic_scale": None,
    },
    "node_embedding": {
        "type": "linear",
    },
    "edge_embedding": {
        "type": "identity",
    },
    "edge_update": {
        "type": "identity",
    },
    "node_update": {
        "magnetic_type": "identity",
    },
    "radial_basis": {
        "bias": False,
        "radial_basis": "j0",
        "num_radial_basis": None,
        "max_eigenvalue": None,
        "use_interpolation": False,
        "num_mag_radial_basis": 10,
        "distance_transform": None,
        "cutoff_fn": "c2poly",
        "polynomial_cutoff": 5,
        "trainable": False,
        "apply_cutoff": True,
        "hidden": [64, 64, 64],
        "mlp_norm": None,
        "gaussian_width": 2.0,
    },
    "angular_basis": {
        "magnetic_Lmax": 2,
        "use_spin_orbit_coupling": True,
    },
    "atomic_basis": {
        "type": "cgtp",
        "bias": True,
        "l1l2": None,
        "scatter_norm": "avg_num_neighbors",
        "nonlinear": "gate",
        "edge_nonlinear": "gate",
        "use_asymmetric_contraction": True,
        "num_head": None,
        "edge_ace_hidden": None,
        "gate_m0": True,
        "scalar_act": None,
        "tensor_act": None,
    },
    "resnet": {
        "type": "BB",
        "linear_type": "aware",
        "use_first_resnet": False,
    },
    "layer_norm": {
        "pre_norm_type": None,
        "final_norm_type": None,
        "use_first_pre_norm": False,
    },
    "product_basis": {
        "type": "cgtp",
        "bias": True,
        "l1l2": None,
        "correlation": 2,
        "return_components": None,
        "num_expert": None,
        "num_channel_per_expert": None,
        "use_shared_expert": False,
        "nonlinear": None,
        "scalar_act": None,
        "agnostic": False,
    },
    "readout_emlp": {
        "bias": False,
        "hidden": [16],
        "use_alllayer": False,
        "use_one_body_magmoms": True,
    },
    "scale_shift": {
        "enable": True,
        "scale_type": "rms_forces",
        "shift_type": None,
        "scale_trainable": False,
        "shift_trainable": False,
        "all_atoms": False,
        "scale_zbl": True,
    },
    "short_range": {
        "zbl": {
            "enable": False,
            "trainable": False,
        }
    },
    "long_range": {
        "les": {
            "enable": False,
            "les_arguments": {
                "sigma": 1.0,
                "dl": 2.0,
                "remove_self_interaction": True,
                "use_dipole": False,
                "use_quad": False,
                "use_induced_charge": False,
                "use_induced_dipole": False,
                "use_anisotropic_polarizability": False,
                "is_periodic": None,
                "N_max": 10,
                "use_epsilon_r_scaling": False,
            },
        },
    },
    "universal_embedding": {
        "electric_field": {
            "enable": False,
            "normalizer": 1.0,
        },
        "magnetic_field": {
            "enable": False,
            "normalizer": 1.0,
        },
    },
    "special": {
        "direct_hessian": {
            "cutoff": 16.0,
            "interaction_type": "cgtp",
        },
        "charges": {
            "method": "lagrangian",
        },
    },
    "dropout": {
        "use_first_dropout": False,
        "stochastic_depth": 0.0,
    },
}


def recursive_update(
    cfg: Mapping[str, Any],
    *,
    default: dict[str, Any] = DEFAULT_MODEL_CONFIG,
) -> dict[str, Any]:
    """Merge nested model defaults with user overrides, retaining extra keys."""
    result = {**default, **cfg}
    for key, value in default.items():
        if key not in cfg:
            result[key] = deepcopy(value)
        elif isinstance(value, dict) and isinstance(cfg[key], Mapping):
            result[key] = recursive_update(cfg[key], default=value)
    return result


def get_invariant_property(ue: dict) -> list[str]:
    invariant_property = []
    for p, v in ue.items():
        if v["enable"] and PROPERTIES[p]["rank"] == 0:
            invariant_property.append(p)
    return invariant_property


def get_equivariant_property(ue: dict) -> list[str]:
    equivariant_property = []
    for p, v in ue.items():
        if v["enable"] and PROPERTIES[p]["rank"] > 0:
            equivariant_property.append(p)
    return equivariant_property


def check_model_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Merge model defaults, validate options, and resolve per-layer settings."""
    cfg = recursive_update(cfg)
    radial = cfg["radial_basis"]
    check_radial_selection(radial["num_radial_basis"], radial["max_eigenvalue"])
    if "jn_max_eigenvalue" in radial:
        raise ValueError("Set radial_basis.max_eigenvalue instead of jn_max_eigenvalue.")
    if radial["radial_basis"] != "spb" and radial["use_interpolation"]:
        raise ValueError("use_interpolation requires radial_basis='spb'.")
    atomic_basis = cfg["atomic_basis"]
    product_basis = cfg["product_basis"]

    if cfg["mlp_type"] not in MLP:
        raise ValueError("mlp_type must be 'mlp_v1' or 'mlp_v2'.")
    if cfg["layer_norm"]["final_norm_type"] == "const_norm":
        raise ValueError(
            "layer_norm.final_norm_type='const_norm' is not supported. "
            "Use 'const_norm' only for layer_norm.pre_norm_type."
        )
    if "multi_cutoff" in radial:
        raise ValueError(
            "Unsupported option radial_basis.multi_cutoff. Set model.config.cutoff instead."
        )
    if "mlp_type" in radial:
        raise ValueError("Set model.config.mlp_type instead of radial_basis.mlp_type.")
    if radial["mlp_norm"] not in (None, "identity", "layer_norm", "rms_norm"):
        raise ValueError(
            "radial_basis.mlp_norm must be null, 'identity', 'layer_norm' or 'rms_norm'."
        )
    embedding = cfg["edge_embedding"]["type"]
    update = cfg["edge_update"]["type"]
    element_paths = embedding in ("element", "element2")
    if element_paths and update != "identity":
        raise ValueError("Element path embeddings require edge_update.type='identity'.")
    if radial["apply_cutoff"] is not False:
        for enabled, option in (
            (element_paths, f"edge_embedding.type='{embedding}'"),
            (atomic_basis["scatter_norm"] == "cutoff", "scatter_norm='cutoff'"),
        ):
            if enabled:
                raise ValueError(f"{option} requires radial_basis.apply_cutoff=false.")

    magnetic_lmax = cfg["angular_basis"]["magnetic_Lmax"]
    if not isinstance(magnetic_lmax, int) or magnetic_lmax < 1:
        raise ValueError("angular_basis.magnetic_Lmax must be a positive integer.")
    num_mag_radial_basis = radial["num_mag_radial_basis"]
    if not isinstance(num_mag_radial_basis, int) or num_mag_radial_basis < 1:
        raise ValueError("radial_basis.num_mag_radial_basis must be positive.")

    if cfg["max_neighbors"] is not None:
        raise ValueError(
            "TACE does not truncate neighbor lists. Set `max_neighbors: null`."
        )

    statistics = cfg["statistics"]
    cfg["atomic_numbers"] = sorted(
        {z for stats in statistics for z in stats["atomic_numbers"]}
    )
    cfg["avg_num_neighbors"] = sum(s["avg_num_neighbors"] for s in statistics) / len(
        statistics
    )

    if isinstance(cfg["fidelity"], Mapping):
        cfg["fidelity"] = [cfg["fidelity"]]
    magnetic_scales = []
    for idx, fidelity_config in enumerate(cfg["fidelity"]):
        magnetic_scale = fidelity_config.get("magnetic_scale")
        if magnetic_scale is None:
            stats = statistics[idx] if idx < len(statistics) else {}
            magnetic_scale = stats.get("max_noncollinear_magmoms_norm_by_element")
            if magnetic_scale is None:
                magnetic_scales = []
                break
            if isinstance(magnetic_scale, Mapping):
                magnetic_scale = {
                    z: 1.2
                    * float(magnetic_scale.get(z, magnetic_scale.get(str(z), 0.0)))
                    + 0.1
                    for z in cfg["atomic_numbers"]
                }
            else:
                magnetic_scale = 1.2 * float(magnetic_scale) + 0.1
        if isinstance(magnetic_scale, Mapping):
            values = {}
            for z in cfg["atomic_numbers"]:
                value = magnetic_scale.get(z, magnetic_scale.get(str(z)))
                if value is None:
                    raise ValueError(f"magnetic_scale is missing atomic number {z}")
                values[z] = float(value)
        elif isinstance(magnetic_scale, Real):
            values = {z: float(magnetic_scale) for z in cfg["atomic_numbers"]}
        else:
            raise TypeError(
                "fidelity.magnetic_scale must be null, a scalar, or an "
                "element-dependent mapping"
            )
        if any(not math.isfinite(value) or value <= 0.0 for value in values.values()):
            raise ValueError("all magnetic_scale values must be finite and positive")
        magnetic_scales.append(values)
    cfg["magnetic_scale"] = magnetic_scales or None

    cfg["atomic_energies"] = (
        [stats["atomic_energy"] for stats in statistics]
        if "energy" in cfg["target_property"]
        else None
    )

    universal_embedding = {
        name: cfg["universal_embedding"][name]
        for name in DEFAULT_MODEL_CONFIG["universal_embedding"]
    }
    cfg["universal_embedding"] = universal_embedding
    cfg["invariant_property"] = get_invariant_property(universal_embedding)
    cfg["equivariant_property"] = get_equivariant_property(universal_embedding)

    for block, keys in (
        (atomic_basis, ("type", "nonlinear", "edge_nonlinear")),
        (product_basis, ("type", "correlation")),
    ):
        for key in keys:
            value = block[key]
            if isinstance(value, (int, str)) or value is None:
                block[key] = [value] * cfg["num_layers"]
            elif not isinstance(value, list):
                raise TypeError(
                    f"{key} must be a scalar or a list of per-layer values."
                )

    components = product_basis["return_components"]
    if components is not None:
        if not isinstance(components, list) or not all(
            isinstance(c, (int, str)) for c in components
        ):
            raise TypeError(
                "product_basis.return_components must be a list of degrees or irreps."
            )
        product_basis["return_components"] = [
            f"{c}{'o' if c % 2 else 'e'}" if isinstance(c, int) else c
            for c in components
        ]

    if (
        cfg["scale_shift"]["enable"]
        and not cfg["scale_shift"]["all_atoms"]
        and "energy" in cfg["target_property"]
    ):
        logging.warning(
            "scale_shift.all_atoms=False is a legacy option that can violate "
            "energy additivity and the dissociation limit by treating isolated "
            "atoms differently when evaluated alone. Do not use it for new "
            "models; set all_atoms=True."
        )
    if atomic_basis["scatter_norm"] == "no_cutoff_density":
        logging.warning(
            "scatter_norm='no_cutoff_density' is a legacy option that can "
            "cause energy jumps when neighbors cross the cutoff and break "
            "force/stress consistency. Do not use it for new models; use "
            "scatter_norm='density' instead."
        )

    return cfg

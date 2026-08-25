################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Any

from ...dataset.quantity import PROPERTY

DEFAULT_MODEL_CONFIG = {
    "mmax": 2,
    "Lmax": 2,
    "lmax": 3,
    "mag_Lmax": 1,
    "parity": False,
    "num_channel": 64,
    "num_layers": 2,
    "target_property": ["energy", "forces"],
    "embedding_property": [],
    "fidelity": {
        "name": "PBE",
        "atomic_energy": None,
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
    "radial_basis": {
        "bias": False,
        "radial_basis": "j0",
        "num_radial_basis": 8,
        "num_mag_radial_basis": 10,
        "distance_transform": None,
        "cutoff_fn": "c2poly",
        "polynomial_cutoff": 5,
        "order": 0,
        "trainable": False,
        "apply_cutoff": True,
        "hidden": [64, 64, 64],
        "gaussian_width": 2.0,
    },
    "angular_basis": {},
    "atomic_basis": {
        "type": "cgtp",
        "l1l2": None,
        "scatter_norm": "avg_num_neighbors",
        "nonlinear": "gate",
        "edge_nonlinear": "so2_sigmoid_gate",
        "use_o2_asymmetric_contraction": False,
        "use_radial_rotary_attention": False,
        "num_head": None,
        "node_wise_hidden": None,
        "edge_ace_hidden": None,
        "edge_wise_hidden": None,
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
        "l1l2": None,
        "correlation": 2,
        "return_components": None,
        "num_expert": None,
        "num_channel_per_expert": None,
        "use_shared_expert": False,
        "nonlinear": None,
        "agnostic": False,
    },
    "readout_emlp": {
        "bias": False,
        "hidden": [16],
        "use_alllayer": False,
        "use_uie": False,
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
        "charges": {
            "enable": False,
            "act": "silu",
        },
        "total_charge": {
            "enable": False,
            "act": "silu",
        },
        "spin_multiplicity": {
            "enable": False,
            "num_embeddings": -1,
        },
        "initial_noncollinear_magmoms": {
            "enable": False,
            "normalizer": 1.0,
        },
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
    cfg: dict[str, Any],
    *,
    default: dict[str, Any] = DEFAULT_MODEL_CONFIG,
) -> dict[str, Any]:
    """
    Recursively update `default` with `cfg`.

    Rules:
    - if key not in cfg: keep default
    - if both values are dict: recurse
    - otherwise: cfg overrides default
    """
    result = {}

    for key, default_val in default.items():
        if key not in cfg:
            result[key] = default_val
            continue

        cfg_val = cfg[key]

        if isinstance(default_val, dict) and isinstance(cfg_val, dict):
            result[key] = recursive_update(cfg_val, default=default_val)
        else:
            result[key] = cfg_val

    # allow cfg to introduce new keys
    for key, cfg_val in cfg.items():
        if key not in result:
            result[key] = cfg_val

    return result


def get_invariant_property(ue: dict) -> list[str]:
    invariant_property = []
    for p, v in ue.items():
        if v["enable"] and PROPERTY[p]["rank"] == 0:
            invariant_property.append(p)
    return invariant_property


def get_equivariant_property(ue: dict) -> list[str]:
    equivariant_property = []
    for p, v in ue.items():
        if v["enable"] and PROPERTY[p]["rank"] > 0:
            equivariant_property.append(p)
    return equivariant_property


def check_model_config(cfg: dict[str, Any]):

    # Update default config with user config
    cfg = recursive_update(cfg)

    if (
        not isinstance(cfg["mag_Lmax"], int)
        or isinstance(cfg["mag_Lmax"], bool)
        or not 1 <= cfg["mag_Lmax"] <= cfg["Lmax"]
    ):
        raise ValueError("mag_Lmax must satisfy 1 <= mag_Lmax <= Lmax.")

    if cfg.get("max_neighbors") is not None:
        raise ValueError(
            "TACE does not "
            "truncate neighbor lists. Set `max_neighbors: null` in the model "
            "configuration."
        )

    # Update statistics info
    cfg["atomic_numbers"] = sorted(
        {z for s in cfg["statistics"] for z in s["atomic_numbers"]}
    )
    cfg["avg_num_neighbors"] = sum(
        s["avg_num_neighbors"] for s in cfg["statistics"]
    ) / len(cfg["statistics"])

    # TODO
    magnetic_statistics = [
        stats["magmoms_norm_by_element"]
        for stats in cfg["statistics"]
        if "magmoms_norm_by_element" in stats
    ]
    cfg["magmoms_norm_by_element"] = (
        {
            z: max(
                float(stats.get(z, stats.get(str(z), 0.0)))
                for stats in magnetic_statistics
            )
            for z in cfg["atomic_numbers"]
        }
        if magnetic_statistics
        else None
    )

    cfg["atomic_energies"] = (
        [stats["atomic_energy"] for stats in cfg["statistics"]]
        if "energy" in cfg["target_property"]
        else None
    )

    # Universal_embedding
    cfg["invariant_property"] = get_invariant_property(cfg["universal_embedding"])
    cfg["equivariant_property"] = get_equivariant_property(cfg["universal_embedding"])

    # Update to list use num_layers
    def _to_list(x):
        if isinstance(x, int) or isinstance(x, str) or x is None:
            return [x for _ in range(cfg["num_layers"])]
        assert isinstance(x, list)
        return x

    # cfg['Lmax'] = _to_list(cfg['Lmax'])
    cfg["product_basis"]["correlation"] = _to_list(cfg["product_basis"]["correlation"])
    cfg["atomic_basis"]["type"] = _to_list(cfg["atomic_basis"]["type"])
    cfg["product_basis"]["type"] = _to_list(cfg["product_basis"]["type"])
    cfg["atomic_basis"]["nonlinear"] = _to_list(cfg["atomic_basis"]["nonlinear"])
    cfg["atomic_basis"]["edge_nonlinear"] = _to_list(
        cfg["atomic_basis"]["edge_nonlinear"]
    )
    # cfg['product_basis']['nonlinear'] = _to_list(cfg['product_basis']['nonlinear'])

    # if cfg['parity']: assert 'so2' not in cfg['atomic_basis']['type'], "When using SO(2) Interaction, set parity: false"
    components = cfg["product_basis"]["return_components"]
    if isinstance(components, list):
        for idx, int_or_irrep in enumerate(components):
            if isinstance(int_or_irrep, int):
                parity = "e" if int_or_irrep % 2 == 0 else "o"
                components[idx] = f"{int_or_irrep}{parity}"
            else:
                assert isinstance(int_or_irrep, str)
    else:
        assert components is None
    cfg["product_basis"]["return_components"] = components

    return cfg

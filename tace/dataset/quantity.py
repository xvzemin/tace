################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple

from .utils import (
    default_value_for_hessian,
    default_value_for_rank0_atom,
    default_value_for_rank0_graph,
    default_value_for_rank1_atom,
    default_value_for_rank1_graph,
    default_value_for_rank2_atom,
    default_value_for_rank2_graph,
    default_value_for_rank3_atom,
    default_value_for_rank3_graph,
    default_value_for_rank4_atom,
    default_value_for_rank4_graph,
    shape_fn_for_abs_fc_mag,
    shape_fn_for_direct_diagonal_hessian,
    shape_fn_for_hessian,
    voigt_to_matrix,
)

PROPERTY = {
    "fidelity_idx": {
        "ase_name": None,
        "type": "int",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",
        "abbreviation": "F_IDX",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "energy": {
        "ase_name": "energy",
        "type": "float",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",
        "abbreviation": "E",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "forces": {
        "ase_name": "forces",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1o",
        "abbreviation": "F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": ["energy"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["positions"],
    },
    "edge_forces": {
        "ase_name": None,
        "type": "float",
        "scope": "per-edge",
        "rank": 1,
        "irreps": "1x1o",
        "abbreviation": "EDGE_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,  # placeholder
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": False,  # can be embedded through uee to achice DeNs
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["edge_vector"],
    },
    "direct_forces": {
        "ase_name": "forces",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1o",
        "abbreviation": "D_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "hessian": {
        "ase_name": None,
        "type": "float",
        "scope": "per-edge",
        "rank": 2,
        "irreps": "1x0e+1x2e",
        "abbreviation": "HESSIAN",
        "shape": {
            "in_data": (-1,),
            "shape_fn": shape_fn_for_hessian,
        },
        "default_value_fn": default_value_for_hessian,
        "must_be_with": ["energy", "forces"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ["positions"],
    },
    # "direct_diagonal_hessian": {
    #     "ase_name": None,
    #     'type': 'float',
    #     "scope": "per-atom",
    #     "rank": 2,
    #     "irreps": '1x0e+1x2e',
    #     "abbreviation": "D_DIAG_H",
    #     "shape": {
    #         "in_data": (-1, 3, 3),
    #         "shape_fn": shape_fn_for_direct_diagonal_hessian,
    #     },
    #     "default_value_fn": default_value_for_rank2_atom,
    #     "must_be_with": [],
    #     "enable_prediction": True,
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    "stress": {
        "ase_name": "stress",
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        # "requires_grad_with": ['displacement'],
        "requires_grad_with": [],  # manual set
    },
    "direct_stress": {
        "ase_name": "stress",
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "D_S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "virials": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        # "requires_grad_with": ['displacement'],
        "requires_grad_with": [],  # manual set
    },
    "direct_virials": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "D_V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "atomic_stresses": {
        "ase_name": "stresses",
        "type": "float",
        "scope": "per-atom",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "A_S",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["edge_vector"],
    },
    "atomic_virials": {
        "ase_name": None,
        "type": "float",
        "scope": "per-atom",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "A_V",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["edge_vector"],
    },
    "direct_dipole": {
        "ase_name": "dipole",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": ["charges"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_dipole": {
        "ase_name": "dipole",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["electric_field"],
    },
    "polarization": {
        "ase_name": "polarization",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "abbreviation": "P",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["electric_field"],
    },
    "direct_polarizability": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_polarizability": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": ["direct_dipole", "conservative_dipole", "polarization"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ["electric_field"],
    },
    "born_effective_charges": {
        "ase_name": "born_effective_charges",
        "type": "float",
        "scope": "per-atom",
        "rank": 2,
        "irreps": "1x2e",
        "abbreviation": "BEC",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": ["direct_dipole", "conservative_dipole", "polarization"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ["electric_field", "positions"],
    },
    # "magnetization": {
    #     "ase_name": None,
    #     'type': 'float',
    #     "scope": "per-system",
    #     "rank": 1,
    #     "irreps": '1x1e',
    #     "abbreviation": "M",
    #     "shape": {
    #         "in_data": (1, 3),
    #         "shape_fn": None,
    #     },
    #     "default_value_fn": default_value_for_rank1_graph,
    #     "must_be_with": [],
    #     "enable_prediction": True,
    #     "enable_embedding": False,
    #     "first_derivative": True,
    #     "second_derivative": False,
    #     "requires_grad_with": ['magnetic_field'],
    # },
    # "magnetic_susceptibility": {
    #     "ase_name": None,
    #     'type': 'float',
    #     "scope": "per-system",
    #     "rank": 2,
    #     "irreps": '1x2e', # TODO, check
    #     "abbreviation": "CHI_M",
    #     "shape": {
    #         "in_data": (1, 3, 3),
    #         "shape_fn": None,
    #     },
    #     "default_value_fn": default_value_for_rank2_graph,
    #     "must_be_with": ['magnetization'],
    #     "enable_prediction": True,
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": True,
    #     "requires_grad_with": ['magnetic_field'],
    # },
    "charges": {
        "ase_name": "charges",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",
        "abbreviation": "C",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": ["total_charge"],
        "enable_prediction": True,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "total_charge": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",
        "abbreviation": "TC",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    # "spin_multiplicity": {
    #     "ase_name": None,
    #     'type': 'int',
    #     "scope": "per-system",
    #     "rank": 0,
    #     "irreps": '1x0e',
    #     "abbreviation": "SM",
    #     "shape": {
    #         "in_data": (1,),
    #         "shape_fn": None,
    #     },
    #     "default_value_fn": default_value_for_rank0_graph,
    #     "must_be_with": [],
    #     "enable_prediction": False,
    #     "enable_embedding": True,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    "initial_collinear_magmoms": {
        "ase_name": "initial_magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",  # TODO, check
        "abbreviation": "I_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "initial_noncollinear_magmoms": {
        "ase_name": "initial_magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1e",  # TODO, check
        "abbreviation": "I_NC_MAG",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "final_collinear_magmoms": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",  # TODO, check
        "abbreviation": "F_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "abs_final_collinear_magmoms": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",  # TODO, check
        "abbreviation": "ABS_F_C_MAG",
        "shape": {
            "in_data": (-1,),
            "shape_fn": shape_fn_for_abs_fc_mag,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "final_noncollinear_magmoms": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1e",  # TODO, check
        "abbreviation": "F_NC_MAG",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "collinear_magnetic_forces": {
        "ase_name": None,
        "type": "float",
        "scope": "per-atom",
        "rank": 0,
        "irreps": "1x0e",  # TODO, check
        "abbreviation": "C_MAG_F",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": ["initial_collinear_magmoms"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["initial_collinear_magmoms"],
    },
    "noncollinear_magnetic_forces": {
        "ase_name": None,
        "type": "float",
        "scope": "per-atom",
        "rank": 1,
        "irreps": "1x1e",  # TODO, check
        "abbreviation": "NC_MAG_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": ["initial_noncollinear_magmoms"],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ["initial_noncollinear_magmoms"],
    },
    "total_collinear_magmom": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-system",
        "rank": 0,
        "irreps": "1x0e",  # TODO, check
        "abbreviation": "TCM",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "total_noncollinear_magmom": {
        "ase_name": "magmoms",
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1e",  # TODO, check
        "abbreviation": "TNCM",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": [],
        "enable_prediction": True,
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "electric_field": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1o",
        "abbreviation": "EF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "magnetic_field": {
        "ase_name": None,
        "type": "float",
        "scope": "per-system",
        "rank": 1,
        "irreps": "1x1e",
        "abbreviation": "MF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": [],
        "enable_prediction": False,
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    # "temperature": {
    #     "ase_name": None,
    #     'type': 'float',
    #     "scope": "per-system",
    #     "rank": 0,
    #     "irreps": '1x0e',
    #     "abbreviation": "TEMP",
    #     "shape": {
    #         "in_data": (1,),
    #         "shape_fn": None,
    #     },
    #     "default_value_fn": default_value_for_rank0_graph,
    #     "must_be_with": [],
    #     "enable_prediction": False,
    #     "enable_embedding": True,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    # "electron_temperature": {
    #     "ase_name": None,
    #     'type': 'float',
    #     "scope": "per-system",
    #     "rank": 0,
    #     "irreps": '1x0e',
    #     "abbreviation": "E_TEMP",
    #     "shape": {
    #         "in_data": (1,),
    #         "shape_fn": None,
    #     },
    #     "default_value_fn": default_value_for_rank0_graph,
    #     "must_be_with": [],
    #     "enable_prediction": False,
    #     "enable_embedding": True,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
}

SUPPORT_PREDICT_PROPERTY = [k for k, v in PROPERTY.items() if v["enable_prediction"]]
SUPPORT_EMBEDDING_PROPERTY = [k for k, v in PROPERTY.items() if v["enable_embedding"]]
KEYS = {f"{k}_key": k for k in PROPERTY}


# should be delete in future versions TODO
class DefaultKeys(Enum):
    # basic
    ENERGY = "energy"
    FORCES = "forces"
    STRESS = "stress"
    VIRIALS = "virials"

    HESSIAN = "hessian"
    EDGE_FORCES = "edge_forces"
    ATOMIC_VIRIALS = "atomic_virials"
    ATOMIC_STRESSES = "atomic_stresses"

    # direct property
    DIRECT_FORCES = "direct_forces"
    DIRECT_STRESS = "direct_stress"
    DIRECT_VIRIALS = "direct_virials"
    DIRECT_DIPOLE = "direct_dipole"
    DIRECT_POLARIZABILITY = "direct_polarizability"
    # DIRECT_DIAGONAL_HESSIAN = "direct_hessian"

    # charges
    CHARGES = "charges"
    TOTAL_CHARGE = "total_charge"

    # external field
    ELECTRIC_FIELD = "electric_field"
    MAGNETIC_FIELD = "magnetic_field"

    CONSERVATIVE_DIPOLE = "conservative_dipole"
    CONSERVATIVE_POLARIZABILITY = "conservative_polarizability"
    BORN_EFFECTIVE_CHARGES = "born_effective_charges"  # do not consider LES
    # MAGNETIZATION = "magnetization"
    # MAGNETIC_SUSCEPTIBILITY = "magnetic_susceptibility"
    POLARIZATION = "polarization"

    # MAG
    INITIAL_COLLINEAR_MAGMOMS = "initial_collinear_magmoms"
    INITIAL_NONCOLLINEAR_MAGMOMS = "initial_noncollinear_magmoms"
    FINAL_COLLINEAR_MAGMOMS = "final_collinear_magmoms"
    ABS_FINAL_COLLINEAR_MAGMOMS = "abs_final_collinear_magmoms"
    FINAL_NONCOLLINEAR_MAGMOMS = "final_noncollinear_magmoms"
    COLLINEAR_MAGNETIC_FORCES = "collinear_magnetic_forces"
    NONCOLLINEAR_MAGNETIC_FORCES = "noncollinear_magnetic_forces"
    TOTAL_COLLINEAR_MAGMOM = "total_collinear_magmom"
    TOTAL_NONCOLLINEAR_MAGMOM = "total_noncollinear_magmom"

    # only for embedding
    FIDELITY_IDX = "fidelity_idx"
    # TEMPERATURE = "temperature"
    # ELECTRON_TEMPERATURE = "electron_temperature"
    # SPIN_MULTIPLICITY = "spin_multiplicity"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict


@dataclass
class KeySpecification:
    """Modify from MACE to simplify reading property"""

    info_keys: Dict[str, str] = field(default_factory=dict)
    arrays_keys: Dict[str, str] = field(default_factory=dict)

    def update(
        self,
        info_keys: Optional[Dict[str, str]] = None,
        arrays_keys: Optional[Dict[str, str]] = None,
    ):
        if info_keys is not None:
            self.info_keys.update(info_keys)
        if arrays_keys is not None:
            self.arrays_keys.update(arrays_keys)
        return self

    @classmethod
    def from_defaults(cls):
        instance = cls()
        return update_keyspec_from_kwargs(instance, DefaultKeys.keydict())


def update_keyspec_from_kwargs(
    keyspec: KeySpecification, keydict: Dict[str, str]
) -> KeySpecification:
    infos = [f"{k}_key" for k, v in PROPERTY.items() if (v["scope"] != "per-atom")]
    arrays = [f"{k}_key" for k, v in PROPERTY.items() if (v["scope"] == "per-atom")]
    info_keys = {}
    arrays_keys = {}
    for key in infos:
        if key in keydict:
            info_keys[key[:-4]] = keydict[key]
    for key in arrays:
        if key in keydict:
            arrays_keys[key[:-4]] = keydict[key]
    keyspec.update(info_keys=info_keys, arrays_keys=arrays_keys)
    return keyspec


def get_target_property(cfg: Dict) -> List[str]:
    loss_property = cfg["loss"].get("loss_property", None)
    assert isinstance(loss_property, list)
    assert set(loss_property).issubset(list(PROPERTY))
    return loss_property


def get_embedding_property(cfg: Dict) -> List[str]:
    embedding_property = []
    for p, v in cfg["model"]["config"].get("universal_embedding", {}).items():
        if v["enable"]:
            assert p in SUPPORT_EMBEDDING_PROPERTY, (
                f"Universal_embedding allowed property are {SUPPORT_EMBEDDING_PROPERTY}"
            )
            embedding_property.append(p)
    atomic_basis_type = cfg["model"]["config"].get("atomic_basis", {}).get("type")
    if isinstance(atomic_basis_type, str):
        atomic_basis_type = [atomic_basis_type]
    if atomic_basis_type is not None and any(
        interaction in {"o3_w6j_mag", "o2_mag"} for interaction in atomic_basis_type
    ):
        if "initial_noncollinear_magmoms" not in embedding_property:
            embedding_property.append("initial_noncollinear_magmoms")
    return embedding_property


def get_need_property(
    target_property: List[str] = [],
    embedding_property: List[str] = [],
    training: bool = False,
) -> List[str]:
    we_should_read = embedding_property
    if training:
        we_should_read = we_should_read + target_property
    joint_property = []
    for name in we_should_read:
        joint_property += PROPERTY[name]["must_be_with"]
    return list(set(joint_property + we_should_read))


# For Metrics

SPECIAL_METRIC_PROPERTY = [
    "polarization",  # for definition
    "abs_final_collinear_magmoms",  # for absolute value
    "direct_forces",  # for DeNS
    "forces",  # for DeNS
]

# INTENSIVE
EXTENSIVE_PROPERTY = ["stress", "direct_stress"]
MAE_PROPERTY = [p for p in SUPPORT_PREDICT_PROPERTY if p not in SPECIAL_METRIC_PROPERTY]
RMSE_PROPERTY = [
    p for p in SUPPORT_PREDICT_PROPERTY if p not in SPECIAL_METRIC_PROPERTY
]
MAE_PER_ATOM_PROPERTY = [
    p
    for p, v in PROPERTY.items()
    if p not in SPECIAL_METRIC_PROPERTY
    and v["scope"] == "per-system"
    and p not in EXTENSIVE_PROPERTY
]
RMSE_PER_ATOM_PROPERTY = [
    p
    for p, v in PROPERTY.items()
    if p not in SPECIAL_METRIC_PROPERTY
    and v["scope"] == "per-system"
    and p not in EXTENSIVE_PROPERTY
]


fields = {f"compute_{k}": False for k, v in PROPERTY.items()}


@dataclass
class ComputeFlag:
    __annotations__ = {k: bool for k in fields}
    locals().update(fields)

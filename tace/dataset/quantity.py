################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import logging
from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass, field


from .utils import (
    voigt_to_matrix,
    shape_fn_for_hessians,
    default_value_for_hessians,
    default_value_for_rank0_atom,
    default_value_for_rank1_atom,
    default_value_for_rank2_atom,
    default_value_for_rank3_atom,
    default_value_for_rank4_atom,
    default_value_for_rank0_graph,
    default_value_for_rank1_graph,
    default_value_for_rank2_graph,
    default_value_for_rank3_graph,
    default_value_for_rank4_graph,
)


# TODO BUG only update exist keys, keep the default keys

PROPERTY = {
    "energy": {
        "rank": 0,
        "type": "graph",
        "abbreviation": "E",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "forces": {
        "rank": 1,
        "type": "atom",
        "abbreviation": "F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {
            1: ['energy'],
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False, 
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['positions'],
    },
    "direct_forces": {
        "rank": 1,
        "type": "atom",
        "abbreviation": "D_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False, 
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "edge_forces": {
        "rank": 1,
        "type": "edge",
        "abbreviation": "EDGE_F",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_atom, # placeholder
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": False, # can be embedded through uee to achice DeNs
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['edge_vector'],
    },
    "hessians": {
        "rank": 2,
        "type": "graph", # type indeed is atom_atom, assume it belongs to graph
        "abbreviation": "HESSIAN",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": shape_fn_for_hessians,
        },
        'class': 'float',
        "default_value_fn": default_value_for_hessians,
        "must_be_with": {
            1: ['energy', 'forces'],
        },
        "conflict_with": {},
        "enable_prediction": False, # In principle, it is also supported, but the code is a bit cumbersome.
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['positions'],
    },
    "direct_hessians": {
        "rank": 2,
        "type": "graph", # type indeed is atom_atom, assume it belongs to graph
        "abbreviation": "D_HESSIAN",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": shape_fn_for_hessians,
        },
        'class': 'float',
        "default_value_fn": default_value_for_hessians,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False, # In principle, it is also supported, but the code is a bit cumbersome.
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "stress": {
        "rank": 2,
        "type": "graph",
        "abbreviation": "S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        # "requires_grad_with": ['displacement'],
        "requires_grad_with": [], # manual set
    },
    "direct_stress": {
        "rank": 2,
        "type": "graph",
        "abbreviation": "D_S",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "virials": {
        "rank": 2,
        "type": "graph",
        "abbreviation": "V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        # "requires_grad_with": ['displacement'],
        "requires_grad_with": [], # manual set
    },
    "direct_virials": {
        "rank": 2,
        "type": "graph",
        "abbreviation": "D_V",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": voigt_to_matrix,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "atomic_stresses": {
        "rank": 2,
        "type": "atom",
        "abbreviation": "A_S",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['edge_vector'],
    },
    "atomic_virials": {
        "rank": 2,
        "type": "atom",
        "abbreviation": "A_V",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['edge_vector'],
    },
    "direct_dipole": {
        "rank": 1,
        "type": "graph",
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {
            1: ['charges'],
        },
        "conflict_with": {
            1: ["polarization"]
        },
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_dipole": {
        "rank": 1,
        "type": "graph",
        "abbreviation": "D",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {
            1: ["polarization"]
        },
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['electric_field'],
    },
    "polarization": {
        "rank": 1,
        "type": "graph",
        "abbreviation": "P",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": [], 
        "must_be_with": {},
        "conflict_with": {
            1: ["direct_dipole", "conservative_dipole"]
        },
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['electric_field'],
    },
    "direct_polarizability": {
        "rank": 2,
        "type": "graph",
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "conservative_polarizability": {
        "rank": 2,
        "type": "graph",
        "abbreviation": "ALPHA",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {
            1: ["direct_dipole"],
            2: ["conservative_dipole"],
            3: ["polarization"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['electric_field'],
    },
    "born_effective_charges": {
        "rank": 2,
        "type": "atom",
        "abbreviation": "BEC",
        "shape": {
            "in_data": (-1, 3, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_atom,
        "must_be_with": {
            1: ["direct_dipole"],
            2: ["conservative_dipole"],
            3: ["polarization"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['electric_field', 'positions'],
    },
    "magnetization": {
        "rank": 1,
        "type": "graph",
        "abbreviation": "M",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['magnetic_field'],
    },
    "magnetic_susceptibility": {
        "rank": 2,
        "type": "graph",
        "abbreviation": "CHI_M",
        "shape": {
            "in_data": (1, 3, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank2_graph,
        "must_be_with": {
            1: ['magnetization']
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": False,
        "second_derivative": True,
        "requires_grad_with": ['magnetic_field'],
    },
    "charges": {
        "rank": 0,
        "type": "atom",
        "abbreviation": "C",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "total_charge": {
        "rank": 0,
        "type": "graph",
        "abbreviation": "TC",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "spin_multiplicity": {
        "rank": 0,
        "type": "graph",
        "abbreviation": "SM",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        'class': 'int',
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "magmoms_0": {
        "rank": 0,
        "type": "atom",
        "abbreviation": "MAG_0",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "magmoms_1": {
        "rank": 1,
        "type": "atom",
        "abbreviation": "MAG_1",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "magnetic_forces_0": {
        "rank": 0,
        "type": "atom",
        "abbreviation": "MAG_F_0",
        "shape": {
            "in_data": (-1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_atom,
        "must_be_with": {
            1: ["magmoms_0"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['magmoms_0'],
    },
    "magnetic_forces_1": {
        "rank": 1,
        "type": "atom",
        "abbreviation": "MAG_F_1",
        "shape": {
            "in_data": (-1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_atom,
        "must_be_with": {
            1: ["magmoms_1"]
        },
        "conflict_with": {},
        "enable_prediction": True,   
        "enable_embedding": False,
        "first_derivative": True,
        "second_derivative": False,
        "requires_grad_with": ['magmoms_1'],
    },
    "total_magmom_0": {
        "rank": 0,
        "type": "graph",
        "abbreviation": "TM_0",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "total_magmom_1": {
        "rank": 1,
        "type": "graph",
        "abbreviation": "TM",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "electric_field": {
        "rank": 1,
        "type": "graph",
        "abbreviation": "EF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "magnetic_field": {
        "rank": 1,
        "type": "graph",
        "abbreviation": "MF",
        "shape": {
            "in_data": (1, 3),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank1_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "level": {
        "rank": 0,
        "type": "graph",
        "abbreviation": "LEVEL",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        'class': 'int',
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "temperature": {
        "rank": 0,
        "type": "graph",
        "abbreviation": "TEMP",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    "electron_temperature": {
        "rank": 0,
        "type": "graph",
        "abbreviation": "E_TEMP",
        "shape": {
            "in_data": (1,),
            "shape_fn": None,
        },
        'class': 'float',
        "default_value_fn": default_value_for_rank0_graph,
        "must_be_with": {},
        "conflict_with": {},
        "enable_prediction": False,   
        "enable_embedding": True,
        "first_derivative": False,
        "second_derivative": False,
        "requires_grad_with": [],
    },
    # "nuclear_shielding": {
    #     "rank": 2,
    #     "type": "atom",
    #     "abbreviation": "NS",
    #     "shape": {
    #         "in_data": (-1, 3, 3),
    #         "shape_fn": None,
    #     },
    #     'class': 'float',
    #     "default_value_fn": default_value_for_rank2_atom,
    #     "must_be_with": {},
    #     "conflict_with": {},
    #     "enable_prediction": True,   
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    # "nuclear_chemical_shift": {
    #     "rank": 0,
    #     "type": "atom",
    #     "abbreviation": "NCS",
    #     "shape": {
    #         "in_data": (-1,),
    #         "shape_fn": None,
    #     },
    #     'class': 'float',
    #     "default_value_fn": default_value_for_rank0_atom,
    #     "must_be_with": {},
    #     "conflict_with": {},
    #     "enable_prediction": True,   
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    # "elasticity_tensor": {
    #     "rank": 4,
    #     "type": "graph",
    #     "abbreviation": "ET",
    #     "shape": {
    #         "in_data": (1, 3, 3, 3, 3),
    #         "shape_fn": None,
    #     },
    #     'class': 'float',
    #     "default_value_fn": default_value_for_rank4_graph,
    #     "must_be_with": {},
    #     "conflict_with": {},
    #     "enable_prediction": True,   
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    # "crystal_system": {
    #     "rank": 0,
    #     "type": "graph",
    #     "abbreviation": "CS",
    #     "shape": {
    #         "in_data": (1,),
    #         "shape_fn": None,
    #     },
    #     'class': 'int',
    #     "default_value_fn": default_value_for_rank0_graph,
    #     "must_be_with": {},
    #     "conflict_with": {},
    #     "enable_prediction": False,   
    #     "enable_embedding": True,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    # "hill_bulk_modulus": {
    #     "rank": 0,
    #     "type": "graph",
    #     "abbreviation": "HILL_K",
    #     "shape": {
    #         "in_data": (1,),
    #         "shape_fn": None,
    #     },
    #     'class': 'float',
    #     "default_value_fn": default_value_for_rank0_graph,
    #     "must_be_with": {
    #         1: ['elasticity_tensor'],
    #     },
    #     "conflict_with": {},
    #     "enable_prediction": True,   
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    # "hill_shear_modulus": {
    #     "rank": 0,
    #     "type": "graph",
    #     "abbreviation": "HILL_G",
    #     "shape": {
    #         "in_data": (1,),
    #         "shape_fn": None,
    #     },
    #     'class': 'float',
    #     "default_value_fn": default_value_for_rank0_graph,
    #     "must_be_with": {
    #         1: ['elasticity_tensor'],
    #     },
    #     "conflict_with": {},
    #     "enable_prediction": True,   
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
    # "hill_young_modulus": {
    #     "rank": 0,
    #     "type": "graph",
    #     "abbreviation": "HILL_E",
    #     "shape": {
    #         "in_data": (1,),
    #         "shape_fn": None,
    #     },
    #     'class': 'float',
    #     "default_value_fn": default_value_for_rank0_graph,
    #     "must_be_with": {
    #         1: ['elasticity_tensor'],
    #     },
    #     "conflict_with": {},
    #     "enable_prediction": True,   
    #     "enable_embedding": False,
    #     "first_derivative": False,
    #     "second_derivative": False,
    #     "requires_grad_with": [],
    # },
}

SUPPORT_PREDICT_PROPERTY = [k for k, v in PROPERTY.items() if v['enable_prediction']]
UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY = [k for k, v in PROPERTY.items() if v['enable_embedding']]
EXTERNAL_FIELD_ALLOWED_PROPERTY = SUPPORT_PREDICT_PROPERTY  # not check


KEYS = {f"{k}_key": k for k in PROPERTY}

# should be delete in future versions TODO
class DefaultKeys(Enum):
    # basic
    ENERGY = "energy"
    FORCES = "forces"
    STRESS = "stress"
    VIRIALS = "virials"

    HESSIANS = "hessians"
    EDGE_FORCES = "edge_forces"
    ATOMIC_VIRIALS = "atomic_virials"
    ATOMIC_STRESSES = "atomic_stresses"
    
    # direct property
    # NUCLEAR_SHIELDING = "nuclear_shielding"
    # NUCLEAR_CHEMICAL_SHIFT = "nuclear_chemical_shift"
    # ELASTICITY_TENSOR = "elasticity_tensor"

    DIRECT_FORCES = "direct_forces"
    DIRECT_STRESS = "direct_stress"
    DIRECT_VIRIALS = "direct_virials"
    DIRECT_DIPOLE = "direct_dipole"
    DIRECT_POLARIZABILITY = "direct_polarizability"

    # charges
    CHARGES = "charges"
    TOTAL_CHARGE = "total_charge"

    # external field
    ELECTRIC_FIELD = "electric_field"
    MAGNETIC_FIELD = "magnetic_field"

    CONSERVATIVE_DIPOLE = "conservative_dipole"
    CONSERVATIVE_POLARIZABILITY = "conservative_polarizability"
    BORN_EFFECTIVE_CHARGES = "born_effective_charges" # do not consider LES
    MAGNETIZATION = "magnetization"
    MAGNETIC_SUSCEPTIBILITY = "magnetic_susceptibility"
    POLARIZATION = "polarization"

    # magnetic systems
    MAGMOMS_0 = "magmoms_0"
    MAGMOMS_1 = "magmoms_1"
    TOTAL_MAGMOM_0 = "total_magmom_0"
    TOTAL_MAGMOM_1 = "total_magmom_1"
    MAGNETIC_FORCES_0 = "magnetic_forces_0"
    MAGNETIC_FORCES_1 = "magnetic_forces_1"

    # only for embedding
    LEVEL = 'level'
    TEMPERATURE = "temperature"
    ELECTRON_TEMPERATURE = "electron_temperature"
    SPIN_MULTIPLICITY = "spin_multiplicity"

    @staticmethod
    def keydict() -> dict[str, str]:
        key_dict = {}
        for member in DefaultKeys:
            key_name = f"{member.name.lower()}_key"
            key_dict[key_name] = member.value
        return key_dict


@dataclass
class KeySpecification:
    '''Modify from MACE to simplify reading property'''
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
    '''Modify from MACE to simplify reading property'''
    infos = [f"{k}_key" for k, v in PROPERTY.items() if (v['type'] == 'graph')]
    arrays = [f"{k}_key" for k, v in PROPERTY.items() if (v['type'] != 'graph')] # not correct,but for convenience now
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
    """
    Automatically infer the physical quantities required for training from the loss function.
    Ensure no conflicting physical quantities appear simultaneously.
    """
    try:
        loss = str(cfg["loss"]["_target_"]).lower()
    except KeyError as e:
        raise KeyError("Missing 'cfg.loss._target_' field in configuration") from e
    loss_property = cfg["loss"].get("loss_property", None)
    if loss_property is None:
        logging.warning(
            "The argument `cfg.loss.loss_property` must be provided by the user. "
            "It is kept optional only for backward compatibility with earlier versions. "
            "Omitting it may lead to bugs when predicted physical property names share common prefixes or substrings."
        )
        loss_property = []
        for p in SUPPORT_PREDICT_PROPERTY:
            clean_property = p.replace("_", "")
            clean_property = p
            if clean_property.lower() in loss:
                loss_property.append(p)

        for p in loss_property:
            conflict_with = PROPERTY[p]['conflict_with']
            for _, conflict_ps in conflict_with.items():
                for conflict_p in conflict_ps:
                    if conflict_p in loss_property:
                        raise ValueError(
                            f"Conflict Property detected: {p} with {conflict_with} cannot be used together."
                        )
    else:
        assert set(loss_property).issubset(list(PROPERTY))

    return loss_property


INVARIANT_DISCRETE_PROPERTY_REQUIRED_FIELD = [
    "type",
    "per",
    "in_dim",
    "out_dim",
    "num_classes",
]

INVARIANT_CONTINUOUS_PROPERTY_REQUIRED_FIELD = [
    "type",
    "per",
    "in_dim",
    "out_dim",
    "bias",
    "act",
]

EQUIVARIANT_PROPERTY_REQUIRED_FIELD = [
    "per",
    "rank",
    "element_trainable",
    "channel_trainable",
]

def get_embedding_property(cfg: Dict) -> Dict[str, List[str]]:

    universal_embedding = cfg["model"]["config"].get("universal_embedding", None)

    invariant_embedding_property = []
    equivariant_embedding_property = []

    if universal_embedding is not None:
        invariant = universal_embedding.get("invariant", None)
        equivariant = universal_embedding.get("equivariant", None)

        if invariant is not None:
            assert isinstance(invariant, Dict)

            for k, v in invariant.items():
                p = k
                assert (
                    p in UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY
                ), f"universal_embedding allowed property are {UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY}, "
                f"if not enough, please contact the author or write by yourself."
                invariant_embedding_property.append(p)

                assert (
                    "type" in v
                ), f"Missing 'type' in cfg.model.config.universal_embedding.{p}."

                _type = v["type"]

                # Validate required fields
                REQUIRED_FIELDS = (
                    INVARIANT_DISCRETE_PROPERTY_REQUIRED_FIELD
                    if _type == "discrete"
                    else (
                        INVARIANT_CONTINUOUS_PROPERTY_REQUIRED_FIELD
                        if _type == "continuous"
                        else None
                    )
                )
                assert (
                    REQUIRED_FIELDS is not None
                ), f"Invalid type '{_type}', got {_type}, allowed type are [continuous, discrete]'"

                missing_fields = [f for f in REQUIRED_FIELDS if f not in v]
                extra_fields = [f for f in v if f not in REQUIRED_FIELDS]

                assert (
                    not missing_fields
                ), f"Missing required fields {missing_fields} for property '{p}'"
                assert (
                    not extra_fields
                ), f"Unexpected fields {extra_fields} for property '{p}'"

        if equivariant is not None:
            assert isinstance(equivariant, Dict)

            for k, v in equivariant.items():
                p = k
                assert (
                    p in UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY
                ), f"universal_embedding allowed property are {UNIVERSAL_EMBEDDING_ALLOWED_PROPERTY}, "
                f"if not enough, please contact the author or write by yourself."
                equivariant_embedding_property.append(p)

                # Validate required fields
                REQUIRED_FIELDS = EQUIVARIANT_PROPERTY_REQUIRED_FIELD

                missing_fields = [f for f in REQUIRED_FIELDS if f not in v]
                extra_fields = [f for f in v if f not in REQUIRED_FIELDS]

                assert (
                    not missing_fields
                ), f"Missing required fields {missing_fields} for property '{p}'"
                assert (
                    not extra_fields
                ), f"Unexpected fields {extra_fields} for property '{p}'"

    return invariant_embedding_property + equivariant_embedding_property

# For Metrics
MAE_PROPERTY = [
        p for p in SUPPORT_PREDICT_PROPERTY 
        if p != "polarization" 
    ]
RMSE_PROPERTY = [   
        p for p in SUPPORT_PREDICT_PROPERTY 
        if p != "polarization" 
    ]
MAE_PER_ATOM_PROPERTY = [
    p for p, v in PROPERTY.items() 
    if v["type"] == "graph" 
    and p != "polarization"
    and p != "stress"   
    and p != "direct_stress"  
]
RMSE_PER_ATOM_PROPERTY = [
    p for p, v in PROPERTY.items() 
    if v["type"] == "graph" 
    and p != "polarization" 
    and p != "stress"    
    and p != "direct_stress"    
]


fields = {f"compute_{k}": False for k, v in PROPERTY.items()}
@dataclass
class ComputeFlag:
    __annotations__ = {k: bool for k in fields} 
    locals().update(fields)


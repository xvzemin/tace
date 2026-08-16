################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import numpy as np
import torch


from ase import units
from ase.io import read, write
import ase.optimize
import ase.optimize.sciopt
from ase.filters import UnitCellFilter, StrainFilter, ExpCellFilter, FrechetCellFilter
filter_cls = {
    None: None,
    "unit": UnitCellFilter,
    "strain": StrainFilter,
    "exp": ExpCellFilter,
    "frechet": FrechetCellFilter, # recommend
}
optimizer_cls = {
    "BFGS": ase.optimize.BFGS,
    "LBFGS": ase.optimize.LBFGS,
    "BFGSLineSearch": ase.optimize.BFGSLineSearch,
    "LBFGSLineSearch": ase.optimize.LBFGSLineSearch,
    "QuasiNewton": ase.optimize.BFGSLineSearch,
    "SciPyFminBFGS": ase.optimize.sciopt.SciPyFminBFGS,
    "SciPyFminCG": ase.optimize.sciopt.SciPyFminCG,
    "FIRE": ase.optimize.FIRE, # recommend
    "FIRE2": ase.optimize.FIRE2,
    "GPMin": ase.optimize.GPMin,
    "GOQN": ase.optimize.GoodOldQuasiNewton,
}

from tace.foundations import tace_foundations
from tace.interface.ase import TACEAseCalc, add_dispersion


# Put your (auto)download model in ~/.cache/tace
model = tace_foundations["TACE-OAM-7M"]

dtype = 'float32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fidelity_idx = 0  # first fidelity

dispersion = False
filter = "frechet"
optimizer = "FIRE"
unrelaxed_atomsList = read('../data/BaTiO3.xyz', index=':')

fmax = 0.05
MAX_STEP = 3000


for idx, atoms in enumerate(unrelaxed_atomsList):

    calc = TACEAseCalc(
        model=model,
        dtype=dtype,
        device=device,
        fidelity_idx=fidelity_idx,
    )
    if dispersion:
        calc = add_dispersion(
            base_calc=calc,
            damping= "bj",  # choices: ["zero", "bj", "zerom", "bjm"]
            dispersion_xc="pbe",
            dispersion_cutoff= 40.0 * units.Bohr,
        )
    atoms.calc = calc

    if filter is None:
        atoms_opt = atoms
    else:
        atoms_opt = filter_cls[filter](atoms)

    opt = optimizer_cls[optimizer](
        atoms_opt,
        trajectory=f'{idx}.traj',
        # logfile='opt.log',
        # maxstep=0.2 # in angstrom
    )

    opt.run(fmax=fmax, steps=MAX_STEP)

    del calc



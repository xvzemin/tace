################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Relax magnetic moments and optional structural degrees of freedom."""

import numpy as np
from ase.io import read, write
import torch

from tace.interface.ase import GeneralTACEAseCalc, MagneticFIRE

# Input and output
MODEL = "/home/xuzemin/mag-o2/checkpoints_epoch/last.ckpt"
INPUT = "/home/xuzemin/mag-o2/Fe-Fmag.xyz"
TRAJECTORY = "magnetic_relax.xyz"
LOGFILE = "-"
STRUCTURE_INDEX = 0

# Calculator
DEVICE ='cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = None

KEYS = {"initial_noncollinear_magmoms": "spin"}

# Nested relaxation always converges magnetic moments at fixed structure before
# taking one coordinate step and then reconverging the magnetic moments.
OPTIMIZE_CELL = False
MAGMOM_KEY = "spin"
MAGNETIC_FORCES_KEY = "noncollinear_magnetic_forces"
MAGMOM_SCALE = 1.0
CELL_FILTER_KWARGS = {}

# FIRE step limits
POSITION_MAXSTEP = 0.1  # Angstrom
MAGMOM_MAXSTEP = 0.02  # Bohr magneton
CELL_MAXSTRAIN = 0.01

# Independent convergence thresholds
POSITION_FMAX = 0.05  # eV / Angstrom
MAGMOM_FMAX = 0.01  # eV / Bohr magneton
CELL_FMAX = 0.005  # eV / Angstrom**3
STRUCTURE_STEPS = 1000
MAGMOM_STEPS = 1000

# FIRE parameters
POSITION_DT = 0.1
MAGMOM_DT = 0.05
DTMAX = 1.0
DOWNHILL_CHECK = True


def main() -> None:
    atoms = read(INPUT, index=STRUCTURE_INDEX)
    atoms.calc = GeneralTACEAseCalc(
        model=MODEL,
        dtype=DTYPE,
        device=DEVICE,
        keys=KEYS,
    )

    optimizer = MagneticFIRE(
        atoms,
        optimize_cell=OPTIMIZE_CELL,
        magmom_key=MAGMOM_KEY,
        magnetic_forces_key=MAGNETIC_FORCES_KEY,
        magmom_scale=MAGMOM_SCALE,
        cell_filter_kwargs=CELL_FILTER_KWARGS,
        position_maxstep=POSITION_MAXSTEP,
        magmom_maxstep=MAGMOM_MAXSTEP,
        cell_maxstrain=CELL_MAXSTRAIN,
        position_fmax=POSITION_FMAX,
        magmom_fmax=MAGMOM_FMAX,
        cell_fmax=CELL_FMAX,
        magmom_steps=MAGMOM_STEPS,
        position_dt=POSITION_DT,
        magmom_dt=MAGMOM_DT,
        dtmax=DTMAX,
        downhill_check=DOWNHILL_CHECK,
        trajectory=TRAJECTORY,
        logfile=LOGFILE,
    )
    converged = optimizer.run(steps=STRUCTURE_STEPS)

    energy = atoms.get_potential_energy()
    maxima = optimizer.get_force_maxima()

    print(f"Converged: {converged}")
    print(f"Energy: {energy:.12f} eV")
    for name, value in maxima.items():
        print(f"Maximum {name} force: {value:.8g}")
    norms = np.linalg.norm(atoms.arrays[MAGMOM_KEY], axis=1)
    print(f"Magnetic-moment norm range: {norms.min():.8g} to {norms.max():.8g}")

if __name__ == "__main__":
    main()

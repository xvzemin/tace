################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
pip install sella
"""

from pathlib import Path

import torch
from ase.io import read, write
from sella import Sella

from tace.interface.ase import TACEAseCalc
from tace.foundations import tace_foundations

model = tace_foundations["TACE-OAM-7M"]
dtype = "float32"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc = TACEAseCalc(
    model=model,
    dtype=dtype,
    device=device,
    fidelity_idx=0,
)

fmax = 0.01
steps = 1000

ts_file = f = Path(str("../data/TS2-AA-on-phase-O5.vasp"))
TS = read(ts_file, 0)
TS.calc = calc   

ts_opt = Sella(
    TS,
    trajectory="o5_aa.ts_opt.traj",
    logfile='-',
    # eta=1e-4,        # Finite difference step size
    # gamma=0.4,       # Convergence criterion for iterative diagonalization
    # delta0=1.3e-3,   # Initial trust radius
    # rho_inc=1.035,   # Threshold for increasing trust radius
    # rho_dec=5.0,     # Threshold for decreasing trust radius
    # sigma_inc=1.15,  # Trust radius increase factor
    # sigma_dec=0.65,  # Trust radius decrease factor
    # order=1,
)
ts_opt.run(fmax, steps)

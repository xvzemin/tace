################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path

import numpy as np
import torch
from ase.io import read, write
from ase.vibrations import Vibrations

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

# ts_file = f = Path(str("o5_aa.ts_opt.traj")) # run ts_opt.py first
# TS = read(ts_file, 0)

ts_file = f = Path(str("o5_aa.neb.traj")) # run ts_opt.py first
neb_atomsList = read(ts_file, index=":")
energies = np.array([atoms.get_potential_energy() for atoms in neb_atomsList])
TS = neb_atomsList[energies.argmax()]


TS.calc = calc   
vib = Vibrations(
    TS,
    name='tmp',
    delta=0.015, # 0.01 ~ 0.03
    nfree=2
)
vib.run()
vib.summary()
vib.clean()
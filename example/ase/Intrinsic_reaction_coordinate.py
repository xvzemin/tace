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
from sella import IRC

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


TS = read("o5_aa.ts_opt.traj", -1)
TS.calc = calc   
irc_file = Path("o5_aa.irc.traj")
irc = IRC(
    TS,
    logfile="-",
    trajectory=str(irc_file),
    dx=0.1,    # default 0.1, unit Angstrom * sqrt(amu), larger, faster, less true traj
    eta=1e-4,  # default 1e-4
    gamma=0.1, # default 0.1
    # gamma=0.4, # default 0.1
    keep_going=False, 
)
irc.run(fmax=fmax, steps=steps, direction='forward')
split = irc.nsteps
irc.run(fmax=fmax, steps=steps, direction='reverse')
tmp_atomsList = read(str(irc_file), ':')
forward = tmp_atomsList[:split+1]
forward.reverse()
backward = tmp_atomsList[split+1:]
irc_file.unlink()
write(str(irc_file), forward + backward)



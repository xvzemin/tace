################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from pathlib import Path

import torch
from ase.io import read, write
from ase.mep import NEB
from ase.optimize import LBFGS, FIRE
from tace.interface.ase import TACEAseCalc
from tace.foundations import tace_foundations

model = tace_foundations["TACE-OAM-7M"]
model = "/share2/vortex/xuzemin/last.ckpt"
dtype = "float32"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
calc = TACEAseCalc(
    model=model,
    dtype=dtype,
    device=device,
    fidelity_idx=0,
    spin_on=False,
)

fmax = 0.03
steps = 1000

irc_file = f = Path("o5_aa.irc.traj") # run ts_irc.py first
irc_atomsList = read(irc_file, ':')
 

IS = irc_atomsList[0].copy()
FS   = irc_atomsList[-1].copy()

del irc_atomsList

for atoms in [IS, FS]:
    atoms.calc = calc
    FIRE(atoms, logfile='-').run(fmax, steps)

n_images = 8
images = [IS]
for _ in range(n_images):
    images.append(IS.copy())
images.append(FS)

neb = NEB(images, climb=True, k=0.1)
neb.interpolate(method="idpp", mic=True)

for img in images:
    img.calc = TACEAseCalc(
        model=model,
        dtype=dtype,
        device=device,
        fidelity_idx=0,
    )

opt = FIRE(
    neb,
    # trajectory="full_neb.traj", 
    logfile='-'
)

converged = opt.run(fmax, steps)

if converged:
    print("NEB succeed")
else:
    print("NEB fail") 

# TS = sorted(images, key=lambda x: x.get_potential_energy())[-1]
for img in images:
    _ = img.get_potential_energy()
write("o5_aa.neb.traj", images)

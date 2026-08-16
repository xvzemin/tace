################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from ase.io import read, write
from ase.optimize.basin import BasinHopping
from ase.optimize import FIRE

import torch

from tace.foundations import tace_foundations
from tace.interface.ase import TACEAseCalc

# Put your (auto)download model in ~/.cache/tace
model = tace_foundations["TACE-OAM-7M"]

dtype = 'float32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fidelity_idx = 0  # first fidelity

dispersion = False
filter = "frechet"
atoms = read('../data/BaTiO3.xyz', index=0)

atoms.calc = TACEAseCalc(
    model=model,
    dtype=dtype,
    device=device,
    fidelity_idx=fidelity_idx,
)

bh = BasinHopping(
    atoms,
    # temperature=0.1,
    dr=0.1,
    optimizer=FIRE,
    fmax=0.05,
)

bh.run(steps=1000)
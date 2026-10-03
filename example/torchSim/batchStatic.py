################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
This is an example for Static caculation for extended systems with autobatching.
'''

import torch
import torch_sim as ts
from ase.io import read, write

from tace.interface.torchsim import TACETorchSimCalc

# === Input ===

# Foundation models are downloaded and cached automatically.
model = "TACE-OAM-7M"

dtype = 'float32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fidelity_idx = 0  # first fidelity
model = TACETorchSimCalc(
    model,
    fidelity_idx=fidelity_idx,
    device=device,
    dtype=dtype, 
    compute_forces=True,
    compute_stress=True,
)

# === input atoms ===
atomsList = read('../data/BaTiO3.xyz', index=':')[:2]

# Automatically manage the memory of multiple Gpus to full capacity
results = ts.static(
    system=atomsList,
    model=model,
    trajectory_reporter={
        "filenames": None, 
        "prop_calculators": {
            1: {
                "potential_energy": lambda state: state.energy,
                "forces": lambda state: state.forces,
                "stress": lambda state: state.stress,
            }
        }
    },
    pbar=True,
    # autobatcher=True,
)

for idx, result in enumerate(results):
    energy = result["potential_energy"].item()
    print(energy)
    forces = result["forces"].detach().cpu().numpy()
    stress = result["stress"].detach().cpu().numpy().squeeze(0)
    atomsList[idx].calc = None
    atomsList[idx].info['energy'] = energy
    atomsList[idx].arrays['forces'] = forces
    atomsList[idx].info['stress'] = stress
    write('static.xyz', atomsList)

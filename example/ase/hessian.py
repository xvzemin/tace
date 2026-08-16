################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch
from ase.io import read, write

from tace.foundations import tace_foundations
from tace.interface.ase import TACEAseCalc, add_dispersion

# Put your (auto)download model in ~/.cache/tace
model = tace_foundations["TACE-OAM-7M"]

dtype = 'float32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
fidelity_idx = 0  # first fidelity
atoms = read('../data/BaTiO3.xyz', index=0)


# The training property of TACE-OAM-L are ['energy', 'forces', 'stress'],
# but we can also predict properties such as hessian, atomic_stresses
target_property = ["energy", "forces", "stress", "atomic_stresses", "hessian"]
calc = TACEAseCalc(
    model=model,
    dtype=dtype,
    device=device,
    fidelity_idx=fidelity_idx,
    target_property=target_property,
)
atoms.calc = calc

energy = atoms.get_potential_energy()
calc_results = atoms.calc.results
print(calc_results.keys())
print(energy)
print(calc_results['energy'])
print(calc_results['forces'].shape)
print(calc_results['stress'].shape)
print(calc_results['stresses'].shape)
print(calc_results['hessian'].shape)






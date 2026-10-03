################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
This is an example for relaxing extended systems.
Optimize the position and the cell simultaneously with autobatching.
Before run, please pip install vesin
'''

from pathlib import Path

import numpy as np
import torch
import torch_sim as ts
optimizer_cls ={
    'fire': ts.Optimizer.fire, # recommend
    'gradient_descent': ts.Optimizer.gradient_descent,
    'bfgs': ts.Optimizer.bfgs,
    'lbfgs': ts.Optimizer.lbfgs,
}
cell_filter_cls = {
    'unit': ts.CellFilter.unit,
    'frechet': ts.CellFilter.frechet, # recommend
}
import ase
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

fmax = 0.05
SAVE_FREQ = 1
MAX_STEP = 3000
outDir = "results"

# === unrelaxed atoms ===
outDir = Path(outDir)
outDir.mkdir(exist_ok=True)
unrelaxed_atomsList = read('../data/BaTiO3.xyz', index=':')[:2]

# filter
# === torchSim ===
optimizer = "fire"
cell_filter = "frechet"

# === md traj ===
filenames = [outDir / f"{i}_batchOptTraj.h5" for i in range(len(unrelaxed_atomsList))]
prop_calculators = {
    SAVE_FREQ: {
        "potential_energy": lambda state: state.energy,
        "forces": lambda state: state.forces,
        "stress": lambda state: state.stress,
        "cell_forces": lambda state: state.cell_forces,
    # can add more SAVE_FREQ
    }
}
batch_reporter = ts.TrajectoryReporter(
    filenames, 
    state_frequency=SAVE_FREQ,
    prop_calculators=prop_calculators,
)


if cell_filter is not None:
    include_cell_forces = True
    init_kwargs=dict(cell_filter=ts.CellFilter.unit)
else:
    include_cell_forces = False
    init_kwargs=None

# convergence_fn = ts.generate_energy_convergence_fn(energy_tol=1e-6)
convergence_fn = ts.generate_force_convergence_fn(
    force_tol=fmax, 
    include_cell_forces=include_cell_forces
)


# Automatically manage the memory of multiple Gpus to full capacity
final_state = ts.optimize(
    system=unrelaxed_atomsList ,
    model=model,
    optimizer=optimizer_cls[optimizer],
    convergence_fn=convergence_fn,
    max_steps=MAX_STEP,
    trajectory_reporter=batch_reporter,
    steps_between_swaps=5,
    pbar=True,
    # autobatcher=True,
    init_kwargs=init_kwargs,
)

# # === convert to Atoms ===
# final_atoms: list[ase.Atoms] = final_state.to_atoms()

# === data processing === 
def check_converge(atoms: ase.Atoms) -> bool:
    max_force = np.linalg.norm(atoms.arrays["forces"], axis=1).max()
    max_cell_force = np.linalg.norm(atoms.info["cell_forces"], axis=1).max()
    if include_cell_forces:
        return (max_force < fmax) and (max_cell_force < fmax)
    else:
        return (max_force < fmax)


num_converged = 0
num_unconverged = 0

for _, filename in enumerate(filenames):
    with ts.TorchSimTrajectory(filename) as traj:
        # traj.write_ase_trajectory(f"{filename}.xyz")
        traj_atomsList = []
        energy = traj.get_array("potential_energy")
        forces = traj.get_array("forces")
        stress = traj.get_array("stress")
        cell_forces = traj.get_array("cell_forces")

        print(filename, len(traj))
        for frame in range(len(traj)):
            atoms = traj.get_atoms(frame)
            atoms.info['energy'] = energy[frame]
            atoms.arrays['forces'] = forces[frame]
            atoms.info['stress'] = stress[frame]
            atoms.info["cell_forces"] = cell_forces[frame]
            traj_atomsList.append(atoms)

        if check_converge(traj_atomsList[-1]):
            saved_filename = f"{filename}.Converged.xyz"
            num_converged += 1
        else:
            saved_filename = f"{filename}.Unconverged.xyz"
            num_unconverged += 1

        write(saved_filename, traj_atomsList)
        traj.close()

print(f"Converged: {num_converged}/{num_converged+num_unconverged}")
print(f"UnConverged: {num_unconverged}/{num_converged+num_unconverged}")
    

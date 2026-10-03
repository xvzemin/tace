################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
'''
This is an example of molecular dynamics simulation for extended systems.
'''
import torch
import torch_sim as ts

integrator_cls = {
    "nve": ts.Integrator.nve,
    "nvt_langevin": ts.Integrator.nvt_langevin,
    "nvt_nose_hoover": ts.Integrator.nvt_nose_hoover,
    "nvt_vrescale": ts.Integrator.nvt_vrescale,
    "npt_langevin_anisotropic": ts.Integrator.npt_langevin_anisotropic,
    "npt_langevin_isotropic": ts.Integrator.npt_langevin_isotropic,
    "npt_nose_hoover_isotropic": ts.Integrator.npt_nose_hoover_isotropic,
    "npt_crescale_isotropic": ts.Integrator.npt_crescale_isotropic,
    "npt_crescale_triclinic": ts.Integrator.npt_crescale_triclinic,
}

from ase.io import read

from tace.interface.torchsim import TACETorchSimCalc

# === model ===
dtype = 'float32'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

BaTiO3 = read('../data/BaTiO3.xyz', '0')
init_conf = BaTiO3

init_atomsList = [init_conf] * 2
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

# === torchSim ===
integrator = "nvt_nose_hoover"
T = 300 # in K
TIME_STEP = 0.001  # in ps
# TOTAL_STEP = 300 * 1000
TOTAL_STEP = 10000
SAVE_FREQ = 10

# === md traj ===
filenames = [f"batchMdTraj{i}.h5" for i in range(len(init_atomsList))]
prop_calculators = {
    SAVE_FREQ: {
        "potential_energy": lambda state: state.energy,
        "kinetic_energy": lambda state: ts.calc_kinetic_energy(
            momenta=state.momenta, masses=state.masses
        ),
        "forces": lambda state: state.forces,
    # can add more SAVE_FREQ
    }
}
batch_reporter = ts.TrajectoryReporter(
    filenames, 
    state_frequency=SAVE_FREQ,
    prop_calculators=prop_calculators,
)

# Automatically manage the memory of multiple Gpus to full capacity
final_state = ts.integrate(
    system=init_atomsList, 
    model=model,  
    integrator=integrator_cls[integrator], 
    n_steps=TOTAL_STEP,  
    temperature=T, # in K
    timestep=TIME_STEP,
    trajectory_reporter=batch_reporter,
    autobatcher=True,
    pbar=True,
    # external_pressure=0.0,
)

# === convert to Atoms ===
# final_atoms: list[ase.Atoms] = final_state.to_atoms()

# # === data processing === 
# final_energies_per_atom = []
# for sys_idx, filename in enumerate(filenames):
#     with ts.TorchSimTrajectory(filename) as traj:
#         final_energy = traj.get_array("potential_energy")[-1].item()
#         n_atoms = len(traj.get_atoms(-1))
#         final_energies_per_atom.append(final_energy / n_atoms)
#         print(
#             f"System {sys_idx}: {final_energy:.6f} eV, {final_energy / n_atoms:.6f} eV/atom"
#         )


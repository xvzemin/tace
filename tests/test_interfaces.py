"""Foundation checkpoints through ASE, TorchSim and LAMMPS.

Set TACE_TEST_MODEL_DIR to a directory containing the three .pt checkpoints.
No models are downloaded during the test run.
"""

import os
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import bulk
from ase.io import write
from ase.md.verlet import VelocityVerlet

from tace.interface.ase import TACEAseCalc


@pytest.fixture(params=["TACE-OAM-7M", "TACE-OAM-L", "TECE-OAM-RRA-1.0"])
def checkpoint(request):
    directory = os.environ.get("TACE_TEST_MODEL_DIR")
    if directory is None:
        pytest.skip("Set TACE_TEST_MODEL_DIR to test foundation checkpoints.")
    path = Path(directory) / (request.param + ".pt")
    assert path.is_file(), path
    return path


@pytest.fixture(params=["cpu", "cuda", "eqx"])
def calculator(checkpoint, request):
    if request.param != "cpu" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return TACEAseCalc(
        checkpoint,
        device="cpu" if request.param == "cpu" else "cuda",
        dtype="float64",
        enable_eqx=request.param == "eqx",
    )


@pytest.fixture
def atoms():
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 1, 1))
    atoms.rattle(stdev=0.035, seed=19)
    atoms.wrap()
    return atoms


def test_ase(checkpoint, calculator, atoms):
    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    assert np.isfinite(energy)
    assert np.isfinite(forces).all() and np.isfinite(stress).all()
    np.testing.assert_allclose(forces.sum(axis=0), 0, atol=1e-8)
    # Explicitly select the reference path, independent of the process environment.
    from tace.utils.env import enable_acceleration

    accelerated = os.environ.get("TACE_USE_EQX") == "1"
    enable_acceleration(force=True)
    reference = atoms.copy()
    reference.calc = TACEAseCalc(checkpoint, device="cpu", dtype="float64")
    np.testing.assert_allclose(energy, reference.get_potential_energy(), atol=2e-7)
    np.testing.assert_allclose(forces, reference.get_forces(), atol=2e-7, rtol=2e-6)
    np.testing.assert_allclose(stress, reference.get_stress(voigt=False), atol=2e-8)
    enable_acceleration(enable_eqx=accelerated, force=True)
    VelocityVerlet(atoms, timestep=0.1 * units.fs, logfile=None).run(2)
    assert np.isfinite(atoms.get_total_energy())


def test_torchsim(calculator, atoms):
    ts = pytest.importorskip("torch_sim")
    from torch_sim.integrators.nve import nve_init, nve_step

    from tace.interface.torchsim import TACETorchSimCalc

    atoms.calc = calculator
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    model = TACETorchSimCalc(
        calculator.model, device=calculator.device, dtype=torch.float64
    )
    state = ts.io.atoms_to_state(
        [atoms, atoms.copy()], device=calculator.device, dtype=torch.float64
    )
    output = model(state)
    np.testing.assert_allclose(output["energy"].cpu(), [energy, energy], atol=2e-7)
    np.testing.assert_allclose(
        output["forces"].cpu(), np.concatenate([forces, forces]), atol=2e-7, rtol=2e-6
    )
    np.testing.assert_allclose(output["stress"].cpu(), [stress, stress], atol=2e-8)
    state = nve_init(state, model, kT=0.01)
    for _ in range(2):
        state = nve_step(state, model, dt=0.001)
    assert torch.isfinite(state.energy).all()
    assert torch.isfinite(state.forces).all()


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("geometry", ["periodic", "isolated"])
def test_lammps(checkpoint, atoms, tmp_path, accelerated, geometry):
    if not torch.cuda.is_available():
        pytest.skip("The LAMMPS interface requires CUDA Kokkos.")
    lammps = pytest.importorskip("lammps")
    mliap = pytest.importorskip("lammps.mliap")
    from tace.interface.lammps import TACELammpsCalc

    if geometry == "isolated":
        atoms = Atoms("Cu", positions=[[10, 10, 10]], cell=[20, 20, 20], pbc=True)
    atoms.calc = TACEAseCalc(
        checkpoint, device="cuda", dtype="float64", enable_eqx=accelerated
    )
    energy, forces, stress = (
        atoms.get_potential_energy(),
        atoms.get_forces(),
        atoms.get_stress(),
    )
    structure = tmp_path / "atoms.data"
    write(structure, atoms, format="lammps-data", atom_style="atomic", masses=True)
    simulation = lammps.lammps(
        cmdargs=[
            "-k",
            "on",
            "g",
            "1",
            "-sf",
            "kk",
            "-pk",
            "kokkos",
            "newton",
            "on",
            "neigh",
            "half",
            "-screen",
            "none",
            "-log",
            "none",
        ]
    )
    try:
        mliap.activate_mliappy_kokkos(simulation)
        simulation.commands_string(
            "units metal\natom_style atomic\nboundary p p p\natom_modify map array\n"
            f'read_data "{structure.as_posix()}"\n'
        )
        mliap.load_unified_kokkos(TACELammpsCalc(atoms.calc.model))
        simulation.commands_string(
            "pair_style mliap unified EXISTS 0\npair_coeff * * Cu\n"
            "neighbor 1.0 bin\nthermo_style custom step pe pxx pyy pzz pyz pxz pxy\n"
            "run 0\n"
        )
        actual_forces = np.ctypeslib.as_array(
            simulation.gather_atoms("f", 1, 3)
        ).reshape(-1, 3)
        actual_stress = (
            -1e-4
            * units.GPa
            * np.array(
                [
                    simulation.get_thermo(key)
                    for key in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")
                ]
            )
        )
        np.testing.assert_allclose(simulation.get_thermo("pe"), energy, atol=2e-7)
        np.testing.assert_allclose(actual_forces, forces, atol=2e-7, rtol=2e-6)
        np.testing.assert_allclose(actual_stress, stress, atol=2e-8)
        simulation.commands_string("fix integrate all nve\ntimestep 0.0001\nrun 2\n")
        assert np.isfinite(simulation.get_thermo("pe"))
    finally:
        simulation.close()

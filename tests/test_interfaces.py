"""Model outputs and foundation checkpoints through ASE, TorchSim and LAMMPS.

Set TACE_TEST_MODEL_DIR to a directory containing the three .pt checkpoints.
No models are downloaded during the test run.
"""

import importlib
import os
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import bulk
from ase.calculators.calculator import Calculator
from ase.io import write
from ase.md.verlet import VelocityVerlet

from tace.interface.ase import MagneticFIRE, TACEAseCalc
from tace.models.adapter import TensorModel


@pytest.mark.parametrize("interface", ["ase", "general_ase", "torchsim", "nvalchemi"])
@pytest.mark.parametrize(
    "enabled, previous", [(False, None), (False, "1"), (True, "0")]
)
def test_luna_acceleration_option(monkeypatch, interface, enabled, previous):
    if interface == "nvalchemi":
        pytest.importorskip("nvalchemi")
        module = importlib.import_module("tace.interface.nvalchemi.wrapper")
        factory = module.TACEWrapper.from_checkpoint
        monkeypatch.setattr(module, "resolve_model_path", Path)
        if enabled:
            monkeypatch.setenv("TACE_USE_CUE", "1")
    elif interface == "torchsim":
        pytest.importorskip("torch_sim")
        module = importlib.import_module("tace.interface.torchsim.torchsim")
        factory = module.TACETorchSimCalc
    else:
        module = importlib.import_module("tace.interface.ase.calculator")
        factory = module.TACEAseCalc
        if interface == "general_ase":
            from tace.interface.ase import GeneralTACEAseCalc

            factory = GeneralTACEAseCalc
    if previous is not None:
        monkeypatch.setenv("TACE_USE_LUNA", previous)
    loader = Mock(side_effect=RuntimeError("model loading reached"))
    monkeypatch.setattr(module, "load_tace", loader)
    options = {"enable_luna": True} if enabled else {}
    with pytest.raises(RuntimeError, match="model loading reached"):
        factory("model.pt2", device="cpu", **options)
    loader.assert_called_once()
    assert os.environ.get("TACE_USE_LUNA") == ("1" if enabled else previous)


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


def test_mixed_boundaries(calculator, atoms, double_precision):
    from torch_geometric.data import Batch

    from tace.dataset.graph import from_atoms

    molecule = Atoms("Cu2", positions=[[0, 0, 0], [2.5, 0, 0]])
    structures = [molecule, molecule.copy(), atoms, atoms.copy()]
    structures[1].cell = [8, 8, 8]
    structures[-1].pbc = [True, True, False]
    model = calculator.model
    graphs = [
        from_atoms(
            calculator.element,
            structure,
            calculator.cutoff,
            max_neighbors=calculator.max_neighbors,
            keyspec=calculator.keySpecification,
            target_property=calculator.target_property,
            embedding_property=calculator.embedding_property,
            training=False,
            neighborlist_backend=calculator.neighborlist_backend,
        )
        for structure in structures
    ]
    batch = Batch.from_data_list(graphs).to(calculator.device).to_dict()
    output = model(batch)
    assert torch.isfinite(output["energy"]).all()
    assert torch.isfinite(output["forces"]).all()
    assert torch.isfinite(output["stress"]).all()
    assert not output["stress"][:2].any()
    for i, atoms in enumerate(structures):
        atoms.calc = calculator
        np.testing.assert_allclose(
            atoms.get_potential_energy(), output["energy"][i].detach().cpu(), atol=2e-7
        )
        start, end = int(batch["ptr"][i]), int(batch["ptr"][i + 1])
        np.testing.assert_allclose(
            atoms.get_forces(),
            output["forces"][start:end].detach().cpu(),
            atol=2e-7,
            rtol=2e-6,
        )
        np.testing.assert_allclose(
            atoms.get_stress(voigt=False),
            output["stress"][i].detach().cpu(),
            atol=2e-8,
        )


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


class QuadraticMagneticCalculator(Calculator):
    implemented_properties = [
        "energy",
        "free_energy",
        "forces",
        "noncollinear_magnetic_forces",
        "stress",
    ]

    def __init__(self, target_positions, target_magmoms):
        super().__init__()
        self.target_positions = np.asarray(target_positions)
        self.target_magmoms = np.asarray(target_magmoms)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms)
        delta_positions = atoms.positions - self.target_positions
        delta_magmoms = atoms.arrays["spin"] - self.target_magmoms
        energy = 0.5 * (
            np.square(delta_positions).sum() + np.square(delta_magmoms).sum()
        )
        self.results = {
            "energy": energy,
            "free_energy": energy,
            "forces": -delta_positions,
            "noncollinear_magnetic_forces": -delta_magmoms,
        }
        if "stress" in self.implemented_properties:
            self.results["stress"] = np.zeros(6)


def make_atoms():
    atoms = Atoms(
        "H2",
        positions=[[0.4, -0.2, 0.1], [1.3, 0.3, -0.2]],
        cell=np.eye(3) * 4.0,
        pbc=True,
    )
    atoms.set_array("spin", np.array([[1.0, 0.5, -0.2], [-0.4, 0.3, 0.8]]))
    target_positions = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    target_magmoms = np.array([[0.2, 0.1, 0.0], [-0.1, 0.0, 0.3]])
    atoms.calc = QuadraticMagneticCalculator(target_positions, target_magmoms)
    return atoms, target_positions, target_magmoms


def test_magnetic_scf_precedes_every_structural_step():
    atoms, _, _ = make_atoms()
    original_positions = atoms.positions.copy()
    optimizer = MagneticFIRE(
        atoms,
        magmom_key="spin",
        magmom_fmax=1e-12,
        magmom_steps=1,
        magmom_maxstep=0.1,
        magmom_dt=0.1,
        logfile=None,
    )

    with pytest.raises(RuntimeError, match="did not converge"):
        optimizer.run(steps=1)

    np.testing.assert_allclose(atoms.positions, original_positions, atol=0.0)


def test_nested_magnetic_and_structural_optimization():
    atoms, target_positions, target_magmoms = make_atoms()
    optimizer = MagneticFIRE(
        atoms,
        magmom_key="spin",
        position_maxstep=0.1,
        magmom_maxstep=0.08,
        position_fmax=1e-5,
        magmom_fmax=1e-5,
        magmom_steps=500,
        position_dt=0.1,
        magmom_dt=0.1,
        downhill_check=True,
        logfile=None,
    )

    converged = optimizer.run(steps=500)

    assert converged
    np.testing.assert_allclose(atoms.positions, target_positions, atol=2e-5)
    np.testing.assert_allclose(atoms.arrays["spin"], target_magmoms, atol=2e-5)
    maxima = optimizer.get_force_maxima()
    assert maxima["positions"] < 1e-5
    assert maxima["magmoms"] < 1e-5


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


DTYPE = torch.float64


class PairReadout(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("cutoff", torch.tensor(3.0, dtype=DTYPE))
        self.register_buffer("atomic_numbers", torch.tensor([13, 14]))
        self.target_property = ["energy", "forces", "stress"]
        self.embedding_property = []
        self.fidelity = {0: "first", 1: "second"}

    def forward(self, data, graph):
        node_energy = data["node_attrs"][:, 1] + 3 * graph.node_fidelity
        node_energy = node_energy.index_add(
            0, data["edge_index"][0], graph.edge_vector.square().sum(-1)
        )
        energy = node_energy.new_zeros(graph.num_graphs).index_add(
            0, data["batch"], node_energy
        )
        return {"energy": energy, "node_energy": node_energy}


@pytest.fixture
def torchsim_state():
    ts = pytest.importorskip("torch_sim")
    return ts.SimState(
        positions=torch.tensor(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.5, 0.0]],
            dtype=DTYPE,
            device=DEVICE,
        ),
        masses=torch.ones(4, dtype=DTYPE, device=DEVICE),
        cell=8 * torch.eye(3, dtype=DTYPE, device=DEVICE).repeat(2, 1, 1),
        atomic_numbers=torch.full((4,), 14, dtype=torch.int64, device=DEVICE),
        system_idx=torch.tensor([0, 0, 1, 1], device=DEVICE),
        pbc=False,
    )


@pytest.fixture
def torchsim_calculator():
    pytest.importorskip("torch_sim")
    from tace.interface.torchsim import TACETorchSimCalc

    def create(**kwargs):
        return TACETorchSimCalc(
            TensorModel(PairReadout()), device=DEVICE, dtype=DTYPE, **kwargs
        )

    return create


@pytest.mark.parametrize("field", ["atomic_numbers", "system_idx"])
def test_torchsim_cache_detects_in_place_changes(
    torchsim_state, torchsim_calculator, field
):
    calc = torchsim_calculator()
    calc(torchsim_state)
    assert getattr(calc, field).data_ptr() != getattr(torchsim_state, field).data_ptr()
    if field == "atomic_numbers":
        torchsim_state.atomic_numbers[0] = 13
    else:
        torchsim_state.system_idx[1] = 1
    actual = calc(torchsim_state)
    expected = torchsim_calculator()(torchsim_state.clone())
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key])
    torch.testing.assert_close(
        calc.ptr,
        torch.tensor([0, 1, 4], device=DEVICE)
        if field == "system_idx"
        else torch.tensor([0, 2, 4], device=DEVICE),
    )


@pytest.mark.parametrize("field", ["atomic_numbers", "system_idx"])
def test_torchsim_fixed_inputs_are_snapshots_and_checked(
    torchsim_state, torchsim_calculator, field
):
    calc = torchsim_calculator(**{field: getattr(torchsim_state, field)})
    calc(torchsim_state)
    if field == "atomic_numbers":
        torchsim_state.atomic_numbers[0] = 13
    else:
        torchsim_state.system_idx[1] = 1
    with pytest.raises(ValueError, match=field):
        calc(torchsim_state)


@pytest.mark.parametrize("compute_forces", [False, True])
@pytest.mark.parametrize("compute_stress", [False, True])
def test_torchsim_derivative_targets(
    torchsim_state, torchsim_calculator, compute_forces, compute_stress
):
    calc = torchsim_calculator(
        target_property=["energy", "virials"],
        compute_forces=compute_forces,
        compute_stress=compute_stress,
    )
    actual = calc(torchsim_state)
    expected = torchsim_calculator()(torchsim_state.clone())
    torch.testing.assert_close(actual["energy"], expected["energy"])
    torch.testing.assert_close(actual["virials"], expected["virials"])
    assert actual["virials"].abs().max() > 0
    assert ("forces" in actual) == compute_forces
    assert ("stress" in actual) == compute_stress


def test_torchsim_energy_only(torchsim_state, torchsim_calculator):
    calc = torchsim_calculator(compute_forces=False, compute_stress=False)
    assert not calc.model.compute_first_derivative
    actual = calc(torchsim_state)
    assert "forces" not in actual and "stress" not in actual
    assert not torchsim_state.positions.requires_grad


@pytest.mark.parametrize("periodic", [False, True])
def test_torchsim_boundary_conditions(torchsim_state, torchsim_calculator, periodic):
    torchsim_state.pbc.fill_(periodic)
    if periodic:
        torchsim_state.positions[1, 0] = 7.0
    actual = torchsim_calculator()(torchsim_state)
    energy = torch.tensor([4.0, 6.5], dtype=DTYPE, device=DEVICE)
    forces = torch.tensor(
        [[4.0, 0.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 6.0, 0.0], [0.0, -6.0, 0.0]],
        dtype=DTYPE,
        device=DEVICE,
    )
    stress = torch.zeros(2, 3, 3, dtype=DTYPE, device=DEVICE)
    if periodic:
        forces[:2] *= -1
        stress[0, 0, 0] = 4.0 / 8**3
        stress[1, 1, 1] = 9.0 / 8**3
    torch.testing.assert_close(actual["energy"], energy, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual["forces"], forces, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(actual["stress"], stress, atol=1e-10, rtol=1e-10)


def test_torchsim_derivative_dependencies_are_retained(
    torchsim_state, torchsim_calculator
):
    calc = torchsim_calculator(
        target_property=["energy", "hessian"],
        compute_forces=False,
        compute_stress=False,
    )
    assert calc.model.compute_first_derivative
    assert calc.model.compute_second_derivative
    actual = calc(torchsim_state)
    expected = torchsim_calculator(target_property=["energy", "forces", "hessian"])(
        torchsim_state
    )
    torch.testing.assert_close(actual["hessian"], expected["hessian"])
    assert "forces" not in actual


@pytest.mark.parametrize("device", [DEVICE, torch.device("cpu")])
@pytest.mark.parametrize("dtype", [torch.float32, DTYPE])
def test_torchsim_direct_call_converts_inputs_without_mutation(
    torchsim_state, torchsim_calculator, device, dtype
):
    torchsim_state = torchsim_state.to(device=device, dtype=dtype)
    original = torchsim_state.clone()
    calc = torchsim_calculator()
    actual = calc(torchsim_state)
    expected = calc(original.clone().to(device=DEVICE, dtype=DTYPE))
    for key in ("energy", "forces", "stress"):
        torch.testing.assert_close(actual[key], expected[key])
        assert actual[key].dtype == DTYPE
        assert actual[key].device.type == DEVICE.type
        assert not actual[key].requires_grad
    for key in ("positions", "cell", "atomic_numbers", "system_idx", "pbc"):
        torch.testing.assert_close(getattr(torchsim_state, key), getattr(original, key))
        assert not getattr(torchsim_state, key).requires_grad


def test_torchsim_system_fidelity_overrides_default(
    torchsim_state, torchsim_calculator
):
    calc = torchsim_calculator(fidelity_idx=1)
    default = calc(torchsim_state)["energy"]
    torchsim_state.system_extras["fidelity_idx"] = torch.tensor([0, 1], device=DEVICE)
    actual = calc(torchsim_state)["energy"]
    torch.testing.assert_close(actual, default - default.new_tensor([6, 0]))
    torchsim_state.system_extras["fidelity_idx"].fill_(0)
    torch.testing.assert_close(calc(torchsim_state)["energy"], default - 6)
    del torchsim_state.system_extras["fidelity_idx"]
    torch.testing.assert_close(calc(torchsim_state)["energy"], default)


@pytest.mark.parametrize(
    "value, error", [([[0], [1]], ValueError), ([0.5, 1.0], TypeError)]
)
def test_torchsim_system_fidelity_validation(
    torchsim_state, torchsim_calculator, value, error
):
    torchsim_state.system_extras["fidelity_idx"] = torch.tensor(value, device=DEVICE)
    with pytest.raises(error, match="fidelity_idx"):
        torchsim_calculator()(torchsim_state)

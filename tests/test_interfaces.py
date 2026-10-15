"""Model outputs and foundation checkpoints through ASE, TorchSim and LAMMPS.

Set TACE_TEST_MODEL_DIR to a directory containing the three .pt checkpoints.
No models are downloaded during the test run.
"""

import os
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from ase import Atoms, units
from ase.build import bulk
from ase.io import write
from ase.md.verlet import VelocityVerlet

from tace.interface.ase import TACEAseCalc


@pytest.fixture
def small_system(double_precision):
    from tace.models._e3nn.default import DEFAULT_MODEL_CONFIG

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2, num_channel=3, Lmax=1, lmax=1, mmax=1,
        cutoff=4.0, max_neighbors=None,
        statistics=[
            dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
        ],
    )
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = dict(
        positions=torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.3, 0.2], [0.4, 1.1, -0.2]]),
        lattice=4.0 * torch.eye(3).unsqueeze(0),
        node_attrs=torch.ones(3, 1),
        edge_index=torch.tensor([[0, 1, 0, 2, 1, 2], [1, 0, 2, 0, 2, 1]]),
        edge_shifts=torch.tensor(
            [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 0.0],
             [1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]
        ),
        batch=torch.zeros(3, dtype=torch.long),
        ptr=torch.tensor([0, 3]),
        fidelity_idx=torch.zeros(1, dtype=torch.long),
        pbc=torch.ones(1, 3, dtype=torch.bool),
    )
    return config, {key: value.to(device) for key, value in data.items()}


@pytest.mark.parametrize("denoising,all_layers", [(False, False), (False, True), (True, False)])
def test_vector_readouts(small_system, monkeypatch, denoising, all_layers):
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel

    monkeypatch.setenv("TACE_USE_DENS", str(int(denoising)))
    config, data = small_system
    config["target_property"] = ["energy", "direct_forces"]
    config["readout_emlp"]["use_alllayer"] = all_layers
    data["forces"] = torch.randn_like(data["positions"])
    model = TensorModel(e3nnTACE(**config)).to(data["positions"].device).train()
    keys = ["direct_forces", "noise_vec"] if denoising else ["direct_forces"]
    output = model(data)
    for key in keys:
        assert output[key].shape == data["positions"].shape
        assert torch.isfinite(output[key]).all()
    sum(output[key].square().sum() for key in keys).backward()
    heads = [model.readout_fn.direct_forces_readouts]
    if denoising:
        heads.append(model.readout_fn.dens_noise_readouts)
    for head in heads:
        gradients = [p.grad for p in head.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert any(g.abs().sum() > 0 for g in gradients)
    model.eval()
    with torch.no_grad():
        output = model(data)
    assert all(torch.isfinite(output[key]).all() for key in keys)


@pytest.mark.parametrize("training", [False, True])
@pytest.mark.parametrize("empty_edges", [False, True])
def test_edge_forces_match_position_derivatives(small_system, training, empty_edges):
    from tace.models._e3nn.tace import e3nnTACE
    from tace.models.adapter import TensorModel
    from tace.models.utils import compute_forces_stress_from_edge_forces

    config, data = small_system
    if empty_edges:
        data["edge_index"] = data["edge_index"][:, :0]
        data["edge_shifts"] = data["edge_shifts"][:0]
    config["target_property"] = ["energy", "forces", "stress"]
    model = TensorModel(e3nnTACE(**config)).to(data["positions"].device).train(training)
    reference = model({key: value.clone() for key, value in data.items()})
    model.reset_target_property(["energy", "edge_forces"])
    edge_data = {key: value.clone() for key, value in data.items()}
    graph = model.prepare_graph(edge_data)
    assert graph.edge_vector.requires_grad
    assert not edge_data["positions"].requires_grad
    output = model(edge_data)
    forces, stress = compute_forces_stress_from_edge_forces(
        output["edge_forces"], graph.edge_vector, data["edge_index"],
        data["batch"], data["lattice"], data["pbc"],
    )
    torch.testing.assert_close(output["energy"], reference["energy"])
    torch.testing.assert_close(forces, reference["forces"], atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(stress, reference["stress"], atol=1e-10, rtol=1e-10)
    if training and not empty_edges:
        parameters = tuple(model.parameters())
        actual = torch.autograd.grad(
            forces.square().sum() + stress.square().sum(), parameters, allow_unused=True
        )
        expected = torch.autograd.grad(
            reference["forces"].square().sum() + reference["stress"].square().sum(),
            parameters, allow_unused=True,
        )
        for value, ref in zip(actual, expected):
            assert (value is None) == (ref is None)
            if value is not None:
                torch.testing.assert_close(value, ref, atol=1e-9, rtol=1e-9)


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

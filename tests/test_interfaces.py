"""Model outputs and foundation checkpoints through ASE, TorchSim and LAMMPS.

Set TACE_TEST_MODEL_DIR to a directory containing the three .pt checkpoints.
No models are downloaded during the test run.
"""

import ctypes
import importlib
import os
import sys
from pathlib import Path
from types import SimpleNamespace
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


@pytest.mark.parametrize("backend", ["matscipy", "ase"])
def test_ase_neighbor_skin_matches_directed_image_pairs(backend, monkeypatch):
    from tace.dataset import neighbour_list as nl

    exact = nl.get_neighborhood
    search = Mock(wraps=exact)
    monkeypatch.setattr(nl, "get_neighborhood", search)
    cache = nl.VerletNeighborList(0.6)
    for invalid in (0, -0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="skin"):
            nl.VerletNeighborList(invalid)

    def rows(edges, shifts):
        return sorted(map(tuple, np.column_stack((edges.T, shifts)).tolist()))

    def check(coords, cell, pbc, cutoff=1.0):
        kw = dict(
            positions=coords,
            lattice=cell,
            pbc=pbc,
            cutoff=cutoff,
            max_neighbors=None,
            backend=backend,
        )
        got, shifts, boundary, lattice = cache(**kw)
        want, images, _, _ = exact(**kw)
        assert rows(got, shifts) == rows(want, images)
        assert tuple(boundary) == tuple(pbc)
        np.testing.assert_array_equal(lattice, cell)
        return got.shape[1]

    xy = np.array([[0.0, 0.0, 0.0], [1.05, 0.0, 0.0]])
    cell = np.diag([5.0, 5.0, 5.0])
    free = (False, False, False)
    assert check(xy, cell, free) == 0
    xy[1, 0] = 0.95
    assert check(xy, cell, free) == 2
    xy[1, 0] = 1.08
    assert check(xy, cell, free) == 0
    assert search.call_count == 1
    xy[1, 0] = 1.0
    check(xy, cell, free)  # Match the backend's exact-shell convention.
    xy[1, 0] = 2.1
    check(xy, cell, free)
    assert search.call_count == 3

    # Include periodic self images, triclinic shifts and wrapped coordinates.
    cell = np.array([[1.85, 0.0, 0.0], [0.3, 2.1, 0.0], [0.1, 0.2, 5.0]])
    slab = (True, True, False)
    for dx in (0.0, 0.07, 1.85):
        check(xy + [[dx, 0, 0], [0, 0, 0]], cell, slab, 2.05)
    assert check(xy[:1], cell, slab, 2.05) > 0
    assert check(xy[:1], cell * 2, slab, 2.05) == 0
    check(xy, cell, (True, False, False), 1.9)

    # Small affine shears reuse image indices but not the old Cartesian shifts.
    pbc = (True,) * 3
    check(xy, cell, pbc, 2.07)
    searches = search.call_count
    strain = np.array([[1.01, 0.07, 0.0], [0.0, 0.99, 0.0], [0.0, 0.0, 1.0]])
    check(xy @ strain, cell @ strain, pbc, 2.07)
    assert search.call_count == searches
    # Compression can move a previously absent periodic image inside the cutoff.
    check(xy * 0.6, cell * 0.6, pbc, 2.07)
    assert search.call_args_list[searches].args[1] == 2.07 + cache.skin

    check(
        np.array([[0.05, 0.1, 0.3], [0.85, 0.6, 0.4]]),
        np.diag([0.0, 3.0, 3.0]),
        (True, False, False),
        0.4,
    )
    rng = np.random.default_rng(3)
    start, direction = rng.normal(size=(2, 3))
    direction /= np.linalg.norm(direction)
    cell = np.array([[3.7, 0.37, 0.15], [-0.75, 3.4, 0.5], [0.5, -0.13, 2.9]])
    for delta in (-1e-10, 0.0, 1e-10):
        r = 1.219123654
        check(
            np.stack((start, start + 20000 * cell[0] + (r + delta) * direction)),
            cell,
            pbc,
            r,
        )


def test_vesin_rejects_singular_periodic_plane():
    from tace.dataset.neighbour_list import VerletNeighborList, get_neighborhood

    for neighbors in (get_neighborhood, VerletNeighborList(0.5)):
        with pytest.raises(ValueError, match="nonsingular neighbor-search cell"):
            neighbors(
                positions=np.zeros((2, 3)),
                cutoff=1.0,
                pbc=(True, True, False),
                lattice=np.array([[2.0, 0, 0], [2.0, 0, 0], [0.0, 0, 2.0]]),
                max_neighbors=None,
                backend="vesin",
            )


def test_ase_neighbor_skin_energy_forces_stress(double_precision):
    from .test_model_loading import make_model

    model = make_model().cpu()
    ordinary = TACEAseCalc(model, device="cpu")
    cached = TACEAseCalc(model, device="cpu", neighbor_skin=0.6)
    atoms = Atoms(
        "HCO",
        positions=[[0, 0, 0], [1.3, 0.2, 0], [4.05, 0.1, 0]],
        pbc=True,
        cell=[7.3, 6.8, 5.6],
    )
    for x, symbol in [(4.05, "O"), (3.92, "C"), (4.08, "H")]:
        atoms.positions[2, 0] = x
        atoms[2].symbol = symbol
        ordinary.calculate(atoms)
        cached.calculate(atoms)
        for key in ("energy", "forces", "stress"):
            np.testing.assert_allclose(
                cached.results[key], ordinary.results[key], rtol=1e-10, atol=1e-10
            )


@pytest.mark.parametrize("edges", [0, 4])
def test_ase_single_graph_staging_and_batch_observers(edges):
    from torch.nn.modules import module as module_hooks
    from torch_geometric.data import Batch, Data

    from tace.interface.ase.calculator import _single_model_input
    from tace.models.adapter import TensorModel

    # Include a noncontiguous input, a scalar, node, cell, edge and pair
    # tensors. A PyG singleton cat isolates user-owned source storage.
    graph = Data(
        num_nodes=3,
        positions=torch.arange(18.0).view(3, 6)[:, ::2],
        entropy=torch.tensor(2.3),
        fidelity_idx=torch.tensor([2]),
        lattice=torch.eye(3)[None],
        edge_index=torch.zeros(2, edges, dtype=torch.long),
        edge_shifts=torch.zeros(edges, 3, dtype=torch.int64),
        initial_noncollinear_magmoms=torch.randn(3, 3),
        direct_hessian=torch.randn(9, 3, 3),
    )
    # This test exercises collation, not a dummy readout.
    model = TensorModel.__new__(TensorModel)
    torch.nn.Module.__init__(model)
    cpu = torch.device("cpu")
    expected = Batch.from_data_list([graph])
    got = _single_model_input(model, graph, cpu)
    assert type(got) is dict and got.keys() == dict(expected.items()).keys()
    for k, v in expected.items():
        if isinstance(v, torch.Tensor):
            torch.testing.assert_close(got[k], v, rtol=0, atol=0, check_stride=True)
        else:
            assert got[k] == v
    before = graph.positions.clone()
    got["positions"].zero_()
    torch.testing.assert_close(graph.positions, before)

    def assert_batch(m, g):
        out = _single_model_input(m, g, cpu)
        assert isinstance(out, Batch) and out.num_graphs == 1
        torch.testing.assert_close(out.to_data_list()[0].positions, g.positions)

    with model.register_forward_pre_hook(lambda *_: None):
        assert_batch(model, graph)
    with module_hooks.register_module_forward_hook(lambda *_: None):
        assert_batch(model, graph)

    class Custom(TensorModel):
        pass

    host = Custom.__new__(Custom)
    torch.nn.Module.__init__(host)
    assert_batch(host, graph)
    graph.note = "nonstandard conditioning"
    assert_batch(model, graph)


def test_ase_native_model_hooks_receive_real_batch(double_precision):
    from torch_geometric.data import Batch

    from .test_model_loading import make_model

    atoms = Atoms(
        "HCO",
        positions=[[0, 0, 0], [1.3, 0.2, 0], [2.1, 0.1, 0]],
        cell=[8.1, 7.3, 6.8],
        pbc=True,
    )
    calc = TACEAseCalc(make_model().cpu(), device="cpu")
    calc.calculate(atoms)
    expected = dict(calc.results)
    seen = []

    def observer(_, args):
        assert isinstance(args[0], Batch)
        seen.append(args[0].get_example(0).num_nodes)

    with calc.model.register_forward_pre_hook(observer):
        calc.calculate(atoms)
    assert seen == [3]
    for k, v in expected.items():
        np.testing.assert_allclose(calc.results[k], v, rtol=0, atol=0)


@pytest.mark.parametrize("custom_keys", [False, True])
def test_diatomic_reference_labels(tmp_path, custom_keys):
    from ase.calculators.singlepoint import SinglePointCalculator

    from tace.scripts.plot_diatom import read_reference

    atoms = Atoms("H2", positions=[[0, 0, 0], [1.2, 0, 0]])
    forces = np.array([[0.25, 0.0, 0.0], [-0.25, 0.0, 0.0]])
    atoms.calc = SinglePointCalculator(atoms, energy=-1.5, forces=forces)
    energy_key, forces_key = "energy", "forces"
    if custom_keys:
        energy_key, forces_key = "ref_energy", "ref_forces"
        atoms.info[energy_key] = -2.0
        atoms.arrays[forces_key] = forces * 2
    path = tmp_path / "diatomic.xyz"
    write(path, atoms)
    rows = read_reference(path, [1], energy_key, forces_key)[1, 1]
    distance, _, energy, radial_force, _ = rows[0]
    assert distance == pytest.approx(1.2)
    assert energy == (-2.0 if custom_keys else -1.5)
    assert radial_force == (-0.5 if custom_keys else -0.25)


@pytest.mark.parametrize("graph_input", [False, True])
@pytest.mark.parametrize("test_mode", [0, 1])
def test_eval_u_shift_only_changes_reference_labels(
    tmp_path, monkeypatch, graph_input, test_mode
):
    from ase.io import read

    from tace.dataset.element import build_element_lookup
    from tace.dataset.graph import from_atoms
    from tace.dataset.quantity import KeySpecification
    from tace.dataset.store import write_graphs
    from tace.scripts import eval as eval_script

    monkeypatch.setenv("TACE_USE_U_SHIFT", "1")
    atoms_list = [Atoms("FeO", positions=[[0, 0, 0], [10, 0, 0]]), Atoms("Fe")]
    for atoms in atoms_list:
        atoms.info["energy"] = -10.0
    if graph_input:
        source = tmp_path / "cache"
        dataset = write_graphs(
            [
                from_atoms(
                    None,
                    atoms,
                    cutoff=2.0,
                    max_neighbors=None,
                    keyspec=KeySpecification.from_defaults(),
                    target_property=["energy"],
                    embedding_property=[],
                    neighborlist_backend="ase",
                )
                for atoms in atoms_list
            ],
            [source],
            "test0",
        )
        dataset.close()
        source = source / "*.lmdb"
    else:
        source = tmp_path / "input.xyz"
        write(source, atoms_list)
    output = tmp_path / "predicted.xyz"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "tace-eval",
            "-i",
            str(source),
            "-o",
            str(output),
            "-m",
            "model.pt",
            "--device",
            "cpu",
            "--nl_backend",
            "ase",
            "--test",
            str(test_mode),
        ],
    )
    prediction = torch.tensor([-11.925, -10.0], dtype=torch.float64)
    model = Mock(side_effect=lambda batch: {"energy": prediction})
    model.parameters.return_value = []
    model.get_max_neighbors.return_value = None
    model.get_cutoff.return_value = 2.0
    model.get_torch_element.return_value = build_element_lookup([8, 26])
    model.get_target_property.return_value = ["energy"]
    model.get_embedding_property.return_value = []
    monkeypatch.setattr(eval_script, "load_tace", lambda *args, **kwargs: model)
    update_metrics = eval_script.update_metrics
    labels_seen = []

    def check_metrics(metrics, prefix, predictions, labels, properties):
        labels_seen.append(labels.energy.clone())
        update_metrics(metrics, prefix, predictions, labels, properties)

    monkeypatch.setattr(eval_script, "update_metrics", check_metrics)
    eval_script.main()
    assert len(labels_seen) == test_mode
    if test_mode:
        torch.testing.assert_close(labels_seen[0], prediction)
    assert [atoms.info["TACE_energy"] for atoms in read(output, ":")] == pytest.approx(
        prediction.tolist()
    )
    if not graph_input:
        assert [atoms.get_potential_energy() for atoms in read(source, ":")] == [
            -10.0,
            -10.0,
        ]


@pytest.mark.parametrize("interface", ["ase", "general_ase", "torchsim", "nvalchemi"])
@pytest.mark.parametrize(
    "enabled, previous, matmul",
    [(False, None, None), (False, "1", "ieee"), (True, "0", "tf32")],
)
def test_calculator_loading_options(monkeypatch, interface, enabled, previous, matmul):
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
    if previous is None:
        monkeypatch.delenv("TACE_USE_LUNA", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_LUNA", previous)
    loader = Mock(side_effect=RuntimeError("model loading reached"))
    monkeypatch.setattr(module, "load_tace", loader)
    options = {"enable_luna": True} if enabled else {}
    if interface != "nvalchemi":
        options["matmul"] = matmul
    with pytest.raises(RuntimeError, match="model loading reached"):
        factory("model.pt2", device="cpu", **options)
    loader.assert_called_once()
    assert os.environ.get("TACE_USE_LUNA") == ("1" if enabled else previous)

    if interface != "nvalchemi":
        assert loader.call_args.kwargs["matmul"] == matmul


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
    with torch.no_grad():
        energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)
    assert "fidelity_idx" not in atoms.info
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


@pytest.fixture(scope="session")
def lammps_kokkos():
    if not torch.cuda.is_available():
        pytest.skip("LAMMPS ML-IAP requires CUDA Kokkos.")
    lammps = pytest.importorskip("lammps")
    probe = lammps.lammps(cmdargs=["-screen", "none", "-log", "none"])
    try:
        if (
            not probe.has_package("ML-IAP")
            or "cuda" not in probe.accelerator_config["KOKKOS"]["api"]
        ):
            pytest.skip("LAMMPS was built without ML-IAP or CUDA Kokkos.")
    finally:
        probe.close()
    try:
        yield lammps, pytest.importorskip("lammps.mliap")
    finally:
        probe.lib.lammps_kokkos_finalize()


@pytest.mark.parametrize("flag", [None, 0, 1, 2, 3, 4])
def test_kokkos_energy_requests_and_directed_pair_rows(flag):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from tace.interface.lammps import TACELammpsCalc

    data = SimpleNamespace(
        npairs=3,
        pair_i=np.array([0, 1, 0], dtype=np.int32),
        pair_j=np.array([1, 3, 2], dtype=np.int32),
        rij=np.ones((3, 3)),
        eatoms=torch.full((2,), -7.0, dtype=torch.float64),
        energy=-11.0,
        update_pair_forces=Mock(),
    )
    if flag is not None:
        data.eflag = flag
    calc = SimpleNamespace(device=torch.device("cpu"))
    batch = TACELammpsCalc._prepare_batch(
        calc, data, 2, 2, torch.zeros(4, dtype=torch.long)
    )
    assert (
        batch["edge_index"].dtype == torch.int64 and batch["edge_index"].is_contiguous()
    )
    assert batch["edge_index"].tolist() == [[1, 3, 2], [0, 1, 0]]
    assert batch["edge_vector"].requires_grad
    forces = torch.arange(18, dtype=torch.float32).reshape(3, 6)[:, ::2]
    TACELammpsCalc._update_lammps_data(data, torch.tensor([1.0, 2.0, 4.0]), forces, 2)
    (sent,) = data.update_pair_forces.call_args.args
    assert sent.is_contiguous() and sent.dtype == torch.float64
    torch.testing.assert_close(sent, forces.double())
    assert data.eatoms.tolist() == ([-7.0, -7.0] if flag == 0 else [1.0, 2.0])
    assert data.energy == (-11.0 if flag == 0 else 3.0)
    data.npairs = 0
    data.pair_i = data.pair_j = data.rij = None
    empty = TACELammpsCalc._prepare_batch(
        calc, data, 2, 2, torch.zeros(4, dtype=torch.long)
    )
    assert empty["edge_index"].shape == (2, 0) and empty["edge_vector"].shape == (0, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA force handoff")
def test_kokkos_force_only_handoff_synchronizes_new_buffer(monkeypatch):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from tace.interface.lammps import TACELammpsCalc

    source = torch.arange(18, dtype=torch.float64, device="cuda").reshape(3, 6)
    scalar = source.new_zeros(1)
    stream = Mock(wraps=torch.cuda.current_stream())
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: stream)
    dense = source[:, :3].contiguous()
    for forces, waits in (
        (dense, 0),
        (torch._neg_view(dense), 1),
        (source[:, ::2], 1),
        (source[:, ::2].float(), 1),
    ):

        def accept(sent, forces=forces, waits=waits):
            assert (
                not sent.is_neg()
                and sent.dtype == torch.float64
                and sent.is_contiguous()
            )
            assert stream.synchronize.call_count == waits
            assert (sent.data_ptr() == forces.data_ptr()) == (waits == 0)
            torch.testing.assert_close(sent, forces.double())

        data = SimpleNamespace(eflag=0, update_pair_forces=Mock(side_effect=accept))
        stream.reset_mock()
        TACELammpsCalc._update_lammps_data(data, scalar, forces, 1)
        data.update_pair_forces.assert_called_once()


@pytest.mark.parametrize("eflag", [0, 3])
def test_kokkos_raw_force_pointer_uses_logical_sign(eflag):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from tace.interface.lammps import TACELammpsCalc

    force = torch.tensor([[1.0, -2.0, 3.0]], dtype=torch.float64)
    negative = torch._neg_view(force)
    assert (
        negative.is_neg()
        and negative.is_contiguous()
        and negative.data_ptr() == force.data_ptr()
    )

    def accept(sent):
        assert not sent.is_neg()
        # Exactly what a C consumer of the pointer sees, without Tensor dispatch.
        assert list((ctypes.c_double * 3).from_address(sent.data_ptr())) == [
            -1.0,
            2.0,
            -3.0,
        ]

    data = SimpleNamespace(
        eflag=eflag, eatoms=torch.zeros(1), update_pair_forces=Mock(side_effect=accept)
    )
    TACELammpsCalc._update_lammps_data(data, torch.ones(1), negative, 1)
    data.update_pair_forces.assert_called_once()


@pytest.mark.parametrize("accelerated", [False, True])
@pytest.mark.parametrize("geometry", ["periodic", "isolated"])
def test_lammps(checkpoint, atoms, tmp_path, accelerated, geometry, lammps_kokkos):
    from tace.interface.lammps import TACELammpsCalc

    lammps, mliap = lammps_kokkos

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
    def __init__(self, dtype=DTYPE):
        super().__init__()
        self.model_dtype = dtype
        self.register_buffer("cutoff", torch.tensor(3.0, dtype=DTYPE))
        self.register_buffer("atomic_numbers", torch.tensor([13, 14]))
        self.target_property = ["energy", "forces", "stress"]
        self.embedding_property = []
        self.fidelity = {0: "first", 1: "second"}

    def forward(self, data, graph):
        node_energy = (data["node_type"] == 1).to(
            torch.float64
        ) + 3 * graph.node_fidelity
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
        dtype = kwargs.pop("dtype", DTYPE)
        return TACETorchSimCalc(
            TensorModel(PairReadout(dtype)), device=DEVICE, dtype=dtype, **kwargs
        )

    return create


@pytest.mark.parametrize("field", ["atomic_numbers", "system_idx"])
@pytest.mark.parametrize("fixed", [False, True])
def test_torchsim_cache_detects_in_place_changes(
    torchsim_state, torchsim_calculator, field, fixed
):
    kwargs = {field: getattr(torchsim_state, field)} if fixed else {}
    calc = torchsim_calculator(**kwargs)
    calc(torchsim_state)
    assert getattr(calc, field).data_ptr() != getattr(torchsim_state, field).data_ptr()
    if field == "atomic_numbers":
        torchsim_state.atomic_numbers[0] = 13
    else:
        torchsim_state.system_idx[1] = 1
    if fixed:
        with pytest.raises(ValueError, match=field):
            calc(torchsim_state)
        return
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_torchsim_geometry_precision(torchsim_state, torchsim_calculator, dtype):
    calc = torchsim_calculator(dtype=dtype)
    inputs = []
    hook = calc.model.register_forward_pre_hook(
        lambda module, args: inputs.append(args[0])
    )
    try:
        output = calc(torchsim_state)
    finally:
        hook.remove()
    assert calc.dtype == dtype
    assert inputs[0]["positions"].dtype == torch.float64
    assert inputs[0]["lattice"].dtype == torch.float64
    assert inputs[0]["node_type"].dtype == torch.int64
    assert inputs[0]["edge_shifts"].dtype == torch.int64
    assert "node_attrs" not in inputs[0]
    assert all(
        output[key].dtype == torch.float64 for key in ("energy", "forces", "stress")
    )


@pytest.mark.parametrize("periodic", [False, True])
def test_torchsim_boundary_conditions(torchsim_state, torchsim_calculator, periodic):
    torchsim_state.pbc.fill_(periodic)
    if periodic:
        torchsim_state.positions[1, 0] = 7.0
    with torch.no_grad():
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


@pytest.mark.parametrize("device", tuple(dict.fromkeys((DEVICE, torch.device("cpu")))))
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


def test_clean_cli_preserves_outputs_on_help_and_invalid_arguments(
    tmp_path, monkeypatch
):
    from tace.scripts.clean import main

    monkeypatch.chdir(tmp_path)
    log = tmp_path / "out.txt"
    log.write_text("training")
    for args, code in ((["--help"], 0), (["--unknown"], 2)):
        with pytest.raises(SystemExit) as exc:
            main(args)
        assert exc.value.code == code
        assert log.read_text() == "training"

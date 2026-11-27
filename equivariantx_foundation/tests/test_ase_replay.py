"""One true graph package, one ASE runner and one bounded CUDA capture."""

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import torch

pytest.importorskip("ase")

from ase import Atoms

from equivariantx_foundation import optimize
from equivariantx_foundation.targets.ase import replay_calculator
from equivariantx_foundation.tests.test_tace_plan import graph, make_model
from tace.interface.ase import TACEAseCalc
from tace.models.compile import aot
from tace.utils.env import ACCELERATION_ENV


def test_replay_factory_never_reloads_an_existing_model():
    with pytest.raises(TypeError, match="path"):
        replay_calculator(torch.nn.Identity())
    with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA device"):
        replay_calculator("potential.pt2", device="cpu")


def test_default_graph_loader_still_uses_default_runner(monkeypatch):
    loads = []

    class Stub:
        def get_metadata(self):
            return {}

    def loader(*args, **kwargs):
        loads.append((args, kwargs))
        return Stub()

    monkeypatch.setattr(aot, "import_custom_ops_libs", lambda *_: None)
    monkeypatch.setattr(torch._inductor, "aoti_load_package", loader)
    with pytest.raises(ValueError, match="not a TACE"):
        aot.load_aotinductor("example.pt2", "cpu")
    assert loads == [(("example.pt2",), {})]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTI replay")
def test_ase_aoti_one_runner_changing_topology(tmp_path, monkeypatch):
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    native = make_model().eval().cuda()
    model = optimize(native, family="tace", task="inference", backend="cuda")
    sample = graph("cuda")
    for key in ("positions", "node_type", "batch"):
        sample[key] = sample[key][:3].contiguous()
    for key in ("lattice", "pbc", "fidelity_idx"):
        sample[key] = sample[key][:1].contiguous()
    sample["ptr"] = sample["ptr"][:2].contiguous()
    sample["edge_index"] = sample["edge_index"][:, :6].contiguous()
    sample["edge_shifts"] = sample["edge_shifts"][:6].contiguous()
    path = aot.export_aotinductor(
        model, tmp_path / "ase-regional.pt2", sample_data=sample
    )
    reference = TACEAseCalc(native, device="cuda", neighborlist_backend="ase")

    original_load = torch._inductor.aoti_load_package
    loads, calls = [], []

    class Probe:
        def __init__(self, real):
            self.real = real

        def get_metadata(self):
            return self.real.get_metadata()

        def __call__(self, *tensors):
            calls.append(len(tensors))
            return self.real(*tensors)

    def tracked_load(*args, **kwargs):
        loads.append(kwargs.copy())
        return Probe(original_load(*args, **kwargs))

    monkeypatch.setattr(torch._inductor, "aoti_load_package", tracked_load)
    calc = replay_calculator(path, device="cuda", neighborlist_backend="ase")
    runner = calc.model.compiled_model
    assert len(loads) == 1 and loads[0]["run_single_threaded"] and calls == []

    first = Atoms(
        numbers=[1, 8, 1, 8],
        positions=[[0.0, 0.1, 0.2], [2.4, 0.2, 0.3], [4.8, 0.5, 0.2], [7.2, 0.1, 0.4]],
        cell=[15.0, 11.0, 12.0],
        pbc=True,
    )
    swapped = first.copy()
    swapped.positions[[1, 2]] = swapped.positions[[2, 1]]
    expanded = swapped.copy()
    expanded += Atoms("H", positions=[[9.7, 0.2, 0.3]])
    denser = first.copy()
    denser.positions[2, 0] = 3.6

    def efs(atoms, calculator):
        atoms = atoms.copy()
        atoms.calc = calculator
        return dict(
            energy=atoms.get_potential_energy(),
            forces=atoms.get_forces(),
            stress=atoms.get_stress(voigt=False),
        )

    first_edges = None
    errors = []
    for ix, (atoms, expected_calls) in enumerate(
        ((first, 2), (swapped, 2), (expanded, 3), (denser, 4), (first, 4))
    ):
        expected, actual = efs(atoms, reference), efs(atoms, calc)
        for k in expected:
            np.testing.assert_allclose(actual[k], expected[k], atol=3e-5, rtol=4e-4)
        errors.append({k: float(np.abs(actual[k] - expected[k]).max()) for k in actual})
        assert len(loads) == 1 and len(calls) == expected_calls
        if ix == 0:
            recorded = runner.replay
            first_edges = recorded.inputs[runner.edge_index].clone()
        else:
            assert runner.replay is recorded
        if ix == 1:
            current_edges = recorded.inputs[runner.edge_index]
            assert current_edges.shape == first_edges.shape and not torch.equal(
                first_edges, current_edges
            )
    with (
        torch.cuda.stream(torch.cuda.Stream()),
        pytest.raises(ValueError),
    ):
        runner(*recorded.inputs)
    with ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(ValueError):
            executor.submit(runner, *recorded.inputs).result(timeout=10)
    small = list(recorded.inputs)
    for setting in (torch.backends.cuda.matmul, torch.backends.cudnn):
        previous = setting.allow_tf32
        try:
            setting.allow_tf32 = not previous
            with pytest.raises(ValueError):
                runner(*small)
        finally:
            setting.allow_tf32 = previous
    assert recorded.context is runner.context
    small[runner.positions] = small[runner.positions][:1]
    with pytest.raises(ValueError, match="at least two atoms"):
        runner(*small)
    assert (
        len(calls) == 4 and runner.get_metadata()["tace_format"] == aot.TACE_AOTI_FORMAT
    )
    (tmp_path / "ase-report.json").write_text(
        json.dumps(dict(loads=loads, raw_calls=len(calls), errors=errors), indent=2)
    )

import json
import os
import subprocess
import sys
import zipfile
from copy import deepcopy
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Union
from unittest.mock import Mock

import pytest
import torch

import tace.models.compile.wrapper as compile_wrapper
from tace.dataset.quantity import (
    SUPPORTED_EMBEDDING_PROPERTIES,
    get_embedding_property,
    get_need_property,
)
from tace.lightning.torch_model import (
    _prune_removed_keys,
    create_model,
)
from tace.models.compile.aot import (
    TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY,
    TACE_AOTI_LUNA_OPS_ENTRY,
    AOTICompiledTensorModel,
    custom_ops_libs_from_model,
    embed_custom_ops_libs,
    ensure_sample_inputs,
    export_metadata,
    graph_aoti_input_keys,
    graph_dynamic_shapes,
    import_custom_ops_libs,
    size_oblivious_export,
    synthetic_graph_sample,
)
from tace.models.compile.compile import compiled_call, trace_and_compile, trace_to_fx
from tace.models.compile.wrapper import CompileTensorModel, FlatTorchModel
from tace.models.spherical.default import DEFAULT_MODEL_CONFIG
from tace.models.spherical.tace import sum_node_energy


@pytest.mark.parametrize("sample_index,selected", [(":", (0, 1, 2)), ("1::2", (1, 3))])
def test_export_sample_reads_only_requested_batch(
    tmp_path, monkeypatch, sample_index, selected
):
    from itertools import accumulate

    import ase.io
    from ase import Atoms

    from tace.dataset.element import TorchElement
    from tace.scripts.export_eval import _build_sample_data

    frames = [
        Atoms("H" * (i + 2), positions=[(i, j * 0.5, 0) for j in range(i + 2)])
        for i in range(6)
    ]
    path = tmp_path / "sample.extxyz"
    ase.io.write(path, frames)
    iread = ase.io.iread
    consumed = []

    def bounded_iread(path, *, index):
        # Ask the reader itself for a bounded default selection: some ASE
        # formats look for the last frame before yielding an open-ended slice.
        if sample_index == ":":
            assert index == slice(0, len(selected))
        for atoms in iread(path, index=index):
            assert len(consumed) < len(selected), "read past the requested batch"
            consumed.append(len(atoms))
            yield atoms

    monkeypatch.setattr(ase.io, "iread", bounded_iread)
    model = Mock()
    model.get_target_property.return_value = ["energy"]
    model.get_embedding_property.return_value = []
    model.get_torch_element.return_value = TorchElement([1])
    model.get_cutoff.return_value = 0.75
    model.get_max_neighbors.return_value = None
    data = _build_sample_data(
        model, str(path), sample_index, len(selected), "ase", "cpu"
    )
    counts = [len(frames[i]) for i in selected]
    assert consumed == counts
    assert data["ptr"].tolist() == list(accumulate(counts, initial=0))
    assert data["batch"].tolist() == [i for i, n in enumerate(counts) for _ in range(n)]
    expected = torch.cat([torch.from_numpy(frames[i].positions) for i in selected])
    torch.testing.assert_close(data["positions"], expected)


@pytest.fixture
def periodic_les_model(request):
    les = pytest.importorskip("les")
    if les.__version__ != "0.2.0":
        pytest.skip("Frozen LES graph inference targets LES 0.2.0")
    from tace.models.adapter import TensorModel
    from tace.models.cartesian.tace import CartesianTACE
    from tace.models.spherical.tace import SphericalTACE

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        cutoff=3.5,
        num_layers=1,
        num_channel=3,
        Lmax=1,
        lmax=1,
        mmax=1,
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=3.0, atomic_energy={1: 0.0})
        ],
        target_property=["energy", "forces", "stress"],
    )
    config["radial_basis"].update(hidden=[4], num_radial_basis=3)
    config["readout_emlp"].update(hidden=[4])
    config["scale_shift"]["enable"] = False
    config["long_range"]["les"]["enable"] = True
    config["long_range"]["les"]["les_arguments"].update(is_periodic=True, N_max=3)
    cls = SphericalTACE
    if getattr(request, "param", None) == "cartesian":
        cls = CartesianTACE
        config["num_layers"] = 2  # Non-scalar Cartesian descriptors in the first layer.
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(41)
        return TensorModel(cls(**config)).eval()


@pytest.mark.parametrize(
    "periodic_les_model", ["spherical", "cartesian"], indirect=True
)
def test_periodic_les_frozen_graph_boundaries(
    periodic_les_model, tmp_path, monkeypatch
):
    import ase.io
    from ase import Atoms

    from tace.models.cartesian.tace import CartesianTACE
    from tace.models.compile import aot
    from tace.models.spherical.les import is_periodic_charge_only_les
    from tace.scripts.export_eval import _build_sample_data

    native = periodic_les_model
    readout = native.readout_fn
    wrapped = aot._frozen_graph_model(native)
    frame = synthetic_graph_sample(wrapped)
    frame["atomic_numbers"] = readout.atomic_numbers[frame["node_type"]]
    native(dict(frame))  # Warming normal e3nn ScriptMethod caches is harmless.
    assert is_periodic_charge_only_les(readout.les)
    if type(readout) is CartesianTACE:
        with pytest.raises(ValueError, match="not joint training"):
            aot.as_joint_export_model(native, frame)
    with pytest.raises(ValueError, match="does not support LES"):
        aot.as_compile_tensor_model(native)
    with pytest.raises(ValueError, match="does not support LES"):
        aot.export_lammps_aotinductor(native, tmp_path / "lammps.pt2")
    for obj, key, value in [
        (readout.les.les, "is_periodic", None),
        (readout.les.les.ewald, "is_periodic", False),
        (readout.les.les, "compute_bec", True),
        (readout.les.les, "use_fixed_atomic_charges", True),
        (readout.les, "_solve", lambda **kwargs: {}),
        (readout.representation, "use_magnetic_interaction", True),
        (readout, "embedding_property", ["electric_field"]),
        (readout, "readout_fn", lambda *args, **kwargs: {}),
        (native, "first_derivative_fn", lambda *args, **kwargs: {}),
    ]:
        with monkeypatch.context() as patch:
            patch.setattr(obj, key, value, raising=False)
            with pytest.raises(ValueError, match="charge-only"):
                aot._frozen_graph_model(native)
    with readout.representation.register_forward_hook(lambda *_: None):
        with pytest.raises(ValueError, match="charge-only"):
            aot._frozen_graph_model(native)
    for pbc, cell in [
        (None, frame["lattice"]),
        (~frame["pbc"], frame["lattice"]),
        (frame["pbc"] & frame["pbc"].new_tensor([True, False, True]), frame["lattice"]),
        (frame["pbc"][:, :2], frame["lattice"]),
        (frame["pbc"], -frame["lattice"]),
        (frame["pbc"], frame["lattice"] * torch.nan),
        (frame["pbc"], frame["lattice"] * 0),
    ]:
        with pytest.raises(ValueError, match="fully periodic"):
            aot.export_aotinductor(
                native, tmp_path / "bad.pt2", frame | {"pbc": pbc, "lattice": cell}
            )
    for name in ("electric_field", "initial_noncollinear_magmoms"):
        with pytest.raises(ValueError, match="fields or spin"):
            aot.export_aotinductor(
                native, tmp_path / "bad.pt2", frame | {name: torch.zeros(1, 3)}
            )
    path = tmp_path / "field.extxyz"
    ase.io.write(
        path,
        Atoms(
            "H2",
            positions=[[0, 0, 0], [1, 0, 0]],
            cell=[3] * 3,
            pbc=True,
            info={"electric_field": [0.1, 0, 0]},
        ),
    )
    with pytest.raises(ValueError, match="fields or spin"):
        _build_sample_data(native, str(path), ":", 1, "ase", "cpu")

    # The CLI's frozen-load exception must not disable compiled ordinary
    # models, or make an LES host eligible for live compilation after loading.
    from tace.models.adapter import TensorModel

    monkeypatch.setenv("TACE_USE_COMPILE", "1")
    cfg = deepcopy(readout.model_config)
    args = (readout.statistics, native.get_target_property(), [])
    token = compile_wrapper._loading_frozen_graph.set(True)
    try:
        assert type(create_model(cfg, *args)) is TensorModel
        ordinary = deepcopy(cfg)
        ordinary["long_range"]["les"]["enable"] = False
        assert isinstance(create_model(ordinary, *args), CompileTensorModel)
    finally:
        compile_wrapper._loading_frozen_graph.reset(token)
    with pytest.raises(RuntimeError) as exc:
        create_model(cfg, *args)
    assert "does not support LES" in str(exc.value.__cause__)


@pytest.fixture
def periodic_les_frames():
    from ase import Atoms

    return [
        Atoms(
            "H" * n,
            scaled_positions=[
                [0.12 + i * 0.36, (i % 2) * 0.2, 0.1 * i] for i in range(n)
            ],
            cell=[[4.2 + 0.2 * n, 0, 0], [0.3 * n, 5, 0], [0.1, 0.2, 4.5]],
            pbc=True,
        )
        for n in (2, 3, 4)
    ]


@pytest.mark.parametrize(
    "periodic_les_model", ["spherical", "cartesian"], indirect=True
)
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA AOTInductor"
            ),
        ),
    ],
)
def test_periodic_les_cli_dynamic_package_without_solver(
    periodic_les_model, periodic_les_frames, tmp_path, device
):
    import ase.io

    from tace.lightning import export_tace
    from tace.models.compile.aot import _frozen_graph_model
    from tace.scripts.export_eval import _build_sample_data

    native = periodic_les_model
    frames = periodic_les_frames
    keys = ("energy", "node_energy", "forces", "stress")
    cases = []
    wrapped = _frozen_graph_model(native)
    input_keys = graph_aoti_input_keys(wrapped)
    for counts in (frames[1:2], frames[:2], frames):
        xyz = tmp_path / "graph.extxyz"
        ase.io.write(xyz, counts)
        d = _build_sample_data(native, str(xyz), ":", len(counts), "ase", "cpu")
        gold = native(dict(d))
        ensure_sample_inputs(d, input_keys, wrapped)
        cases.append((d, {k: gold[k].detach() for k in keys}))
    assert (
        len(
            {
                (d["positions"].size(0), d["edge_index"].size(1), d["lattice"].size(0))
                for d, _ in cases
            }
        )
        == 3
    )
    assert any(
        abs(float(y["energy"].sum() - y["node_energy"].sum())) > 1e-14 for _, y in cases
    )
    torch.save(cases, tmp_path / "ref.pt")
    export_tace(native, str(tmp_path / "fixture.pt"))
    env = dict(os.environ, OMP_NUM_THREADS="2", TORCHINDUCTOR_COMPILE_THREADS="2")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "tace.scripts.export_eval",
            "-m",
            str(tmp_path / "fixture.pt"),
            "--backend",
            "aoti",
            "--device",
            device,
            "-o",
            str(tmp_path / "model.pt2"),
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json, sys
sys.modules.update(tace=None, les=None)
import torch
import equivariantx  # Register generic operators, without either host package.
from torch._inductor import aoti_load_package
runner = aoti_load_package(sys.argv[1])
metadata = runner.get_metadata()
in_keys, out_keys = (json.loads(metadata["tace_" + side + "_keys"]) for side in ("input", "output"))
assert metadata["tace_periodic_les"] == "charge_only_v1"
for d, expected in torch.load(sys.argv[2], weights_only=True):
    actual = dict(zip(out_keys, runner(*(d[k].to(sys.argv[3]) for k in in_keys)), strict=True))
    for key, value in expected.items():
        torch.testing.assert_close(actual[key].cpu(), value, rtol=1e-4, atol=3e-6)
""",
            str(tmp_path / "model.pt2"),
            str(tmp_path / "ref.pt"),
            device,
        ],
        env=env,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import sys, torch
sys.modules['les'] = None
from tace.lightning import load_tace
device = sys.argv[3]
model = load_tace(sys.argv[1], device=device)
assert model.metadata["tace_periodic_les"] == "charge_only_v1"
for d, gold in torch.load(sys.argv[2], weights_only=True):
    d = {k: v.to(device) if torch.is_tensor(v) else v for k,v in d.items()}
    out = model(d)
    for k, v in gold.items():
        torch.testing.assert_close(out[k].cpu(), v, rtol=1e-4, atol=3e-6)
# Device-side asserts poison the CUDA context; only check synchronous rejection here.
bad = [{"electric_field": torch.zeros(1, 3)}, {"pbc": d["pbc"][:,:1]}]
if device == "cpu":
    bad.extend([{"pbc": ~d["pbc"]}, {"pbc": d["pbc"] & d["pbc"].new_tensor([True, False, True])},
                {"lattice": -d["lattice"]}])
for invalid in bad:
    try:
        model(d | invalid)
    except ValueError:
        continue
    raise AssertionError("accepted invalid frozen LES inputs")
""",
            str(tmp_path / "model.pt2"),
            str(tmp_path / "ref.pt"),
            device,
        ],
        env=env,
        check=True,
    )


@pytest.mark.parametrize("dynamic", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA AOTInductor"
            ),
        ),
    ],
)
def test_periodic_les_joint_package_updates_all_live_state(
    periodic_les_model, periodic_les_frames, tmp_path, dynamic, device, monkeypatch
):
    """Full E/F/S mixed derivatives, including the charge head and trainable loss."""
    import ase.io

    from equivariantx_foundation.targets import aoti
    from tace.scripts.export_eval import _build_sample_data
    from tace.utils.loss.uncertainty import UncertaintyLoss

    model = periodic_les_model.to(device).train()
    objective = (
        UncertaintyLoss(
            ["energy", "forces", "stress"],
            ["mae_energy_per_atom", "l2mae_forces", "mae_voigt_stress"],
            [1.0, 2.0, 3.0],
        )
        .to(device)
        .train()
    )
    reference, reference_loss = deepcopy(model), deepcopy(objective)
    batches = []
    for structures in (
        periodic_les_frames[:2],
        periodic_les_frames[1:] if dynamic else periodic_les_frames[:2],
    ):
        xyz = tmp_path / "cells.xyz"
        ase.io.write(xyz, structures, format="extxyz")
        batch = _build_sample_data(model, str(xyz), ":", 2, "ase", "cpu")
        n = batch["positions"].size(0)
        batch = {
            k: batch[k]
            for k in (
                "positions",
                "atomic_numbers",
                "edge_index",
                "edge_shifts",
                "lattice",
                "pbc",
                "batch",
                "ptr",
            )
        }
        batch["entropy"] = torch.ones(2, dtype=torch.float64)
        for key, shape in (("energy", (2,)), ("forces", (n, 3)), ("stress", (2, 3, 3))):
            batch[key] = (
                torch.arange(torch.Size(shape).numel(), dtype=torch.float64).reshape(
                    shape
                )
                * 0.003
                + 0.12
            )
            batch[key + "_weight"] = batch["entropy"].clone()
        if not batches:
            # Only force/stress derivatives train the charge head on this step.
            batch["energy_weight"].zero_()
        batches.append({k: v.to(device) for k, v in batch.items()})
    path = tmp_path / "joint.pt2"
    if device == "cpu":
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA device"):
            aoti.export_training(
                model, objective, batches[0], path, dynamic_shapes=dynamic
            )
        return
    aoti.export_training(model, objective, batches[0], path, dynamic_shapes=dynamic)
    step = aoti.load_training(path)
    prepared, params, buffers = aoti.training_inputs(model, objective, batches[-1])
    for module in (model, model.readout_fn.les, model.readout_fn.les.les.ewald):
        with module.register_forward_pre_hook(lambda *_: None):
            with pytest.raises(ValueError, match="charge-only"):
                aoti.validate_training_model(step, model, objective, prepared)
    original_autograd_threading = torch.autograd.is_multithreading_enabled()
    aoti.validate_training_model(step, model, objective, prepared)
    assert torch.autograd.is_multithreading_enabled() == original_autograd_threading
    _, ref_params, _ = aoti.training_inputs(reference, reference_loss, batches[-1])
    optim = torch.optim.AdamW(params.values(), lr=0.001, eps=1e-5)
    ref_optim = torch.optim.AdamW(ref_params.values(), lr=0.001, eps=1e-5)
    for batch in batches:
        data = aoti.training_batch(model, objective, batch)
        pred, loss, grads = step(data, params, buffers)
        expected = reference(batch)
        target = reference_loss(expected, data)
        ref_optim.zero_grad(set_to_none=True)
        target.backward()
        torch.testing.assert_close(loss, target, rtol=1e-4, atol=5e-6)
        for k in ("energy", "node_energy", "forces", "stress"):
            torch.testing.assert_close(pred[k], expected[k], rtol=1e-4, atol=3e-6)
        assert grads.keys() == ref_params.keys() == params.keys()
        assert any(
            g is not None and g.abs().max() > 0
            for k, g in grads.items()
            if "latent_charges_readouts" in k
        )
        assert any(
            g is not None and g.abs().max() > 0
            for k, g in grads.items()
            if k.startswith("loss_fn.")
        )
        optim.zero_grad(set_to_none=True)
        for k, p in params.items():
            grad = ref_params[k].grad
            torch.testing.assert_close(grads[k], grad, rtol=1e-3, atol=3e-6)
            p.grad = grads[k]
        if dynamic and batch is batches[0]:
            # The raw training ABI must retain the physical input contract and
            # all gradients without importing TACE or the optional solver.
            inputs = tmp_path / "standalone.pt"
            gold = (
                {k: expected[k].detach() for k in pred},
                target.detach(),
                {k: p.grad for k, p in ref_params.items()},
            )
            torch.save(((data, params, buffers), gold), inputs)
            subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys, torch
sys.modules["tace"] = sys.modules["les"] = None
from equivariantx_foundation.targets.aoti import load_training
args, expected = torch.load(sys.argv[2], weights_only=True)
step = load_training(sys.argv[1])
torch.testing.assert_close(step(*args), expected, rtol=1e-3, atol=5e-6)
""",
                    str(path),
                    str(inputs),
                ],
                check=True,
            )
        optim.step()
        ref_optim.step()
        for k, p in params.items():
            torch.testing.assert_close(p, ref_params[k], rtol=1e-4, atol=3e-6)
    with pytest.raises(ValueError, match="electric_field"):
        step(data | {"electric_field": torch.ones(2, 3)}, params, buffers)
    if device == "cpu":
        with pytest.raises(RuntimeError, match="fully periodic"):
            step(data | {"pbc": ~data["pbc"]}, params, buffers)
    elif dynamic:
        # An async CUDA assertion poisons its context. Exercise the checked
        # standalone loader in separate processes, with neither host installed.
        for bad in ("pbc", "lattice"):
            proc = subprocess.run(
                [
                    sys.executable,
                    "-c",
                    """
import sys, torch
sys.modules["tace"] = sys.modules["les"] = None
from equivariantx_foundation.targets.aoti import load_training
(d,p,b), _ = torch.load(sys.argv[2],weights_only=True)
k=sys.argv[3]
d[k] = ~d[k] if k=="pbc" else -d[k]
load_training(sys.argv[1])(d,p,b)
torch.cuda.synchronize()
""",
                    str(path),
                    str(inputs),
                    bad,
                ],
                capture_output=True,
                text=True,
            )
            assert proc.returncode != 0 and "finite positive volume" in proc.stderr
    with pytest.raises(ValueError, match="fully periodic"):
        aoti.export_training(
            model,
            objective,
            batch | {"lattice": -batch["lattice"]},
            tmp_path / "bad.pt2",
        )
    with monkeypatch.context() as patch:
        patch.setattr(torch.distributed, "is_initialized", lambda: True)
        patch.setattr(torch.distributed, "get_world_size", lambda: 2)
        with pytest.raises(ValueError, match="single-rank"):
            step(data, params, buffers)
        with pytest.raises(ValueError, match="single-rank"):
            aoti.export_training(model, objective, batch, tmp_path / "bad.pt2")


def test_export_node_energy_allows_one_or_more_structures():
    class Readout(torch.nn.Module):
        def forward(self, energy, batch, ptr):
            return sum_node_energy(energy, batch, ptr.size(0) - 1)

    def example(counts):
        counts = torch.tensor(counts)
        ptr = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        batch = torch.repeat_interleave(torch.arange(len(counts)), counts)
        energy = torch.arange(len(batch), dtype=torch.float64) - 0.75
        return energy, batch, ptr

    n = torch.export.Dim("nodes", min=2, max=24)
    p = torch.export.Dim("ptr_length", min=2, max=5)
    # Use the same size-oblivious policy as TACE artifact export: without it
    # even torch.zeros(ptr.size(0) - 1) may guard that its size exceeds one.
    with size_oblivious_export():
        exported = torch.export.export(
            Readout(),
            example([2, 3]),
            strict=True,
            dynamic_shapes=({0: n}, {0: n}, {0: p}),
        ).module()
    for counts in ([5], [2, 3], [3, 1, 2]):
        energy, batch, ptr = example(counts)
        expected = torch.stack([part.sum() for part in energy.split(counts)])
        torch.testing.assert_close(exported(energy, batch, ptr), expected)


def test_fixed_frozen_graph_export_preserves_values_and_rejects_new_shapes(
    tmp_path, monkeypatch
):
    from tace.models.adapter import TensorModel
    from tace.models.compile import aot
    from tace.models.spherical.tace import SphericalTACE

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["scale_shift"]["enable"] = False
    config.update(
        cutoff=3.0,
        Lmax=1,
        lmax=1,
        mmax=1,
        num_layers=1,
        num_channel=2,
        target_property=["energy", "forces"],
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: -1.0})
        ],
    )
    native = TensorModel(SphericalTACE(**config)).eval()
    observed = {}

    def inspect(program, *, package_path, inductor_configs):
        assert not program.range_constraints
        observed["runner"] = AOTICompiledTensorModel(
            program.module(), inductor_configs["aot_inductor.metadata"], "cpu"
        )
        return package_path

    monkeypatch.setattr(aot, "compile_and_package", inspect)
    monkeypatch.setattr(aot, "embed_custom_ops_libs", lambda *_: None)
    monkeypatch.setattr(aot, "ensure_cxx_compiler", lambda: None)
    path = tmp_path / "fixed.pt2"
    with pytest.raises(ValueError, match="sample_data"):
        aot.export_aotinductor(native, path, fixed_shapes=True)
    sample = synthetic_graph_sample(aot.as_compile_tensor_model(native))
    shifts = sample["edge_shifts"].to(torch.float32)
    unused = torch.zeros(2, 3, 3, dtype=torch.float32).transpose(1, 2)
    sample.update(edge_shifts=shifts, direct_hessian=unused)
    wrapped = aot.as_compile_tensor_model(native)
    keys = aot.graph_aoti_input_keys(wrapped)
    prepared = dict(sample)
    aot.ensure_sample_inputs(prepared, keys, wrapped)
    assert prepared["direct_hessian"] is unused
    assert aot.canonicalize_inputs(sample, keys, 0)["direct_hessian"] is unused
    torch.testing.assert_close(prepared["edge_shifts"], shifts.long())
    assert sample["edge_shifts"] is shifts
    aot.export_aotinductor(native, path, sample_data=sample, fixed_shapes=True)
    for shift in (0.0, 0.15):
        frame = sample | {"positions": sample["positions"].clone()}
        frame["positions"][0, 0] += shift
        if shift:
            frame["positions"] = torch._neg_view(frame["positions"])
            assert not aot.canonicalize_inputs(frame, ("positions",), 0)[
                "positions"
            ].is_neg()
        expected, actual = native(frame), observed["runner"](frame)
        for key in config["target_property"]:
            torch.testing.assert_close(actual[key], expected[key], rtol=1e-4, atol=1e-5)
    with pytest.raises(ValueError, match="Fixed AOTI input 'edge_index'"):
        observed["runner"](sample | {"edge_index": sample["edge_index"][:, :-2]})


@pytest.mark.parametrize("target", ["graph", "lammps"])
def test_compiled_element_inputs(target):
    from tace.models.compile import aot

    metadata = {
        "tace_input_keys": '["node_type"]',
        "tace_output_keys": '["energy", "node_energy", "edge_forces"]',
        "tace_target_property": '["energy"]',
        "tace_embedding_property": "[]",
        "tace_atomic_numbers": "[1, 8, 26]",
        "tace_cutoff": "6.0",
        "tace_max_neighbors": "null",
        "tace_fidelity_idx": "0",
        "tace_dtype": "float32",
    }
    wrapper = (
        aot.AOTICompiledTensorModel
        if target == "graph"
        else aot.AOTICompiledLammpsModel
    )
    model = wrapper(lambda indices: (indices, indices, indices), metadata, "cpu")
    for numbers, expected in [([26, 1, 8], [2, 0, 1]), ([8, 8], [1, 1]), ([], [])]:
        for key, values in [("atomic_numbers", numbers), ("node_type", expected)]:
            data = {key: torch.tensor(values, dtype=torch.int64)}
            out = model(data)
            assert (out["energy"] if target == "graph" else out[0]).tolist() == expected
            assert set(data) == {key}
    for z in (-1, 27):
        with pytest.raises((IndexError, RuntimeError)):
            model({"atomic_numbers": torch.tensor([z])})
    with pytest.raises(RuntimeError, match="not supported"):
        model({"atomic_numbers": torch.tensor([2])})
    for idx in (-1, 3):
        with pytest.raises(RuntimeError, match="outside"):
            model({"node_type": torch.tensor([idx])})
    with pytest.raises(TypeError, match="one-dimensional"):
        model({"node_type": torch.zeros(1, 1, dtype=torch.int64)})


@pytest.mark.parametrize("n,guard", [(2, "Eq"), (3, "Ne")])
def test_frozen_graph_rejects_python_branch_on_independent_axes(tmp_path, n, guard):
    from tace.models.adapter import TensorModel
    from tace.models.compile import aot
    from tace.models.spherical.tace import SphericalTACE

    class DimerReadout(SphericalTACE):
        def readout_fn(self, data, graph, from_representation, edge_feats):
            out = super().readout_fn(data, graph, from_representation, edge_feats)
            if data["positions"].shape[0] == data["edge_index"].shape[1]:
                out["node_energy"] = out["node_energy"] + 2.0
                out["energy"] = out["energy"] + 2.0 * data["ptr"].diff()
            return out

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["scale_shift"]["enable"] = False
    config.update(
        cutoff=3.0,
        Lmax=1,
        lmax=1,
        mmax=1,
        num_layers=1,
        num_channel=2,
        target_property=["energy"],
        statistics=[
            dict(atomic_numbers=[1], avg_num_neighbors=2.0, atomic_energy={1: 0.0})
        ],
    )
    model = TensorModel(DimerReadout(**config)).eval()
    # A real isolated dimer or triangle with all reciprocal edges: N=E or N!=E.
    edges = (~torch.eye(n, dtype=torch.bool)).nonzero().T.contiguous()
    data = {
        "positions": torch.eye(3, dtype=torch.float64)[:n],
        "node_type": torch.zeros(n, dtype=torch.long),
        "edge_index": edges,
        "edge_shifts": torch.zeros(edges.shape[1], 3, dtype=torch.long),
        "lattice": torch.eye(3, dtype=torch.float64).unsqueeze(0) * 16,
        "pbc": torch.zeros(1, 3, dtype=torch.bool),
        "batch": torch.zeros(n, dtype=torch.long),
        "ptr": torch.tensor([0, n]),
        "fidelity_idx": torch.zeros(1, dtype=torch.long),
    }
    path = tmp_path / "unsafe.pt2"
    with pytest.raises(ValueError, match=rf"{guard}\(.*independent.*fixed_shapes"):
        aot.export_aotinductor(model, path, sample_data=data)
    assert not path.exists()


def test_frozen_trace_accepts_shared_derived_and_disjoint_input_axes():
    def readout(nodes, labels, groups, ptr, edges):
        if nodes.size(0) != labels.size(0) or ptr.size(0) != groups.size(0) + 1:
            return nodes - 1
        if nodes.size(0) == edges.size(0):
            return nodes - 2
        return nodes + 3

    n = torch.export.Dim("nodes", min=2, max=6)
    g = torch.export.Dim("groups", min=1, max=3)
    e = torch.export.Dim("edges", min=8, max=14)
    specs = ({0: n}, {0: n}, {0: g}, {0: g + 1}, {0: e})
    examples = tuple(torch.zeros(k) for k in (5, 5, 3, 4, 9))
    with size_oblivious_export():
        traced = trace_to_fx(readout, examples, dynamic_shapes=specs)
    for counts in ((5, 5, 3, 4, 9), (2, 2, 1, 2, 14)):
        args = tuple(torch.randn(k) for k in counts)
        torch.testing.assert_close(traced(*args), readout(*args))


@pytest.mark.parametrize("exported_op", [False, True])
def test_lammps_exchange_packs_rows_and_gradients(exported_op):
    from torch._subclasses.fake_tensor import FakeTensorMode

    from tace.models.lammps import GhostExchange, use_lammps_mliap_data

    for n in (4, 0):
        # The last two rows are ghost copies of the same owner.
        owner = torch.tensor([0, 1, 0, 0][:n], dtype=torch.long)
        calls = []

        class Exchange:
            def check(self, x, out, width):
                assert x.is_contiguous() and not x.is_neg() and out.is_contiguous()
                assert width == 6 and x.shape == out.shape == (n, width)

            def forward_exchange(self, x, out, width):
                self.check(x, out, width)
                calls.append("forward")
                out.copy_(x[owner])

            def reverse_exchange(self, x, out, width):
                self.check(x, out, width)
                calls.append("reverse")
                out.zero_().index_add_(0, owner, x)

        exchange = Exchange()
        x = torch.arange(6 * n, dtype=torch.float64).reshape(n, 3, 2)
        x = x.transpose(1, 2).requires_grad_()
        with use_lammps_mliap_data(exchange):
            op = torch.ops.tace.lammps_forward_exchange.default
            y = op(x) if exported_op else GhostExchange.apply(x, exchange)
            torch.testing.assert_close(y, x[owner])
            assert y.is_contiguous()
            negative = torch._neg_view(torch.arange(6 * n).to(x).reshape_as(x))
            for grad in (x.new_ones(()).expand_as(y), x.detach(), negative):
                (dx,) = torch.autograd.grad(y, x, grad, retain_graph=True)
                torch.testing.assert_close(
                    dx, torch.zeros_like(x).index_add_(0, owner, grad)
                )
            z = op(negative) if exported_op else GhostExchange.apply(negative, exchange)
            torch.testing.assert_close(z, negative[owner])
            assert calls == ["forward", "reverse", "reverse", "reverse", "forward"]
        if exported_op:
            with FakeTensorMode() as mode:
                fake = mode.from_tensor(x)
                for fn in (op, torch.ops.tace.lammps_reverse_exchange.default):
                    assert fn(fake).stride() == y.stride()

    if exported_op:
        singleton = torch.empty_strided((3, 1), (1, 11))
        exchange = Mock()
        with use_lammps_mliap_data(exchange):
            for fn in (op, torch.ops.tace.lammps_reverse_exchange.default):
                actual = fn(singleton)
                with FakeTensorMode() as mode:
                    assert fn(mode.from_tensor(singleton)).stride() == actual.stride()


def test_lammps_aoti_dynamic_receiver_rows(tmp_path, monkeypatch):
    from tace.models.adapter import TensorModel
    from tace.models.compile import aot
    from tace.models.compile.wrapper import FlatLammpsModel
    from tace.models.lammps import use_lammps_mliap_data
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV

    for name in ACCELERATION_ENV.values():
        monkeypatch.setenv(name, "0")
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg.update(
        cutoff=2.5,
        Lmax=1,
        lmax=1,
        mmax=1,
        num_layers=2,
        num_channel=2,
        statistics=[
            dict(
                atomic_numbers=[1, 2],
                avg_num_neighbors=4.0,
                atomic_energy={1: 0.0, 2: 0.0},
            )
        ],
    )
    cfg["radial_basis"].update(num_radial_basis=4, hidden=[4])
    cfg["product_basis"]["correlation"] = 1
    cfg["scale_shift"]["enable"] = False
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        native = TensorModel(SphericalTACE(**cfg)).eval().requires_grad_(False)

    # This baseline uses all source rows as destinations before truncating
    # ghosts; the package should instead allocate only the real receiver rows.
    padded = deepcopy(native)
    for layer in padded.readout_fn.representation.interactions:
        assert layer.rejector.supports_receiver_count
        layer.rejector.supports_receiver_count = False
    padded = FlatLammpsModel(
        aot.as_compile_tensor_model(padded), aot.LAMMPS_AOTI_INPUT_KEYS
    )

    class Exchange:
        def __init__(self, owner):
            self.owner, self.calls = owner, []
            self.nlocal = int(owner.max()) + 1

        def forward_exchange(self, features, out, width):
            self.calls.append(("forward", width))
            out.copy_(features[self.owner])

        def reverse_exchange(self, grad, out, width):
            self.calls.append(("reverse", width))
            assert grad[self.nlocal :].abs().max() > 1e-10
            out.zero_().index_add_(0, self.owner, grad)

    def ring(n, *, two_shells=True):
        # Reciprocal edges across a periodic boundary reference image atoms,
        # including multiple edges with the same ghost owner.
        offsets = torch.tensor([-2, 1, -1, 2] if two_shells else [1, -1])
        target = torch.arange(n).repeat_interleave(len(offsets))
        j = target + offsets.repeat(n)
        source, image = j % n, j // n
        image_rows, ghost = torch.unique(j[image != 0], return_inverse=True)
        owner = torch.cat((torch.arange(n), image_rows % n))
        local_src = source.clone()
        local_src[image != 0] = n + ghost
        types = torch.arange(n) % 2
        x = torch.arange(n, dtype=torch.float64)
        spacing = 1.0 if two_shells else 1.5
        pos = torch.stack((x * spacing, x.cos() * 0.20, x.sin() * 0.17), dim=-1)
        lattice = torch.diag(x.new_tensor([n * spacing, 5, 5])).unsqueeze(0)
        shifts = torch.zeros((len(j), 3), dtype=torch.long)
        shifts[:, 0] = -image
        common = dict(
            batch=torch.zeros(n, dtype=torch.long),
            ptr=torch.tensor([0, n]),
            fidelity_idx=torch.tensor([0]),
        )
        periodic = dict(
            **common,
            positions=pos,
            node_type=types,
            edge_index=torch.stack((source, target)),
            edge_shifts=shifts,
            lattice=lattice,
            pbc=torch.tensor([[True, False, False]]),
        )
        rank = dict(
            **common,
            node_type=types[owner],
            edge_index=torch.stack((local_src, target)),
            edge_vector=pos[target] - pos[source] + shifts.double() @ lattice[0],
        )
        return periodic, rank, Exchange(owner)

    # Trace at one shape and load a real native CPU artifact for both calls.
    _, sample, _ = ring(4)
    package = aot.export_lammps_aotinductor(native, tmp_path / "owners.pt2", sample)
    runner = aot.load_lammps_aotinductor(package, "cpu")
    for n, two_shells in ((3, True), (7, False)):
        periodic, rank, exchange = ring(n, two_shells=two_shells)
        with use_lammps_mliap_data(exchange):
            energy, nodes, pair = runner(rank)
            assert [name for name, _ in exchange.calls] == ["forward", "reverse"]
            expected = padded(*(rank[key] for key in aot.LAMMPS_AOTI_INPUT_KEYS))
        torch.testing.assert_close(
            (energy, nodes, pair), expected, atol=2e-6, rtol=1e-5
        )
        # Independent periodic graph: no ghost exchange, derivatives are with
        # respect to the real coordinates, not ML-IAP pair displacements.
        reference = native(periodic)
        source, target = periodic["edge_index"]
        force = torch.zeros_like(periodic["positions"]).index_add_(0, source, pair)
        force.index_add_(0, target, pair, alpha=-1)
        torch.testing.assert_close(energy, reference["energy"][0], atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(
            nodes, reference["node_energy"], atol=2e-6, rtol=1e-5
        )
        torch.testing.assert_close(force, reference["forces"], atol=2e-6, rtol=1e-5)
        assert pair.abs().max() > 1e-6


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA AOTInductor")
def test_single_structure_reference_energy_aoti(tmp_path):
    from tace.models.compile.compile import inductor_options

    class Readout(torch.nn.Module):
        def forward(self, base, basis, weight, scalar, types, heads, batch):
            nodes = torch.arange(types.numel(), device=types.device)
            ref = base.T[types][nodes, heads[batch]]
            learned = (basis * weight).sum(
                -1
            ).double() / 11**0.5 + scalar.double() * 0.25
            return (
                sum_node_energy(ref, batch, 1),
                sum_node_energy(learned, batch, 1),
                ref + learned,
            )

    x = torch.randn(16, 11, dtype=torch.float32, device="cuda")
    zeros = torch.zeros(16, dtype=torch.long, device="cuda")
    args = (
        x.new_tensor([[-3409.0]], dtype=torch.float64),
        x,
        x.new_ones(11),
        x[:, 0].contiguous(),
        zeros,
        zeros[:1].clone(),
        zeros.clone(),
    )
    model = Readout()
    export = torch.export.export(model, args)
    package = torch._inductor.aoti_compile_and_package(
        export,
        package_path=str(tmp_path / "energy.pt2"),
        inductor_configs=inductor_options(),
    )
    result = torch._inductor.aoti_load_package(package)(*args)
    torch.testing.assert_close(tuple(result), model(*args))
    torch.testing.assert_close(result[0], args[0].reshape(1) * len(x))
    # With two structures, atoms need not be ordered by graph.
    groups = torch.arange(16, device="cuda") % 2
    torch.testing.assert_close(
        sum_node_energy(result[2], groups, 2),
        torch.stack([result[2][groups == i].sum() for i in range(2)]),
    )


@pytest.mark.parametrize("backend", ["gloo", "nccl"])
def test_manual_gradient_sync_preserves_dtype_and_views(monkeypatch, backend):
    from types import SimpleNamespace

    from tace.utils.strategy import SimpleDDPStrategy

    model = torch.nn.ParameterList(
        [
            torch.nn.Parameter(torch.zeros(2, 3, dtype=dtype))
            for dtype in (torch.float32, torch.float64, torch.float32)
        ]
    )
    for i, parameter in enumerate(model[:2]):
        parameter.grad = torch.full((3, 2), float(i + 1), dtype=parameter.dtype).T
    gradients = [parameter.grad for parameter in model]
    seen = []
    waited = []

    def reduce(flat, op, async_op):
        assert async_op
        assert op == (
            torch.distributed.ReduceOp.SUM
            if backend == "gloo"
            else torch.distributed.ReduceOp.AVG
        )
        seen.append(flat.dtype)

        # The peer has identical local values and gradient availability.
        def wait():
            waited.append(flat.dtype)
            if op == torch.distributed.ReduceOp.SUM:
                flat.mul_(2)

        return SimpleNamespace(wait=wait)

    monkeypatch.setattr(torch.distributed, "get_backend", lambda: backend)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    strategy = SimpleDDPStrategy()
    strategy.model = model
    strategy.post_backward(None)
    assert seen == waited == [torch.float32, torch.float64]
    for i, parameter in enumerate(model[:2]):
        assert parameter.grad is gradients[i]
        torch.testing.assert_close(
            parameter.grad, torch.full_like(parameter.grad, i + 1)
        )
    assert model[2].grad is None


def _simple_ddp_worker(rank, rendezvous):
    from datetime import timedelta
    from unittest.mock import patch

    import torch.distributed as dist
    from lightning.fabric.plugins.environments import LightningEnvironment

    from tace.utils.strategy import SimpleDDPStrategy

    dist.init_process_group(
        "gloo",
        init_method=rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    try:
        env = LightningEnvironment()
        env.set_world_size(2)
        env.set_global_rank(rank)
        strategy = SimpleDDPStrategy(
            cluster_environment=env, parallel_devices=[torch.device("cpu")] * 2
        )
        model = torch.nn.Module()
        for i, name in enumerate(("a", "b", "unused")):
            model.register_parameter(
                name,
                torch.nn.Parameter(
                    torch.full((2, 3), i + rank + 1.0, dtype=torch.float32)
                ),
            )
        model.loss_weight = torch.nn.Parameter(
            torch.tensor(1.0 + rank, dtype=torch.float64)
        )
        model.register_buffer("persistent_stat", torch.tensor(11 + rank))
        strategy.model = model
        strategy.configure_ddp()
        assert strategy.model is model
        assert model.a.eq(1).all() and model.b.eq(2).all() and model.unused.eq(3).all()
        assert model.loss_weight.item() == 1 and model.persistent_stat.item() == 11
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=0.1)

        # Opposite ranks use different weights of identical shape. The old
        # local-only buckets silently averaged a's gradient with b's.
        local = model.b if rank else model.a
        other = model.a if rank else model.b
        view = torch.zeros(3, 2, dtype=local.dtype).T
        local.grad = view
        with patch.object(dist, "all_reduce", wraps=dist.all_reduce) as collect:
            with strategy.block_backward_sync():
                first = ((1 + 2 * rank) * local.sum() + model.loss_weight) / 2
                first.backward()
                strategy.post_backward(first)
                assert not collect.called
                assert other.grad is None and local.grad is view
            second = (
                (5 + 2 * rank) * local.sum() + (3 + 2 * rank) * model.loss_weight
            ) / 2
            second.backward()
            strategy.post_backward(second)
            assert collect.call_count == 2  # one data + presence bucket per dtype
        assert local.grad is view and not view.is_contiguous()
        torch.testing.assert_close(model.a.grad, torch.full_like(model.a, 1.5))
        torch.testing.assert_close(model.b.grad, torch.full_like(model.b, 2.5))
        torch.testing.assert_close(
            model.loss_weight.grad, torch.full_like(model.loss_weight, 2.5)
        )
        assert model.unused.grad is None
        optimizer.step()
        assert model.unused not in optimizer.state
        after = [p.detach().clone() for p in (model.a, model.b, model.unused)]
        for weight in model.parameters():
            root = weight.detach().clone()
            dist.broadcast(root, src=0)
            torch.testing.assert_close(weight, root, rtol=0, atol=0)

        # Now rank 1 has no model gradients at all. Global absence must
        # restore None even for formerly active parameters (no AdamW decay).
        optimizer.zero_grad(set_to_none=True)
        loss = torch.zeros((), requires_grad=True) if rank else 4 * model.loss_weight
        loss.backward()
        strategy.post_backward(loss)
        assert model.a.grad is model.b.grad is model.unused.grad is None
        torch.testing.assert_close(
            model.loss_weight.grad, torch.full_like(model.loss_weight, 2)
        )
        optimizer.step()
        for p, expected in zip((model.a, model.b, model.unused), after, strict=True):
            torch.testing.assert_close(p, expected, rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        strategy.post_backward(torch.zeros(()))
        assert all(p.grad is None for p in model.parameters())
    finally:
        dist.destroy_process_group()


def test_simple_ddp_gloo_state_unused_and_accumulation(tmp_path):
    if not (torch.distributed.is_available() and torch.distributed.is_gloo_available()):
        pytest.skip("requires Gloo")
    torch.multiprocessing.spawn(
        _simple_ddp_worker,
        args=((tmp_path / "gloo").as_uri(),),
        nprocs=2,
        join=True,
    )


@pytest.mark.parametrize(
    "backend,device", [("eager", "cpu"), ("inductor", "cpu"), ("inductor", "cuda")]
)
@pytest.mark.parametrize("dtype", [torch.float64, torch.float32])
@pytest.mark.parametrize("uncertainty", [False, True])
def test_joint_training_gradients(monkeypatch, backend, device, dtype, uncertainty):
    import inspect
    from functools import wraps

    from tace.models.adapter import TensorModel
    from tace.models.compile import joint
    from tace.utils.loss.normal import NormalLoss
    from tace.utils.loss.uncertainty import UncertaintyLoss

    if "dynamic_shapes" not in inspect.signature(torch.compile).parameters:
        pytest.skip("Joint training requires explicit dynamic shapes")
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    torch.set_default_dtype(dtype)
    monkeypatch.setenv("TACE_USE_COMPILE", "1")
    monkeypatch.setenv("TACE_USE_EQX", "1")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["product_basis"]["type"] = "nonlinear_cgtp"
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    targets = ["energy", "forces", "stress"]
    config["target_property"] = targets
    stats = [dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)]
    model = create_model(config, stats, targets, [])
    model.readout_fn.compile_backend = backend
    model = model.to(device)
    next(model.parameters()).requires_grad_(False)
    model.readout_fn.register_parameter(
        "unused", torch.nn.Parameter(next(model.parameters()).new_ones(2))
    )
    loss_fn = (UncertaintyLoss if uncertainty else NormalLoss)(
        targets,
        ["mae_energy_per_atom", "l2mae_forces", "mae_voigt_stress"],
        [1.0, 2.0, 3.0],
    )
    loss_fn.to(next(model.parameters()).device)
    parameters = tuple(
        p for p in (*model.parameters(), *loss_fn.parameters()) if p.requires_grad
    )
    data = synthetic_graph_sample(model)
    data["entropy"] = data["positions"].new_ones(2)
    for key, shape in (("energy", (2,)), ("forces", (4, 3)), ("stress", (2, 3, 3))):
        data[key] = data["positions"].new_full(shape, 0.1)
        data[f"{key}_weight"] = data["entropy"].clone()
    data["stress_weight"][1] = 0
    data["stress"][1] = float("nan")
    accumulated = [None] * len(parameters)
    for step in range(2):
        batch = {key: value.clone() for key, value in data.items()}
        batch["positions"] *= 1 + step * 0.1
        if step:
            batch["edge_index"] = batch["edge_index"][:, :3]
            batch["edge_shifts"] = batch["edge_shifts"][:3]
            batch["stress_weight"].zero_()
        expected = TensorModel.forward(model, dict(batch))
        reference = loss_fn(expected, batch)
        gradients = torch.autograd.grad(reference, parameters, allow_unused=True)
        state = (*model.parameters(), *model.buffers(), *loss_fn.parameters())
        values = [value.detach().clone() for value in state]
        input_values = {key: value.detach().clone() for key, value in batch.items()}
        if step and backend == "inductor":
            program = next(iter(model._compiled_cache.values()))
            assert program.artifact_info["status"] in (
                "saved",
                "hit",
            ), program.artifact_info
            model._compiled_cache.clear()

            @wraps(joint.make_fx)
            def unexpected_compile(*args, **kwargs):
                pytest.fail(
                    "A matching training artifact must load without tracing or compiling"
                )

            with monkeypatch.context() as cached:
                cached.setattr(joint, "make_fx", unexpected_compile)
                cached.setattr(
                    torch._inductor, "standalone_compile", unexpected_compile
                )
                actual, loss = model.forward_loss(batch, loss_fn)
            assert (
                next(iter(model._compiled_cache.values())).artifact_info["status"]
                == "hit"
            )
        else:
            actual, loss = model.forward_loss(batch, loss_fn)
        current = (*model.parameters(), *model.buffers(), *loss_fn.parameters())
        assert len(current) == len(state)
        for original, value, saved in zip(state, current, values):
            assert original is value
            torch.testing.assert_close(value, saved, atol=0, rtol=0)
        for key, value in input_values.items():
            torch.testing.assert_close(
                batch[key], value, atol=0, rtol=0, equal_nan=True
            )
        tolerance = 2e-4 if dtype == torch.float32 else 2e-9
        torch.testing.assert_close(loss, reference, atol=tolerance, rtol=tolerance)
        for key in targets:
            torch.testing.assert_close(
                actual[key], expected[key], atol=tolerance, rtol=tolerance
            )
        (loss / 2).backward()
        for i, (parameter, grad) in enumerate(zip(parameters, gradients)):
            if grad is None:
                assert parameter.grad is None
            else:
                accumulated[i] = (
                    grad / 2 if accumulated[i] is None else accumulated[i] + grad / 2
                )
                torch.testing.assert_close(
                    parameter.grad, accumulated[i], atol=tolerance, rtol=tolerance
                )
        assert not any(
            value.grad is not None for value in batch.values() if value.is_leaf
        )
        with torch.no_grad():
            for parameter in parameters:
                parameter.mul_(0.99)
    assert len(model._compiled_cache) == 1
    expected_parameters = [torch.nn.Parameter(p.detach().clone()) for p in parameters]
    for parameter, gradient in zip(expected_parameters, accumulated):
        parameter.grad = gradient
    for values in (parameters, expected_parameters):
        torch.nn.utils.clip_grad_norm_(values, 1.0)
        torch.optim.AdamW(values, lr=1e-3).step()
    for actual, expected in zip(parameters, expected_parameters):
        torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)

    if backend == "eager" and dtype == torch.float32:
        from tace.utils.loss.registry import LOSS_FN

        # Replace one registered function after a joint program is cached.
        name = "mae_energy_per_atom"
        native = LOSS_FN[name]
        monkeypatch.setitem(
            LOSS_FN, name, lambda *args, **kw: scale * native(*args, **kw)
        )
        for scale in (2.0, 5.0):
            expected = loss_fn(TensorModel.forward(model, dict(data)), data)
            _, actual = model.forward_loss(data, loss_fn)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            for a, b in zip(
                torch.autograd.grad(expected, parameters, allow_unused=True),
                torch.autograd.grad(actual, parameters, allow_unused=True),
                strict=True,
            ):
                torch.testing.assert_close(a, b, atol=0, rtol=0)
        assert len(model._compiled_cache) == 1
        with pytest.raises(ValueError, match="built-in loss"):
            joint.JointTraining(model, loss_fn, data)
        with pytest.raises(ValueError, match="built-in loss"):
            joint.joint_training(model, loss_fn, data, return_gradients=True)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_joint_loss_aliases_and_scaling(device):
    from tace.models.compile.joint import ParameterLoss

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    a, b = (torch.nn.Parameter(torch.ones(3, device=device)) for _ in range(2))
    shared = torch.ones(3, device=device)
    loss = ParameterLoss.apply(shared.new_tensor(1.0), (shared, shared), a, b)
    (loss * 4).backward(retain_graph=True)
    assert a.grad.data_ptr() != b.grad.data_ptr()
    a.grad.zero_()
    loss.backward()
    torch.testing.assert_close(a.grad, torch.ones_like(shared))
    torch.testing.assert_close(b.grad, torch.full_like(shared, 5.0))
    torch.testing.assert_close(shared, torch.ones_like(shared))


def test_joint_artifact_metadata():
    from types import SimpleNamespace

    from e3nn import o3

    from equivariantx import co2, co3, o2
    from tace.models.compile.artifact import module_context

    class Settings:
        __slots__ = ("representations", "options")

        def __getstate__(self):
            raise AssertionError("Compilation must not invoke serialization hooks")

    module = torch.nn.Linear(2, 2)
    settings = Settings()
    settings.representations = (
        o2.Irreps("2x0ee+3x1mo"),
        co2.Irreps("2x0ee+3x1mo"),
        o3.Irreps("2x0e+3x1o"),
        co3.Irreps("2x0e+3x1o"),
    )
    settings.options = SimpleNamespace(layout="ir_mul", weight=module.weight)
    module.settings = settings
    expected = module_context(module)
    settings.representations = tuple(type(v)(v) for v in settings.representations)
    assert module_context(module) == expected
    with torch.no_grad():
        module.weight.add_(1)
    assert module_context(module) == expected
    for irreps in ("2x0oe+3x1mo", "2x0ee+3x1me", "3x1mo+2x0ee"):
        settings.representations = (o2.Irreps(irreps), *settings.representations[1:])
        assert module_context(module) != expected
    settings.options.weight = torch.ones(2, requires_grad=True)
    with pytest.raises(ValueError, match="Unregistered trainable"):
        module_context(module)
    settings.options.weight = module.weight
    settings.options.parent = settings
    with pytest.raises(ValueError, match="Cyclic"):
        module_context(module)


def test_joint_cache_separates_broadcast_and_dynamic_reciprocal_rows(monkeypatch):
    from tace.models.compile import joint
    from tace.utils.loss.normal import NormalLoss

    model = torch.nn.Linear(2, 1)
    model._compiled_cache = {}
    objective = NormalLoss(["energy"], ["mse_energy"], [1.0])
    monkeypatch.setattr(joint, "supports_joint_loss", lambda *args: True)
    # Cache dispatch only. One representative row is a static dimension;
    # two or more rows belong to the same dynamic program.
    monkeypatch.setattr(joint, "JointTraining", lambda *a: Mock(return_value=object()))
    data = dict(positions=torch.zeros(2, 3), ptr=torch.tensor([0, 2]))
    runs = [
        joint.joint_training(
            model,
            objective,
            dict(
                data,
                edge_index=torch.zeros(2, 2 * n, dtype=torch.long),
                edge_pair_rows=torch.arange(n),
            ),
        )
        for n in (1, 2, 3, 1)
    ]
    assert runs[0] is runs[-1] and runs[1] is runs[2] and runs[0] is not runs[1]
    objective.eval()
    joint.joint_training(
        model,
        objective,
        dict(
            data,
            edge_index=torch.zeros(2, 2, dtype=torch.long),
            edge_pair_rows=torch.arange(1),
        ),
    )
    assert len(model._compiled_cache) == 3


def test_joint_artifact_fixed_and_dynamic_shapes(tmp_path, monkeypatch):
    import inspect
    from types import SimpleNamespace

    from tace.models.compile import joint
    from tace.models.compile.artifact import JointArtifactCache
    from tace.utils.loss.normal import NormalLoss

    if "dynamic_shapes" not in inspect.signature(
        torch.compile
    ).parameters or not hasattr(torch._inductor, "CompiledArtifact"):
        pytest.skip("Joint training artifact cache unavailable")
    model = SimpleNamespace(readout_fn=SimpleNamespace(), shape_padding=False)
    objective = NormalLoss(["energy"], ["mse_energy"], [1.0])

    class Loss(torch.nn.Linear):
        def forward(self, edge_index, positions):
            prediction = super().forward(positions).sum() / positions.shape[0]
            return prediction.square(), (prediction,)

    module = Loss(3, 1)
    module.model = torch.nn.Identity()
    module.model.output_keys = ("energy",)
    monkeypatch.setattr(joint, "TrainingLoss", lambda *args: module)
    monkeypatch.setattr(JointArtifactCache, "load", lambda self: module)
    monkeypatch.setenv("TORCHINDUCTOR_CACHE_DIR", str(tmp_path))

    def batch(n, e):
        return {
            "positions": torch.ones(n, 3),
            "edge_index": torch.zeros(2, e, dtype=torch.long),
        }

    def key(n, e, fixed):
        return joint.JointTraining(
            model, objective, batch(n, e), _fixed_shapes=fixed
        ).artifact_info["key"]

    dynamic = key(4, 6, False)
    assert key(5, 8, False) == dynamic
    assert len({dynamic, key(4, 6, True), key(5, 6, True), key(4, 8, True)}) == 4
    model.readout_fn.compile_backend = "eager"
    fixed = joint.JointTraining(model, objective, batch(4, 6), _fixed_shapes=True)
    out, loss = fixed(batch(4, 6))
    value, (pred,) = module(**batch(4, 6))
    torch.testing.assert_close(out["energy"], pred)
    torch.testing.assert_close(loss, value)
    for shape in ((5, 6), (4, 8)):
        with pytest.raises(ValueError, match="input shape changed"):
            fixed(batch(*shape))


def test_joint_artifact_contract(tmp_path, monkeypatch):
    from torch._inductor.output_code import CompiledFxGraph

    from tace.models.compile.artifact import (
        FORMAT_VERSION,
        JointArtifactCache,
        compiled_source_identity,
        module_context,
        pack_constants,
        runner_source_identities,
        runtime_signature,
        storage_aliases,
        tensor_contract,
        unpack_constants,
    )

    assert FORMAT_VERSION == "v1"
    module = torch.nn.Linear(2, 2)
    module.register_buffer("scale", torch.ones(2))
    module.settings = {"weight": 1.0, "statistics": [2.0]}
    context = module_context(module)
    with torch.no_grad():
        module.weight.add_(1)
        module.scale.mul_(2)
    assert module_context(module) == context
    assert module_context(deepcopy(module)) == context
    module.settings["weight"] = 2.0
    assert module_context(module) != context
    module.settings["weight"] = 1.0
    module.settings["statistics"][0] = 3.0
    assert module_context(module) != context

    values = (torch.randn(4, 3), torch.randn(4), torch.randn(2, 2).T)
    contract = {
        "tensor_inputs": [
            tensor_contract(value, {0: "nodes"} if i < 2 else {}, i < 2)
            for i, value in enumerate(values)
        ],
        "storage_aliases": storage_aliases(values),
        "dynamic_min": 2,
    }
    cache = JointArtifactCache(tmp_path, contract)
    assert cache.load() is None
    cache.binary.write_bytes(b"incomplete")
    cache.manifest.write_text(
        json.dumps({"contract": contract, "binary_sha256": "invalid"})
    )
    assert cache.load() is None
    assert cache.info["status"] == "invalid"
    assert cache.validate_inputs(values) == {"nodes": 4}
    unseen = (torch.randn(6, 3), torch.randn(6), values[2])
    assert cache.validate_inputs(unseen) == {"nodes": 6}
    for bad in (
        (unseen[0], values[1], values[2]),
        (values[0].double(), values[1], values[2]),
        (values[0], values[1], values[2].contiguous()),
        (
            values[0],
            values[1],
            torch.empty(8, dtype=values[2].dtype).as_strided((2, 2), (1, 2), 4),
        ),
        (values[0][:1], values[1][:1], values[2]),
    ):
        with pytest.raises(ValueError):
            cache.validate_inputs(bad)

    raw = torch.arange(12, dtype=torch.float64)
    constants = (
        raw[2:10:2],
        raw.view(3, 4).T,
        torch.empty(0),
        torch.empty(0, dtype=torch.int64),
    )
    packed, specs = pack_constants(constants)
    restored = unpack_constants(packed, specs)
    for a, b in zip(constants, restored):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert a.stride() == b.stride()
        assert a.storage_offset() == b.storage_offset()
    assert torch._C._is_alias_of(restored[0], restored[1])
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda device: "same GPU")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device: (9, 0))
    assert runtime_signature(torch.device("cuda:0"), {}) != runtime_signature(
        torch.device("cuda:1"), {}
    )

    source_path = tmp_path / "compiled.py"
    monkeypatch.setattr(
        "torch._inductor.codecache.get_path",
        lambda *args: (None, None, str(source_path)),
    )
    compiled = Mock(
        spec=CompiledFxGraph,
        source_code=(
            "def call(args):\n    assert args[0].shape[0] >= 2\n    return args\n"
        ),
        cache_key="test_source",
        guards_expr=None,
    )
    for newline in ("\n", "\r\n"):
        source_path.write_bytes(
            compiled.source_code.replace("\n", newline).encode("utf-8")
        )
        assert runner_source_identities(compiled) == [
            compiled_source_identity(compiled)
        ]
    source_path.write_bytes(compiled.source_code.replace(">= 2", ">= 1").encode())
    with pytest.raises(ValueError, match="Actual generated source file differs"):
        runner_source_identities(compiled)
    source_path.unlink()
    with pytest.raises(ValueError, match="Actual generated source file differs"):
        runner_source_identities(compiled)


def test_compile_metadata_comments_restore():
    from torch._inductor import ir
    from torch._inductor.codegen import triton

    from tace.models.compile.compile import omit_provenance_comments

    original = (ir.get_kernel_metadata, triton.get_kernel_metadata)
    with pytest.raises(RuntimeError, match="compiler error"):
        with omit_provenance_comments():
            assert ir.get_kernel_metadata() == ("", "")
            assert triton.get_kernel_metadata() == ("", "")
            raise RuntimeError("compiler error")
    assert (ir.get_kernel_metadata, triton.get_kernel_metadata) == original


def test_compiled_shared_state(double_precision):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(3, 3)
            self.linear.register_buffer("scale", torch.randn(3))
            self.linear.weight_alias = self.linear.weight
            self.linear.register_buffer("scale_alias", self.linear.scale)
            self.alias = self.linear
            self.tied = torch.nn.Linear(3, 3)
            self.tied.weight = self.linear.weight
            self.tied.register_buffer("scale", self.linear.scale)

        def forward(self, x):
            return (
                self.linear(x) * self.linear.scale
                + self.alias(x) * self.alias.scale
                + self.tied(x) * self.tied.scale
                + (x @ self.linear.weight_alias.T) * self.linear.scale_alias
            ).sin()

    torch.compiler.reset()
    model = Model()
    state = {**dict(model.named_parameters()), **dict(model.named_buffers())}
    x = torch.randn(4, 3, requires_grad=True)
    compiled, parameters, buffers = trace_and_compile(model, (x,), backend="aot_eager")
    for name, value in (*model.named_parameters(), *model.named_buffers()):
        assert value is state[name]
    for size in (4, 7):
        x = torch.randn(size, 3, requires_grad=True)
        actual = compiled_call(compiled, model, (x,), parameters, buffers)
        expected = model(x)
        torch.testing.assert_close(actual, expected)
        gradients = [
            torch.autograd.grad(value.square().sum(), (x, *model.parameters()))
            for value in (actual, expected)
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        with torch.no_grad():
            model.linear.weight.add_(0.1)
            model.linear.scale.mul_(0.9)


@pytest.mark.parametrize("supports_mixed_reduction", [False, True])
def test_compile_avoids_size_dependent_reduction_fusion(
    monkeypatch, supports_mixed_reduction
):
    from torch._inductor import config

    key = "triton.mix_order_reduction"
    supported = dict(config.get_config_copy())
    if supports_mixed_reduction:
        supported[key] = True
    else:
        supported.pop(key, None)
    monkeypatch.setitem(vars(config), "get_config_copy", lambda: supported)
    received = {}

    def compile_graph(graph, **kwargs):
        received.update(kwargs)
        return graph

    monkeypatch.setattr(torch, "compile", compile_graph)
    model = torch.nn.Linear(3, 2)
    inputs = (torch.randn(4, 3),)
    compiled, parameters, buffers = trace_and_compile(model, inputs, backend="inductor")
    assert received["dynamic"] is True
    assert received["fullgraph"] is True
    if supports_mixed_reduction:
        assert received["options"][key] is False
    else:
        assert key not in received["options"]
    torch.testing.assert_close(
        compiled_call(compiled, model, inputs, parameters, buffers), model(*inputs)
    )


@pytest.mark.parametrize(
    "backend", [None, "eqx", "openequivariance", "cuequivariance", "lunaequivariance"]
)
def test_aoti_dependencies(tmp_path, monkeypatch, backend):
    model = torch.nn.Linear(2, 2)
    expected_libs = set()
    if backend == "eqx":
        import equivariantx as eqx  # noqa: F401

        graph = torch.fx.Graph()
        operator = torch.ops.eqx.gather_sum.default
        inputs = tuple(
            graph.placeholder(arg.name) for arg in operator._schema.arguments
        )
        graph.output(graph.call_function(operator, inputs))
        model = torch.fx.GraphModule(torch.nn.Module(), graph)
        expected_libs = {"equivariantx"}
    elif backend is not None:
        model = type(
            "Convolution",
            (torch.nn.Module,),
            {"__module__": f"tace.models.backends.{backend}._tp_scatter"},
        )()
        expected_libs = {
            "openequivariance": {"openequivariance"},
            "cuequivariance": {"cuequivariance", "cuequivariance_torch"},
            "lunaequivariance": {"tace.models.backends.lunaequivariance"},
        }[backend]
        if backend == "lunaequivariance":
            model.aoti_config = {"dtype": "float64"}
    libs = custom_ops_libs_from_model(model)
    assert libs == expected_libs

    path = tmp_path / "model.pt2"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("model/archive_format", "pt2")
    embed_custom_ops_libs(path, libs, model)
    if libs:
        with zipfile.ZipFile(path) as archive:
            assert (
                set(
                    archive.read(f"model/{TACE_AOTI_CUSTOM_OPS_LIBS_ENTRY}")
                    .decode()
                    .split()
                )
                == libs
            )
            if backend == "lunaequivariance":
                assert json.loads(
                    archive.read(f"model/{TACE_AOTI_LUNA_OPS_ENTRY}")
                ) == [model.aoti_config]

    register = Mock()
    if backend == "lunaequivariance":
        from types import SimpleNamespace

        monkeypatch.setitem(
            sys.modules,
            "tace.models.backends.lunaequivariance",
            SimpleNamespace(register_luna_ops=register),
        )
    importer = Mock()
    monkeypatch.setattr("tace.models.compile.aot.importlib.import_module", importer)
    import_custom_ops_libs(path)
    assert [call.args[0] for call in importer.call_args_list] == sorted(libs)
    if backend == "lunaequivariance":
        register.assert_called_once_with(model.aoti_config)


@pytest.mark.parametrize(
    "backend", ["native", "eqx", "luna", "cartesian", "cartesian_eqx", "cartesian_o2"]
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_compiled_training_and_export(tmp_path, dtype, backend):
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    use_luna = backend == "luna"
    if use_luna:
        if sys.platform != "linux" or torch.cuda.get_device_capability()[0] < 8:
            pytest.skip("LunaEquivariance requires Linux and compute capability >= 8.0")
        pytest.importorskip("LunaEquivariance")
    from tace.models.compile.aot import export_aotinductor, load_aotinductor
    from tace.utils.env import enable_acceleration

    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["atomic_basis"]["type"] = "uv_o2" if backend == "cartesian_o2" else "cgtp"
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    targets = ["energy", "forces", "stress"]
    config["target_property"] = targets
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    enable_acceleration(force=True)
    reference = create_model(config, statistics, targets, []).cuda()
    state = reference.state_dict()
    if backend.startswith("cartesian"):
        from tace.lightning import convert_basis

        state = convert_basis(reference).state_dict()
        config["_target_"] = "tace.models.CartesianTACE"
    enable_acceleration(
        enable_luna=use_luna,
        enable_eqx=backend in ("eqx", "cartesian_eqx", "cartesian_o2"),
        enable_compile=True,
        force=True,
    )
    model = create_model(config, statistics, targets, []).cuda()
    model.load_state_dict(state)
    sample = synthetic_graph_sample(model)
    tolerance = 2e-4 if dtype == torch.float32 else 2e-9
    for step in range(2):
        data = {key: value.clone() for key, value in sample.items()}
        data["positions"] *= 1 + 0.05 * step
        if step:
            data["edge_index"] = data["edge_index"].flip(1)[:, :3]
            data["edge_shifts"] = data["edge_shifts"].flip(0)[:3]
        expected = reference(dict(data))
        actual = model(dict(data))
        for key in targets:
            assert actual[key] is not None and expected[key] is not None
            torch.testing.assert_close(
                actual[key], expected[key], atol=tolerance, rtol=tolerance
            )
        gradients = [
            torch.autograd.grad(
                sum(out[key].square().sum() for key in targets),
                tuple(module.parameters()),
                allow_unused=True,
            )
            for out, module in ((actual, model), (expected, reference))
        ]
        for got, wanted in zip(*gradients):
            assert (got is None) == (wanted is None)
            if got is not None:
                torch.testing.assert_close(got, wanted, atol=tolerance, rtol=tolerance)
        with torch.no_grad():
            for module, grads in zip((model, reference), gradients):
                for parameter, grad in zip(module.parameters(), grads):
                    if grad is not None:
                        parameter.add_(grad, alpha=-1e-4)
    assert model._compiled_cache

    path = export_aotinductor(model, tmp_path / "model.pt2", sample)
    loaded = load_aotinductor(path, "cuda")
    sample = data  # Use a different edge count from the export example.
    expected = reference.eval()(dict(sample))
    actual = loaded(dict(sample))
    for key in targets:
        torch.testing.assert_close(
            actual[key], expected[key], atol=tolerance, rtol=tolerance
        )
    if backend == "native" and dtype == torch.float32 and torch.cuda.device_count() > 1:
        current = torch.cuda.current_device()
        other = torch.device("cuda", (current + 1) % torch.cuda.device_count())
        second = load_aotinductor(path, other)
        assert second.compile_device == other
        relocated = second({key: value.to(other) for key, value in sample.items()})
        for key in targets:
            assert relocated[key].device == other
            torch.testing.assert_close(
                relocated[key].cpu(),
                expected[key].cpu(),
                atol=tolerance,
                rtol=tolerance,
            )
        assert torch.cuda.current_device() == current
    # Loading must restore registrations without constructing the original model.
    case = tmp_path / "case.pt"
    torch.save(
        {
            "sample": sample,
            "expected": {key: expected[key].detach() for key in targets},
        },
        case,
    )
    script = """
import sys
import torch
from tace.models.compile.aot import load_aotinductor
torch.set_num_threads(4)
case = torch.load(sys.argv[2], weights_only=True)
model = load_aotinductor(sys.argv[1], "cuda")
actual = model(case["sample"])
for key, expected in case["expected"].items():
    torch.testing.assert_close(actual[key], expected,
                               atol=float(sys.argv[3]), rtol=float(sys.argv[3]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), str(case), str(tolerance)],
        cwd=Path(__file__).resolve().parents[1],
        env={**os.environ, "TACE_USE_LUNA": "0", "TACE_USE_COMPILE": "0"},
        capture_output=True,
        text=True,
        timeout=900 if use_luna else 180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_constant_construction_precision():
    from e3nn import o3

    from equivariantx.nn import MergeLayerNorm
    from tace.models.angular import CartesianHarmonics
    from tace.models.blocks import AtomicEnergy, ScaleShift
    from tace.models.linear import e3nnLinear
    from tace.models.radial import (
        AgnesiTransform,
        C2PolynomialCutoff,
        C3PolynomialCutoff,
        CosineCutoff,
        GaussianBasis,
        MollifierCutoff,
        SoftTransform,
        ZBLBasis,
        j0SphericalBesselBasis,
        jnSphericalBesselBasis,
    )
    from tace.models.spherical.basis_change import DirectVirials
    from tace.models.spherical.mtace import MagneticBasis
    from tace.models.spherical.symmetric_contraction import Contraction

    constants = []
    for dtype in (torch.float32, torch.float64):
        torch.set_default_dtype(dtype)
        modules = torch.nn.ModuleList(
            [
                j0SphericalBesselBasis(cutoff=4.123456789),
                jnSphericalBesselBasis(cutoff=4.123456789, order=2),
                GaussianBasis(cutoff=4.123456789),
                CosineCutoff(4.123456789),
                MollifierCutoff(4.123456789),
                C2PolynomialCutoff(4.123456789),
                C3PolynomialCutoff(4.123456789),
                AgnesiTransform(),
                SoftTransform(),
                ZBLBasis("cosine"),
                AtomicEnergy([{1: -0.123456789}], [1]),
                ScaleShift([1], [{1: 0.123456789}], [{1: -0.123456789}]),
                MagneticBasis([3.123456789], 4, 1, [1]),
                CartesianHarmonics(3),
                DirectVirials(),
                MergeLayerNorm("2x0e + 2x1o + 2x2e"),
                e3nnLinear("2x0e + 2x1o", "2x0e + 2x1o"),
                Contraction(
                    o3.Irreps("2x0e + 2x1o"), o3.Irreps("0e"), 2, num_elements=1
                ),
            ]
        )
        buffers = {
            f"{name}.{key}": value
            for name, module in modules.named_modules()
            if type(module).__module__.startswith("tace.")
            for key, value in module.named_buffers(recurse=False)
            if value.is_floating_point()
        }
        assert buffers
        assert all(value.dtype == torch.float64 for value in buffers.values())
        assert all(
            parameter.dtype in (dtype, torch.float64)
            for parameter in modules.parameters()
        )
        constants.append(buffers)

    assert constants[0].keys() == constants[1].keys()
    for name in constants[0]:
        torch.testing.assert_close(
            constants[0][name], constants[1][name], atol=0, rtol=0
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("num_fidelities", [1, 2])
def test_atomic_energy_lookup(dtype, num_fidelities):
    from tace.models.blocks import AtomicEnergy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    elements = list(range(1, 103))
    energies = [
        {z: -100_000.123456789 * z - fidelity for z in elements}
        for fidelity in range(num_fidelities)
    ]
    module = AtomicEnergy(energies, elements).to(device=device)
    node_type = torch.arange(8192, device=device) % len(elements)
    expected = module.atomic_energy.T[node_type]
    allow_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        for n in (0, 1, 128, 3072, 8192):
            torch.testing.assert_close(
                module(node_type[:n]), expected[:n], atol=0, rtol=0
            )
        compiled = torch.compile(module, fullgraph=True)
        torch.testing.assert_close(compiled(node_type), expected, atol=0, rtol=0)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32


@pytest.mark.parametrize("space", ["spherical", "cartesian"])
def test_model_dtype_conversion(space):
    from tace.models.adapter import TensorModel
    from tace.utils.precision import configure_model_precision

    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        _target_=f"tace.models.{space.title()}TACE",
        num_layers=1,
        num_channel=2,
        Lmax=1,
        lmax=1,
        cutoff=4.1234567890123,
        max_neighbors=None,
    )
    config["radial_basis"].update(hidden=[4], trainable=True, apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        {
            "atomic_numbers": [1],
            "atomic_energy": {1: -100000.1234567890123},
            "avg_num_neighbors": 2.0,
        }
    ]
    model = create_model(config, statistics, ["energy", "forces"], [])
    readout = model.readout_fn
    atomic_energy = readout.atomic_energy_layer
    atomic_energy.register_buffer("alias", atomic_energy.atomic_energy)
    atomic_energy.float64_buffers = ("atomic_energy", "alias")
    energy = atomic_energy.atomic_energy.clone()
    cutoff = readout.cutoff.clone()
    radial = next(
        module for module in readout.modules() if type(module).__name__ == "RadialBasis"
    )
    radial.scripted = torch.jit.script(torch.nn.BatchNorm1d(1, affine=False).double())
    scripted_mean = radial.scripted.running_mean.fill_(0.1234567890123).clone()
    physical = next(radial.parameters())
    physical.grad = torch.full_like(physical, 0.1234567890123)
    value, gradient = physical.detach().clone(), physical.grad.clone()
    for wrapper in (TensorModel, CompileTensorModel):
        model = wrapper(readout)
        for dtype, convert in (
            (torch.float32, lambda m: m.float()),
            (torch.float64, lambda m: m.double()),
            (torch.float32, lambda m: m.to(dtype=torch.float32)),
            (torch.float64, lambda m: configure_model_precision(m, torch.float64)),
        ):
            if isinstance(model, CompileTensorModel):
                model._compiled_cache["test"] = None
            assert convert(model) is model
            assert model.get_model_dtype() == dtype
            torch.testing.assert_close(
                radial.scripted.running_mean, scripted_mean, atol=0, rtol=0
            )
            assert readout.representation.model_dtype == dtype
            assert (
                next(readout.representation.node_embedding.parameters()).dtype == dtype
            )
            assert atomic_energy.atomic_energy is atomic_energy.alias
            torch.testing.assert_close(
                atomic_energy.atomic_energy, energy, atol=0, rtol=0
            )
            torch.testing.assert_close(readout.cutoff, cutoff, atol=0, rtol=0)
            torch.testing.assert_close(physical, value, atol=0, rtol=0)
            torch.testing.assert_close(physical.grad, gradient, atol=0, rtol=0)
            if isinstance(model, CompileTensorModel):
                assert not model._compiled_cache
        with pytest.raises(ValueError, match="use autocast"):
            model.bfloat16()
        assert model.get_model_dtype() == torch.float64
        assert (
            next(readout.representation.node_embedding.parameters()).dtype
            == torch.float64
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "interaction,compiled,skip,u_shift",
    [
        ("cgtp", False, False, False),
        ("uv_o2", False, False, False),
        ("cgtp", True, False, False),
        ("cgtp", True, True, False),
        ("cgtp", False, False, True),
        ("cgtp", True, False, True),
    ],
)
def test_lightning_model_training_precision(
    monkeypatch, dtype, interaction, compiled, skip, u_shift
):
    import inspect

    import lightning as L
    from torch.utils.data import DataLoader

    from tace.lightning import create_model
    from tace.lightning.lit_model import LightningWrapperModel

    monkeypatch.setenv("TACE_USE_U_SHIFT", "1" if u_shift else "0")
    if compiled:
        if "dynamic_shapes" not in inspect.signature(torch.compile).parameters:
            pytest.skip("Joint training requires explicit dynamic shapes")
        monkeypatch.setenv("TACE_USE_COMPILE", "1")

    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=2,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["atomic_basis"]["type"] = interaction
    config["atomic_basis"]["num_head"] = 1
    config["radial_basis"]["hidden"] = [4]
    config["radial_basis"]["apply_cutoff"] = False
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    numbers = [8, 26] if u_shift else [1]
    statistics = [
        {
            "atomic_numbers": numbers,
            "atomic_energy": dict.fromkeys(numbers, 0.0),
            "avg_num_neighbors": 2.0,
        }
    ]
    model = create_model(config, statistics, ["energy", "forces"], [])
    model.readout_fn.compile_backend = "eager"
    assert all(p.dtype == dtype for p in model.parameters())
    assert model.readout_fn.cutoff.dtype == torch.float64
    assert model.readout_fn.atomic_energy_layer.atomic_energy.dtype == torch.float64
    assert model.readout_fn.atomic_numbers.dtype == torch.int64

    class TrainingModel(LightningWrapperModel):
        def _process_batch(self, batch):
            torch.testing.assert_close(
                batch["energy"],
                torch.full_like(batch["energy"], -2.85 if u_shift else 1.0),
            )
            return super()._process_batch(batch)

        def training_step(self, batch, batch_idx):
            loss = super().training_step(batch, batch_idx)
            assert loss.dtype == torch.float64 and torch.isfinite(loss)
            assert (batch["energy"] == 1.0).all()
            if compiled:
                assert len(self.model._compiled_cache) == 1
            return loss

        def on_after_backward(self):
            gradients = [p.grad for p in self.parameters() if p.grad is not None]
            assert gradients
            assert all(g.dtype == dtype and torch.isfinite(g).all() for g in gradients)
            if skip:
                assert all(torch.count_nonzero(g) == 0 for g in gradients)

    trainer = L.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="32-true" if dtype == torch.float32 else "64-true",
        max_steps=2,
        accumulate_grad_batches=2,
        gradient_clip_val=1.0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    batches = []
    for _ in range(4):
        batch = _magnetic_embedding_sample()
        batch.pop("initial_noncollinear_magmoms")
        if u_shift:
            batch["atomic_numbers"] = torch.tensor([26, 26, 8])
            batch["node_type"] = torch.tensor([1, 1, 0])
        batch.update(
            energy=torch.ones(1),
            forces=torch.ones(3, 3),
            entropy=torch.ones(1),
            energy_weight=torch.ones(1),
            forces_weight=torch.ones(1),
        )
        batches.append(batch)
    cfg = dict(
        loss=dict(
            _target_="tace.utils.loss.NormalLoss",
            loss_property=["energy", "forces"],
            loss_function_name=["mse_energy", "mse_forces"],
            loss_property_weights=[1.0, 1.0],
        ),
        optimizer=dict(_target_="torch.optim.SGD", lr=1e-3),
        misc=dict(
            LossSkipController=dict(
                enable=skip,
                manual_threshold=0.0,
                start_step=0,
                ema_window=10,
                multiplier=2.0,
            )
        ),
    )
    module = TrainingModel(cfg, model, ["energy", "forces"], [], statistics)
    trainer.fit(module, train_dataloaders=DataLoader(batches, batch_size=None))
    assert trainer.global_step == 2


@pytest.mark.parametrize("path_class", [PurePosixPath, PureWindowsPath])
def test_foundation_aoti_output_keeps_version(monkeypatch, path_class):
    from tace.scripts import export_eval

    monkeypatch.setattr(export_eval, "Path", path_class)
    assert (
        export_eval._default_aoti_output_path("TECE-OAM-RRA-1.0")
        == "TECE-OAM-RRA-1.0.pt2"
    )
    assert path_class(
        export_eval._default_aoti_output_path("models/TACE-OAM-7M.pt")
    ) == path_class("models/TACE-OAM-7M.pt2")


@pytest.mark.parametrize("value", [None, "0", "1"])
@pytest.mark.parametrize("training", [False, True])
def test_tf32_environment(monkeypatch, value, training):
    from tace.utils.env import set_tf32
    from tace.utils.precision import set_precision

    previous = (
        torch.backends.cuda.matmul.allow_tf32,
        torch.backends.cudnn.allow_tf32,
    )
    if value is None:
        monkeypatch.delenv("TACE_USE_TF32", raising=False)
    else:
        monkeypatch.setenv("TACE_USE_TF32", value)
    expected = training if value is None else value == "1"
    configurations = [lambda: set_tf32(training=training)]
    if training:
        configurations.append(lambda: set_precision({"trainer": {"precision": 32}}))
    try:
        for configure in configurations:
            torch.backends.cuda.matmul.allow_tf32 = not expected
            torch.backends.cudnn.allow_tf32 = not expected
            configure()
            assert torch.backends.cuda.matmul.allow_tf32 is expected
            assert torch.backends.cudnn.allow_tf32 is expected
    finally:
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = (
            previous
        )


@pytest.mark.parametrize(
    "enabled,supported,expected",
    [
        ((), ("eqx", "luna", "oeq", "eqt", "cue"), None),
        (
            ("eqx", "luna", "oeq", "eqt", "cue"),
            ("cue", "eqt", "oeq", "luna", "eqx"),
            "eqx",
        ),
        (("luna", "oeq", "eqt", "cue"), ("cue", "eqt", "oeq", "luna", "eqx"), "luna"),
        (("oeq", "eqt", "cue"), ("cue", "eqt", "oeq", "luna", "eqx"), "oeq"),
        (("eqt", "cue"), ("cue", "eqt", "oeq", "luna", "eqx"), "eqt"),
        (("cue",), ("cue", "eqt", "oeq", "luna", "eqx"), "cue"),
        (("eqx", "luna", "oeq"), ("oeq", "cue"), "oeq"),
        (("eqx", "luna", "oeq"), ("eqt",), None),
    ],
)
def test_acceleration_priority(monkeypatch, enabled, supported, expected):
    from tace.utils.env import ACCELERATION_ENV, select_acceleration

    for name in ("eqx", "luna", "oeq", "eqt", "cue"):
        monkeypatch.setenv(ACCELERATION_ENV[name], "1" if name in enabled else "0")
    monkeypatch.setenv("TACE_USE_COMPILE", "1")
    assert select_acceleration(*supported, kernel="conv") == expected


def test_acceleration_priority_respects_eqx_kernel_switch(monkeypatch):
    from tace.utils.env import EQX_KERNELS, select_acceleration

    monkeypatch.setenv("TACE_USE_EQX", "1")
    monkeypatch.setenv("TACE_USE_OEQ", "1")
    monkeypatch.setitem(EQX_KERNELS, "conv", False)
    assert select_acceleration("eqx", "oeq", kernel="conv") == "oeq"
    assert select_acceleration("eqx", "eqt", kernel="product") == "eqx"


def test_model_input_properties_include_required_embeddings():
    assert {
        "charges",
        "total_charge",
        "initial_collinear_magmoms",
        "initial_noncollinear_magmoms",
    }.issubset(SUPPORTED_EMBEDDING_PROPERTIES)
    cfg = {
        "loss": {
            "loss_property": [
                "charges",
                "collinear_magnetic_forces",
                "noncollinear_magnetic_forces",
            ]
        },
        "model": {
            "config": {
                "universal_embedding": {
                    "electric_field": {"enable": True},
                    "magnetic_field": {"enable": False},
                },
                "atomic_basis": {"type": "cgtp"},
            }
        },
    }

    assert get_embedding_property(cfg) == [
        "electric_field",
        "total_charge",
        "initial_collinear_magmoms",
        "initial_noncollinear_magmoms",
    ]


def test_model_loading_prunes_removed_architecture_keys():
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config["atomic_basis"]["removed_atomic_option"] = True
    config["product_basis"]["removed_product_option"] = True
    config["radial_basis"]["unrelated_option"] = True

    cleaned = _prune_removed_keys(config)

    assert "removed_atomic_option" not in cleaned["atomic_basis"]
    assert "removed_product_option" not in cleaned["product_basis"]
    assert cleaned["radial_basis"]["unrelated_option"] is True
    assert "removed_atomic_option" in config["atomic_basis"]
    assert "removed_product_option" in config["product_basis"]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("interaction_type", ["cgtp", "o2"])
def test_compile_direct_hessian(dtype, interaction_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_default_dtype(dtype)
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    targets = ["energy", "forces", "direct_hessian"]
    config["special"]["direct_hessian"]["interaction_type"] = interaction_type
    config.update(
        num_layers=1,
        num_channel=2,
        Lmax=2,
        lmax=2,
        cutoff=4.0,
        max_neighbors=None,
        fidelity=[{"name": "DFT"}],
        target_property=targets,
    )
    config["radial_basis"].update(hidden=[4], apply_cutoff=False)
    config["readout_emlp"]["hidden"] = [4]
    config["scale_shift"]["enable"] = False
    statistics = [
        dict(atomic_numbers=[1], atomic_energy={1: 0.0}, avg_num_neighbors=2.0)
    ]
    eager = create_model(config, statistics, targets, []).to(device)
    model = CompileTensorModel(deepcopy(eager.readout_fn)).to(device)
    model.readout_fn.compile_backend = "aot_eager"
    sample = synthetic_graph_sample(model)
    sample["pbc"].zero_()
    expected = eager(dict(sample))
    actual = model(dict(sample))
    assert "direct_hessian_pair_index" not in sample
    assert "direct_hessian_edge_pairs" not in sample
    assert "direct_hessian_edge_shifts" not in sample
    for key in targets:
        torch.testing.assert_close(actual[key], expected[key])
    expected_grads = torch.autograd.grad(
        sum(expected[key].square().sum() for key in targets),
        tuple(eager.parameters()),
        allow_unused=True,
    )
    actual_grads = torch.autograd.grad(
        sum(actual[key].square().sum() for key in targets),
        tuple(model.parameters()),
        allow_unused=True,
    )
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        if expected_grad is None:
            assert actual_grad is None
        else:
            torch.testing.assert_close(actual_grad, expected_grad)

    from tace.utils.loss.normal import NormalLoss

    training_data = model.prepare_property_data(dict(sample))
    training_data.update(
        direct_hessian=torch.zeros_like(actual["direct_hessian"]),
        entropy=torch.ones(2, device=device),
        direct_hessian_weight=torch.ones(2, device=device),
    )
    check_joint_property_loss(
        model,
        training_data,
        NormalLoss(["direct_hessian"], ["mse_direct_hessian"], [1.0]),
    )

    model.eval()
    eager.eval()
    keys = graph_aoti_input_keys(model)
    outputs = model._output_keys()
    ensure_sample_inputs(sample, keys, model)
    flat_model = FlatTorchModel(model, keys, outputs)
    inputs = tuple(sample[key] for key in keys)
    with size_oblivious_export():
        dynamic_shapes = graph_dynamic_shapes(keys, num_graphs=2)
        traced = trace_to_fx(flat_model, inputs, dynamic_shapes=dynamic_shapes)
        exported = torch.export.export(
            traced,
            inputs,
            dynamic_shapes=dynamic_shapes,
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    metadata = export_metadata(model, keys, outputs)
    assert metadata["tace_format"] == "tace_graph_v1"
    loaded = AOTICompiledTensorModel(exported.module(), metadata, device)
    sample.pop("direct_hessian_pair_index")
    sample.pop("direct_hessian_edge_pairs")
    sample.pop("direct_hessian_edge_shifts")
    original_positions = sample["positions"].clone()
    for distance in (1.0, model.readout_fn.pair_representation.cutoff + 1):
        sample["positions"] = original_positions * distance
        expected = eager(dict(sample))
        actual = loaded(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])
        assert "direct_hessian_pair_index" not in sample
        assert "direct_hessian_edge_pairs" not in sample
        assert actual["direct_hessian"].shape == (8, 3, 3)
        if distance > 1:
            assert not actual["direct_hessian"][[1, 2, 5, 6]].any()

    isolated = {key: value[:2] for key, value in sample.items()}
    isolated["ptr"] = sample["ptr"].new_tensor([0, 1, 2])
    isolated["batch"] = sample["batch"].new_tensor([0, 1])
    isolated["edge_index"] = sample["edge_index"][:, :0]
    isolated["edge_shifts"] = sample["edge_shifts"][:0]
    expected = eager(dict(isolated))
    actual = loaded(dict(isolated))
    for key in targets:
        torch.testing.assert_close(actual[key], expected[key])
    assert actual["direct_hessian"].shape == (2, 3, 3)
    isolated = {key: value[:1] for key, value in isolated.items()}
    isolated["ptr"] = sample["ptr"].new_tensor([0, 1])
    isolated["edge_index"] = sample["edge_index"][:, :0]
    with pytest.raises(ValueError, match="at least two atoms"):
        loaded(isolated)

    sample["positions"] = original_positions
    sample["lattice"][0] = sample["lattice"].new_tensor(
        [[12, 0, 0], [2, 14, 0], [1, 2, 16]]
    )
    sample["pbc"][0, 0] = True
    expected = eager(dict(sample))
    for module in (model, loaded):
        actual = module(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])
    # Moving one atom by a lattice vector must leave all predictions unchanged.
    sample["positions"] = original_positions.clone()
    sample["positions"][1] += sample["lattice"][0, 0]
    sample["edge_shifts"][:2, 0] = sample["edge_shifts"].new_tensor([-1, 1])
    for module in (eager, model, loaded):
        actual = module(dict(sample))
        for key in targets:
            torch.testing.assert_close(actual[key], expected[key])

    model.reset_target_property(["direct_hessian"])
    assert model._output_keys() == ("direct_hessian",)


class _MagneticEmbeddingReadout(torch.nn.Module):
    def __init__(
        self,
        embedding_property: list[str],
        atomic_basis_type: Union[str, None],
    ) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer("cutoff", torch.tensor(4.0))
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.int64))
        self.target_property = ["energy", "forces"]
        self.embedding_property = embedding_property
        self.model_config = (
            {"atomic_basis": {"type": atomic_basis_type}}
            if atomic_basis_type is not None
            else {}
        )
        self.max_neighbors = None

    def forward(self, data, graph):
        node_energy = self.scale * (
            graph.positions.square().sum(dim=-1)
            + data["initial_noncollinear_magmoms"].square().sum(dim=-1)
        )
        energy = torch.zeros(
            graph.num_graphs,
            dtype=node_energy.dtype,
            device=node_energy.device,
        ).index_add(0, data["batch"], node_energy)
        return {"energy": energy, "node_energy": node_energy}


class _ChargeReadout(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0), requires_grad=False)
        self.register_buffer("cutoff", torch.tensor(4.0))
        self.register_buffer("atomic_numbers", torch.tensor([1], dtype=torch.int64))
        self.target_property = ["charges"]
        self.embedding_property = []
        self.model_config = {}
        self.max_neighbors = None

    def forward(self, data, graph):
        raw_charges = self.scale * graph.positions[:, 0]
        graph_charges = torch.zeros_like(data["total_charge"]).index_add(
            0,
            data["batch"],
            raw_charges,
        )
        num_atoms = data["ptr"][1:] - data["ptr"][:-1]
        correction = (data["total_charge"] - graph_charges) / num_atoms
        return {
            "energy": graph_charges,
            "charges": raw_charges + correction[data["batch"]],
        }


def _magnetic_embedding_sample() -> dict[str, torch.Tensor]:
    return {
        "pbc": torch.ones(1, 3, dtype=torch.bool),
        "positions": torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        "node_type": torch.zeros(3, dtype=torch.int64),
        "edge_index": torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]]),
        "edge_shifts": torch.zeros(4, 3),
        "lattice": torch.eye(3).unsqueeze(0) * 10.0,
        "batch": torch.zeros(3, dtype=torch.int64),
        "ptr": torch.tensor([0, 3], dtype=torch.int64),
        "fidelity_idx": torch.zeros(1, dtype=torch.int64),
        "initial_noncollinear_magmoms": torch.tensor(
            [[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
        ),
    }


def test_aoti_charges_requires_total_charge_input():
    import inspect

    model = CompileTensorModel(_ChargeReadout()).eval()
    sample = _magnetic_embedding_sample()
    sample.pop("initial_noncollinear_magmoms")

    with pytest.raises(KeyError, match="total_charge"):
        model._input_keys(sample)

    input_keys = graph_aoti_input_keys(model)
    assert input_keys[-1] == "total_charge"
    with pytest.raises(KeyError, match="total_charge"):
        ensure_sample_inputs(dict(sample), input_keys, model)

    sample["total_charge"] = torch.tensor([1.5])
    assert model._input_keys(sample)[-1] == "total_charge"
    output_keys = model._output_keys()
    assert output_keys == ("charges",)
    flat_model = FlatTorchModel(model, input_keys, output_keys).eval()
    inputs = tuple(sample[key] for key in input_keys)
    (charges,) = flat_model(*inputs)
    torch.testing.assert_close(charges.sum(), sample["total_charge"].sum())
    traced_model = trace_to_fx(flat_model, inputs)
    (traced_charges,) = traced_model(*inputs)
    torch.testing.assert_close(traced_charges, charges)

    synthetic = synthetic_graph_sample(model)
    assert synthetic["total_charge"].shape == (2,)
    metadata = export_metadata(model, input_keys, output_keys)
    assert "total_charge" in json.loads(metadata["tace_input_keys"])
    assert "total_charge" in json.loads(metadata["tace_embedding_property"])
    assert json.loads(metadata["tace_output_keys"]) == ["charges"]

    from tace.utils.loss.normal import NormalLoss

    model.readout_fn.scale.requires_grad_(True)
    loss_fn = NormalLoss(["charges"], ["mse_charges"], [1.0])
    for n in (3, 5, 1):
        sample.update(
            positions=torch.randn(n, 3),
            node_type=torch.zeros(n, dtype=torch.long),
            batch=torch.zeros(n, dtype=torch.long),
            ptr=torch.tensor([0, n]),
            charges=torch.zeros(n),
            entropy=torch.ones(1),
            charges_weight=torch.ones(1),
        )
        if n == 1:
            sample.update(
                edge_index=torch.empty(2, 0, dtype=torch.long),
                edge_shifts=torch.empty(0, 3),
            )
        check_joint_property_loss(model, sample, loss_fn)
        if "dynamic_shapes" in inspect.signature(torch.compile).parameters:
            assert len(model._compiled_cache) == (2 if n == 1 else 1)


def check_joint_property_loss(model, data, loss_fn):
    import inspect

    from tace.models.adapter import TensorModel

    if "dynamic_shapes" not in inspect.signature(torch.compile).parameters:
        return
    training = model.training
    backend = getattr(model.readout_fn, "compile_backend", "inductor")
    model.train()
    model.readout_fn.compile_backend = "eager"
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    expected = TensorModel.forward(model, dict(data))
    expected_loss = loss_fn(expected, data)
    expected_grads = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    _, loss = model.forward_loss(dict(data), loss_fn)
    gradients = torch.autograd.grad(loss, parameters, allow_unused=True)
    torch.testing.assert_close(loss, expected_loss)
    for actual, expected in zip(gradients, expected_grads):
        if expected is None:
            assert actual is None
        else:
            torch.testing.assert_close(actual, expected)
    model.train(training)
    model.readout_fn.compile_backend = backend


def test_compiled_evaluation_disables_autograd(monkeypatch):
    model = CompileTensorModel(_ChargeReadout()).eval()
    sample = _magnetic_embedding_sample()
    sample.pop("initial_noncollinear_magmoms")
    sample["total_charge"] = torch.tensor([1.5])
    observed = []

    monkeypatch.setattr(
        compile_wrapper,
        "trace_and_compile",
        lambda *args, **kwargs: (object(), (), ()),
    )

    def fake_compiled_call(*args, **kwargs):
        observed.append(torch.is_grad_enabled())
        return (torch.zeros(3),)

    monkeypatch.setattr(compile_wrapper, "compiled_call", fake_compiled_call)

    with torch.enable_grad():
        model._compiled_forward(sample)

    assert observed == [False]


def _slice_update(x: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
    output = x.clone()
    output[:, 1:2, :] = value
    return output


def test_trace_to_fx_removes_full_dynamic_slice_scatter():
    x = torch.randn(4, 3, 5)
    value = torch.randn(4, 1, 5)
    traced = trace_to_fx(_slice_update, (x, value), functionalize=True)

    for node in traced.graph.nodes:
        if node.op != "call_function":
            continue
        assert node.target != torch.ops.aten.copy_.default
        if node.target == torch.ops.aten.slice_scatter.default:
            start = node.args[3] if len(node.args) > 3 else None
            end = node.args[4] if len(node.args) > 4 else None
            assert (start, end) != (0, sys.maxsize)

    torch.testing.assert_close(traced(x, value), _slice_update(x, value))


@pytest.mark.parametrize(
    ("embedding_property", "atomic_basis_type"),
    [
        pytest.param([], "uv_o2_mag", id="o2-magnetic"),
        pytest.param(
            ["initial_noncollinear_magmoms"],
            "cgtp",
            id="universal-embedding",
        ),
    ],
)
def test_aoti_keeps_magnetic_embedding_without_magnetic_force_target(
    embedding_property,
    atomic_basis_type,
):
    model = CompileTensorModel(
        _MagneticEmbeddingReadout(embedding_property, atomic_basis_type)
    ).eval()
    sample = _magnetic_embedding_sample()

    assert "initial_noncollinear_magmoms" in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )
    assert "noncollinear_magnetic_forces" not in get_need_property(
        model.get_target_property(),
        model.get_embedding_property(),
        training=True,
    )

    input_keys = graph_aoti_input_keys(model)
    output_keys = model._output_keys()
    assert "initial_noncollinear_magmoms" in input_keys
    assert "noncollinear_magnetic_forces" not in output_keys

    flat_model = FlatTorchModel(model, input_keys, output_keys).eval()
    output = flat_model(*(sample[key] for key in input_keys))
    zero_magmoms = dict(sample)
    zero_magmoms["initial_noncollinear_magmoms"] = torch.zeros_like(
        sample["initial_noncollinear_magmoms"]
    )
    output_without_magmoms = flat_model(*(zero_magmoms[key] for key in input_keys))
    assert output_keys == ("energy", "node_energy", "forces")
    assert not torch.equal(output[0], output_without_magmoms[0])
    torch.testing.assert_close(output[2], output_without_magmoms[2])
    torch.testing.assert_close(output[2], -2.0 * sample["positions"])

    metadata = export_metadata(model, input_keys, output_keys)
    assert "initial_noncollinear_magmoms" in json.loads(metadata["tace_input_keys"])
    assert "initial_noncollinear_magmoms" in json.loads(
        metadata["tace_embedding_property"]
    )
    assert "noncollinear_magnetic_forces" not in json.loads(
        metadata["tace_output_keys"]
    )

    from tace.utils.loss.normal import NormalLoss

    model.readout_fn.scale.requires_grad_(True)
    model.reset_target_property(["energy", "noncollinear_magnetic_forces"])
    sample.update(
        noncollinear_magnetic_forces=torch.zeros(3, 3),
        entropy=torch.ones(1),
        noncollinear_magnetic_forces_weight=torch.ones(1),
    )
    check_joint_property_loss(
        model,
        sample,
        NormalLoss(
            ["noncollinear_magnetic_forces"],
            ["mse_noncollinear_magnetic_forces"],
            [1.0],
        ),
    )


def test_import_without_cpu_affinity_or_cuda(tmp_path):
    script = """
import os
import sys

if hasattr(os, "sched_getaffinity"):
    del os.sched_getaffinity
sys.modules["torch.utils.cpp_extension"] = None

from tace.interface.ase import TACEAseCalc
from tace.lightning import load_tace
from tace.scripts.train import main
from equivariantx.ops.common.cuda import runtime as cuda

assert not hasattr(os, "sched_getaffinity")
assert not cuda._POOL._threads
assert not cuda.runtime.cache_info().currsize
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("training", [False, True])
def test_precision_policy(monkeypatch, training):
    from tace.utils.precision import PrecisionPolicy, resolve_precision

    monkeypatch.delenv("TACE_USE_TF32", raising=False)
    monkeypatch.delenv("TACE_MATMUL_PRECISION", raising=False)
    default_dtype = torch.get_default_dtype()
    cfg = {"dtype": {"model": "float32", "autocast": None, "matmul": None}}
    policy = resolve_precision(cfg, training=training)
    assert policy.matmul == ("tf32" if training else "ieee")
    assert policy.torch_dtype == torch.float32
    assert torch.get_default_dtype() == default_dtype
    assert cfg["dtype"]["matmul"] == policy.matmul

    mixed = resolve_precision({"dtype": {"autocast": "bfloat16"}}, training=training)
    assert mixed.matmul == "ieee"
    for options in (
        {"model": "bfloat16"},
        {"model": "float64", "autocast": "bfloat16"},
        {"autocast": "float16", "matmul": "tf32"},
        {"matmul": "bf16x3"},
    ):
        with pytest.raises(ValueError):
            PrecisionPolicy(**options)

    monkeypatch.setenv("TACE_USE_TF32", "0")
    assert resolve_precision({}, training=training).matmul == "ieee"
    with pytest.raises(ValueError, match="conflicts"):
        resolve_precision({"dtype": {"matmul": "tf32"}}, training=training)
    monkeypatch.setenv("TACE_MATMUL_PRECISION", "tf32")
    with pytest.raises(ValueError, match="conflicts"):
        resolve_precision({}, training=training)


def test_model_dtype_metadata(tmp_path, monkeypatch):
    from tace.lightning import export_tace, load_tace

    monkeypatch.setenv("TACE_USE_TF32", "0")
    config = deepcopy(DEFAULT_MODEL_CONFIG)
    config.update(
        num_layers=1,
        num_channel=2,
        Lmax=1,
        lmax=1,
        mmax=1,
        cutoff=4.0,
        max_neighbors=None,
    )
    config["radial_basis"]["hidden"] = [4]
    config["readout_emlp"]["hidden"] = [4]
    model = create_model(
        model_config=config,
        statistics=[
            {
                "atomic_numbers": [1],
                "atomic_energy": {1: -1.23456789123},
                "avg_num_neighbors": 2.0,
                "rms_forces": {1: 1.0},
            }
        ],
        target_property=["energy"],
        embedding_property=[],
    )
    path = tmp_path / "model.pt"
    export_tace(model, str(path))
    package = torch.load(path, weights_only=False)
    assert package["dtype"]["model"] == str(model.get_model_dtype()).removeprefix(
        "torch."
    )
    restored = load_tace(path)
    assert restored.get_model_dtype() == model.get_model_dtype()
    assert restored.readout_fn.atomic_energy_layer.atomic_energy.dtype == torch.float64
    with pytest.raises(ValueError, match="FP64 geometry"):
        load_tace(path, device="mps")

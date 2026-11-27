"""End-to-end regional AOTInductor deployment, including a clean runtime."""

import json
import os
import subprocess
import sys
import zipfile
from collections import Counter
from copy import deepcopy

import pytest
import torch

from equivariantx_foundation import optimize
from equivariantx_foundation.models.tace.planner import plan
from equivariantx_foundation.targets import aoti
from equivariantx_foundation.tests.test_tace_plan import graph, make_model
from tace.models.adapter import TensorModel
from tace.models.compile import aot
from tace.models.compile.compile import inductor_options, trace_to_fx
from tace.models.compile.wrapper import FlatTorchModel
from tace.models.spherical.tace import SphericalTACE
from tace.utils.env import ACCELERATION_ENV

RUNTIME = """
import json, sys
import torch
from equivariantx_foundation.targets.aoti import load
runner = load(sys.argv[1])
cases = torch.load(sys.argv[2], weights_only=True)
assert "tace" not in sys.modules, "loading must not construct a TACE host model"
report = []
with torch.inference_mode():
    for case in cases:
        outputs = runner(*(v.cuda() for v in case["inputs"]))
        errors = {}
        for name, expected, actual in zip(case["properties"], case["expected"], outputs, strict=True):
            actual = actual.cpu()
            torch.testing.assert_close(actual, expected, rtol=4e-4, atol=4e-5)
            if name == "direct_hessian" and case.get("hessian_edges") == 0:
                offdiag = ~torch.eye(3, dtype=torch.bool)
                blocks = actual.reshape(2,3,3,3,3)
                assert not blocks[:,offdiag].any() and blocks[:,~offdiag].abs().max() > 1e-7
            errors[name] = (actual - expected).abs().max().item()
        report.append({"shape": case["shape"], "max_abs_errors": errors})
    args = [tuple(x.cuda() for x in case["inputs"]) for case in cases]
    replay = load(sys.argv[1], replay_inputs=args[0])
first = replay(*args[0])
saved = tuple(x.clone() for x in first)
for case_inputs, case in [(args[2], cases[2]), (args[0], cases[0])]:
    got = replay(*case_inputs)
    for actual, expected in zip(got, case["expected"], strict=True):
        torch.testing.assert_close(actual.cpu(), expected, rtol=4e-4, atol=4e-5)
    for before, snapshot, new in zip(first, saved, got, strict=True):
        assert before.data_ptr() != new.data_ptr()
        torch.testing.assert_close(before, snapshot, atol=0, rtol=0)
assert any(not torch.equal(a, b) for a, b in zip(cases[0]["expected"], cases[2]["expected"]))
try:
    replay(*args[1])
except ValueError as e:
    assert "shape" in str(e)
else:
    raise AssertionError("Replay accepted an input with different atom and edge counts")
with torch.cuda.stream(torch.cuda.Stream()):
    try:
        replay(*args[0])
    except ValueError as e:
        assert "stream" in str(e)
    else:
        raise AssertionError("Replay accepted a different caller stream")
assert "tace" not in sys.modules
print(json.dumps(report))
"""


def _cases():
    sample = graph("cuda")
    other = {k: v.clone() for k, v in sample.items()}
    keep = (other["edge_index"] != 5).all(0)
    for key in ("positions", "node_type", "batch"):
        other[key] = other[key][:-1].contiguous()
    other["ptr"][-1] -= 1
    other["edge_index"] = other["edge_index"][:, keep].flip(1).contiguous()
    other["edge_shifts"] = other["edge_shifts"][keep].flip(0).contiguous()
    other["positions"] *= 0.99
    moved = {k: v.clone() for k, v in sample.items()}
    moved["positions"][1, 2] += 0.29
    moved["edge_index"] = moved["edge_index"].flip(1).contiguous()
    moved["edge_shifts"] = moved["edge_shifts"].flip(0).contiguous()
    moved["node_type"] = moved["node_type"].roll(1)
    return sample, other, moved


def _model(basis):
    if basis == "hip":
        from equivariantx_foundation.tests.test_tace_plan import physics_model

        return physics_model("direct_hessian")
    if basis == "w6j_mag":
        from equivariantx_foundation.tests.test_interaction_schedule import (
            _wigner6j_model,
        )

        return _wigner6j_model()
    if basis == "uv_o2_mag":
        from equivariantx_foundation.tests.test_tace_plan import physics_model

        return physics_model("noncollinear_magnetic_forces")
    if basis == "cgtp":
        return make_model()
    if basis == "rotary":
        from equivariantx_foundation.tests.test_tece import _rotary_model

        return _rotary_model()
    if basis in ("cartesian", "nonlinear_ictd"):
        from tace.models.cartesian.tace import CartesianTACE

        def cartesian_model(**cfg):
            if basis == "nonlinear_ictd":
                cfg["product_basis"].update(type=basis, correlation=3)
            return CartesianTACE(**cfg)

        return make_model(cartesian_model)
    cfg = deepcopy(make_model().readout_fn.model_config)
    cfg["atomic_basis"]["type"] = basis
    cfg["radial_basis"]["apply_cutoff"] = False
    return TensorModel(SphericalTACE(**cfg))


def _hip_reordered(data):
    """Reorder atoms within each structure, edges independently, and labelled Hessian pairs on both axes."""
    positions = data["positions"]
    order = torch.tensor([2, 0, 1, 4, 5, 3], device=positions.device)
    out = dict(data)
    for key in ("positions", "node_type", "forces"):
        if key in data:
            out[key] = data[key][order]
    out["edge_index"] = order.argsort()[data["edge_index"].flip(1)].contiguous()
    out["edge_shifts"] = data["edge_shifts"].flip(0).contiguous()
    if "direct_hessian" in data:
        out["direct_hessian"] = _hip_reordered_blocks(data["direct_hessian"], order)
    return out


def _hip_reordered_blocks(blocks, order):
    # Row-major ordered pairs are local to each structure, not a global NxN tensor.
    p = order.reshape(2, 3) % 3
    b = torch.arange(2, device=p.device)[:, None, None]
    return blocks.reshape(2, 3, 3, 3, 3)[b, p[:, :, None], p[:, None, :]].reshape(
        -1, 3, 3
    )


def _spin_inputs(batch, basis):
    if basis == "uv_o2_mag":
        batch["node_type"].zero_()  # The SOC fixture contains only Fe.
    pos = batch["positions"]
    batch["initial_noncollinear_magmoms"] = torch.sin(pos + pos.roll(1, dims=1))


def test_joint_replay_static_graph_contract():
    class Program(torch.nn.Module):
        def __init__(self, fn):
            super().__init__()
            self.fn = fn

        def forward(self, x):
            return self.fn(x)

    x = torch.tensor([1.0, -1.0])
    for fn, allowed in (
        (torch.sin, True),
        (lambda t: t[t.long().abs()], True),
        (torch.rand_like, False),
        (torch.nonzero, False),
        (lambda t: torch.cond(t.sum() > 0, torch.sin, torch.cos, (t,)), False),
    ):
        program = torch.export.export(Program(fn), (x,), strict=False)
        assert aoti._static_training_graph(program) is allowed


def _operators(exported):
    return Counter(
        str(node.target)
        for node in exported.graph.nodes
        if isinstance(node.target, torch._ops.OpOverload)
        and node.target.namespace in ("eqx", "eqx_foundation")
    )


def _has_indexed_o3(exported):
    from equivariantx.utils.metadata import parse_metadata

    for node in exported.graph.nodes:
        if node.target != torch.ops.eqx.edge_program.default:
            continue
        expression, _ = parse_metadata(node.args[0])
        if any(
            op == "convolution" and data[0] == "o3" for op, _, _, data in expression
        ) and any(
            op == "input" and isinstance(data[1], tuple) and data[1][0] == "indexed"
            for op, _, _, data in expression
        ):
            return True
    return False


def _rowwise_o3_calls(program):
    # None source/target means no indexed message passing. Count in the same
    # exported graph passed to the packager, including any retained adjoints.
    return sum(
        n.target is torch.ops.eqx.o3_contraction.default and n.args[2:4] == (None, None)
        for n in program.graph.nodes
    )


def _install_hip_rowwise(host, monkeypatch):
    """Activate only this HIP pair at construction; preserve the existing trunk."""
    old = host.readout_fn.pair_representation
    with monkeypatch.context() as enabled:
        enabled.setenv("TACE_USE_EQX", "1")
        pair = _model("hip").readout_fn.pair_representation.to(
            next(old.parameters()).device
        )
    pair.load_state_dict(old.state_dict(), strict=True)
    host.readout_fn.pair_representation = pair.train(old.training)


def test_foreign_aoti_export_registers_only_referenced_eqx_namespaces(
    tmp_path, monkeypatch
):
    package = tmp_path / "model.pt2"
    operators = (
        "eqx_oam_l_conv::affine",
        "eqx_tace_product::bilinear_ace",
        "aten::mm",
        "vendor::user_op",
    )
    with zipfile.ZipFile(package, "w") as archive:
        for model, targets in (("model", operators), ("other", ("eqx_unused::f",))):
            archive.writestr(
                f"model/data/aotinductor/{model}/program.wrapper.json",
                json.dumps({"nodes": [{"node": {"target": t}} for t in targets]}),
            )
    imports = []
    monkeypatch.setattr(aoti, "import_module", imports.append)

    class Compiled:
        def get_metadata(self):
            return {"AOTI_DEVICE_KEY": "cuda"}

    runner = Compiled()
    monkeypatch.setattr(torch._inductor, "aoti_load_package", lambda *a, **kw: runner)
    with zipfile.ZipFile(package) as archive:
        assert aoti._package_namespaces(archive) == [
            "eqx_oam_l_conv",
            "eqx_tace_product",
        ]
    if not torch.cuda.is_available():
        with pytest.raises(RuntimeError, match="requires an NVIDIA CUDA device"):
            aoti.load(package)
        assert not imports
        return
    assert aoti.load(package) is runner
    assert imports == [
        "equivariantx_foundation.models.tace.tace_oam_l.convolution",
        "equivariantx_foundation.ops.bilinear",
    ]


def test_standalone_fixed_input_guard(monkeypatch):
    calls = []

    class Compiled:
        def get_metadata(self):
            return {
                "tace_fixed_inputs": '[["x", [2, 2], "float32"]]',
                "AOTI_DEVICE_KEY": "cpu",
            }

        def __call__(self, x):
            calls.append(x)
            return x.square()

    monkeypatch.setattr(aoti, "_load_runner", lambda *a, **kw: Compiled())
    runner = aoti.load("unused.pt2")
    x = torch.rand(2, 2)
    torch.testing.assert_close(runner(x), x.square())
    for other in (x[:1], x.double(), x.T, torch._neg_view(x)):
        with pytest.raises(ValueError, match="Fixed AOTI"):
            runner(other)
    with pytest.raises(ValueError):
        runner(x, x)
    assert len(calls) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph inference")
def test_replay_preserves_output_tree(tmp_path):
    class Potential(torch.nn.Module):
        def forward(self, x):
            return {"energy": x.square().sum(), "derivatives": (2 * x, None)}

    x = torch.arange(9, device="cuda", dtype=torch.float32).view(3, 3)
    path = aoti.export(torch.export.export(Potential(), (x,)), tmp_path / "nested.pt2")
    step = aoti.load(path, replay_inputs=(x,))
    before = step(x)
    after = step(x + 2)
    for value, result in [(x, before), (x + 2, after)]:
        torch.testing.assert_close(result, Potential()(value))
        assert isinstance(result["derivatives"], tuple)
    with torch.autocast("cuda"):
        with pytest.raises(ValueError, match="autocast"):
            step(x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="regional AOTI needs CUDA")
@pytest.mark.parametrize(
    "basis", ["cgtp", "uu_o2", "rotary", "uv_o2_mag", "w6j_mag", "hip", "hip_rowwise"]
)
def test_regional_efs_package(tmp_path, monkeypatch, basis):
    for variable in ACCELERATION_ENV.values():
        monkeypatch.setenv(variable, "0")
    torch.set_default_dtype(torch.float32)
    rowwise = basis == "hip_rowwise"
    native = _model("hip" if rowwise else basis).eval().cuda()
    chosen = plan(native, task="inference", backend="cuda")
    assert chosen.schedule == "regional" and any(
        r.implementation == "cuda" and r.path.startswith("interactions.")
        for r in chosen.regions
    )
    prepared = optimize(native, family="tace", task="inference", backend="cuda")
    if rowwise:
        _install_hip_rowwise(prepared, monkeypatch)
        assert not hasattr(
            native.readout_fn.pair_representation.interactions[0].rejector,
            "contraction",
        )
    tensor_model = aot.as_compile_tensor_model(prepared)
    keys = aot.graph_aoti_input_keys(tensor_model)
    properties = ("energy", "forces", "stress")
    magnetic = basis in ("uv_o2_mag", "w6j_mag")
    if magnetic:
        properties += ("noncollinear_magnetic_forces",)
    cases, records = _cases(), []
    if basis == "hip" or rowwise:
        properties += ("direct_hessian",)
        cases = (*cases[:2], _hip_reordered(cases[0]))
        if rowwise:
            # Two triangles with all distances between the pair (3 Å) and
            # trunk (4 Å) cutoffs: twelve physical trunk edges, no pair edges.
            isolated = dict(cases[0])
            isolated["positions"] = isolated["positions"].new_tensor(
                [[0, 0, 0], [3.4, 0, 0], [1.7, 2.945, 0]] * 2
            )
            isolated["lattice"] = isolated["lattice"] * 4
            isolated["edge_shifts"] = torch.zeros_like(isolated["edge_shifts"])
            cases = (*cases, isolated)
    for data in cases:
        if magnetic:
            _spin_inputs(data, basis)
        expected = native(dict(data))
        assert all(expected[k].abs().max() > 1e-12 for k in properties)
        aot.ensure_sample_inputs(data, keys, tensor_model)
        if rowwise and data is cases[-1]:
            assert data["direct_hessian_edge_pairs"].numel() == 0
            offdiag = ~torch.eye(3, dtype=torch.bool, device="cuda")
            blocks = expected["direct_hessian"].reshape(2, 3, 3, 3, 3)
            assert not blocks[:, offdiag].any()
            assert blocks[:, ~offdiag].abs().max() > 1e-7
        records.append(
            dict(
                properties=properties,
                inputs=tuple(data[k].detach().cpu() for k in keys),
                expected=tuple(expected[k].detach().cpu() for k in properties),
                shape=[len(data["positions"]), data["edge_index"].shape[1]],
            )
        )
        if rowwise:
            records[-1]["hessian_edges"] = data["direct_hessian_edge_pairs"].numel()
    torch.save(records, tmp_path / "cases.pt")
    if basis == "hip":
        h = records[0]["expected"][-1]
        assert h.reshape(2, 3, 3, 3, 3)[:, 0, 1, 0, 2].abs().max() > 1e-7
        torch.testing.assert_close(
            records[2]["expected"][-1],
            _hip_reordered_blocks(h, torch.tensor([2, 0, 1, 4, 5, 3])),
            atol=4e-6,
            rtol=1e-4,
        )
    flat = FlatTorchModel(tensor_model, keys, properties).eval()
    args = tuple(cases[0][k] for k in keys)
    with aot.size_oblivious_export():
        traced = trace_to_fx(flat, args, functionalize=magnetic)
        exported = torch.export.export(
            traced,
            args,
            dynamic_shapes=aot.graph_dynamic_shapes(keys, num_graphs=2),
            strict=False,
            prefer_deferred_runtime_asserts_over_guards=True,
        )
    ops = _operators(exported)
    if rowwise:
        assert _rowwise_o3_calls(exported) > 0
    print("retained EQX operators:", dict(ops), flush=True)
    if basis in ("cgtp", "hip"):
        assert any("o3_" in op or "edge_program" in op for op in ops), ops
    elif basis == "w6j_mag":
        assert any("sparse_bilinear" in op or "o3_contraction" in op for op in ops), ops
    elif basis == "rotary":
        assert _has_indexed_o3(exported), ops
        assert any("tece_interaction" in op for op in ops), ops
        assert any("tece_contraction" in op for op in ops), ops
    else:
        # Depending on packing the native local contraction is either one
        # edge expression or indexed frame operations (ACE uses ace_contract).
        assert any("frame" in op or "edge_program" in op for op in ops), ops
    package = aoti.export(
        exported, tmp_path / f"{basis}.pt2", inductor_configs=inductor_options()
    )
    with zipfile.ZipFile(package) as archive:
        (manifest,) = [n for n in archive.namelist() if n.endswith("/" + aoti.MANIFEST)]
        assert "eqx" in archive.read(manifest).decode().splitlines()
    result = subprocess.run(
        [sys.executable, "-c", RUNTIME, str(package), str(tmp_path / "cases.pt")],
        env={**os.environ, "TACE_USE_COMPILE": "0", "TACE_USE_EQX": "0"},
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = {
        "model": basis,
        "operators": dict(ops),
        "child_process": json.loads(result.stdout.strip().splitlines()[-1]),
    }
    (tmp_path / "report.json").write_text(json.dumps(report, indent=2))


TRAINING_RUNTIME = """
import json, sys
import torch
from equivariantx_foundation.targets.aoti import load_training
step = load_training(sys.argv[1])
test = torch.load(sys.argv[2], map_location="cuda", weights_only=True)
params = {k: torch.nn.Parameter(v) for k, v in test["parameters"].items()}
opt = torch.optim.SGD(params.values(), lr=test["lr"])
replay = None
if test.get("check_replay"):
    replay = load_training(sys.argv[1], replay_inputs=(test["batch"], params, test["buffers"]))
history = []
for expected in test["steps"]:
    batch = {**test["batch"], **expected.get("head_input", {})}
    pred, loss, grads = step(batch, params, test["buffers"])
    if replay is not None:
        actual = replay(batch, params, test["buffers"])
        torch.testing.assert_close(actual, (pred, loss, grads), atol=2e-5, rtol=3e-4)
        pred, loss, grads = actual
        if history:
            torch.testing.assert_close(first_loss, first_snapshot, atol=0, rtol=0)
        else:
            first_loss, first_snapshot = loss, loss.clone()
    torch.testing.assert_close(loss, expected["loss"], atol=2e-5, rtol=3e-4)
    errors = {"loss": (loss - expected["loss"]).abs().item()}
    for key in test["properties"]:
        torch.testing.assert_close(pred[key], expected[key], atol=2e-5, rtol=3e-4)
        errors[key] = (pred[key] - expected[key]).abs().max().item()
    assert grads.keys() == expected["grads"].keys()
    opt.zero_grad(set_to_none=True)
    errors["gradient"] = 0.
    for name,g in grads.items():
        ref = expected["grads"][name]
        assert (ref is None) == (g is None)
        if g is not None:
            torch.testing.assert_close(g, ref, atol=2e-5, rtol=3e-4, msg=name)
            errors["gradient"] = max(errors["gradient"], (g-ref).abs().max().item())
            params[name].grad = g.clone()
    opt.step()
    errors["parameter"] = 0.
    for name, ref in expected["state"].items():
        torch.testing.assert_close(params[name], ref, atol=2e-6, rtol=3e-5)
        errors["parameter"] = max(errors["parameter"],(params[name]-ref).abs().max().item())
    history.append([loss.item(), errors])
assert abs(history[1][0]-history[0][0]) > 1e-6, history
try:
    step({**test["batch"], "positions":test["batch"]["positions"][:-1]}, params, test["buffers"])
except ValueError:
    pass
else:
    raise AssertionError("wrong fixed input shape not rejected")
if replay is not None:
    for mapping in (params, test["buffers"]):
        key = next(k for k, v in mapping.items() if v.numel())
        original = mapping[key]
        mapping[key] = original.clone()
        try:
            replay(batch, params, test["buffers"])
        except ValueError as error:
            assert "storage" in str(error)
        else:
            raise AssertionError("captured training state silently went stale")
        mapping[key] = original
    key = next(k for k,p in params.items() if p.numel())
    params[key].data = params[key].detach().clone()
    try:
        replay(batch, params, test["buffers"])
    except ValueError as error:
        assert "storage" in str(error)
    else:
        raise AssertionError("same Parameter object with new storage was accepted")
assert "tace" not in sys.modules
print(json.dumps(history))
"""


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="regional training package needs CUDA"
)
@pytest.mark.parametrize(
    "basis",
    [
        "cgtp",
        "uu_o2",
        "uv_o2",
        "cartesian",
        "rotary",
        "nonlinear_ictd",
        "uv_o2_mag",
        "w6j_mag",
        "hip",
    ],
)
def test_regional_aoti_live_joint_training(tmp_path, monkeypatch, basis):
    from tace.utils.loss.normal import NormalLoss

    for variable in ACCELERATION_ENV.values():
        monkeypatch.setenv(variable, "0")
    torch.set_default_dtype(torch.float32)

    def build():
        m = _model(basis)
        if basis == "cgtp":
            cfg = deepcopy(m.readout_fn.model_config)
            cfg["fidelity"] = [{"name": "a"}, {"name": "b"}]
            cfg["statistics"] = [
                {
                    **cfg["statistics"][0],
                    "atomic_energy": {1: -h, 8: -2 * h},
                    "rms_forces": {1: 1.0 + h, 8: 1.5 + h},
                    "mean_energy_per_atom": {1: 0.2 + h, 8: 0.5 + h},
                }
                for h in range(2)
            ]
            cfg["scale_shift"].update(
                enable=True,
                all_atoms=True,
                scale_type="rms_forces",
                shift_type="mean_energy_per_atom",
                scale_trainable=True,
                shift_trainable=True,
            )
            m = TensorModel(SphericalTACE(**cfg))
        return m.cuda().train()

    ref, host = build(), build()
    for item in (ref, host):
        # An unused trainable parameter must round-trip as None, not a zero
        # gradient (weight decay gives the two different SGD semantics).
        item.readout_fn.register_parameter(
            "unused_parameter", torch.nn.Parameter(torch.ones(2, device="cuda"))
        )
        item.readout_fn.register_buffer("unused_lookup", torch.ones(3, device="cuda"))
    host.load_state_dict(ref.state_dict(), strict=True)
    chosen = plan(host, task="training", backend="cuda")
    assert chosen.schedule == "regional" and any(
        r.implementation == "cuda" and r.path.startswith("interactions.")
        for r in chosen.regions
    ), chosen
    model = optimize(host, family="tace", task="training", backend="cuda")
    properties = ["energy", "forces", "stress"]
    loss_functions = ["mse_energy_per_atom", "mse_forces", "mse_stress"]
    weights = [1.0, 2.0, 15.0]
    if basis in ("uv_o2_mag", "w6j_mag"):
        properties.append("noncollinear_magnetic_forces")
        loss_functions.append("mse_noncollinear_magnetic_forces")
        weights.append(3.0)
    if basis == "hip":
        properties.append("direct_hessian")
        loss_functions.append("mse_direct_hessian")
        weights.append(3.0)
    obj = NormalLoss(properties, loss_functions, weights).train()
    batch = graph("cuda")
    if basis == "cgtp":
        del batch["fidelity_idx"]  # The export sample deliberately has no head IDs.
    pos = batch["positions"]
    ones = pos.new_ones(len(batch["ptr"]) - 1)
    if basis == "hip":
        # Nonconstant symmetric labels with nonzero Cartesian and atom-pair off-diagonals.
        u = torch.cos(pos + 0.7).reshape(2, 3, 3)
        batch.update(
            direct_hessian=torch.einsum("gia,gjb->gijab", u, u).reshape(-1, 3, 3),
            direct_hessian_weight=ones,
        )
    if basis in ("uv_o2_mag", "w6j_mag"):
        _spin_inputs(batch, basis)
        batch.update(
            noncollinear_magnetic_forces=0.4 * torch.cos(pos),
            noncollinear_magnetic_forces_weight=ones,
        )
    batch.update(
        energy=pos.new_tensor([1.2, -0.6]),
        forces=0.3 * torch.sin(pos),
        stress=0.07 * batch["lattice"],
        entropy=ones,
        energy_weight=ones,
        forces_weight=ones,
        stress_weight=ones,
    )
    # Inspect the exact same graph we package (no second trace). The child
    # must subsequently execute it with live state and no TACE installed in memory.
    package_export = aoti.export

    def inspected_export(exported, output, **kwargs):
        schema = json.loads(
            kwargs["inductor_configs"]["aot_inductor.metadata"][aoti.TRAINING_SCHEMA]
        )
        active = {schema["buffers"][i] for i in schema["active_buffers"]}
        assert any(k.endswith(".unused_lookup") for k in schema["buffers"])
        assert not any(k.endswith(".unused_lookup") for k in active)
        assert any(k.endswith(".atomic_energy") for k in active)
        assert len(exported.graph_signature.user_inputs) == (
            len(schema["batch"]) + len(schema["parameters"]) + len(active)
        )
        ops = _operators(exported)
        (tmp_path / "joint-operators.json").write_text(
            json.dumps(
                {
                    "model": basis,
                    "operators": ops,
                    "regions": [
                        [r.path, r.implementation, r.reason] for r in chosen.regions
                    ],
                },
                indent=2,
            )
        )
        if basis == "cartesian":
            assert (
                ops["eqx.o3_contraction.default"] and ops["eqx.stf_rank_two.default"]
            ), ops
        elif basis == "nonlinear_ictd":
            assert ops["eqx.stf_rank_two.default"], ops
            assert any(
                r.path.startswith("products.") and "projected ICTD" in r.reason
                for r in chosen.regions
            )
        elif basis == "rotary":
            assert _has_indexed_o3(exported), ops
            assert any("tece_interaction" in op for op in ops), ops
            assert any("tece_contraction" in op for op in ops), ops
        elif basis == "w6j_mag":
            assert ops["eqx.o3_contraction.default"], ops
        else:
            assert any("edge_program" in op or "frame" in op for op in ops), ops
        return package_export(exported, output, **kwargs)

    monkeypatch.setattr(aoti, "export", inspected_export)
    package = aoti.export_training(model, obj, batch, tmp_path / "joint-training.pt2")
    with zipfile.ZipFile(package) as archive:
        (manifest,) = [n for n in archive.namelist() if n.endswith("/" + aoti.MANIFEST)]
        assert "eqx" in archive.read(manifest).decode().splitlines()
    data, params, buffers = aoti.training_inputs(model, obj, batch)
    test = dict(
        properties=properties,
        check_replay=basis in ("cgtp", "uv_o2_mag"),
        batch=data,
        parameters={k: v.detach() for k, v in params.items()},
        buffers=buffers,
        # Keep both updates on a stable E/F/S trajectory for this initialization.
        lr=5e-4 if basis == "rotary" else 0.02,
        steps=[],
    )
    opt = torch.optim.SGD(ref.parameters(), lr=test["lr"])
    selections = (
        [(0, None), (0, [1, 0]), (1, None)] if basis == "cgtp" else [(0, None)] * 2
    )
    for step_idx, (default, ids) in enumerate(selections):
        raw = dict(batch)
        heads = {}
        if basis == "hip" and step_idx:
            raw = _hip_reordered(raw)
            # Rebuild every geometry-dependent pair map as a deployment caller must.
            heads, _, _ = aoti.training_inputs(model, obj, raw)
        if basis == "cgtp":
            ref.reset_fidelity_idx(default)
            model.reset_fidelity_idx(default)
            if ids is not None:
                raw["fidelity_idx"] = torch.tensor(
                    ids, dtype=torch.int32, device="cuda"
                )
            runtime, _, _ = aoti.training_inputs(model, obj, raw)
            want = torch.tensor(
                [default, default] if ids is None else ids, device="cuda"
            )
            torch.testing.assert_close(runtime["fidelity_idx"], want)
            assert ("fidelity_idx" in raw) == (ids is not None)
            heads = {"fidelity_idx": runtime["fidelity_idx"]}
        opt.zero_grad(set_to_none=True)
        y = ref(raw)
        loss = obj(y, raw)
        loss.backward()
        grads = {
            "model." + k: p.grad.detach().clone() if p.grad is not None else None
            for k, p in ref.named_parameters()
            if p.requires_grad
        }
        assert grads["model.readout_fn.unused_parameter"] is None
        if basis == "hip":
            assert any(
                ".direct_hessian_readout.edge_readout." in k
                and g is not None
                and g.abs().max() > 1e-7
                for k, g in grads.items()
            )
        if basis == "cgtp":
            shift = grads["model.readout_fn.scale_shift.shift"]
            for h in range(2):
                assert bool(shift[h].abs().sum()) == bool(
                    (heads["fidelity_idx"] == h).any()
                )
        assert any(
            ".interactions." in k and g is not None and g.abs().max() > 1e-6
            for k, g in grads.items()
        )
        assert any(
            ".products." in k and g is not None and g.abs().max() > 1e-6
            for k, g in grads.items()
        )
        opt.step()
        test["steps"].append(
            dict(
                loss=loss.detach(),
                grads=grads,
                head_input=heads,
                state={
                    "model." + k: v.detach().clone() for k, v in ref.named_parameters()
                },
                **{k: y[k].detach() for k in properties},
            )
        )
    torch.save(test, tmp_path / "joint-case.pt")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            TRAINING_RUNTIME,
            str(package),
            str(tmp_path / "joint-case.pt"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    (tmp_path / "joint-report.json").write_text(
        json.dumps(json.loads(result.stdout.strip().splitlines()[-1]), indent=2)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP pair CUDA operator")
def test_hip_pair_table_free_joint_keeps_rowwise(monkeypatch):
    """A functional joint with no live CG operands may still embed the fixed kernel."""
    from e3nn import o3
    from torch.fx.experimental.proxy_tensor import make_fx

    from tace.models.spherical.edge_inter import _RowwisePairProduct
    from tace.models.spherical.paths import generate_paths

    monkeypatch.setenv("TACE_USE_OEQ", "0")
    x_ir, y_ir = o3.Irreps("4x1o"), o3.Irreps("1x0e")
    paths, out = generate_paths(o3.Irreps("1o"), x_ir, y_ir)
    pair = _RowwisePairProduct(x_ir, y_ir, out, paths, shared_weights=False).cuda()
    names, buffers = zip(*pair.named_buffers(), strict=True)
    assert not any("_w3j_" in name for name in names)
    args = tuple(
        torch.randn(9, d, device="cuda", requires_grad=True)
        for d in (x_ir.dim, y_ir.dim, pair.weight_numel)
    )

    def joint(*values):
        result = torch.func.functional_call(
            pair, dict(zip(names, values[3:], strict=True)), values[:3]
        )
        return (result, *torch.autograd.grad(result.square().sum(), values[:3]))

    fx = make_fx(joint, tracing_mode="fake", _allow_non_fake_inputs=True)(
        *args, *buffers
    )
    program = torch.export.export(fx, (*args, *buffers), strict=True)
    assert _rowwise_o3_calls(program) > 0
    ref = pair.tp(*args)
    torch.testing.assert_close(
        program.module()(*args, *buffers),
        (ref, *torch.autograd.grad(ref.square().sum(), args)),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="HIP pair CUDA operator")
@pytest.mark.parametrize("package", [False, True], ids=["eager", "aoti"])
def test_hip_rowwise_two_adamw_steps(tmp_path, monkeypatch, package):
    """Same signed block labels and two live AdamW updates; the AOTI graph owns no optimizer."""
    from tace.utils.loss.normal import NormalLoss

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    torch.manual_seed(103)
    torch.set_default_dtype(torch.float32)
    native = _model("hip").cuda().train()
    # Isolate the pair change: activate the production option at construction
    # of the HIP pair module, retaining exactly the reference trunk and readouts.
    host = _model("hip").cuda().train()
    _install_hip_rowwise(host, monkeypatch)
    for model in (native, host):
        model.readout_fn.register_parameter(
            "unused", torch.nn.Parameter(torch.ones(1, device="cuda"))
        )
    host.load_state_dict(native.state_dict(), strict=True)
    maps = [dict(m.named_parameters()) for m in (native, host)]
    assert maps[0].keys() == maps[1].keys()
    opts = [
        torch.optim.AdamW(d.values(), lr=3e-4, eps=1e-4, weight_decay=0.02)
        for d in maps
    ]
    keys = ("energy", "forces", "stress", "direct_hessian")
    obj = (
        NormalLoss(
            list(keys),
            ["mse_energy_per_atom", "mse_forces", "mse_stress", "mse_direct_hessian"],
            [1.0, 2.0, 15.0, 3.0],
        )
        .cuda()
        .train()
    )
    raw = graph("cuda")
    pos = raw["positions"]
    ones = pos.new_ones(2)
    u = torch.sin(2.1 * pos - 1.5).reshape(2, 3, 3)
    label = torch.einsum("gia,gjb->gijab", u, u).reshape(-1, 3, 3)
    assert label.min() < 0 < label.max()
    raw.update(
        energy=pos.new_tensor([0.7, -0.6]),
        forces=0.3 * torch.sin(pos),
        stress=0.07 * raw["lattice"],
        direct_hessian=label,
        entropy=ones,
        **{k + "_weight": ones for k in keys},
    )
    counts = {}
    if package:
        # These are external state, not frozen constants in the training ABI.
        pair_cgs = {
            "model." + name: value
            for name, value in native.named_buffers()
            if "pair_representation." in name
            and "rejector.tp." in name
            and "_w3j_" in name
        }
        assert len(pair_cgs) == 8
        export = aoti.export

        def audited(program, output, **kwargs):
            counts["rowwise"] = _rowwise_o3_calls(program)
            assert counts["rowwise"] == 0
            schema = json.loads(
                kwargs["inductor_configs"]["aot_inductor.metadata"][
                    aoti.TRAINING_SCHEMA
                ]
            )
            live = [
                schema["buffers"][i]
                for i in schema.get("active_buffers", range(len(schema["buffers"])))
            ]
            assert pair_cgs.keys() <= set(live)
            return export(program, output, **kwargs)

        monkeypatch.setattr(aoti, "export", audited)
        runtime = aoti.load_training(
            aoti.export_training(host, obj, raw, tmp_path / "hip-rowwise.pt2")
        )
    else:
        hits = []
        hook = host.readout_fn.pair_representation.interactions[
            0
        ].rejector.contraction.register_forward_hook(lambda *_: hits.append(1))
    initial = {n: p.detach().clone() for n, p in maps[0].items()}
    errors = []
    for step in (1, 2):
        data = raw if step == 1 else _hip_reordered(raw)
        for opt in opts:
            opt.zero_grad(set_to_none=True)
        # Both force calculations may enable coordinate gradients; do not share
        # a differentiable gather from the second step's permutation.
        data = {k: v.detach() for k, v in data.items()}
        if package and step == 2:
            original_h = native(dict(data))["direct_hessian"].detach()
            for value in pair_cgs.values():
                value.mul_(-2).add_(0.25)
        ref = native({k: v.detach() for k, v in data.items()})
        if package and step == 2:
            assert (ref["direct_hessian"] - original_h).abs().max() > 1e-5
        loss = obj(ref, data)
        loss.backward()
        if package:
            inputs, params, buffers = aoti.training_inputs(host, obj, data)
            if step == 2:
                buffers = {**buffers, **pair_cgs}
            pred, value, grad = runtime(inputs, params, buffers)
            assert grad.keys() == params.keys()
            for name, p in maps[1].items():
                assert params["model." + name] is p
                p.grad = grad["model." + name]
        else:
            pred = host({k: v.detach() for k, v in data.items()})
            value = obj(pred, data)
            value.backward()
            assert len(hits) == step
        for k in keys:
            torch.testing.assert_close(pred[k], ref[k], atol=2e-5, rtol=3e-4)
        torch.testing.assert_close(value, loss, atol=2e-6, rtol=2e-5)
        active = []
        max_grad = max_param = max_slot = 0.0
        for n, p in maps[0].items():
            q = maps[1][n]
            assert (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                active.append(n)
                torch.testing.assert_close(q.grad, p.grad, atol=2e-5, rtol=3e-4, msg=n)
                if p.numel():
                    max_grad = max(max_grad, (q.grad - p.grad).abs().max().item())
        assert "readout_fn.unused" in maps[0] and "readout_fn.unused" not in active
        assert any(
            "pair_representation." in n and maps[0][n].grad.abs().max() > 1e-7
            for n in active
            if maps[0][n].numel()
        )
        for opt in opts:
            opt.step()
        for n, p in maps[0].items():
            q = maps[1][n]
            torch.testing.assert_close(q, p, atol=2e-6, rtol=3e-5, msg=n)
            if p.numel():
                max_param = max(max_param, (q - p).abs().max().item())
            assert (p in opts[0].state) == (q in opts[1].state) == (n in active)
            if n not in active:
                torch.testing.assert_close(q, initial[n], atol=0, rtol=0)
                continue
            a, b = opts[0].state[p], opts[1].state[q]
            assert a.keys() == b.keys() == {"step", "exp_avg", "exp_avg_sq"}
            assert a["step"].item() == b["step"].item() == step
            for slot in ("exp_avg", "exp_avg_sq"):
                torch.testing.assert_close(
                    b[slot],
                    a[slot],
                    atol=2e-6 if slot == "exp_avg" else 2e-7,
                    rtol=5e-4,
                    msg=n,
                )
                if p.numel():
                    max_slot = max(max_slot, (b[slot] - a[slot]).abs().max().item())
        assert any(
            not torch.equal(initial[n], maps[0][n])
            for n in active
            if "pair_representation" in n
        )
        errors.append(
            dict(
                step=step,
                loss=loss.item(),
                grad=max_grad,
                parameter=max_param,
                moment=max_slot,
                parameters=len(maps[0]),
                active=len(active),
            )
        )
    if not package:
        hook.remove()
    (tmp_path / "hip-rowwise-report.json").write_text(
        json.dumps(dict(operators=counts, steps=errors), indent=2)
    )

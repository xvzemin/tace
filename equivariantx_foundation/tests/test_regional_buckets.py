"""CUDA Graph bucket replay on changing real graphs/labels and live parameters."""

import os
from contextlib import nullcontext
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from equivariantx_foundation import optimize
from equivariantx_foundation.models.tace.training import (
    BucketedEFSTrainingStep,
    PaddedEFSLoss,
    _efs_loss,
    _pack,
    pack_efs_training_batch,
    supports_training_buckets,
)
from equivariantx_foundation.tests.test_regional_aoti import _cases, _model
from equivariantx_foundation.tests.test_tace_plan import graph
from tace.models.adapter import TensorModel
from tace.models.spherical.tace import SphericalTACE
from tace.utils.env import ACCELERATION_ENV


@pytest.mark.parametrize("watch", [False, True])
def test_replay_registration_updates(monkeypatch, watch):
    from equivariantx_foundation.models.tace import replay_loss

    if not watch:
        monkeypatch.setattr(replay_loss, "_dict_version", None)

    class Option:
        pass

    class Linear(torch.nn.Linear):
        pass

    class Model(torch.nn.Module):
        def parameters(self):
            return iter(())

        def buffers(self):
            return iter(())

        def get_target_property(self):
            return ["energy"]

        def get_embedding_property(self):
            return []

    model = Model()
    model.flags, model.option = SimpleNamespace(forces=False), Option()
    model.linear = Linear(4, 2)
    model.register_buffer("table", torch.ones(4))
    state = replay_loss._Bindings(model)

    model.__dict__ = vars(model).copy()
    model.linear._parameters = dict(model.linear._parameters)
    assert state.matches(model)
    with monkeypatch.context() as patch:
        patch.setitem(model.linear._parameters, "weight", model.linear.bias)
        assert not state.matches(model)
    with monkeypatch.context() as patch:
        replaced = {**model.linear._parameters, "weight": model.linear.bias}
        patch.setattr(
            Linear, "_parameters", property(lambda _: replaced), raising=False
        )
        assert not state.matches(model)
    with monkeypatch.context() as patch:
        patch.setattr(Option, "__call__", lambda _: None, raising=False)
        assert not state.matches(model)
    with torch.nn.modules.module.register_module_forward_hook(lambda *_: None):
        assert not state.matches(model)
    with torch.no_grad():
        model.linear.weight.add_(1)
    assert state.matches(model)
    with model.linear.weight.register_hook(lambda g: g):
        assert not state.matches(model)
    prior = model.table.detach()
    model.table.data = prior.clone()
    assert not state.matches(model)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA magnetic EFSM replay")
@pytest.mark.parametrize("microbatches", [1, 2])
def test_magnetic_bucket_replays_spin_labels_and_parameter_adjoints(
    monkeypatch, microbatches
):
    from equivariantx_foundation.models.tace import training
    from tace.models.time_reversal import supports_time_reversal

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(112)
    spin, mag = "initial_noncollinear_magmoms", "noncollinear_magnetic_forces"
    native, source = (_model("uv_o2_mag").float().train().cuda() for _ in range(2))
    source.load_state_dict(native.state_dict())
    model = optimize(
        source, family="tace", task="training", backend="cuda", regional_only=True
    )
    assert not supports_training_buckets(source)
    if supports_time_reversal():
        assert not supports_training_buckets(model)  # distinct T-e3nn node products
        return
    assert supports_training_buckets(model)
    rep = model.readout_fn.representation
    for owner, field, bad in (
        (rep.magnetic_basis.angular_basis, "normalize", True),
        (rep.magnetic_basis, "use_spin_orbit_coupling", False),
        (rep.interactions[0], "_magnetic_weights", lambda *_: None),
        (rep, "use_time_reversal", True),
        (rep.interactions[-1].rejector.eqx_tp, "backend", "torch"),
    ):
        with monkeypatch.context() as patch:
            patch.setattr(owner, field, bad)
            assert not supports_training_buckets(model)
    batches = _cases()
    for i, b in enumerate(batches):
        b["node_type"].zero_()
        x = b["positions"]
        b[spin] = (x.cos() * (0.4 + i) + x.roll(1, 1).sin()).detach()
        b[spin][0].zero_()  # physical zero spin with nonzero real neighbors
        b[mag] = (i - 0.3) * x.sin() + 0.1
        b.update(
            energy=b["ptr"].diff().double() * (-0.3 + i),
            forces=0.1 * x.cos(),
            stress=b["lattice"] * 0.02,
        )
        b[f"{mag}_weight"] = x.new_ones(2)

    # Make the disconnected padding actively contribute nonzero messages and
    # deliberately large magnetic labels. Neither may enter magnetic gradients.
    def active_pack(data, *args, **kwargs):
        p = _pack(data, *args, **kwargs)
        n = len(data["positions"])
        assert p[spin][n:].shape[1] == 3
        p[mag][n:].fill_(12.3)
        p["lattice"][-1] = torch.diag(p["lattice"].new_tensor([0.8, 1.1, 1.3]))
        if data is not batches[0]:
            p[spin][n:].fill_(0.7)
        return p

    monkeypatch.setattr(training, "_pack", active_pack)
    opt, ropt = (torch.optim.AdamW(m.parameters(), lr=2e-4) for m in (model, native))
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as profile:
        step = BucketedEFSTrainingStep(
            model, opt, batches[0], max_atoms=7, max_edges=14, microbatches=microbatches
        )
    assert any(
        e.key == "eqx::edge_program" and e.count >= 6 for e in profile.key_averages()
    )
    assert not opt.state
    graph_id = step.step.graph
    for i, group in enumerate(((batches[0], batches[1]), (batches[1], batches[2]))):
        group = group[:microbatches]
        ropt.zero_grad(set_to_none=True)
        ref = [native(dict(b)) for b in group]
        loss = (
            torch.cat(
                [
                    (r["energy"] - b["energy"]) / b["ptr"].diff()
                    for r, b in zip(ref, group, strict=True)
                ]
            )
            .square()
            .mean()
        )
        for k in ("forces", "stress", mag):
            residual = torch.cat([r[k] - b[k] for r, b in zip(ref, group, strict=True)])
            term = residual.square().mean()
            if k == mag and i == 0:
                magnetic_grads = torch.autograd.grad(
                    term,
                    tuple(native.parameters()),
                    allow_unused=True,
                    retain_graph=True,
                )
                assert any(
                    g is not None and g.numel() and g.abs().max() > 1e-7
                    for g in magnetic_grads
                )
            loss = loss + term
        loss.backward()
        ropt.step()
        actual = step(group if microbatches > 1 else group[0])
        assert step.step.graph is graph_id
        torch.testing.assert_close(actual["loss"], loss, rtol=4e-4, atol=3e-6)
        assert actual[mag].abs().max() > 1e-9
        for k in ("energy", "forces", "stress", mag):
            torch.testing.assert_close(
                actual[k], torch.cat([r[k] for r in ref]), rtol=4e-4, atol=3e-5
            )
        for (name, p), (other, q) in zip(
            model.named_parameters(), native.named_parameters(), strict=True
        ):
            assert name == other and (p.grad is None) == (q.grad is None)
            if q.grad is not None:
                torch.testing.assert_close(
                    p.grad, q.grad, rtol=4e-4, atol=3e-6, msg=name
                )
                torch.testing.assert_close(p, q, rtol=3e-4, atol=3e-6, msg=name)
                torch.testing.assert_close(
                    opt.state[p]["exp_avg"],
                    ropt.state[q]["exp_avg"],
                    rtol=4e-4,
                    atol=1e-6,
                    msg=name,
                )
                assert opt.state[p]["step"].item() == i + 1
    for key in (spin, mag):
        with pytest.raises(ValueError, match="spin and a labelled magnetic force"):
            step._pack_batch({**batches[0], key: batches[0][key][:, 0]})


def test_magnetic_bucket_rejects_nonunit_or_missing_labels_on_cpu():
    b = graph("cpu")
    x = b["positions"]
    spin, mag = "initial_noncollinear_magmoms", "noncollinear_magnetic_forces"
    b.update(
        energy=x.new_zeros(2),
        stress=torch.zeros_like(b["lattice"]),
        forces=torch.zeros_like(x),
    )
    b[spin] = b[mag] = torch.zeros_like(x)

    def pack(data):
        return _pack(data, 7, 14, 2, True, magnetic=True, edge_layout="directed")

    for key in (spin, mag):
        with pytest.raises(ValueError, match="spin and a labelled magnetic force"):
            pack({k: v for k, v in b.items() if k != key})
        with pytest.raises(RuntimeError, match=f"unit {key}_weight"):
            pack({**b, f"{key}_weight": x.new_tensor([0, 1.0])})
        with pytest.raises(ValueError, match=f"one {key}_weight per structure"):
            pack({**b, f"{key}_weight": x.new_empty(0)})


def _weighted_sample():
    from torch_geometric.data import Data

    b = graph("cpu")
    x = b["positions"]
    # Move one atom to the second structure; discard inter-structure edges.
    b["ptr"][1] = 2
    b["batch"][2] = 1
    i, j = b["edge_index"]
    keep = b["batch"][i] == b["batch"][j]
    b["edge_index"] = b["edge_index"][:, keep]
    b["edge_shifts"] = b["edge_shifts"][keep]
    b.update(
        energy=x.new_tensor([4.1, -2.5]),
        forces=x.cos(),
        stress=b["lattice"] * 0.1,
        entropy=x.new_tensor([0.55, 1.7]),
        energy_weight=x.new_tensor([1.5, 0.0]),
        forces_weight=x.new_tensor([0.8, 0.4]),
        stress_weight=x.new_tensor([0.0, 1.1]),
    )
    b["stress"][0].fill_(float("nan"))  # the native stress term masks operands
    model = SimpleNamespace(readout_fn=SimpleNamespace(cutoff=x.new_tensor(7.0)))
    return model, Data.from_dict({**b, "_num_nodes": [2, 4], "num_nodes": 6})


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay")
@pytest.mark.parametrize(
    "kind,update,stress",
    [
        ("cgtp", "identity", True),
        ("uu_o2", "identity", True),
        ("uv_o2", "element2", True),
        ("cgtp", "identity", False),
        ("cartesian", "identity", False),
        ("nonlinear_ictd", "identity", True),
        ("cartesian_uu", "identity", True),
        ("cartesian_uv", "element2", True),
    ],
)
def test_regional_weighted_training_loss(monkeypatch, kind, update, stress):
    from weakref import ref

    from equivariantx_foundation.models.tace.replay_loss import BucketedTrainingLoss
    from tace.utils.loss.normal import NormalLoss

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    planar = kind in ("cartesian_uu", "cartesian_uv")
    host_template = _model("cartesian" if planar else kind).readout_fn
    cfg = deepcopy(host_template.model_config)
    if planar:
        cfg["atomic_basis"]["type"] = "uu_o2" if kind == "cartesian_uu" else "uv_o2"
        cfg["num_channel"] = 64
        cfg["radial_basis"]["apply_cutoff"] = False
    host_type = type(host_template)
    cfg["edge_update"]["type"] = update
    cfg["readout_emlp"]["use_alllayer"] = True
    targets = ["energy", "forces"] + (["stress"] if stress else [])
    cfg["target_property"] = targets
    native, host = (
        TensorModel(host_type(**deepcopy(cfg))).train().cuda() for _ in range(2)
    )
    host.load_state_dict(native.state_dict())
    model = optimize(host, family="tace", task="training", backend="cuda")
    assert model.readout_fn.representation._foundation_regional
    assert model.readout_fn.representation.reciprocal_radial == (update == "identity")
    objective = NormalLoss(
        targets,
        ["mse_energy_per_atom", "huber_forces", "mse_voigt_stress"][: len(targets)],
        [0.6, 0.4, 0.2][: len(targets)],
        [{}, {"element_weights": [0.4, 1.4]}, {}][: len(targets)],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-4)
    capacity = (7, 8192 if kind == "cartesian_uv" else 15, 3)
    with (
        model.readout_fn.energy_readouts[0].register_forward_hook(lambda *_: None),
        pytest.raises(ValueError),
    ):
        BucketedTrainingLoss(model, optimizer, objective, capacity)
    if kind == "cartesian":
        from e3nn import o3

        with monkeypatch.context() as patch:
            layer = model.readout_fn.energy_readouts[-1].linear2[0]
            patch.setitem(layer._modules, "linear", o3.Linear("1x0e", "1x0e"))
            with pytest.raises(ValueError):
                BucketedTrainingLoss(model, optimizer, objective, capacity)
        with monkeypatch.context() as patch:
            patch.setitem(
                model.readout_fn.representation._modules,
                "o3_angular_basis",
                torch.nn.Identity(),
            )
            with pytest.raises(ValueError):
                BucketedTrainingLoss(model, optimizer, objective, capacity)
    if planar:
        conv = model.readout_fn.representation.interactions[1].rejector
        for frame in (conv.local_frame_in, conv.local_frame_out):
            with monkeypatch.context() as patch:
                patch.setattr(frame, "basis_change", not frame.basis_change)
                with pytest.raises(ValueError):
                    BucketedTrainingLoss(model, optimizer, objective, capacity)
    if kind == "cartesian_uv":
        first, second = (
            m.rejector for m in model.readout_fn.representation.interactions
        )
        assert first.local_frame_in.lmax == 0 and first.eqx_tp is not None
        assert second.local_frame_in.lmax == 2 and second.eqx_tp is not None
        with monkeypatch.context() as patch:
            patch.setitem(
                second.nonlinearity.act_scalars.acts[0]._modules, "f", torch.nn.Tanh()
            )
            with pytest.raises(ValueError):
                BucketedTrainingLoss(model, optimizer, objective, capacity)
        with monkeypatch.context() as patch:
            paths = second.nonlinearity._paths
            assert len(paths) > 1
            patch.setattr(second.nonlinearity, "_paths", tuple(reversed(paths)))
            with pytest.raises(ValueError):
                BucketedTrainingLoss(model, optimizer, objective, capacity)
        with monkeypatch.context() as patch:
            patch.setattr(first.local_frame_in, "to_local", lambda *args: None)
            with pytest.raises(ValueError):
                BucketedTrainingLoss(model, optimizer, objective, capacity)
        captured_edges = {first.eqx_tp: [], second.eqx_tp: []}
        forward = type(second.eqx_tp).forward

        def witness(module, x, w, idx, *args, **kwargs):
            if module in captured_edges and torch.cuda.is_current_stream_capturing():
                captured_edges[module].append(idx.shape[1])
            return forward(module, x, w, idx, *args, **kwargs)

        monkeypatch.setattr(type(second.eqx_tp), "forward", witness)
    model.reset_target_property(["energy", "forces"] + ([] if stress else ["stress"]))
    with pytest.raises(ValueError):
        BucketedTrainingLoss(model, optimizer, objective, capacity)
    model.reset_target_property(targets)
    multi = not stress
    small = (3, 5, 1)
    replay = BucketedTrainingLoss(
        model, optimizer, objective, [capacity, small] if multi else capacity
    )
    tables = []
    padded_forward = PaddedEFSLoss.forward

    def track_table(self, pred, data):
        tables.append(ref(self.element_weights_1))
        return padded_forward(self, pred, data)

    monkeypatch.setattr(PaddedEFSLoss, "forward", track_table)

    _, first = _weighted_sample()
    first = {k: v.cuda() for k, v in first.items() if torch.is_tensor(v)}
    packed = pack_efs_training_batch(model, first, 8, 18, stress=stress)
    for key in ("edge_pair_rows", "edge_pair_inverse"):
        assert (key in packed) == model.uses_reciprocal_weights()
    del packed
    # The first two atoms form the structure with an unlabelled stress tensor.
    edge = first["edge_index"][0] < 2
    second = {k: v[:1] for k, v in first.items()}
    second.update(
        positions=0.98 * first["positions"][:2],
        forces=first["forces"][:2],
        node_type=first["node_type"][:2],
        batch=first["batch"][:2],
        ptr=first["ptr"][:2],
        edge_index=first["edge_index"][:, edge].contiguous(),
        edge_shifts=first["edge_shifts"][edge],
    )
    if not stress:
        # A cached periodic batch and a fresh molecule share the E/F capture.
        del second["stress"], second["stress_weight"]
        for name in ("pbc", "lattice", "edge_shifts"):
            second[name] = torch.zeros_like(second[name])
        monkeypatch.setattr(
            "tace.models.adapter.compute_symmetric_displacement",
            lambda *_: pytest.fail("E/F replay must not differentiate strain"),
        )
    if multi:
        oversized = {
            **first,
            "edge_index": first["edge_index"].repeat(1, 3),
            "edge_shifts": first["edge_shifts"].repeat(3, 1),
        }
        with monkeypatch.context() as patch:
            patch.setattr(
                "equivariantx_foundation.models.tace.replay_loss.pack_efs_training_batch",
                lambda *_args, **_kw: pytest.fail("oversized batch was padded"),
            )
            with pytest.raises(ValueError, match="(?i)bucket|capacity"):
                replay(oversized, objective)
        assert replay.step is None and replay.captures == replay.replays == 0
        # Keep this loss live through the first-ever capture of the other shape.
        pred, a = replay(first, objective)
        saved = {k: v.clone() for k, v in pred.items()}
    steps = {}
    batches = (first, second, first, second) if multi else (first, second)
    for batch in batches:
        model.zero_grad(set_to_none=True)
        native.zero_grad(set_to_none=True)
        actual, loss = replay(batch, objective)
        if multi:
            shape = capacity if batch is first else small
            assert len(replay.step.static["positions"]) == shape[0] + 1
            assert steps.setdefault(shape, ref(replay.step))() is replay.step
        assert set(actual) == set(targets)
        if not stress:
            assert {"stress", "stress_weight"}.isdisjoint(replay.step.static)
        # CUDA replay reads this constant without executing Python again.
        assert tables[-1]() is not None
        torch.testing.assert_close(
            loss, objective(actual, batch), rtol=1e-11, atol=1e-12
        )
        predicted = native(batch)
        expected = objective(predicted, batch)
        for key in targets:
            torch.testing.assert_close(
                actual[key], predicted[key], rtol=4e-4, atol=3e-5
            )
        torch.testing.assert_close(loss, expected, rtol=2e-5, atol=2e-6)
        loss.backward()
        expected.backward()
        for (name, p), (ref_name, q) in zip(
            model.named_parameters(), native.named_parameters(), strict=True
        ):
            assert name == ref_name and (p.grad is None) == (q.grad is None)
            if q.grad is not None:
                torch.testing.assert_close(
                    p.grad, q.grad, rtol=4e-4, atol=3e-6, msg=name
                )
        del actual, loss, predicted, expected
    assert replay.captures == 1 + multi and replay.replays == len(batches) + multi
    if kind == "cartesian_uv":
        assert list(captured_edges.values()) == [[capacity[1] + capacity[0]]] * 2
    if multi:
        model.zero_grad(set_to_none=True)
        native.zero_grad(set_to_none=True)
        _, b = replay(second, objective)
        changed = {**first, "positions": first["positions"] * 0.96}
        replay(changed, objective)
        torch.testing.assert_close(pred, saved, atol=0, rtol=0)
        (a + b).backward()
        (objective(native(first), first) + objective(native(second), second)).backward()
        for p, q in zip(model.parameters(), native.parameters(), strict=True):
            if q.grad is not None:
                torch.testing.assert_close(p.grad, q.grad, rtol=4e-4, atol=4e-6)
        del a, b
        objective.loss_property_weights[0] *= 1.5
        for batch in (second, first):
            _, loss = replay(batch, objective)
            torch.testing.assert_close(
                loss, objective(native(batch), batch), rtol=2e-5, atol=2e-6
            )
            del loss
        assert replay.captures == 4 and all(r() is None for r in steps.values())
        replay.clear()
        assert replay.step is None and all(r() is None for r in tables)


@pytest.mark.parametrize(
    "family,stress",
    [("mse", True), ("mae", True), ("huber", True), ("l2mae", True), ("huber", False)],
)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_weighted_efs_preserves_native_terms_and_gradients_on_cpu(
    family, dtype, stress
):
    from tace.utils.loss.normal import NormalLoss

    model, b = _weighted_sample()
    for key, value in b.items():
        if torch.is_tensor(value) and value.is_floating_point():
            b[key] = value.to(dtype)
    names = ["energy", "energy_per_atom", "forces"]
    if stress:
        names.extend(["stress", "voigt_stress"])
    objective = NormalLoss(
        ["energy", "forces"] + (["stress"] if stress else []),
        [f"{family}_{p}" for p in names] + ["huber_forces"],
        [0.3, 0.2, 1.5, 0.6, 0.4][: len(names)] + [0.8],
        [
            {},
            {},
            {
                "element_weights": [0.3, 2.1],
                **({"padded": True} if family == "l2mae" else {}),
            },
            {},
            {},
        ][: len(names)]
        + [{"huber_delta": 1.1}],
    )
    padded_loss = PaddedEFSLoss(objective)
    assert objective.loss_function_kwargs[2]["element_weights"] == [0.3, 2.1]
    assert padded_loss.element_weights_2.dtype == torch.float64
    assert not padded_loss.element_weights_2.requires_grad
    params = torch.tensor([0.45, 1.23], dtype=dtype, requires_grad=True)

    def predict(data):
        x = data["positions"]
        y = (x + params[0]).square() * params[1]
        energy = x.new_zeros(len(data["ptr"]) - 1).index_add(0, data["batch"], y.sum(1))
        out = {"energy": energy, "forces": y}
        if stress:
            out["stress"] = (data["lattice"] - params[1]) * params[0]
        return out

    for stress_weight in (b["stress_weight"], b["stress_weight"].bool() & False):
        b["stress_weight"] = stress_weight
        native = objective(predict(b), b)
        for n, max_graphs in ((6, None), (6, 4), (10, 2), (10, 4)):
            padded = pack_efs_training_batch(
                model, b, n, 18, max_graphs=max_graphs, stress=stress
            )
            assert ("stress" in padded) is stress
            actual = padded_loss(predict(padded), padded)
            # Registered native terms evaluate FP32 operands in data precision.
            assert actual.dtype == native.dtype == torch.float64
            torch.testing.assert_close(actual, native)
            torch.testing.assert_close(
                torch.autograd.grad(actual, params),
                torch.autograd.grad(native, params, retain_graph=True),
            )
    b["energy"][1] = float("nan")  # zero energy weight does not sanitize NaNs
    p = pack_efs_training_batch(model, b, 8, 18, max_graphs=4, stress=stress)
    assert torch.isnan(objective(predict(b), b)) and torch.isnan(
        padded_loss(predict(p), p)
    )


def test_graph_capacity_preserves_real_components_and_replay_shapes():
    model, sample = _weighted_sample()
    two = sample.to_dict()
    one = {
        **two,
        "batch": torch.zeros_like(two["batch"]),
        "ptr": two["ptr"][[0, -1]],
    }
    for key in (
        "pbc",
        "lattice",
        "fidelity_idx",
        "energy",
        "stress",
        "entropy",
        "energy_weight",
        "forces_weight",
        "stress_weight",
    ):
        one[key] = two[key][:1]
    specs = []
    for g, raw in ((2, two), (1, one)):
        packed = pack_efs_training_batch(model, raw, 6, 18, max_graphs=2)
        specs.append({key: (v.shape, v.dtype) for key, v in packed.items()})
        assert len(packed["positions"]) == 7 and packed["edge_index"].shape == (2, 24)
        counts = packed["ptr"].diff()
        assert counts.tolist() == [*raw["ptr"].diff().tolist(), 1, *([0] * (2 - g))]
        assert torch.equal(torch.bincount(packed["batch"], minlength=3), counts)
        assert torch.equal(packed["batch"][:6], raw["batch"])
        e = raw["edge_index"].size(1)
        assert torch.equal(packed["edge_index"][:, :e], raw["edge_index"])
        assert (packed["edge_index"][:, e:] == 6).all()
        assert torch.equal(packed["edge_shifts"][:e], raw["edge_shifts"])
        assert (packed["edge_shifts"][e:] == 1).all()
        assert (torch.linalg.det(packed["lattice"][g:]) > 0).all()
        for key in ("entropy", "energy_weight", "forces_weight", "stress_weight"):
            assert torch.equal(packed[key][:g], raw[key])
            assert torch.count_nonzero(packed[key][g:]) == 0
    assert specs[0] == specs[1]


def test_weighted_efs_rejects_custom_inputs_and_objectives_on_cpu(monkeypatch):
    from tace.utils.loss.normal import NormalLoss
    from tace.utils.loss.registry import LOSS_FN

    model, data = _weighted_sample()
    b = data.to_dict()
    with pytest.raises(ValueError, match="graph|structure"):
        pack_efs_training_batch(model, b, 8, 18, max_graphs=1)
    for bad in (
        "edge_pair_rows",
        "lmp_data",
        "initial_noncollinear_magmoms",
        "user_field",
    ):
        with pytest.raises(ValueError, match=bad):
            pack_efs_training_batch(model, {**b, bad: b["positions"]}, 8, 18)
    with pytest.raises(ValueError, match="per structure"):
        pack_efs_training_batch(model, {**b, "entropy": torch.zeros(2, 1)}, 8, 18)
    with pytest.raises(ValueError, match="geometry or label shape"):
        pack_efs_training_batch(model, {**b, "energy": b["energy"][:, None]}, 8, 18)
    names = ["mse_energy", "huber_forces", "mae_stress"]
    obj = NormalLoss(["energy", "forces", "stress"], names, [1.0, 1.0, 1.0])
    with (
        obj.register_forward_hook(lambda *_: None),
        pytest.raises(ValueError, match="unmodified"),
    ):
        PaddedEFSLoss(obj)
    with monkeypatch.context() as m:
        m.setitem(LOSS_FN, names[0], lambda *_: None)
        with pytest.raises(ValueError, match="Replaced"):
            PaddedEFSLoss(obj)
    with pytest.raises(ValueError, match="constant numeric"):
        PaddedEFSLoss(
            NormalLoss(
                ["energy", "forces", "stress"],
                names,
                [1, 1, 1],
                [{}, {"element_weights": [torch.tensor(0.7, requires_grad=True)]}, {}],
            )
        )
    with pytest.raises(ValueError, match="Unsupported"):
        PaddedEFSLoss(
            NormalLoss(
                ["energy", "forces", "stress"], names[:2] + ["mse_virials"], [1] * 3
            )
        )
    with pytest.raises(ValueError, match="Unsupported"):
        PaddedEFSLoss(
            NormalLoss(["energy", "forces"], ["mse_energy", "mse_stress"], [1, 1])
        )
    with pytest.raises(ValueError, match="cover"):
        PaddedEFSLoss(
            NormalLoss(["energy", "forces"], ["mse_energy", "mae_energy"], [1, 1])
        )


@pytest.mark.parametrize(
    "fallback,chunked", [(False, False), (False, True), (True, True)]
)
def test_replay_loss_owns_mixed_dtype_gradients(monkeypatch, fallback, chunked):
    from equivariantx_foundation.models.tace import replay_loss

    if fallback:
        monkeypatch.delattr(torch, "_foreach_clone", raising=False)
    if chunked:
        monkeypatch.setattr(replay_loss, "_VJP_CHUNK_BYTES", 24)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
    tensors = [x, x.double() + 7, None, x]
    params = [
        torch.nn.Parameter(torch.zeros_like(x if t is None else t)) for t in tensors
    ]
    attach = replay_loss._bind_parameter_loss(params, tensors)
    saved = []

    def pack(tensor):
        saved.append(tensor.detach())
        return saved[-1]

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        first = attach(x.new_zeros(()))
    assert len(saved) == 3 and len({t.data_ptr() for t in saved}) == 3
    assert [t.stride() for t in saved] == [t.stride() for t in tensors if t is not None]
    refs = [t.clone() for t in saved]
    x.add_(4)
    tensors[1].sub_(2)
    second = attach(x.new_ones(()))
    for loss in (first, second, first):
        actual = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=True)
        expected = [*refs[:2], None, refs[2]] if loss is first else tensors
        for g, ref in zip(actual, expected, strict=True):
            if ref is None:
                assert g is None
            else:
                torch.testing.assert_close(g, ref, atol=0, rtol=0)


def test_local_replay_vjp_shares_storage_preserves_ties_and_skips_live_accumulators():
    from equivariantx_foundation.models.tace.training import _LocalParameterVJP
    from tace.models.compile.joint import ParameterLoss

    class Tied(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.p = torch.nn.Parameter(torch.tensor(0.4, dtype=torch.float64))
            self.branch = torch.nn.Module()
            self.branch.same = self.p
            self.unused = torch.nn.Parameter(torch.zeros_like(self.p))

        def forward(self, x):
            assert self.p is self.branch.same
            return (self.p * x + self.branch.same * x.square()).sin()

    model = Tied()
    params = tuple(model.parameters())
    local = _LocalParameterVJP(model, params)
    assert all(
        a is not b and a.data_ptr() == b.data_ptr()
        for a, b in zip(params, local.leaves)
    )
    calls = []
    edge = torch.autograd.graph.get_gradient_edge(model.p).node
    handle = edge.register_hook(lambda *_: calls.append(1))
    previous = model.p.grad = model.p.new_tensor(0.3)
    x = model.p.new_tensor([0.5, 1.4])
    try:
        for change in (0.0, 0.7):
            with torch.no_grad():
                model.p.add_(change)
            out, loss, grads = local(x, lambda y, _: y.sum())
            assert model.p is model.branch.same is params[0]
            expected = ((x + x.square()) * model.p.detach()).cos() * (x + x.square())
            torch.testing.assert_close(out, model(x))
            torch.testing.assert_close(grads[0], expected.sum())
            assert grads[1] is None and model.p.grad is previous and calls == []
        ParameterLoss.apply(loss.detach(), (grads[0],), model.p).backward()
        assert model.unused.grad is None and calls == [1]
        torch.testing.assert_close(model.p.grad, expected.sum() + 0.3)
    finally:
        handle.remove()


@pytest.mark.parametrize("world_size", [1, 2])
def test_ddp_bucket_requires_nccl_and_global_counts(monkeypatch, world_size):
    from torch.utils._python_dispatch import TorchDispatchMode

    from equivariantx_foundation.models.tace.replay_loss import (
        BucketedTrainingLoss,
        _validate_model,
    )
    from tace.utils.loss.common import prepare_tensor_loss_counts
    from tace.utils.loss.normal import NormalLoss
    from tace.utils.loss.registry import LOSS_FN

    model, data = _weighted_sample()
    obj = NormalLoss(
        ["energy", "forces", "stress"],
        ["mse_energy", "huber_forces", "mse_forces", "mae_voigt_stress"],
        [1.0, 1.3, 0.732, 0.4],
    )
    with pytest.raises(ValueError, match="NCCL"):
        BucketedTrainingLoss(model, None, obj, (10, 12), ddp=True)
    with pytest.raises(ValueError, match="whole training plan"):
        _validate_model(model)
    dist = torch.distributed
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: world_size)
    monkeypatch.setattr(dist, "get_backend", lambda: "nccl")
    loss = PaddedEFSLoss(obj, _ddp=True)
    packed = pack_efs_training_batch(model, data, 8, 16, max_graphs=4)
    with pytest.raises(ValueError, match="outside CUDA capture"):
        loss({}, packed)
    with pytest.raises(ValueError, match="single-device"):
        PaddedEFSLoss(obj)
    calls = []

    def include_other_rank(counts, **kwargs):
        calls.append(counts.shape)
        if world_size == 2:
            counts.add_(3)  # the other rank has three positive stress weights

    monkeypatch.setattr(dist, "all_reduce", include_other_rank)
    packed = prepare_tensor_loss_counts(packed, ("stress",))
    assert calls == [(1,)] and packed["_loss_world_size"] == world_size

    def predict(d):
        return {
            "energy": d["ptr"].diff().double(),
            "forces": (d["positions"].cos() * 0.37).float().requires_grad_(),
            "stress": d["lattice"].sin(),
        }

    unused = torch.empty_like(packed["lattice"], dtype=torch.float32)

    class CheckConversions(TorchDispatchMode):
        def __torch_dispatch__(self, fn, types, args=(), kwargs=None):
            if fn == torch.ops.aten._to_copy.default and args[0] is unused:
                pytest.fail("The E/F/S objective converted an unused tensor")
            return fn(*args, **(kwargs or {}))

    pred = predict(packed)
    with CheckConversions():
        actual = loss(
            {**pred, "scalar_descriptor": unused}, {**packed, "lattice": unused}
        )
    expected = sum(
        LOSS_FN[name](pred, packed) * weight * packed["_efs_term_scales"][axis]
        for name, weight, axis in zip(
            obj.loss_function_name, obj.loss_property_weights, (0, 1, 1, 2), strict=True
        )
    )
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(
        torch.autograd.grad(actual, pred["forces"])[0],
        torch.autograd.grad(expected, pred["forces"])[0],
        rtol=0,
        atol=0,
    )
    assert calls == [(1,)]  # no collective from the actual local objective
    raw = data.to_dict()
    native = obj(predict(raw), prepare_tensor_loss_counts(raw, ("stress",)))
    torch.testing.assert_close(actual, native)
    calls.clear()
    ef = NormalLoss(["energy", "forces"], ["mse_energy", "huber_forces"], [1.0, 1.0])
    packed = pack_efs_training_batch(model, data, 8, 16, stress=False)
    torch.testing.assert_close(
        PaddedEFSLoss(ef, _ddp=True)(predict(packed), packed), ef(predict(raw), raw)
    )
    assert not calls


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.distributed.is_nccl_available(),
    reason="CUDA/NCCL replay",
)
def test_ddp_bucket_rejects_disconnected_parameters(monkeypatch, tmp_path):
    from equivariantx_foundation.models.tace.replay_loss import BucketedTrainingLoss
    from tace.utils.loss.normal import NormalLoss

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    model = optimize(
        _model("uv_o2").float().cuda().train(),
        family="tace",
        backend="cuda",
        task="training",
    )
    model.register_parameter(
        "unused", torch.nn.Parameter(torch.zeros(1, device="cuda"))
    )
    opt = torch.optim.SGD(model.parameters(), lr=1e-4)
    loss = NormalLoss(
        ["energy", "forces", "stress"],
        ["mse_energy", "mse_forces", "mse_stress"],
        [1.0] * 3,
    )
    _, sample = _weighted_sample()
    data = {k: v.cuda() for k, v in sample.items() if torch.is_tensor(v)}
    dist = torch.distributed
    dist.init_process_group(
        "nccl", init_method=(tmp_path / "store").as_uri(), rank=0, world_size=1
    )
    try:
        replay = BucketedTrainingLoss(model, opt, loss, (7, 15, 3), ddp=True)
        with pytest.raises(ValueError, match="disconnected parameters"):
            replay(data, loss)
        assert replay.step is None and replay.captures == 0
        assert all(p.grad is None for p in model.parameters())
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host-submission check")
def test_weighted_pack_does_not_wait_for_cuda_scalar_transfers():
    # Recording one fixed example catches an accidental pageable host->device
    # copy or host scalar read. Normal use packs varying real shapes outside capture.
    model, data = _weighted_sample()
    model.readout_fn.cutoff = model.readout_fn.cutoff.cuda()
    data = data.cuda()
    for cap in (len(data.positions), len(data.positions) + 5):
        expected = pack_efs_training_batch(model, data, cap, 18)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            packed = pack_efs_training_batch(model, data, cap, 18)
        graph.replay()
        for key in expected:
            torch.testing.assert_close(packed[key], expected[key], equal_nan=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay")
def test_pretrained_mixed_rotary_bucket(monkeypatch):
    """Optional full checkpoint check; never downloads weights."""
    from tace.lightning import load_tace

    path = os.environ.get("EQX_TECE_CHECKPOINT")
    if not path:
        pytest.skip("Set EQX_TECE_CHECKPOINT to test the complete released TECE")
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    native, host = (
        load_tace(
            path,
            device="cuda",
            dtype="float32",
            matmul="ieee",
            strict=True,
            target_property=["energy", "forces", "stress"],
        ).train()
        for _ in range(2)
    )
    model = optimize(
        host, family="tace", task="training", backend="cuda", regional_only=True
    )
    assert supports_training_buckets(model)
    batches = _cases()[:2]
    for b in batches:
        b["energy"] = b["positions"].new_tensor([-2.0, -1.0])
        b["forces"] = b["positions"].sin() * 0.2
        b["stress"] = b["lattice"] * 0.02
    # Small live SGD updates; native and regional use different FP32 reduction orders.
    opts = [torch.optim.SGD(m.parameters(), lr=1e-8) for m in (native, model)]
    capture = BucketedEFSTrainingStep(
        model, opts[1], batches[0], max_atoms=6, max_edges=12
    )
    assert capture.edge_layout == "directed"
    for b in batches:
        opts[0].zero_grad(set_to_none=True)
        out = native(b)
        loss = ((out["energy"] - b["energy"]) / b["ptr"].diff()).square().mean()
        for k in ("forces", "stress"):
            loss = loss + (out[k] - b[k]).square().mean()
        loss.backward()
        opts[0].step()
        actual = capture(b)
        for k in ("energy", "forces", "stress", "loss"):
            torch.testing.assert_close(
                actual[k], loss if k == "loss" else out[k], atol=5e-4, rtol=3e-4
            )
        total, error = 0.0, 0.0
        for (name, p), (key, ref) in zip(
            model.named_parameters(), native.named_parameters(), strict=True
        ):
            assert name == key and (p.grad is None) == (ref.grad is None)
            if p.grad is not None:
                delta = (p.grad - ref.grad).double()
                error += delta.square().sum()
                total += ref.grad.double().square().sum()
                torch.testing.assert_close(
                    p.grad, ref.grad, atol=2e-3, rtol=1e-3, msg=name
                )
        assert (error / total).sqrt() < 1e-4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay")
@pytest.mark.parametrize(
    "basis", ["cgtp", "uv_o2", "cartesian", "nonlinear_ictd", "rotary"]
)
def test_multihead_bucket_switches_fidelity_without_recapture(monkeypatch, basis):
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    torch.set_default_dtype(torch.float32)
    torch.manual_seed(102)
    original = _model(basis).readout_fn
    config = deepcopy(original.model_config)
    if basis == "rotary":
        monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
        config["atomic_basis"]["scatter_norm"] = "density"
        config["product_basis"].update(
            nonlinear="bilinear_gate",
            num_expert=2,
            num_channel_per_expert=2,
            use_shared_expert=True,
        )
    config["fidelity"] = [{"name": "a"}, {"name": "b"}]
    config["statistics"] = [
        {
            **config["statistics"][0],
            "atomic_energy": {1: -h, 8: -2 * h},
            "rms_forces": {1: 1.0 + h, 8: 1.5 + h},
            "mean_energy_per_atom": {1: 0.2 + h, 8: 0.5 + h},
        }
        for h in range(2)
    ]
    config["scale_shift"].update(
        enable=True,
        all_atoms=True,
        scale_type="rms_forces",
        shift_type="mean_energy_per_atom",
        scale_trainable=True,
        shift_trainable=True,
    )
    host, ref = (
        TensorModel(type(original)(**deepcopy(config))).train().cuda() for _ in range(2)
    )
    if basis == "rotary":
        # Density weights otherwise have zero derivative at beta=0.
        with torch.no_grad():
            host.readout_fn.representation.interactions[0].scatter_norm.beta.fill_(0.2)
    ref.load_state_dict(host.state_dict())
    model = optimize(host, family="tace", task="training", backend="cuda")
    assert supports_training_buckets(model)
    if basis == "nonlinear_ictd":
        product = model.readout_fn.representation.products[0]
        with monkeypatch.context() as patch:
            patch.setattr(product.embedding, "scale_grad_by_freq", True)
            assert not supports_training_buckets(model)
        with monkeypatch.context() as patch:
            patch.setattr(product.aces[0].tp, "backend", "torch")
            assert not supports_training_buckets(model)
    if basis == "rotary":
        from equivariantx_foundation.models.tace.interactions import (
            _HostFrameRadialInteraction,
        )
        from equivariantx_foundation.models.tace.products import (
            _ScheduledBilinearMoEACE,
        )
        from equivariantx_foundation.models.tace.tece_oam_rra.regional import (
            StreamedRotaryInteraction,
        )
        from tace.models.spherical.edge import Element2EdgeUpdate

        rep = model.readout_fn.representation
        assert [type(m) for m in rep.interactions] == [
            _HostFrameRadialInteraction,
            StreamedRotaryInteraction,
        ]
        assert all(type(p) is _ScheduledBilinearMoEACE for p in rep.products)
        assert all(type(u) is Element2EdgeUpdate for u in rep.edge_updates)
        for target, attr, value in (
            (rep.interactions[1].rejector, "_eqx_metadata", None),
            (rep.interactions[1].rejector, "attention_eps", 0.0),
            (rep.interactions[0].rejector.eqx_tp, "radial_mode", "last"),
            (rep.products[0].scheduled_contraction.shared, "backend", "torch"),
            (rep, "use_dens", True),
        ):
            with monkeypatch.context() as patch:
                patch.setattr(target, attr, value)
                assert not supports_training_buckets(model)
        with monkeypatch.context() as patch:
            patch.setenv("TACE_MEMORY_FIRST", "0")
            assert not supports_training_buckets(model)
    opt, ropt = (torch.optim.SGD(m.parameters(), lr=0.005) for m in (model, ref))
    samples = list(_cases())
    for sample in samples:
        sample["energy"] = sample["lattice"].new_tensor([0.4, -0.2])
        sample["forces"] = sample["positions"].sin() * 0.1
        sample["stress"] = sample["lattice"] * 0.01
        del sample["fidelity_idx"]
    profile = (
        torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU])
        if basis == "rotary"
        else nullcontext()
    )
    with profile:
        bucket = BucketedEFSTrainingStep(
            model, opt, samples[0], max_atoms=8, max_edges=16
        )
    if basis == "rotary":
        assert (
            bucket.edge_layout == "directed"
            and "edge_pair_rows" not in bucket.step.static
        )
        hits = {event.key: event.count for event in profile.key_averages()}
        # Two warm calls plus recording. Include the actual adjoints, not just
        # the registered model classes or their Python forwards.
        assert hits["eqx_foundation::tece_interaction"] == 3
        assert hits["eqx_foundation::tece_contraction"] >= 3
        assert hits["eqx::edge_program"] >= 3 and hits["eqx::bilinear_ace"] >= 6
    capture = bucket.step.graph
    for i, (default, ids) in enumerate(
        [(0, None), (0, [1, 0]), (1, None), (1, [0, 1])]
    ):
        data = dict(samples[i % len(samples)])
        if basis == "rotary" and i == 1:
            # Active self-images on a disconnected dummy: neither its attention
            # denominator nor trainable density may affect real outputs/grads.
            bucket.dummy_cell = torch.diag(data["lattice"].new_tensor([0.8, 1.1, 1.3]))[
                None
            ]
        for m in (model, ref):
            m.reset_fidelity_idx(default)
        if ids is not None:
            data["fidelity_idx"] = torch.tensor(ids, device="cuda", dtype=torch.int32)
        ropt.zero_grad(set_to_none=True)
        expected = ref(data)
        loss = (
            ((expected["energy"] - data["energy"]) / data["ptr"].diff()).square().mean()
        )
        for name in ("forces", "stress"):
            loss = loss + (expected[name] - data[name]).square().mean()
        loss.backward()
        ropt.step()
        actual = bucket(data)
        assert bucket.step.graph is capture
        for name in ("energy", "forces", "stress", "loss"):
            torch.testing.assert_close(
                actual[name],
                loss if name == "loss" else expected[name],
                atol=3e-5,
                rtol=4e-4,
            )
        for (name, p), (rname, r) in zip(
            model.named_parameters(), ref.named_parameters(), strict=True
        ):
            assert name == rname
            torch.testing.assert_close(p.grad, r.grad, atol=5e-5, rtol=6e-4)
            torch.testing.assert_close(p, r, atol=3e-5, rtol=4e-4)
        shift_grad = model.readout_fn.scale_shift.shift.grad
        if ids is None:
            assert shift_grad[default].abs().sum() > 0
            assert torch.count_nonzero(shift_grad[1 - default]) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay")
@pytest.mark.parametrize(
    "layers,rank,order",
    [
        (("cgtp",) * 2, 2, 2),
        (("uu_o2",) * 2, 2, 2),
        (("uv_o2",) * 2, 2, 2),
        (("cgtp",), 1, 2),
        (("uu_o2",), 2, 2),
        (("cgtp",) * 3, 2, 3),
        (("uv_o2", "uu_o2", "uv_o2"), 1, 2),
        (("cgtp",) * 4, 3, 2),
        (("uu_o2", "uv_o2") * 2, 2, 2),
        (("ictd",) * 2, 2, 2),
        (("ictd",) * 3, 2, 2),
        (("cartesian_uu",) * 2, 2, 2),
    ],
    ids=(
        "cgtp2",
        "uu2",
        "uv2",
        "cgtp1-l1",
        "uu1",
        "cgtp3-corr3",
        "mixed3-l1",
        "cgtp4-l3",
        "mixed4",
        "ictd2",
        "ictd3",
        "cartesian-uu64-threshold",
    ),
)
def test_regional_training_bucket(monkeypatch, layers, rank, order):
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    torch.set_default_dtype(torch.float32)
    from tace.models.cartesian.tace import CartesianTACE

    cartesian_uu = layers[0] == "cartesian_uu"
    cartesian = cartesian_uu or layers[0] == "ictd"
    host_type = CartesianTACE if cartesian else SphericalTACE
    if cartesian:
        layers = (("uu_o2" if cartesian_uu else "cgtp"),) * len(layers)
    config = deepcopy(_model("cgtp").readout_fn.model_config)
    config.update(num_layers=len(layers), Lmax=rank, lmax=rank, mmax=rank)
    if cartesian_uu:
        torch.manual_seed(17)
        config["num_channel"] = 64
        config["statistics"][0]["avg_num_neighbors"] = 18
    config["atomic_basis"]["type"] = list(layers)
    config["product_basis"]["correlation"] = order
    config["radial_basis"]["apply_cutoff"] = layers[0] == "cgtp"
    if len(layers) >= 3 and (cartesian or layers[0] == "cgtp"):
        # Exercise a live node norm in spherical and Cartesian storage.
        config["layer_norm"]["final_norm_type"] = "soft_cap_rms_norm"
    host, ref = (
        TensorModel(host_type(**deepcopy(config))).train().cuda() for _ in range(2)
    )
    ref.load_state_dict(host.state_dict())
    assert not supports_training_buckets(host)
    opt = torch.optim.SGD(host.parameters(), lr=0.02)
    ref_opt = torch.optim.SGD(ref.parameters(), lr=0.02)
    model = optimize(host, family="tace", task="training", backend="cuda")
    if not cartesian:
        ref = optimize(ref, family="tace", task="training", backend="cuda")
    assert all(
        a is b for a, b in zip(model.parameters(), host.parameters(), strict=True)
    )
    assert supports_training_buckets(model)
    with model.readout_fn.energy_readouts[0].register_forward_hook(lambda *_: None):
        assert not supports_training_buckets(model)
    with monkeypatch.context() as patch:
        patch.setitem(
            model.readout_fn.energy_readouts[0].acts._modules,
            "0",
            torch.nn.Softmax(dim=0),
        )
        assert not supports_training_buckets(model)
    with monkeypatch.context() as patch:
        patch.setattr(
            model.readout_fn.representation.node_embedding.embedding,
            "scale_grad_by_freq",
            True,
        )
        assert not supports_training_buckets(model)
    # A batch-axis softmax couples real nodes to padding even with no edges
    # between components. Neither its standard PyTorch origin nor a previous
    # regional marker is proof that this modified representation is safe.
    rep = model.readout_fn.representation
    previous_norm = getattr(rep, "final_norm", None)
    with monkeypatch.context() as patch:
        # nn.Module registers children in _modules, not its instance dict.
        patch.setitem(rep._modules, "final_norm", torch.nn.Softmax(dim=0))
        assert not supports_training_buckets(model)
    assert getattr(rep, "final_norm", None) is previous_norm
    batches = _cases()
    if cartesian_uu:
        from equivariantx_foundation.tests.test_cartesian_schedule import _periodic

        batches = [_periodic(8, "cuda"), _periodic(7, "cuda")]
        assert [b["edge_index"].size(1) for b in batches] == [9216, 6174]
        program = rep.interactions[1].rejector.eqx_tp
        with monkeypatch.context() as patch:
            patch.setattr(program, "backend", "torch")
            assert not supports_training_buckets(model)
        with monkeypatch.context() as patch:
            patch.setitem(
                rep.interactions[1].rejector._modules, "eqx_tp", deepcopy(program)
            )
            assert not supports_training_buckets(model)
        with monkeypatch.context() as patch:
            patch.setattr(rep, "_can_pack_wigner", True)
            assert not supports_training_buckets(model)
        captured = []
        original = type(program).forward

        def audit(module, x, w, idx, rotation, cutoff, **kwargs):
            if module is program and torch.cuda.is_current_stream_capturing():
                captured.append(idx.shape[1])
            return original(module, x, w, idx, rotation, cutoff, **kwargs)

        monkeypatch.setattr(type(program), "forward", audit)
    for i, b in enumerate(batches):
        b["energy"] = (
            torch.linspace(
                0.2,
                -0.5,
                b["ptr"].numel() - 1,
                dtype=b["positions"].dtype,
                device=b["positions"].device,
            )
            + i * 0.1
        )
        b["forces"] = 0.2 * torch.sin(b["positions"])
        b["stress"] = 0.03 * b["lattice"]
    # Input node_type is authoritative even when atomic_numbers are also supplied.
    batches[1]["node_type"] = batches[1]["node_type"].flip(0)
    # Keep the input field signature fixed; the physical values change.
    for b in batches:
        b["atomic_numbers"] = torch.full_like(
            b["node_type"], host.get_atomic_numbers()[0]
        )

    def eager(batch):
        ref_opt.zero_grad(set_to_none=True)
        p = ref(dict(batch))
        loss = (((p["energy"] - batch["energy"]) / batch["ptr"].diff()) ** 2).mean()
        for key in ("forces", "stress"):
            loss = loss + (p[key] - batch[key]).square().mean()
        loss.backward()
        ref_opt.step()
        return {k: p[k] for k in ("energy", "forces", "stress")} | {"loss": loss}

    initial = [p.detach().clone() for p in model.parameters()]
    shapes = [(len(b["positions"]), b["edge_index"].size(1)) for b in batches]
    capacity = dict(
        max_atoms=max(n for n, _ in shapes), max_edges=max(e for _, e in shapes)
    )
    with pytest.raises(ValueError, match="private CUDA replay only"):
        BucketedEFSTrainingStep(model, opt, batches[0], joint=True, **capacity)

    def activate_dummy(packed, real_atoms):
        packed["lattice"][-1] = torch.diag(
            packed["lattice"].new_tensor(
                [2.1, 2.1, 2.1] if cartesian_uu else [0.8, 1.1, 1.3]
            )
        )
        packed["energy"][-1] = 100
        packed["forces"][real_atoms:].fill_(20)
        packed["stress"][-1].fill_(-30)
        return packed

    if cartesian:
        # Feed an active, labelled dummy through capture AND every replay;
        # the default production cell outside cutoff tests a different case.
        from equivariantx_foundation.models.tace import training

        def active_pack(data, *args, **kwargs):
            return activate_dummy(_pack(data, *args, **kwargs), len(data["positions"]))

        monkeypatch.setattr(training, "_pack", active_pack)
    bucket = BucketedEFSTrainingStep(model, opt, batches[0], **capacity)
    if cartesian_uu:
        assert captured == [9216 + 512]
    for p, old in zip(model.parameters(), initial, strict=True):
        torch.testing.assert_close(p, old, atol=0, rtol=0)
    recorded = bucket.step.graph
    for step_index, batch in enumerate(batches):
        # For deeper stacks, deliberately activate the dummy component. Its
        # nonzero periodic messages and large labels still cannot change real
        # predictions, the masked objective or any trainable parameter gradient.
        check_active = cartesian or len(layers) >= 3
        if check_active:
            packed = _pack(batch, *bucket.cap, edge_layout=bucket.edge_layout)
            inactive_energy = model(packed)["energy"][-1].detach()
            activate_dummy(packed, len(batch["positions"]))
            changed = model(packed)
            assert (changed["energy"][-1].detach() - inactive_energy).abs() > 1e-12
            masked = _efs_loss(changed, packed)
            dummy_grads = torch.autograd.grad(
                masked, tuple(model.parameters()), allow_unused=True
            )
            changed = bucket.trim(
                {k: changed[k].detach() for k in ("energy", "forces", "stress")}
                | {"loss": masked.detach()},
                batch,
            )
        if cartesian:
            unpadded = model(dict(batch))
        expected, actual = eager(batch), bucket(batch)
        assert bucket.step.graph is recorded
        for name in expected:
            if cartesian and name != "loss":
                torch.testing.assert_close(
                    unpadded[name],
                    expected[name],
                    atol=3e-5,
                    rtol=4e-4,
                    msg=lambda diff: (
                        f"regional vs native step {step_index} {name}: {diff}"
                    ),
                )
            if check_active:
                torch.testing.assert_close(
                    changed[name],
                    expected[name],
                    atol=3e-5,
                    rtol=4e-4,
                    msg=lambda diff: f"padded eager step {step_index} {name}: {diff}",
                )
            torch.testing.assert_close(
                actual[name],
                expected[name],
                atol=3e-5,
                rtol=4e-4,
                msg=lambda diff: f"replayed step {step_index} {name}: {diff}",
            )
        for index, ((k, p), (q, v)) in enumerate(
            zip(model.named_parameters(), ref.named_parameters(), strict=True)
        ):
            assert k == q and (p.grad is None) == (v.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(p.grad, v.grad, atol=3e-5, rtol=4e-4, msg=k)
            if check_active:
                torch.testing.assert_close(
                    dummy_grads[index], v.grad, atol=3e-5, rtol=4e-4, msg=k
                )
            torch.testing.assert_close(p, v, atol=2e-6, rtol=3e-5, msg=k)
        if check_active:
            for blocks in (
                model.readout_fn.representation.interactions,
                model.readout_fn.representation.products,
            ):
                for block in blocks:
                    assert any(
                        p.grad is not None and p.grad.numel() and p.grad.abs().max() > 0
                        for p in block.parameters()
                    )
    assert any(
        not torch.equal(p, old)
        for p, old in zip(model.parameters(), initial, strict=True)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph replay")
@pytest.mark.parametrize(
    "embedding,scatter",
    [
        ("element", "cutoff"),
        ("element2", "sqrt_avg_num_neighbors"),
        ("identity", "identity"),
    ],
)
def test_regional_microbatches_reproduce_native_adamw(monkeypatch, embedding, scatter):
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    cfg = deepcopy(_model("cgtp").readout_fn.model_config)
    cfg["cutoff"] = 12.0
    cfg["edge_embedding"]["type"] = embedding
    cfg["atomic_basis"]["scatter_norm"] = scatter
    cfg["radial_basis"]["apply_cutoff"] = False
    native, host = (
        TensorModel(SphericalTACE(**deepcopy(cfg))).train().cuda() for _ in range(2)
    )
    # Distinct nonzero endpoint weights expose reversed or stale element indices.
    with torch.no_grad():
        for p in native.readout_fn.representation.edge_embedding.parameters():
            p.normal_(std=0.1)
    host.load_state_dict(native.state_dict())
    model = optimize(host, family="tace", task="training", backend="cuda")
    assert supports_training_buckets(model)
    a, b, c = _cases()
    for i, d in enumerate((a, b, c)):
        d.update(
            energy=(i + 1) * d["lattice"][:, 0, 0],
            forces=d["positions"].sin(),
            stress=0.03 * d["lattice"],
        )
    reference = torch.optim.AdamW(native.parameters(), lr=3e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    step = BucketedEFSTrainingStep(
        model, optimizer, a, max_atoms=7, max_edges=15, microbatches=2
    )
    assert step.dummy_cell.sum(dim=1).norm() > cfg["cutoff"]
    assert not optimizer.state
    for index, group in enumerate(((a, b), (b, c)), start=1):
        reference.zero_grad(set_to_none=True)
        preds = [native(dict(d)) for d in group]
        # One loss on all real labels, not an average of the microbatch means.
        per_atom = torch.cat(
            [
                (p["energy"] - d["energy"]) / d["ptr"].diff()
                for p, d in zip(preds, group, strict=True)
            ]
        )
        loss = per_atom.square().mean()
        for key in ("forces", "stress"):
            residuals = torch.cat(
                [p[key] - d[key] for p, d in zip(preds, group, strict=True)]
            )
            loss = loss + residuals.square().mean()
        loss.backward()
        reference.step()
        actual = step(group)
        torch.testing.assert_close(actual["loss"], loss, rtol=2e-5, atol=2e-6)
        for key in ("energy", "forces", "stress"):
            torch.testing.assert_close(
                actual[key], torch.cat([p[key] for p in preds]), rtol=3e-4, atol=3e-5
            )
        for (name, p), (other, q) in zip(
            model.named_parameters(), native.named_parameters(), strict=True
        ):
            assert name == other and (p.grad is None) == (q.grad is None)
            if q.grad is not None:
                torch.testing.assert_close(
                    p.grad, q.grad, rtol=4e-4, atol=3e-6, msg=name
                )
                torch.testing.assert_close(p, q, rtol=3e-4, atol=3e-6, msg=name)
                assert optimizer.state[p]["step"].item() == index
                for moment in ("exp_avg", "exp_avg_sq"):
                    torch.testing.assert_close(
                        optimizer.state[p][moment],
                        reference.state[q][moment],
                        rtol=4e-4,
                        atol=3e-6,
                        msg=name,
                    )
        for p in model.readout_fn.representation.edge_embedding.parameters():
            assert p.grad is not None and p.grad.abs().max() > 0


@pytest.mark.parametrize("layout", ["identity_pairs", "directed"])
def test_regional_packing_rejects_ghosts_and_overridden_topology(layout):
    b = graph("cpu")
    n, e = len(b["positions"]), b["edge_index"].size(1)
    b.update(
        energy=b["positions"].new_zeros(2), forces=torch.zeros_like(b["positions"])
    )

    def pack(data):
        return _pack(data, n, e, 2, False, edge_layout=layout)

    with pytest.raises(ValueError, match="local/ghost"):
        pack({**b, "lmp_data": object()})
    with pytest.raises(ValueError, match="raw edges"):
        pack({**b, "edge_pair_rows": b["edge_index"][0]})
    with pytest.raises(ValueError, match="one three-component shift"):
        pack({**b, "edge_shifts": b["edge_shifts"][:1]})
    invalid = b["edge_index"].clone()
    invalid[0, 0] = n
    with pytest.raises(RuntimeError, match="ghost edge"):
        pack({**b, "edge_index": invalid})
    invalid[0, 0] = n - 1
    with pytest.raises(RuntimeError, match="within their structure"):
        pack({**b, "edge_index": invalid})
    with pytest.raises(RuntimeError, match="membership"):
        pack({**b, "batch": b["batch"] + 1})


def test_matching_architecture_does_not_admit_cpu_or_eval_buckets(monkeypatch):
    from equivariantx_foundation.models.tace import planner

    host = _model("cgtp")
    monkeypatch.setattr(planner, "release_schedule", lambda *a, **kw: ("release", ""))
    assert not supports_training_buckets(host.train())
    assert not supports_training_buckets(host.eval())
    if torch.cuda.is_available():
        host.cuda()
        assert not supports_training_buckets(host)
        assert not supports_training_buckets(
            host.train()
        )  # Unvalidated release cutoff.
        host.readout_fn.cutoff.fill_(6.0)
        assert supports_training_buckets(host)
        with next(host.parameters()).register_hook(lambda g: g):
            assert not supports_training_buckets(host)
        assert not supports_training_buckets(host.requires_grad_(False))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA regional plan")
@pytest.mark.parametrize("kind,width,mmax", [("uu_o2", 32, 2), ("uv_o2", 64, 1)])
def test_cartesian_local_frames_do_not_claim_ictd_bucket(
    monkeypatch, kind, width, mmax
):
    from tace.models.cartesian.tace import CartesianTACE

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    config = deepcopy(_model("cgtp").readout_fn.model_config)
    config.update(num_channel=width, mmax=mmax)
    config["atomic_basis"]["type"] = kind
    config["radial_basis"]["apply_cutoff"] = False
    host = TensorModel(CartesianTACE(**config)).float().train().cuda()
    planned = optimize(host, family="tace", task="training", backend="cuda")
    assert planned.readout_fn.representation._foundation_regional
    assert not supports_training_buckets(planned)

"""Ownership of live optimizer state, export snapshots and captured storage."""

import json
import subprocess
import sys

import pytest
import torch
from torch import nn

from equivariantx_foundation.models.tace.foundation import (
    frozen_export_copy,
    shared_parameter_copy,
)


def test_training_aliases_and_frozen_export_storage_are_different_contracts():
    source = nn.Sequential(nn.Linear(3, 3), nn.Linear(3, 3)).double().train()
    source[1].weight = source[0].weight
    source[1].bias.requires_grad_(False)
    source[0].eval()
    source.register_buffer("scale", torch.ones(2))
    source.register_buffer("workspace", torch.zeros(2), persistent=False)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1, momentum=0.9)
    x = torch.arange(6, dtype=torch.float64).reshape(2, 3)
    source(x).sum().backward()
    optimizer.step()
    grad = source[0].weight.grad
    momentum = optimizer.state[source[0].weight]["momentum_buffer"].clone()
    training = shared_parameter_copy(source)
    snapshot = frozen_export_copy(training)
    # Both identity and storage matter: a distinct, detached Parameter sharing
    # storage would silently change a frozen graph on the next optimizer step.
    assert training[0].weight is source[0].weight
    assert snapshot[0].weight is snapshot[1].weight
    for sp, fp in zip(source.parameters(), snapshot.parameters(), strict=True):
        assert sp.data_ptr() != fp.data_ptr() and not fp.requires_grad
    assert [p.requires_grad for p in source.parameters()] == [True, True, False]
    assert training.training and not training[0].training and training[1].training
    assert source.training and not source[0].training and source[1].training
    assert not any(m.training for m in snapshot.modules())
    assert snapshot.state_dict().keys() == source.state_dict().keys()
    snapshot.scale.zero_()
    snapshot.workspace.add_(1)
    assert training.scale.eq(1).all() and training.workspace.eq(0).all()
    assert source[0].weight.grad is grad
    torch.testing.assert_close(
        optimizer.state[source[0].weight]["momentum_buffer"], momentum
    )
    before = snapshot(x)
    torch.testing.assert_close(before, source(x))
    optimizer.step()
    torch.testing.assert_close(before, snapshot(x))
    assert not torch.equal(before, source(x))


class AlreadyOptimized(nn.Module):
    """Minimal structural stand-in for idempotent release optimizers."""

    def __init__(self):
        super().__init__()
        self.readout_fn = nn.Module()
        rep = self.readout_fn.representation = nn.Module()
        rep.interactions = nn.ModuleList()
        rep.products = nn.ModuleList([nn.Linear(3, 3)])


@pytest.mark.parametrize("kind", ["7m", "rra"])
@pytest.mark.parametrize("fail", [False, True])
def test_release_export_does_not_freeze_or_rewrite_existing_adapter(
    kind, fail, monkeypatch
):
    pytest.importorskip("lammps.mliap.mliap_unified_abc")
    from equivariantx_foundation.models.tace import foundation

    source = AlreadyOptimized().train()
    rep = source.readout_fn.representation
    initial_product = rep.products[0]
    initial_product.eval()
    initial_product.bias.requires_grad_(False)
    optimizer = torch.optim.SGD(source.parameters(), lr=0.1)
    x = torch.ones(1, 3)
    prediction = initial_product(x).detach()

    def compile_or_fail(model, _output, *, receiver_model, **kwargs):
        frozen = model.readout_fn.representation
        receiver = receiver_model.readout_fn.representation
        assert not any(p.requires_grad for p in model.parameters())
        assert not any(p.requires_grad for p in receiver_model.parameters())
        if kind == "rra":
            # Folded and indexed products must stay separate, sharing weights
            # within this export only.
            assert isinstance(frozen.products[0], nn.Sequential)
            assert type(receiver.products[0]) is nn.Linear
            assert receiver.products[0].weight is frozen.products[0][0].weight
        else:
            assert type(frozen).__name__ == "SevenDistanceTangent"
        p = next(receiver_model.parameters())
        assert p.data_ptr() != initial_product.weight.data_ptr()
        torch.testing.assert_close(frozen.products[0](x), prediction)
        p.add_(10)
        if fail:
            raise RuntimeError("compile failure")
        return "completed"

    if kind == "7m":
        from equivariantx_foundation.models.tace.tace_oam_7m import lammps as entry

        monkeypatch.setattr(entry, "optimize_oam_7m", lambda model: model)
        monkeypatch.setattr(entry, "tangent_embedding", lambda rep: True)
        monkeypatch.setattr(entry, "export_local_lammps", compile_or_fail)
    else:
        from equivariantx_foundation.models.tace.tece_oam_rra import frozen_product
        from equivariantx_foundation.models.tace.tece_oam_rra import lammps as entry

        monkeypatch.setattr(entry, "optimize_rra", lambda model: model)
        monkeypatch.setattr(frozen_product, "FrozenBilinearProduct", nn.Sequential)
        monkeypatch.setattr(foundation, "export_local_lammps", compile_or_fail)

    if fail:
        with pytest.raises(RuntimeError, match="compile failure"):
            entry.export_lammps(source, "unused.pt", capacity=True)
    else:
        assert entry.export_lammps(source, "unused.pt", capacity=True) == "completed"
    assert type(rep) is nn.Module and rep.products[0] is initial_product
    assert source.training and rep.training and not initial_product.training
    assert (
        initial_product.weight.requires_grad and not initial_product.bias.requires_grad
    )
    torch.testing.assert_close(initial_product(x), prediction)
    initial_product(x).sum().backward()
    optimizer.step()
    assert not torch.equal(initial_product(x), prediction)


def test_oam_l_tangent_variant_keeps_preoptimized_base_and_compact_sibling(monkeypatch):
    from equivariantx_foundation.models.tace.tace_oam_l import model as oam_l
    from equivariantx_foundation.models.tace.tace_oam_l import tangent

    source = AlreadyOptimized().train()
    base = frozen_export_copy(source)
    # The released optimizer can return the input module unchanged if its
    # memory/geometry/grouping options exactly match the request.
    monkeypatch.setattr(oam_l, "optimize_oam_l", lambda model, **kw: model)
    monkeypatch.setattr(
        tangent, "radial_embedding", lambda rep: rep.products[0].weight.detach()
    )
    base.readout_fn.representation.products[0].scalar_norms = True
    variant = tangent._tangent_from_frozen(base, keep_base=True)
    src_rep, base_rep, tangent_rep = (
        m.readout_fn.representation for m in (source, base, variant)
    )
    assert type(src_rep) is type(base_rep) is nn.Module
    assert not hasattr(base_rep, "frozen_embedding") and not hasattr(
        src_rep, "stream_heads"
    )
    assert type(tangent_rep).__name__ == "RadialTangentRepresentation"
    assert tangent_rep.products[0].weight is base_rep.products[0].weight
    assert (
        tangent_rep.products[0].weight.data_ptr()
        != src_rep.products[0].weight.data_ptr()
    )
    assert source.training and src_rep.products[0].weight.requires_grad


@pytest.mark.parametrize("write_buffer", [True, False])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="Foundation CUDA export")
def test_joint_export_preserves_modes_and_rejects_external_writes(
    write_buffer, monkeypatch
):
    from equivariantx_foundation.targets import aoti
    from equivariantx_foundation.tests.test_tace_plan import graph
    from tace.models.adapter import TensorModel
    from tace.models.compile.joint import supports_joint_loss
    from tace.utils.loss.normal import NormalLoss

    class Readout(nn.Module):
        target_property = ["energy"]
        embedding_property = []

        def __init__(self):
            super().__init__()
            self.weight = nn.Parameter(torch.tensor(2.0))
            self.register_buffer("atomic_numbers", torch.tensor([1, 8]))
            self.register_buffer("offset", torch.zeros(2))

        def forward(self, data, g):
            offset = self.offset if write_buffer else self.offset.clone()
            offset[1:].add_(1)
            nodes = self.weight * g.positions[:, 0] + offset[1]
            energy = nodes.new_zeros(g.num_graphs).index_add(0, data["batch"], nodes)
            return {"energy": energy, "node_energy": nodes}

    source = TensorModel(Readout()).cuda().train()
    source.readout_fn.eval()
    objective = NormalLoss(["energy"], ["mse_energy"], [1.0])
    if not supports_joint_loss(source, objective):
        pytest.skip("joint training trace is not available")
    batch = graph("cuda")
    ones = batch["positions"].new_ones(2)
    batch.update(energy=ones, entropy=ones, energy_weight=ones)
    # Exercise the real trace and export; stop only at native code generation.
    monkeypatch.setattr(aoti, "export", lambda *args, **kwargs: "packaged")
    if write_buffer:
        with pytest.raises(ValueError, match="cannot mutate"):
            aoti.export_training(source, objective, batch, "unused.pt2")
    else:
        assert (
            aoti.export_training(source, objective, batch, "unused.pt2") == "packaged"
        )
    assert source.training and not source.readout_fn.training
    torch.testing.assert_close(
        source.readout_fn.offset, torch.zeros_like(source.readout_fn.offset)
    )


def test_training_contract_normalizes_only_independent_pure_work():
    from types import SimpleNamespace

    from equivariantx_foundation.models.tace.training_contract import _graph_contract

    ops = torch.ops.aten

    def signature(case, reverse):
        g = torch.fx.Graph()
        x, y = g.placeholder("x"), g.placeholder("y")

        def op(target, *args):
            return g.call_function(target, args)

        if case in ("independent", "args", "outputs"):
            if reverse and case == "independent":
                b, a = op(ops.neg.default, y), op(ops.neg.default, x)
            else:
                a, b = op(ops.neg.default, x), op(ops.neg.default, y)
            if reverse and case == "outputs":
                a, b = b, a
            result = (
                (a, b)
                if case == "outputs"
                else op(
                    ops.add.Tensor, *((b, a) if reverse and case == "args" else (a, b))
                )
            )
        elif case == "sharing":
            a = op(ops.neg.default, x)
            result = (a, op(ops.neg.default, x) if reverse else a)
        elif case == "rng":
            a, b = op(ops.rand_like.default, x), op(ops.rand_like.default, x)
            result = (b, a) if reverse else (a, b)
        elif case == "assert":
            for check in (False, True) if reverse else (True, False):
                op(ops._assert_scalar.default, check, "bad")
            result = x
        elif case == "uninitialized":
            if not reverse:
                a = op(ops.neg.default, x)
            alloc = op(ops.new_empty.default, x, [2])
            if reverse:
                a = op(ops.neg.default, x)
            result = (a, alloc)
        else:
            alloc = op(ops.new_zeros.default, x, [2])
            alias = op(ops.as_strided.default, alloc, [2], [1], 0)
            src = alloc if case == "scratch_alias" else y
            if reverse:
                op(ops.copy_.default, alias, x)
                result = op(ops.neg.default, src)
            else:
                result = op(ops.neg.default, src)
                op(ops.copy_.default, alias, x)
        g.output(result)
        trace = SimpleNamespace(
            graph=torch.fx.GraphModule(nn.Module(), g),
            args=(torch.ones(2), torch.ones(2)),
            batch_keys=("x", "y"),
            parameters=(),
            buffers=(),
        )
        return _graph_contract(trace)[0]

    for case in ("independent", "scratch_independent"):
        assert signature(case, False) == signature(case, True), case
    for case in (
        "args",
        "outputs",
        "sharing",
        "rng",
        "assert",
        "scratch_alias",
        "uninitialized",
    ):
        assert signature(case, False) != signature(case, True), case


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Foundation CUDA training package"
)
def test_dynamic_joint_package_uses_live_labels_and_checks_graph_axes(
    tmp_path, subprocess_package_env
):
    from copy import deepcopy

    from torch._inductor import config

    from equivariantx_foundation.targets import aoti
    from tace.models.adapter import TensorModel
    from tace.models.compile.joint import supports_joint_loss
    from tace.utils.loss.uncertainty import UncertaintyLoss

    class Springs(nn.Module):
        target_property = ["energy", "forces"]
        embedding_property = []

        def __init__(self):
            super().__init__()
            self.k = nn.Parameter(torch.tensor(1.25, dtype=torch.float64))
            self.cutoff = 4.0
            self.absolute = False
            self.reference = torch.tensor(0.25, dtype=torch.float64, device="cuda")
            self.register_buffer("atomic_numbers", torch.tensor([1]))
            self.register_buffer("scale", torch.tensor(0.75, dtype=torch.float64))
            self.register_buffer("scale_alias", self.scale)
            self.register_buffer("unused", torch.zeros(2))

        def forward(self, data, graph):
            # A relation between independent axes, not implied by either range.
            torch._check(graph.positions.shape[0] <= graph.edge_vector.shape[0])
            vector = graph.edge_vector
            potential = vector.abs() if self.absolute else vector.square()
            pair = self.k * self.scale * potential.sum(-1) + self.reference
            nodes = pair.new_zeros(graph.positions.shape[0]).index_add(
                0, data["edge_index"][0], pair
            )
            return {"energy": nodes.sum().reshape(1), "node_energy": nodes}

    def batch(n, e, weight):
        xyz = torch.arange(3 * n, dtype=torch.float64).reshape(n, 3) / 10
        src = torch.arange(e) % n
        return {
            k: v.cuda()
            for k, v in dict(
                positions=xyz,
                node_type=torch.zeros(n, dtype=torch.long),
                edge_index=torch.stack((src, (src + 1) % n)),
                edge_shifts=torch.zeros(e, 3, dtype=torch.long),
                lattice=torch.eye(3, dtype=xyz.dtype).unsqueeze(0),
                ptr=torch.tensor([0, n]),
                batch=torch.zeros(n, dtype=torch.long),
                energy=xyz.new_full((1,), 0.5),
                forces=torch.cos(xyz),
                entropy=xyz.new_ones(1),
                energy_weight=xyz.new_tensor([weight]),
                forces_weight=xyz.new_tensor([weight]),
            ).items()
        }

    model = TensorModel(Springs()).cuda().train()
    model.readout_fn.scale_alias = model.readout_fn.scale
    objective = UncertaintyLoss(
        ["energy", "forces"], ["mse_energy", "mse_forces"], [1, 2]
    ).cuda()
    if not supports_joint_loss(model, objective):
        pytest.skip("joint training trace is not available")
    sample = batch(3, 4, 0.0)
    with config.patch(scalar_asserts=False, unsafe_skip_scalar_range_asserts=True):
        path = aoti.export_training(
            model, objective, sample, tmp_path / "dynamic.pt2", dynamic_shapes=True
        )
    with pytest.raises(ValueError, match="Use load_training"):
        aoti.load(path)
    step = aoti.load_training(path)
    # An independent host, different in-domain atom/edge counts, new live
    # weights and a new live reference buffer must bind to the same program.
    replica, loss_copy = deepcopy(model), deepcopy(objective)
    prepared = aoti.training_batch(replica, loss_copy, batch(5, 7, 1.0))
    with torch.no_grad():
        replica.readout_fn.k.add_(0.5)
        replica.readout_fn.scale.mul_(1.25)
        loss_copy.log_sigmas["energy"].add_(0.5)
    aoti.validate_training_model(step, replica, loss_copy, prepared)
    for name, value, component in (
        ("cutoff", 5.0, "geometry"),
        ("reference", replica.readout_fn.reference + 1, "constants"),
        ("absolute", True, "graph"),
    ):
        original = getattr(replica.readout_fn, name)
        setattr(replica.readout_fn, name, value)
        with pytest.raises(ValueError, match=component):
            aoti.validate_training_model(step, replica, loss_copy, prepared)
        setattr(replica.readout_fn, name, original)
    replica.readout_fn.scale_alias = replica.readout_fn.scale.clone()
    with pytest.raises(ValueError, match="state"):
        aoti.validate_training_model(step, replica, loss_copy, prepared)
    replica.readout_fn.scale_alias = replica.readout_fn.scale
    replica.readout_fn.k.requires_grad_(False)
    with pytest.raises(ValueError, match="state"):
        aoti.validate_training_model(step, replica, loss_copy, prepared)
    replica.readout_fn.k.requires_grad_(True)
    loss_copy.loss_function_name[0] = "mae_energy"
    with pytest.raises(ValueError, match="objective"):
        aoti.validate_training_model(step, replica, loss_copy, prepared)
    # A single native package, including force mixed derivatives and the
    # uncertainty regularizers, must not specialize to the sample's zero mask.
    _, params, buffers = aoti.training_inputs(model, objective, sample)
    for raw, labelled in ((sample, False), (batch(5, 7, 1.0), True)):
        data = aoti.training_batch(model, objective, raw)
        if labelled:
            buffers = dict(buffers)
            del buffers["model.readout_fn.unused"]
        pred, loss, grads = step(data, params, buffers)
        eager = model(raw)
        eager_loss = objective(eager, raw)
        expected = torch.autograd.grad(eager_loss, tuple(params.values()))
        torch.testing.assert_close(loss, eager_loss)
        assert grads.keys() == params.keys()
        for key, grad in zip(params, expected, strict=True):
            torch.testing.assert_close(grads[key], grad)
        for key in ("energy", "node_energy", "forces"):
            torch.testing.assert_close(pred[key], eager[key])
        assert (grads["model.readout_fn.k"].item() != 0) == labelled

    # Reuse this native package and the last (labelled, resized) batch. The
    # standalone runtime needs tensors, not a Python host model or objective.
    fixture = tmp_path / "standalone-tensors.pt"
    torch.save(
        {
            "args": (data, {k: v.detach() for k, v in params.items()}, buffers),
            "pred": {k: eager[k].detach() for k in ("energy", "node_energy", "forces")},
            "loss": eager_loss.detach(),
            "grads": dict(zip(params, expected, strict=True)),
        },
        fixture,
    )
    runtime = """
import sys

class WithoutHostOrOperators:
    def find_spec(self, name, *args):
        if name.split('.')[0] in {'tace', 'equivariantx'}:
            raise ModuleNotFoundError(name, name=name)

assert not {'tace', 'equivariantx'}.intersection(sys.modules)
sys.meta_path.insert(0, WithoutHostOrOperators())
import torch
from equivariantx_foundation.targets.aoti import load_training

case = torch.load(sys.argv[2], map_location='cuda', weights_only=True)
pred, loss, grads = load_training(sys.argv[1])(*case['args'])
torch.testing.assert_close({k: pred[k] for k in case['pred']}, case['pred'])
torch.testing.assert_close(loss, case['loss'])
torch.testing.assert_close(grads, case['grads'])
assert not {'tace', 'equivariantx'}.intersection(sys.modules)
"""
    child = subprocess.run(
        [sys.executable, "-c", runtime, str(path), str(fixture)],
        cwd=tmp_path,
        env=subprocess_package_env("equivariantx", "equivariantx_foundation"),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert child.returncode == 0, child.stdout + child.stderr

    for bad in (
        {"energy": data["energy"].float()},  # dtype
        {"ptr": data["ptr"].new_tensor([0, 2, 5])},  # G is fixed
        {"edge_shifts": data["edge_shifts"][:-1]},  # edge_index still has E=7
        {"forces": data["forces"][:-1]},  # positions still has N=5
        {
            "edge_index": data["edge_index"][:, :1],
            "edge_shifts": data["edge_shifts"][:1],
        },
    ):
        with pytest.raises(ValueError, match="shape|dtype"):
            step(data | bad, params, buffers)

    with pytest.raises(KeyError, match="scale"):
        step(
            data,
            params,
            {k: v for k, v in buffers.items() if k != "model.readout_fn.scale"},
        )

    # All tensor layouts, per-axis ranges and shared dimensions are valid;
    # only the cross-axis assertion in the compiled native program can reject it.
    with pytest.raises(
        RuntimeError, match="Expected.*to be True|Runtime assertion failed"
    ):
        step(*aoti.training_inputs(model, objective, batch(5, 3, 1.0)))


def _joint_runner(monkeypatch, x, p):
    from equivariantx_foundation.targets import aoti

    class Runner:
        def get_metadata(self):
            schema = {
                "static_replay": True,
                "device": str(x.device),
                "inputs": [aoti._signature(x), aoti._signature(p)],
                "batch": ["positions"],
                "parameters": ["p"],
                "buffers": [],
                "predictions": [],
                "gradients": ["p"],
                "present": [True, True],
            }
            return {aoti.TRAINING_SCHEMA: json.dumps(schema)}

        def __call__(self, x, p):
            return (x * p).sum(), x

    monkeypatch.setattr(aoti, "_load_runner", lambda *args, **kw: Runner())
    return aoti


def test_joint_runner_rejects_lazy_negative_inputs(monkeypatch):
    x = torch.ones(2)
    negative = torch._neg_view(x)
    assert negative.is_contiguous() and negative.is_neg()
    aoti = _joint_runner(monkeypatch, x, x)
    step = aoti.load_training("unused.pt2")
    with pytest.raises(ValueError, match="missing host contract; re-export"):
        aoti.validate_training_model(step, None, None, {})
    for positions, p in ((x, negative), (negative, x)):
        with pytest.raises(ValueError, match="materialized"):
            step({"positions": positions}, {"p": p}, {})


def test_joint_gradients_are_independently_writable(monkeypatch):
    from equivariantx_foundation.targets import aoti

    x = torch.tensor([3.0, 4.0])
    p, q = (nn.Parameter(torch.zeros_like(x)) for _ in range(2))

    class Runner:
        def get_metadata(self):
            return {
                aoti.TRAINING_SCHEMA: json.dumps(
                    dict(
                        device="cpu",
                        inputs=[aoti._signature(x)] * 3,
                        batch=["x"],
                        parameters=["p", "q"],
                        buffers=[],
                        gradients=["p", "q"],
                        predictions=["energy"],
                        present=[True] * 4,
                    )
                )
            }

        def __call__(self, x, p, q):
            return (x * (p + q)).sum(), x, x, x

    monkeypatch.setattr(aoti, "_load_runner", lambda *a, **kw: Runner())
    step = aoti.load_training("unused.pt2")
    prediction, _, gradients = step({"x": x}, {"p": p, "q": q}, {})
    p.grad, q.grad = gradients.values()
    torch.nn.utils.clip_grad_norm_([p, q], 1.0)
    expected = torch.tensor([3.0, 4.0]) / (50**0.5)
    for g in (p.grad, q.grad):
        torch.testing.assert_close(g, expected)
    torch.testing.assert_close(prediction["energy"], torch.tensor([3.0, 4.0]))
    torch.testing.assert_close(x, torch.tensor([3.0, 4.0]))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph training replay")
def test_joint_replay_rejects_relocated_allocation(monkeypatch):
    x = torch.ones(2, device="cuda")
    p = nn.Parameter(x.clone())
    aoti = _joint_runner(monkeypatch, x, p)
    inputs = ({"positions": x}, {"p": p}, {})
    step = aoti.load_training("unused.pt2", replay_inputs=inputs)
    opt = torch.optim.SGD([p], lr=0.25)
    _, first, grads = step(*inputs)
    p.grad = grads["p"]
    opt.step()
    _, second, _ = step(*inputs)
    torch.testing.assert_close(first, x.new_tensor(2.0))
    torch.testing.assert_close(second, x.new_tensor(1.5))
    address = p.data_ptr()
    p.detach().resize_(65536)
    assert p.shape == x.shape and p.data_ptr() != address
    with pytest.raises(ValueError, match="storage"):
        step(*inputs)

"""Derivative and execution parity in raw and irreducible Cartesian storage."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3

from equivariantx import co3
from equivariantx_foundation.models.tace.cartesian import (
    cartesian_interaction_rule,
    cartesian_product_rule,
)
from equivariantx_foundation.models.tace.foundation import shared_parameter_copy
from equivariantx_foundation.tests.test_interaction_schedule import ordinary_layer
from tace.models.adapter import TensorModel
from tace.models.cartesian.layers import cartesian_modules
from tace.models.cartesian.tace import CartesianTACE
from tace.models.spherical.default import DEFAULT_MODEL_CONFIG
from tace.models.spherical.prod import CgtpACE


def converted(source, rule, backend):
    transform, reason = rule(source, backend=backend, task="training")
    assert transform is not None, reason
    fast = transform(shared_parameter_copy(source))
    assert source.state_dict().keys() == fast.state_dict().keys()
    source.load_state_dict(fast.state_dict(), strict=True)
    fast.load_state_dict(source.state_dict(), strict=True)
    expected = dict(source.named_parameters())
    actual = dict(fast.named_parameters())
    assert actual.keys() == expected.keys() and all(
        expected[k] is actual[k] for k in expected
    )
    return fast


@pytest.fixture(params=["torch", "cuda"])
def setup(request, monkeypatch, double_precision):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    return request.param, "cuda" if request.param == "cuda" else "cpu"


def test_raw_ictd_convolution(setup):
    backend, device = setup
    host = cartesian_modules(ordinary_layer(width=3, lmax=2)).to(device)
    fast = converted(host, cartesian_interaction_rule, backend)
    conv = host.rejector
    # Arbitrary tensor components, deliberately not symmetrized. Matching
    # only on STF values can hide a change to gradients of projected tensors.
    nodes, edges = 5, 13
    x, a, w = [
        torch.randn(n, size, device=device, requires_grad=True)
        for n, size in (
            (nodes, conv.irreps_in1.dim),
            (edges, conv.irreps_in2.dim),
            (edges, conv.weight_numel),
        )
    ]
    receivers = 3
    index = torch.randint(receivers - 1, (2, edges), device=device)
    # Ghost senders participate; the last owned receiver is isolated.
    index[0, ::2] = nodes - 1

    def evaluate(layer, compact):
        count = receivers if compact else None
        result = layer(x, a, w, index, num_nodes=count)
        if compact:
            assert result.shape[0] == receivers
            assert (
                result.untyped_storage().nbytes()
                == result.numel() * result.element_size()
            )
        result = result[:receivers]
        gradients = torch.autograd.grad(
            result.sin().sum(), (x, a, w), create_graph=True
        )
        assert gradients[0][-1].abs().sum() > 0
        mixed = torch.autograd.grad(sum(g.square().sum() for g in gradients), (x, a, w))
        return result, gradients, mixed

    calls = []
    fast.rejector.scheduled_convolution.register_forward_hook(
        lambda mod, args, output: calls.append(output.shape[0])
    )
    oracle = evaluate(conv, False)
    for layer in (conv, fast.rejector):
        torch.testing.assert_close(oracle, evaluate(layer, True), rtol=1e-8, atol=1e-9)
    # Check the actual indexed convolution, before the final channel packing.
    assert calls == [receivers]


@pytest.mark.parametrize("edges", [0, 9])
def test_ictd_streamed_receiver_count(setup, edges):
    backend, device = setup
    host = cartesian_modules(ordinary_layer(width=2, lmax=2)).to(device)
    conv = host.rejector
    conv.eqx_tp.backend = backend
    senders, receivers = 5, 3
    index = torch.randint(receivers - 1, (2, edges), device=device)
    index[0, ::2] = senders - 1
    x, a, r, w, vec, amp = [
        torch.randn(shape, device=device, requires_grad=True)
        for shape in (
            (senders, conv.irreps_in1.dim),
            (edges, conv.irreps_in2.dim),
            (edges, 4),
            (4, conv.weight_numel),
            (edges, 3),
            (edges, 1),
        )
    ]

    # The streamed geometry path packs angular channels and uses a
    # different backward program from the explicit-attribute schedule.
    def run(count):
        return conv.forward_stream(x, a, r, w, index, amp, vec, num_nodes=count)

    actual = run(receivers)
    expected = run(None)[:receivers]
    torch.testing.assert_close(actual, expected)
    assert actual.shape[0] == receivers
    assert actual.untyped_storage().nbytes() == actual.numel() * actual.element_size()
    if not edges:
        weights = r @ w
        fast = converted(host, cartesian_interaction_rule, backend)
        for count, reference in ((receivers, actual), (0, actual[:0])):
            torch.testing.assert_close(run(count), reference)
            for layer in (conv, fast.rejector):
                value = layer(x, a, weights, index, num_nodes=count)
                torch.testing.assert_close(value, reference)
        return

    def derivatives(value):
        variables = (x, r, w, vec, amp)
        first = torch.autograd.grad(value.sin().sum(), variables, create_graph=True)
        assert first[0].shape == x.shape and first[0][-1].abs().sum() > 0
        mixed = torch.autograd.grad(sum(g.square().sum() for g in first), variables)
        return first, mixed

    torch.testing.assert_close(
        derivatives(actual), derivatives(expected), rtol=1e-8, atol=1e-9
    )


def test_ictd_product_live_weights(setup):
    backend, device = setup
    irreps = o3.Irreps("3x0e+3x1o+3x2e")
    host = cartesian_modules(
        CgtpACE(
            layer=0,
            num_layers=2,
            num_elements=3,
            Lmax=2,
            lmax=2,
            num_channel=3,
            num_expert=None,
            num_channel_per_expert=None,
            target_irreps=o3.Irreps("0e+1o+2e"),
            irreps_in=irreps,
            correlation=[2, 2],
            l1l2=None,
            bias=True,
            nonlinear=None,
            parity=True,
        )
    ).to(device)
    fast = converted(host, cartesian_product_rule, backend)
    assert fast.scheduled_contraction.backend == backend
    assert fast.scheduled_contraction is not host.eqx_ace
    x = torch.randn(7, irreps.dim, device=device, requires_grad=True)
    attrs = co3.ChangeOfBasis(irreps, backend="torch").to(x)
    z = torch.arange(7, device=device) % 3
    batch = torch.zeros_like(z)

    def evaluate(layer):
        result = layer(attrs(x), z, None, batch)
        grads = torch.autograd.grad(
            result.sin().sum(), (x, *layer.parameters()), create_graph=True
        )
        mixed = torch.autograd.grad(
            grads[0].square().sum(), (x, *layer.parameters()), allow_unused=True
        )
        return result, grads, mixed

    calls = []
    fast.scheduled_contraction.register_forward_hook(lambda *args: calls.append(1))
    torch.testing.assert_close(evaluate(host), evaluate(fast), rtol=1e-8, atol=1e-9)
    assert calls == [1]


def _local_model(device, width, basis="uu_o2"):
    cfg = deepcopy(DEFAULT_MODEL_CONFIG)
    cfg.update(
        num_layers=2,
        num_channel=width,
        Lmax=2,
        lmax=2,
        mmax=2,
        parity=True,
        cutoff=2.6,
        statistics=[
            dict(
                atomic_numbers=[1, 8],
                avg_num_neighbors=18.0,
                atomic_energy={1: 0.0, 8: 0.0},
            )
        ],
        target_property=["energy", "forces", "stress"],
    )
    cfg["atomic_basis"]["type"] = basis
    cfg["scale_shift"]["enable"] = False
    cfg["radial_basis"].update(hidden=[8], num_radial_basis=4, apply_cutoff=False)
    cfg["readout_emlp"].update(hidden=[4], use_one_body_magmoms=False)
    return TensorModel(CartesianTACE(**cfg)).to(device)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native ICTD CUDA products")
def test_nonlinear_ictd_automatic_scope(monkeypatch):
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.models.tace.regional import block_rules
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.utils.env import ACCELERATION_ENV

    for var in ACCELERATION_ENV.values():
        monkeypatch.setenv(var, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")

    def build(**cfg):
        cfg.update(num_channel=64, lmax=3, mmax=3, Lmax=2, parity=None)
        cfg["product_basis"].update(
            type="nonlinear_ictd", nonlinear=None, correlation=3
        )
        return CartesianTACE(**cfg)

    model = make_model(build).float().cuda()
    rep = model.readout_fn.representation

    def products(task="training", backend=None):
        return [
            r
            for r in plan(model, task=task, backend=backend).regions
            if r.path in ("products.0", "products.1")
        ]

    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        for task in ("inference", "training"):
            choices = products(task)
            assert [r.provider for r in choices] == ["eqx_cuda", "eqx_cuda"]
            assert ["1/2" in choices[0].reason, "2/2" in choices[1].reason] == [
                True,
                True,
            ]
        with rep.products[0].gate.register_forward_hook(lambda *args: None):
            assert [r.provider for r in products()] == [None, "eqx_cuda"]
        _, match = block_rules(
            model, backend="cuda", task="inference", automatic=True, rank_local=True
        )
        for product in rep.products:
            select, _ = match(product)
            assert select(product, backend="cuda", task="inference")[0] is None
        with monkeypatch.context() as m:
            m.setattr(rep.interactions[1], "mmax", 2)
            assert not any(r.lower for r in products())
        with monkeypatch.context() as m:
            m.setenv("TACE_MEMORY_FIRST", "0")
            assert not any(r.lower for r in products())
            assert all(r.lower for r in products(backend="cuda"))
        torch.set_float32_matmul_precision("high")
        assert not any(r.lower for r in products())
        torch.set_float32_matmul_precision("highest")
        model.double()
        assert not any(r.lower for r in products())
        assert all(r.lower for r in products(backend="cuda"))
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native ICTD CUDA products")
def test_nonlinear_ictd_regional_training(monkeypatch, double_precision):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        efs_loss,
        graph,
        make_model,
    )
    from tace.utils.env import ACCELERATION_ENV

    for var in ACCELERATION_ENV.values():
        monkeypatch.setenv(var, "0")

    def build(**cfg):
        cfg["product_basis"].update(type="nonlinear_ictd", correlation=3)
        return CartesianTACE(**cfg)

    host = make_model(build).double().cuda().train()
    options = dict(backend="cuda", task="training", regional_only=True)
    report = plan(host, **options)
    assert any(
        r.path.startswith("products.") and "projected ICTD" in r.reason
        for r in report.regions
    )
    fast = optimize(host, family="tace", **options)
    assert host.state_dict().keys() == fast.state_dict().keys()
    assert all(
        a is b for a, b in zip(host.parameters(), fast.parameters(), strict=True)
    )
    stages = [f.tp for m in fast.readout_fn.representation.products for f in m.aces]
    assert any(t.backend == "cuda" and t.projection.groups for t in stages)
    assert all(
        f.tp.backend == "torch"
        for m in host.readout_fn.representation.products
        for f in m.aces
    )

    def result(m):
        y = m(graph("cuda"))
        return [y[k] for k in EFS], torch.autograd.grad(
            efs_loss(y), tuple(m.parameters()), allow_unused=True
        )

    expected = result(host)
    from equivariantx.ops.common import channel_product as channel_ops

    called = []
    product = channel_ops.tensor_product

    def record(metadata, *args, **kwargs):
        called.append((metadata, kwargs.get("backend")))
        return product(metadata, *args, **kwargs)

    monkeypatch.setattr(channel_ops, "tensor_product", record)
    actual = result(fast)
    assert all(
        (t.kernel_metadata, "cuda") in called for t in stages if t.backend == "cuda"
    )
    torch.testing.assert_close(actual, expected, rtol=3e-7, atol=3e-8)


def _periodic(side, device):
    axis = torch.arange(side)
    sites = torch.cartesian_prod(axis, axis, axis)
    nearest = torch.arange(-1, 2)
    nearest = torch.cartesian_prod(nearest, nearest, nearest)
    distance = nearest.abs().sum(1)
    nearest = nearest[(distance > 0) & (distance < 3)]
    n = len(sites)
    destinations = sites[:, None] + nearest
    images = destinations.div(side, rounding_mode="floor")
    indices = destinations.remainder(side) @ torch.tensor([side**2, side, 1])
    pos = sites.double() * 1.6 + 0.07 * torch.rand(n, 3, dtype=torch.float64)
    data = dict(
        positions=pos,
        node_type=torch.arange(n) % 2,
        edge_index=torch.stack(
            (torch.arange(n).repeat_interleave(len(nearest)), indices.flatten())
        ),
        edge_shifts=images.flatten(0, 1),
        lattice=torch.eye(3, dtype=torch.float64)[None] * (side * 1.6),
        pbc=torch.ones(1, 3, dtype=torch.bool),
        batch=torch.zeros(n, dtype=torch.long),
        ptr=torch.tensor([0, n]),
        fidelity_idx=torch.zeros(1, dtype=torch.long),
    )
    return {k: v.to(device) for k, v in data.items()}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="paired ICTD CUDA program")
@pytest.mark.parametrize("product", ["cgtp", "nonlinear_ictd"])
def test_paired_cartesian_ef_keeps_directed_attributes_and_live_radial(
    monkeypatch, request, product
):
    from equivariantx.conv.co3 import CartesianTensorProductConv
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.models.tace.replay_loss import BucketedTrainingLoss
    from equivariantx_foundation.tests.test_tace_plan import graph, make_model
    from tace.utils.env import ACCELERATION_ENV
    from tace.utils.loss.normal import NormalLoss

    for var in ACCELERATION_ENV.values():
        monkeypatch.setenv(var, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    precision = torch.get_float32_matmul_precision()
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(precision))
    torch.set_float32_matmul_precision("highest")

    def build(**cfg):
        cfg.update(
            num_channel=64,
            Lmax=2,
            lmax=3,
            mmax=3,
            parity=None,
            target_property=["energy", "forces"],
        )
        cfg["radial_basis"].update(hidden=[64, 1024], bias=False, apply_cutoff=False)
        cfg["product_basis"]["correlation"] = 3
        if product == "nonlinear_ictd":
            cfg["product_basis"].update(type=product, nonlinear=None)
        return CartesianTACE(**cfg)

    source = make_model(build).float().cuda().train()
    auto = plan(source, task="training", regional_only=True)
    assert not any("paired scalar radial" in r.reason for r in auto.regions)
    model = optimize(
        source, family="tace", backend="cuda", task="training", regional_only=True
    )
    rep = model.readout_fn.representation
    data = graph("cuda")
    # Drop one reverse: the periodic pair and singleton edge must coexist.
    data["edge_index"] = data["edge_index"][:, :-1]
    data["edge_shifts"] = data["edge_shifts"][:-1]
    assert "edge_pair_rows" not in source.prepare_property_data(data)
    assert model.prepare_property_data(data)["edge_pair_rows"].numel() == 6
    assert source.state_dict().keys() == model.state_dict().keys()
    assert all(
        a is b for a, b in zip(source.parameters(), model.parameters(), strict=True)
    )
    seen = []
    captured = []
    forward = CartesianTensorProductConv.forward

    def witness(self, feats, attrs, radial, proj, index, **kw):
        if torch.cuda.is_current_stream_capturing():
            captured.append((kw.get("edge_pairs") is None, proj.numel() == 0))
        if kw.get("edge_pairs") is not None:
            seen.append(
                (
                    radial.shape[0],
                    attrs.shape[0],
                    index.shape[1],
                    kw["radial_network"] is not None,
                )
            )
        return forward(self, feats, attrs, radial, proj, index, **kw)

    monkeypatch.setattr(CartesianTensorProductConv, "forward", witness)

    def derivatives(m):
        y = m(data)
        loss = y["energy"].square().mean() + y["forces"].square().mean()
        return (y["energy"], y["forces"]), torch.autograd.grad(
            loss, tuple(m.parameters()), allow_unused=True
        )

    expected, actual = derivatives(source), derivatives(model)
    torch.testing.assert_close(actual, expected, rtol=5e-4, atol=3e-5)
    del actual, expected  # Release the default-stream force graphs before capture.
    assert seen == [(6, 11, 11, True)] * 2

    # A late hook can observe or mutate individual directed rows, so restore
    # the original dense call even when someone supplies a stale compact map.
    prepared = model.prepare_property_data(data)
    rows = []
    with rep.interactions[0].edge_info.register_forward_pre_hook(
        lambda _, xs: rows.append(xs[0].shape[0])
    ):
        assert "edge_pair_rows" not in model.prepare_property_data(prepared)
        model(data)
    assert rows == [11] and len(seen) == 2
    with monkeypatch.context() as m:
        m.setenv("TACE_MEMORY_FIRST", "0")
        assert "edge_pair_rows" not in model.prepare_property_data(data)
    differentiable_shifts = dict(
        data, edge_shifts=data["edge_shifts"].double().requires_grad_()
    )
    assert "edge_pair_rows" not in model.prepare_property_data(differentiable_shifts)
    with torch.autograd.forward_ad.dual_level():
        assert "edge_pair_rows" not in model.prepare_property_data(data)
    model.reset_target_property(["energy", "edge_forces"])
    assert "edge_pair_rows" not in model.prepare_property_data(data)

    # The prior private CUDA bucket still uses the directed program; timesteps
    # may change the real number of reciprocal pairs without a new capture.
    from equivariantx_foundation.models.tace.training import (
        BucketedEFSTrainingStep,
        supports_training_buckets,
    )

    model.reset_target_property(["energy", "forces"])
    objective = NormalLoss(
        ["energy", "forces"], ["mse_energy_per_atom", "mse_forces"], [0.3, 0.7]
    )
    if product == "nonlinear_ictd":
        # This radial schedule must not turn the retained Torch factor into
        # a CUDA factor or imply support for the all-CUDA replay contract.
        assert [f.tp.backend for p in rep.products for f in p.aces] == [
            "cuda",
            "torch",
            "cuda",
            "cuda",
        ]
        assert not supports_training_buckets(model)
        with pytest.raises(ValueError):
            BucketedTrainingLoss(model, None, objective, (6, 12, 2))
        return
    assert supports_training_buckets(model)

    class UnknownAdapter(type(model)):
        pass

    with monkeypatch.context() as m:
        m.setattr(model, "__class__", UnknownAdapter)
        assert not supports_training_buckets(model)
    labels = dict(
        data,
        energy=data["positions"].new_tensor([0.1, -0.2]),
        forces=0.05 * data["positions"],
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    bucket = BucketedEFSTrainingStep(
        model, optimizer, labels, max_atoms=6, max_edges=12
    )
    assert captured == [(True, True)] * 2
    assert bucket.step._replay_model.readout_fn is model.readout_fn
    assert "edge_pair_rows" not in bucket.step.static
    for batch in (
        labels,
        {
            **labels,
            "edge_index": data["edge_index"][:, :-2],
            "edge_shifts": data["edge_shifts"][:-2],
        },
    ):
        y = source(batch)
        loss = (
            (y["energy"].reshape(-1) - labels["energy"]) / data["ptr"].diff()
        ).square().mean() + (y["forces"] - labels["forces"]).square().mean()
        params = tuple(source.parameters())
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        updates = [
            p.detach().clone() if g is None else p.detach() - 0.01 * g
            for p, g in zip(params, grads, strict=True)
        ]
        result = bucket(batch)
        torch.testing.assert_close(
            (result["energy"], result["forces"], result["loss"]),
            (y["energy"], y["forces"], loss),
            rtol=1e-3,
            atol=5e-5,
        )
        torch.testing.assert_close(
            [p.grad for p in params], grads, rtol=1e-3, atol=5e-5
        )
        torch.testing.assert_close(
            [p.detach() for p in params], updates, rtol=1e-5, atol=1e-6
        )
    del bucket, y, loss, result

    x = data["positions"]
    labels.update(
        entropy=x.new_tensor([0.5, 1.3]),
        energy_weight=x.new_tensor([1.2, 0.7]),
        forces_weight=x.new_tensor([0.6, 1.1]),
    )
    replay = BucketedTrainingLoss(model, optimizer, objective, (6, 12, 2))
    for key in ("edge_pair_rows", "edge_pair_inverse"):
        with pytest.raises(ValueError, match=key):
            replay({**labels, key: prepared[key]}, objective)
    assert replay.captures == 0
    captured.clear()
    actual, loss = replay(labels, objective)
    assert captured == [(True, True)] * 2
    assert "edge_pair_rows" not in replay.step.static
    expected = source(labels)
    torch.testing.assert_close(
        actual, {k: expected[k] for k in actual}, rtol=5e-4, atol=3e-5
    )
    expected_loss = objective(expected, labels)
    torch.testing.assert_close(loss, expected_loss, rtol=4e-5, atol=1e-6)
    torch.testing.assert_close(
        torch.autograd.grad(loss, params, allow_unused=True),
        torch.autograd.grad(expected_loss, params, allow_unused=True),
        rtol=5e-4,
        atol=3e-5,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cartesian_uu_real_efs_training(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    torch.set_default_dtype(torch.float32)
    host = _local_model("cuda", 64).train()
    fast = shared_parameter_copy(host)
    interaction = fast.readout_fn.representation.interactions[1]
    lower, reason = cartesian_interaction_rule(
        interaction, backend="cuda", task="training"
    )
    assert lower is not None, reason
    lower(interaction)
    calls = []
    interaction.rejector.eqx_tp.register_forward_hook(lambda *args: calls.append(True))
    assert host.state_dict().keys() == fast.state_dict().keys()
    params = dict(host.named_parameters())
    assert all(params[n] is p for n, p in fast.named_parameters())
    data = _periodic(8, "cuda")
    assert data["edge_index"].size(1) == 9216

    def evaluate(model):
        result = model(dict(data))
        loss = (
            (result["energy"] / 512 - 0.1).square().mean()
            + (result["forces"] - 0.2).square().mean()
            + (20 * result["stress"] - 0.3).square().mean()
        )
        grads = torch.autograd.grad(loss, tuple(params.values()), allow_unused=True)
        return ({k: result[k].detach() for k in ("energy", "forces", "stress")}, grads)

    original, planned = evaluate(host), evaluate(fast)
    assert calls, "the real mixed-precision geometry must reach CO2"
    torch.testing.assert_close(original, planned, rtol=8e-4, atol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cartesian UV CUDA program")
def test_cartesian_uv_automatic_scope(monkeypatch, request):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.models.tace.regional import block_rules
    from tace.utils.env import ACCELERATION_ENV

    for name in ACCELERATION_ENV.values():
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    original = torch.get_float32_matmul_precision()
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(original))
    torch.set_float32_matmul_precision("highest")
    model = _local_model("cuda", 64, "uv_o2").float()

    def installs(source, task="training", backend=None):
        regions = plan(source, backend=backend, task=task, regional_only=True).regions
        return [
            region.lower is not None
            for region in regions
            if region.path in ("interactions.0", "interactions.1")
        ]

    first, second = (
        layer.rejector for layer in model.readout_fn.representation.interactions
    )
    assert (first.weight_numel, second.weight_numel) == (128, 1408)
    assert first.local_frame_out.output_dim == 192 and not first.nonlinearity._paths
    for task in ("training", "inference"):
        assert installs(model, task) == [True, True]
    actual = optimize(
        model, family="tace", backend=None, task="training", regional_only=True
    )
    assert [
        m.rejector.eqx_tp is not None
        for m in actual.readout_fn.representation.interactions
    ] == [True, True]
    assert installs(actual) == [False, False]
    rank_rules, _ = block_rules(
        model, backend="cuda", task="inference", automatic=True, rank_local=True
    )
    for layer in model.readout_fn.representation.interactions:
        choose, _ = rank_rules(layer)
        assert choose(layer, backend="cuda", task="inference")[0] is None

    cfg = deepcopy(model.readout_fn.model_config)
    cfg["parity"] = None
    other = TensorModel(CartesianTACE(**cfg)).float().cuda()
    assert other.readout_fn.representation.interactions[1].rejector.weight_numel != 1408
    assert installs(other) == [True, False]
    assert installs(other, backend="cuda") == [True, True]
    with monkeypatch.context() as patch:
        patch.setenv("TACE_USE_EQX", "1")
        host = _local_model("cuda", 64, "uv_o2").float()
        assert (
            host.readout_fn.representation.interactions[1].rejector.eqx_tp is not None
        )
        assert installs(host) == installs(host, backend="cuda") == [False, False]
    with monkeypatch.context() as patch:
        patch.setenv("TACE_MEMORY_FIRST", "0")
        assert installs(model) == [False, False]
    torch.set_float32_matmul_precision("high")
    assert installs(model) == [False, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Cartesian UV CUDA program")
@pytest.mark.parametrize("stage", [0, 1], ids=["scalar", "tensor"])
def test_cartesian_uv_streamed_endpoints_and_frames(monkeypatch, stage):
    from equivariantx_foundation import optimize
    from tace.models.cartesian.local import RotationMatrices
    from tace.utils.env import ACCELERATION_ENV

    for name in ACCELERATION_ENV.values():
        monkeypatch.setenv(name, "0")
    model = _local_model("cuda", 64, "uv_o2").float().train()
    ref = model.readout_fn.representation.interactions[stage].rejector
    ref.local_frame_in.basis_change = False
    with monkeypatch.context() as patch:
        patch.setitem(
            ref.nonlinearity.act_scalars.acts[0]._modules, "f", torch.nn.Softplus()
        )
        retained = optimize(
            model, family="tace", backend="cuda", task="training", regional_only=True
        )
        assert (
            retained.readout_fn.representation.interactions[stage].rejector.eqx_tp
            is None
        )
    fast = optimize(
        model, family="tace", backend="cuda", task="training", regional_only=True
    )
    conv = fast.readout_fn.representation.interactions[stage].rejector
    assert ref.eqx_tp is None and conv.eqx_tp is not None
    assert ref.state_dict().keys() == conv.state_dict().keys()
    assert all(v is dict(ref.named_parameters())[k] for k, v in conv.named_parameters())
    calls = []
    conv.eqx_tp.register_forward_hook(
        lambda _, args, _out: calls.append(args[2].shape[1])
    )

    # 64 distinct directed edges, each repeated 128 times with 1/128 of its
    # cutoff: exact receiver sums match the small, genuinely native reference.
    nodes, edges, copies = 16, 64, 128
    x = torch.randn(nodes, conv.irreps_in.dim, device="cuda", requires_grad=True)
    weights = torch.randn(edges, conv.weight_numel, device="cuda", requires_grad=True)
    index = torch.randint(nodes, (2, edges), device="cuda")
    vectors = torch.randn(edges, 3, device="cuda", requires_grad=True)
    cutoff = torch.rand(edges, 1, device="cuda", requires_grad=True)
    probe = torch.randn(nodes, conv.irreps_out.dim, device="cuda")
    variables = (x, weights, vectors, cutoff, *conv.parameters())

    def evaluate(op, count, distinct=False, derivatives=True):
        frame = RotationMatrices()
        rot = frame(vectors)[0].repeat(count, 1, 1)
        inv = frame(vectors.roll(1, dims=0))[0].repeat(count, 1, 1) if distinct else rot
        out = op(
            x,
            weights.repeat(count, 1),
            index.repeat(1, count),
            rot,
            inv,
            edge_cutoff=cutoff.repeat(count, 1) / count,
        )
        if not derivatives:
            return out
        loss = (out * probe).sum() / nodes
        grad = torch.autograd.grad(loss, variables, create_graph=True)
        mixed = torch.autograd.grad(
            (grad[2] * vectors.detach().cos()).sum(), variables, allow_unused=True
        )
        return out.detach(), tuple(g.detach() for g in grad), mixed

    expected = evaluate(ref, 1)
    actual = evaluate(conv, copies)
    assert calls == [8192]
    torch.testing.assert_close(actual, expected, rtol=8e-4, atol=2e-4)
    with torch.no_grad():
        native = evaluate(ref, 1, distinct=True, derivatives=False)
        result = evaluate(conv, copies, distinct=True, derivatives=False)
    assert calls == [8192]
    assert not torch.allclose(native, expected[0])
    torch.testing.assert_close(result, native, rtol=8e-4, atol=2e-4)


def test_rotation_cast_preserves_distinct_inverse(monkeypatch):
    from tace.models.cartesian.local import RotationMatrices

    monkeypatch.setenv("TACE_USE_EQX", "0")
    torch.set_default_dtype(torch.float32)
    host = _local_model("cpu", 3)

    class IndependentInverse(RotationMatrices):
        def forward(self, vectors):
            rot, _ = super().forward(vectors)
            return rot, rot.clone()

    rep = host.readout_fn.representation
    rep.o2_angular_basis = IndependentInverse()
    seen = []

    def inspect(_, args):
        a, b = args[3:5]
        seen.append((a.dtype, b.dtype, a is b))

    rep.interactions[1].rejector.register_forward_pre_hook(inspect)
    result = host(_periodic(3, "cpu"))
    assert result["forces"].isfinite().all()
    assert seen == [(torch.float32, torch.float32, False)]

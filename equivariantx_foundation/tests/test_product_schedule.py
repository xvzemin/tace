"""Structural product conversion and mixed-gradient parity for unnamed models."""

import pytest
import torch

pytest.importorskip("tace.models.spherical.prod")
from e3nn import o3

from equivariantx_foundation.models.tace.foundation import shared_parameter_copy
from equivariantx_foundation.models.tace.products import product_rule
from tace.models.spherical.prod import CgtpACE, MaceACE, NonlinearCgtpACE


def make_product(kind, degree=2, agnostic=False, channels=3):
    return kind(
        layer=0,
        num_layers=2,
        num_elements=3,
        Lmax=1,
        lmax=1,
        num_channel=channels,
        num_expert=None,
        num_channel_per_expert=None,
        target_irreps=o3.Irreps("0e+1o"),
        irreps_in=o3.Irreps(f"{channels}x0e+{channels}x1o"),
        correlation=[degree, degree],
        l1l2=None,
        bias=True,
        nonlinear=None,
        parity=True,
        agnostic=agnostic,
    )


@pytest.mark.parametrize(
    "kind,degree,agnostic",
    [(CgtpACE, 2, False), (CgtpACE, 3, True), (NonlinearCgtpACE, 3, False)],
)
@pytest.mark.parametrize(
    "backend,device", [("torch", "cpu"), ("torch", "cuda"), ("cuda", "cuda")]
)
def test_local_product_execution(
    monkeypatch, double_precision, kind, degree, agnostic, backend, device
):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    source = make_product(kind, degree, agnostic).to(device)
    with torch.no_grad():
        for name, p in source.named_parameters():
            if name.endswith("bias"):
                p.normal_(std=0.2)
    lower, _ = product_rule(source, backend=backend, task="training")
    candidate = lower(shared_parameter_copy(source))
    assert candidate.state_dict().keys() == source.state_dict().keys()
    before = dict(source.named_parameters())
    after = dict(candidate.named_parameters())
    assert before.keys() == after.keys() and all(
        p is after[n] for n, p in before.items()
    )
    candidate.load_state_dict(source.state_dict(), strict=True)
    assert type(source) is kind
    assert not hasattr(source, "scheduled_contraction")
    x = torch.randn(7, source.irreps_in.dim, device=device, requires_grad=True)
    types = torch.tensor([0, 1, 2, 0, 1, 2, 1], device=device)
    batch = torch.zeros_like(types)
    residual = torch.randn(7, source.irreps_out.dim, device=device)

    def evaluate(layer):
        pred = layer(x, types, residual, batch)
        (deriv,) = torch.autograd.grad(pred.square().sum(), x, create_graph=True)
        mixed = torch.autograd.grad(
            deriv.square().sum(), (x, *before.values()), allow_unused=True
        )
        return pred, deriv, *mixed

    actual, expected = evaluate(candidate), evaluate(source)
    for a, b in zip(actual, expected, strict=True):
        if a is None or b is None:
            assert a is b
        else:
            torch.testing.assert_close(a, b, atol=1e-9, rtol=1e-9)


def test_unsupported_products_are_left_native(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")

    class DifferentForward(CgtpACE):
        pass

    hooked = make_product(CgtpACE)
    hooked.coefs[0].register_forward_hook(lambda module, inputs, output: output * 2)
    for module in (
        make_product(DifferentForward),
        make_product(CgtpACE, degree=1),
        hooked,
    ):
        assert product_rule(module, backend="torch", task="inference")[0] is None


def make_bilinear(experts, shared, agnostic, *, matrix=False, elements=4):
    from tace.models.linear import switch_e3nn_weight_layout
    from tace.models.spherical.tece_oam_rra import BilinearMoEACE

    layer = BilinearMoEACE(
        layer=0,
        num_layers=2,
        num_elements=elements,
        Lmax=1,
        lmax=1,
        num_channel=4,
        num_expert=experts,
        num_channel_per_expert=4 // experts,
        target_irreps=o3.Irreps("0e+1o"),
        irreps_in=o3.Irreps("4x0e+4x1o"),
        correlation=[2, 2],
        l1l2=None,
        bias=True,
        nonlinear="bilinear_gate",
        parity=True,
        agnostic=agnostic,
        use_shared_expert=shared,
    )
    return switch_e3nn_weight_layout(layer, "matrix" if matrix else "flat")


@pytest.mark.parametrize("existing", [False, True])
def test_bilinear_host_selection(monkeypatch, existing):
    from equivariantx_foundation.models.tace.products import host_product_provider

    monkeypatch.setenv("TACE_USE_EQX", str(int(existing)))
    p = make_bilinear(2, True, False)
    assert hasattr(p, "eqx_ace") == existing
    assert host_product_provider(p) is None  # The host never calls eqx_ace on CPU.
    assert product_rule(p, backend="cuda", task="training")[0] is None
    if torch.cuda.is_available():
        p.cuda()
        assert host_product_provider(p) == ("eqx_cuda" if existing else None)
        assert product_rule(p, backend="torch", task="training")[0] is None
        assert (
            product_rule(p, backend="cuda", task="training", automatic=True)[0] is None
        )
        assert (product_rule(p, backend="cuda", task="training")[0] is None) == existing
        if not existing:
            with monkeypatch.context() as m:
                m.setattr(p, "_merge_shared_expert", lambda a, b: a - b)
                assert product_rule(p, backend="cuda", task="training")[0] is None
            with monkeypatch.context() as m:
                m.setattr(p.coefs[1], "torch_linear", torch.nn.Identity())
                assert product_rule(p, backend="cuda", task="training")[0] is None
            with monkeypatch.context() as m:
                m.setattr(p.shared_coefs[1], "use_lora", True, raising=False)
                m.setattr(p.shared_coefs[1], "lora_A", object(), raising=False)
                assert product_rule(p, backend="cuda", task="training")[0] is None
        p.aces[0].register_forward_hook(lambda module, args, output: 2 * output)
        assert product_rule(p, backend="cuda", task="training")[0] is None


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native CUDA bilinear contraction"
)
@pytest.mark.parametrize(
    "experts,shared,agnostic,matrix,elements",
    [
        (1, False, True, False, 4),
        (1, False, False, True, 4),
        (2, False, False, False, 4),
        (2, True, False, True, 4),
        (2, True, False, True, 32),
    ],
)
def test_bilinear_live_second_order(
    monkeypatch, double_precision, experts, shared, agnostic, matrix, elements
):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    p = make_bilinear(
        experts, shared, agnostic, matrix=matrix, elements=elements
    ).cuda()
    with torch.no_grad():
        for n, v in p.named_parameters():
            if n.endswith("bias"):
                v.normal_(std=0.1)
    lower, _ = product_rule(p, backend="cuda", task="training")
    q = lower(shared_parameter_copy(p))
    assert list(q.state_dict()) == list(p.state_dict())
    q.load_state_dict(p.state_dict(), strict=True)
    params = tuple(p.parameters())
    assert all(a is b for a, b in zip(params, q.parameters(), strict=True))
    assert not tuple(q.scheduled_contraction.parameters())
    assert not q.scheduled_contraction.state_dict()
    assert not hasattr(p, "scheduled_contraction")
    x = torch.randn(6, p.irreps_in.dim, device="cuda", requires_grad=True)
    sc = torch.randn(6, p.irreps_out.dim, device="cuda", requires_grad=True)
    direction = torch.randn_like(x)
    types = torch.tensor([3, 1, 0, 1, 2, 3], device="cuda")
    calls = []
    q.scheduled_contraction.register_forward_hook(
        lambda m, a, out: calls.append((a[3].shape[0], a[4].clone()))
    )

    def run(layer):
        y = layer(x, types, sc, torch.zeros_like(types))
        energy = y.square().mean()
        (force,) = torch.autograd.grad(energy, x, create_graph=True)
        grads = torch.autograd.grad(
            energy + (force * direction).sum(), (x, sc, *params)
        )
        return y.detach(), force.detach(), grads

    history = []
    for step in range(2):
        a, da, ga = run(p)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as prof:
            b, db, gb = run(q)
        assert len(calls) == step + 1
        rows, routing = calls[-1]
        compact = elements >= 4 * types.numel()
        assert rows == (types.numel() if compact else 1 if agnostic else elements)
        expected_types = (
            torch.arange(types.numel(), device=types.device)
            if compact
            else torch.zeros_like(types)
            if agnostic
            else types
        )
        torch.testing.assert_close(routing, expected_types)
        assert any(e.key == "eqx::bilinear_ace" for e in prof.key_averages())
        for expected, actual in zip((a, da, *ga), (b, db, *gb), strict=True):
            torch.testing.assert_close(expected, actual, atol=5e-11, rtol=5e-10)
        history.append(b)
        if step == 0:
            with torch.no_grad():
                for param, grad in zip(params, gb[2:], strict=True):
                    param.add_(grad, alpha=-0.02)
            types = types.roll(1)
    assert not torch.allclose(*history)
    if compact:
        q.eval()
        with torch.no_grad():
            result = q(x, types, sc, torch.zeros_like(types))
            torch.testing.assert_close(result, p(x, types, sc, torch.zeros_like(types)))
        rows, routing = calls[-1]
        assert rows == elements  # Inference already needs no species-table adjoints.
        torch.testing.assert_close(routing, types)


def test_existing_eqx_schedule_is_preserved(monkeypatch):
    from equivariantx_foundation.models.tace.products import host_product_provider

    monkeypatch.delenv("TACE_USE_EQX", raising=False)
    source = make_product(CgtpACE).float()
    lower, reason = product_rule(source, backend="cuda", task="inference")
    assert lower is None and "existing" in reason
    assert host_product_provider(source) == "eqx_torch"
    if torch.cuda.is_available():
        assert host_product_provider(source.cuda()) == "eqx_cuda"


def test_nonlinear_torch_is_opt_in(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    source = make_product(NonlinearCgtpACE)
    assert (
        product_rule(source, backend="torch", task="training", automatic=True)[0]
        is None
    )
    assert (
        product_rule(source, backend="torch", task="training", automatic=False)[0]
        is not None
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_existing_cuda_nonlinear_schedule_is_preserved(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "1")
    source = make_product(NonlinearCgtpACE).float().cuda()
    for backend in ("torch", "cuda"):
        lower, reason = product_rule(source, backend=backend, task="training")
        assert lower is None and "existing" in reason


@pytest.mark.parametrize("host,other", [("torch", "cuda"), ("cuda", "torch")])
def test_mace_selects_existing_program_without_rebuilding(monkeypatch, host, other):
    monkeypatch.setattr("equivariantx.ops.config.DEFAULT_BACKEND", host)
    monkeypatch.setenv("TACE_USE_EQX", "1")
    monkeypatch.setenv("TACE_USE_CUE", "0")
    source = make_product(MaceACE, degree=3)
    native, reason = product_rule(source, backend=host, task="inference")
    assert native is None and "already" in reason
    lower, _ = product_rule(source, backend=other, task="training")
    clone = shared_parameter_copy(source)
    program = clone.product
    candidate = lower(clone)
    assert candidate.product is program and candidate.product.backend == other
    assert source.product.backend == host and candidate.weight is source.weight
    assert candidate.state_dict().keys() == source.state_dict().keys()
    assert candidate.product.metadata == source.product.metadata
    candidate.product.register_forward_hook(lambda module, args, output: output * 2)
    assert product_rule(candidate, backend=host, task="inference")[0] is None


def test_existing_mace_cue_provider_is_preserved(monkeypatch):
    pytest.importorskip("cuequivariance_torch")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_USE_CUE", "1")
    source = make_product(MaceACE)
    assert source.product.use_cue and source.product.fused_tp is not None
    for backend in ("torch", "cuda"):
        lower, reason = product_rule(source, backend=backend, task="training")
        assert lower is None and "CuEq" in reason


@pytest.mark.parametrize(
    "host,device,backend,provider",
    [
        ("torch", "cpu", "torch", "eqx_torch"),
        ("cueq", "cpu", "torch", "eqx_torch"),
        ("eqx", "cuda", "cuda", "eqx_cuda"),
        ("cueq", "cuda", "torch", "cueq"),
    ],
)
def test_plan_identifies_host_mace_provider(
    monkeypatch, host, device, backend, provider
):
    from unittest.mock import patch

    from equivariantx.ops.cuda.mbe.ace import symmetric_contraction as cuda_kernel
    from equivariantx.ops.torch.mbe.ace import symmetric_contraction as torch_kernel
    from equivariantx_foundation.models.tace.planner import plan
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.default import DEFAULT_MODEL_CONFIG
    from tace.utils.env import ACCELERATION_ENV

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if host == "cueq":
        pytest.importorskip("cuequivariance_torch")
    for variable in ACCELERATION_ENV.values():
        monkeypatch.setenv(variable, "0")
    monkeypatch.setenv("TACE_USE_EQX", "1" if host == "eqx" else "0")
    monkeypatch.setenv("TACE_USE_CUE", "1" if host == "cueq" else "0")
    monkeypatch.setitem(DEFAULT_MODEL_CONFIG["product_basis"], "type", "mace")
    source = make_model().eval().to(device)
    report = plan(source, backend=backend)
    regions = [
        r
        for r in report.regions
        if r.path.startswith("products.") and r.path.count(".") == 1
    ]
    assert regions and all(
        (r.implementation, r.provenance, r.provider)
        == ("native", "host_selected", provider)
        for r in regions
    )
    layer = source.readout_fn.representation.products[0]
    op = layer.product
    if provider == "cueq":
        target, member = op.fused_tp, "forward"
    else:
        target, member = (
            (cuda_kernel if provider == "eqx_cuda" else torch_kernel),
            "uuu",
        )
    with patch.object(target, member, wraps=getattr(target, member)) as hit:
        x = layer.weight.new_zeros(3, layer.irreps_in.dim)
        types = torch.tensor([0, 1, 0], device=device)
        layer(x, types, None, torch.zeros_like(types))
        hit.assert_called_once()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_mace_plan_retarget_efs_and_live_sgd(monkeypatch, double_precision):
    from equivariantx.ops.cuda.mbe.ace import symmetric_contraction as cuda_kernel
    from equivariantx.ops.torch.mbe.ace import symmetric_contraction as torch_kernel
    from equivariantx_foundation import optimize
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        efs_loss,
        graph,
        make_model,
    )
    from tace.models.spherical.default import DEFAULT_MODEL_CONFIG
    from tace.utils.env import ACCELERATION_ENV

    for variable in ACCELERATION_ENV.values():
        monkeypatch.setenv(variable, "0")
    monkeypatch.setitem(DEFAULT_MODEL_CONFIG["product_basis"], "type", "mace")
    source, reference = (make_model().cuda().train() for _ in range(2))
    reference.load_state_dict(source.state_dict())
    params = dict(source.named_parameters())
    # The optimizer predates all of the retargetings.
    optimizer = torch.optim.SGD(source.parameters(), lr=0.005)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.005)
    data = graph("cuda")
    expected = reference(data)
    active = source
    called = []
    for kernel, name in ((torch_kernel, "torch"), (cuda_kernel, "cuda")):

        def observe(*args, name=name, forward=kernel.uuu):
            called.append(name)
            return forward(*args)

        monkeypatch.setattr(kernel, "uuu", observe)
    for backend in ("cuda", "torch", "cuda"):
        active = optimize(active, family="tace", task="training", backend=backend)
        assert (
            optimize(active, family="tace", task="training", backend=backend) is active
        )
        assert active.state_dict().keys() == source.state_dict().keys()
        assert all(param is params[key] for key, param in active.named_parameters())
        assert all(
            p.product.backend == "torch"
            for p in source.readout_fn.representation.products
        )
        called.clear()
        got = active(data)
        assert called == [backend] * len(active.readout_fn.representation.products)
        for key in EFS:
            torch.testing.assert_close(got[key], expected[key], rtol=1e-7, atol=1e-9)
    efs_loss(expected).backward()
    efs_loss(got).backward()
    ref_params = dict(reference.named_parameters())
    assert active.readout_fn.representation.products[0].weight.grad.abs().max() > 0
    for name, param in params.items():
        torch.testing.assert_close(
            param.grad, ref_params[name].grad, rtol=2e-7, atol=1e-9
        )
    optimizer.step()
    ref_optimizer.step()
    for name, param in params.items():
        torch.testing.assert_close(param, ref_params[name], rtol=2e-7, atol=1e-9)
    updated, ref_updated = active(data), reference(data)
    assert not torch.equal(updated["energy"], got["energy"])
    for key in EFS:
        torch.testing.assert_close(updated[key], ref_updated[key], rtol=2e-7, atol=1e-9)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph training")
@pytest.mark.parametrize("kind", ["mace", "nonlinear_cgtp", "bilinear"])
def test_spherical_product_training_bucket(kind, monkeypatch):
    from equivariantx.ops.cuda.mbe.ace import bilinear, symmetric_contraction
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import products, training
    from equivariantx_foundation.models.tace.replay_loss import _validate_model
    from equivariantx_foundation.tests.test_regional_aoti import _cases
    from equivariantx_foundation.tests.test_tace_plan import EFS, make_model
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    torch.set_default_dtype(torch.float32)

    def build(**cfg):
        if kind == "bilinear":
            cfg["num_channel"] = 4
            cfg["product_basis"].update(
                type="cgtp",
                nonlinear="bilinear_gate",
                correlation=2,
                num_expert=2,
                num_channel_per_expert=2,
                use_shared_expert=True,
            )
        else:
            cfg["product_basis"].update(type=kind, correlation=3)
        return SphericalTACE(**cfg)

    src, ref = (make_model(build).train().cuda() for _ in range(2))
    ref.load_state_dict(src.state_dict())
    opt, ref_opt = (torch.optim.SGD(m.parameters(), lr=0.02) for m in (src, ref))
    model = optimize(src, family="tace", backend="cuda", task="training")
    rep = model.readout_fn.representation
    assert training.supports_training_buckets(model)
    assert _validate_model(model) == EFS
    assert not products.spherical_bucket_product(
        ref.readout_fn.representation.products[0]
    )
    first = rep.products[0]
    if kind == "mace":
        target, kernel = symmetric_contraction, "uuu"
        with monkeypatch.context() as mp:
            mp.setattr(first.product, "use_cue", True)
            assert not training.supports_training_buckets(model)
    elif kind == "bilinear":
        target, kernel = bilinear, "evaluate"
        with monkeypatch.context() as mp:
            mp.setattr(first.scheduled_contraction.shared, "backend", "torch")
            assert not training.supports_training_buckets(model)
    else:
        target, kernel = products, "channel_product"
        with monkeypatch.context() as mp:
            mp.setattr(
                first, "eqx_metadata", "different whole-product fusion", raising=False
            )
            assert not training.supports_training_buckets(model)
        with monkeypatch.context() as mp:
            mp.setattr(first.aces[0], "schedule_backend", "torch")
            assert not training.supports_training_buckets(model)

    cases = _cases()
    for i, b in enumerate(cases):
        b["energy"] = b["positions"].new_tensor([0.2, -0.3]) + i * 0.1
        b["forces"] = 0.15 * b["positions"].sin()
        b["stress"] = 0.02 * b["lattice"]
    raw_pack = training._pack

    def active(data, *args, **kwargs):
        packed = raw_pack(data, *args, **kwargs)
        packed["lattice"][-1] = torch.diag(
            packed["lattice"].new_tensor([0.8, 1.1, 1.3])
        )
        packed["energy"][-1] = 110
        packed["forces"][len(data["positions"]) :] = -15
        packed["stress"][-1] = 20
        return packed

    monkeypatch.setattr(training, "_pack", active)
    called = []
    implementation = getattr(target, kernel)

    def observe(*args, **kwargs):
        called.append(True)
        return implementation(*args, **kwargs)

    monkeypatch.setattr(target, kernel, observe)
    bucket = training.BucketedEFSTrainingStep(
        model, opt, cases[0], max_atoms=6, max_edges=12
    )
    assert len(called) >= sum(
        len(p.aces) if kind == "nonlinear_cgtp" else 1 for p in rep.products
    )
    graph = bucket.step.graph
    for data in cases:
        ref_opt.zero_grad(set_to_none=True)
        expected = ref(dict(data))
        loss = (
            ((expected["energy"] - data["energy"]) / data["ptr"].diff()).square().mean()
        )
        for k in ("forces", "stress"):
            loss = loss + (expected[k] - data[k]).square().mean()
        loss.backward()
        ref_opt.step()
        result = bucket(data)
        assert bucket.step.graph is graph
        for k in EFS:
            torch.testing.assert_close(
                result[k], expected[k], atol=3e-5, rtol=4e-4, msg=k
            )
        torch.testing.assert_close(result["loss"], loss, atol=3e-5, rtol=4e-4)
        for (name, p), (ref_name, q) in zip(
            model.named_parameters(), ref.named_parameters(), strict=True
        ):
            assert name == ref_name and (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(
                    p.grad, q.grad, atol=3e-5, rtol=4e-4, msg=name
                )
            torch.testing.assert_close(p, q, atol=2e-6, rtol=3e-5, msg=name)
        for product in rep.products:
            if kind == "mace":
                live = (product.weight,)
            elif kind == "bilinear":
                live = (product.coefs[1].weight, product.shared_coefs[1].weight)
            else:
                live = (product.embedding.weight, product.gate[0].weight)
            assert all(p.grad is not None and p.grad.abs().max() > 0 for p in live)


@pytest.mark.parametrize("existing_fusion", [False, True])
def test_tace_path_gate_portable_and_existing_program(
    monkeypatch, double_precision, existing_fusion
):
    from equivariantx_foundation.models.tace.products import host_product_provider
    from tace.models.spherical.prod import TaceACE

    monkeypatch.setenv("TACE_USE_EQX", str(int(existing_fusion)))
    module = make_product(TaceACE, degree=3)
    assert hasattr(module, "eqx_metadata") == existing_fusion
    assert (
        host_product_provider(module) == "eqx_torch"
    )  # CPU path, even with prepared CUDA metadata
    for backend in ("torch", "cuda"):
        assert product_rule(module, backend=backend, task="training")[0] is None
    x = torch.randn(5, module.irreps_in.dim, requires_grad=True)
    y = module(x, torch.zeros(5, dtype=torch.long), None, None)
    (grad,) = torch.autograd.grad(y.square().sum(), x, create_graph=True)
    mixed = torch.autograd.grad(grad.square().sum(), (x, *module.parameters()))
    assert all(g.isfinite().all() and g.abs().max() > 0 for g in mixed)


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="native CUDA path-gate fusion"
)
def test_tace_path_gate_regional_live_derivatives_and_retarget(
    monkeypatch, double_precision
):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import nonlinear_ace
    from equivariantx_foundation.models.tace.products import _ScheduledPathGates
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        efs_loss,
        graph,
        make_model,
    )
    from tace.models.spherical.default import DEFAULT_MODEL_CONFIG
    from tace.models.spherical.prod import TaceACE
    from tace.utils.env import ACCELERATION_ENV

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setitem(DEFAULT_MODEL_CONFIG["product_basis"], "type", "tace")
    monkeypatch.setitem(DEFAULT_MODEL_CONFIG["product_basis"], "correlation", 3)
    source, reference = (make_model().double().cuda().train() for _ in range(2))
    with torch.no_grad():
        for name, p in source.named_parameters():
            if name.startswith("readout_fn.representation.products.") and name.endswith(
                "bias"
            ):
                p.normal_(std=0.1)
    reference.load_state_dict(source.state_dict())
    opt, ref_opt = (
        torch.optim.SGD(m.parameters(), lr=0.01) for m in (source, reference)
    )
    native = source.readout_fn.representation.products[0]
    assert (
        product_rule(native, backend="cuda", task="training", automatic=True)[0] is None
    )
    with native.generalized.blocks[0].register_forward_hook(lambda *_: None):
        assert product_rule(native, backend="cuda", task="training")[0] is None
    with monkeypatch.context() as mp:
        mp.setitem(
            native.linear_up._buffers, "_bias_mask", 2 * native.linear_up._bias_mask
        )
        assert product_rule(native, backend="cuda", task="training")[0] is None
    with monkeypatch.context() as mp:
        mp.setattr(native, "fully_connected", True)
        assert product_rule(native, backend="cuda", task="training")[0] is None
    model = optimize(source, family="tace", backend="cuda", task="training")
    assert all(
        type(p) is _ScheduledPathGates for p in model.readout_fn.representation.products
    )
    assert all(
        a is b for a, b in zip(model.parameters(), source.parameters(), strict=True)
    )
    assert not hasattr(native, "_foundation_path_program")
    assert model.state_dict().keys() == source.state_dict().keys()
    assert optimize(model, family="tace", backend="cuda", task="training") is model
    called = []
    original = nonlinear_ace.run

    def audited(module, *args, **kwargs):
        called.append(True)
        return original(module, *args, **kwargs)

    monkeypatch.setattr(nonlinear_ace, "run", audited)

    data = graph("cuda")
    for _ in range(2):
        ref_opt.zero_grad(set_to_none=True)
        opt.zero_grad(set_to_none=True)
        expected, actual = reference(data), model(data)
        for key in EFS:
            torch.testing.assert_close(actual[key], expected[key], rtol=3e-7, atol=2e-9)
        efs_loss(expected).backward()
        efs_loss(actual).backward()
        for p, q in zip(model.parameters(), reference.parameters(), strict=True):
            torch.testing.assert_close(p.grad, q.grad, rtol=4e-7, atol=2e-9)
        opt.step()
        ref_opt.step()
        # The adapter enables coordinate differentiation in place. Start a new
        # physical leaf for the next step, independent of both backward tapes.
        data["positions"] = data["positions"].detach() * 0.98
    assert len(called) == 4
    restored = optimize(model, family="tace", backend="torch", task="training")
    assert all(
        type(p) is TaceACE and not hasattr(p, "_foundation_path_program")
        for p in restored.readout_fn.representation.products
    )
    actual, expected = restored(data), reference(data)
    for key in EFS:
        torch.testing.assert_close(actual[key], expected[key], rtol=4e-7, atol=2e-9)
    assert len(called) == 4

"""Parity of a scheduled, non-release O3 interaction with the original layer."""

import io
from functools import wraps
from types import SimpleNamespace

import pytest
import torch
from e3nn import o3

from equivariantx_foundation.models.tace.foundation import shared_parameter_copy
from equivariantx_foundation.models.tace.interactions import (
    StreamedInteraction,
    canonical_geometry,
    interaction_rule,
)
from equivariantx_foundation.models.tace.matching import (
    canonical_component_harmonics,
    custom_forward,
    customized_codegen,
)
from tace.models.spherical.inter import (
    O2CgtpInteraction,
    O3CgtpInteraction,
    O3IctdInteraction,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_component_polynomial_preserves_derivatives_and_host(device):
    from equivariantx_foundation.models.tace.interactions import (
        restore_polynomial,
        schedule_polynomial,
    )

    if device == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("requires NVIDIA CUDA")
    basis = o3.SphericalHarmonics(list(range(4)), False, "component")
    if not hasattr(basis, "sph_func"):
        pytest.skip("this e3nn release does not expose its polynomial evaluator")
    original = basis.sph_func
    v = torch.randn(11, 6, device=device, dtype=torch.float64)[:, ::2].requires_grad_()
    cot = torch.randn(11, 16, device=device, dtype=v.dtype)

    def derivatives():
        value = basis(v)
        g = torch.autograd.grad((value * cot).sum(), v, create_graph=True)[0]
        h = torch.autograd.grad(g.sum(), v, create_graph=True)[0]
        d3 = torch.autograd.grad(h.sum(), v)[0]
        return value, g, h, d3

    expected = derivatives()
    schedule_polynomial(basis)
    assert canonical_component_harmonics(basis, 3)
    assert shared_parameter_copy(basis).sph_func is basis.sph_func
    observed = []
    with basis.register_forward_hook(lambda m, a, y: observed.append(y.shape)):
        for y, ref in zip(derivatives(), expected, strict=True):
            torch.testing.assert_close(y, ref, atol=1e-11, rtol=1e-11)
    assert observed == [(11, 16)]
    assert basis(v[:0]).shape == (0, 16)
    assert basis(v[:1]).shape == (1, 16)
    restore_polynomial(basis)
    assert basis.sph_func is original


def test_component_polynomial_selection(monkeypatch):
    from equivariantx_foundation.models.tace.interactions import harmonic_region
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV

    for name in ACCELERATION_ENV.values():
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")

    def host(**cfg):
        cfg.update(num_channel=64, lmax=3, mmax=3, Lmax=2)
        return SphericalTACE(**cfg)

    rep = make_model(host).float().readout_fn.representation
    basis = rep.o3_angular_basis
    if not hasattr(basis, "sph_func") or torch.version.hip:
        pytest.skip("requires replaceable e3nn polynomials and a non-ROCm build")
    original = basis.sph_func
    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        assert harmonic_region(rep, "torch", automatic=True) is None
        assert harmonic_region(rep, "cuda", automatic=True).lower is not None
        with basis.register_forward_hook(lambda *_: None):
            assert harmonic_region(rep, "cuda", automatic=True) is None
        torch.set_float32_matmul_precision("high")
        assert harmonic_region(rep, "cuda", automatic=True) is None
        pinned = harmonic_region(rep, "cuda", automatic=False)
        assert basis.sph_func is original
        pinned.lower(basis)
        assert harmonic_region(rep, "cuda", automatic=False).lower is None
        default = harmonic_region(rep, "cuda", automatic=True)
        assert default.provider == "e3nn"
        default.lower(basis)
        assert basis.sph_func is original
        monkeypatch.setenv("TACE_MEMORY_FIRST", "0")
        assert harmonic_region(rep, "cuda", automatic=False) is None
    finally:
        torch.set_float32_matmul_precision(precision)


def ordinary_layer(width=4, lmax=1, *, interaction=O3CgtpInteraction, layer=1):
    irreps = o3.Irreps([(width, (l, (-1) ** l)) for l in range(lmax + 1)])
    return interaction(
        layer=layer,
        num_layers=3,
        num_elements=3,
        avg_num_neighbors=7,
        mmax=lmax,
        Lmax=lmax,
        lmax=lmax,
        correlation=[2] * 3,
        num_channel=width,
        edge_feats_channel=5,
        num_radial_basis=5,
        target_irreps=o3.Irreps("0e"),
        radial_mlp=[8],
        radial_bias=True,
        irreps_in=irreps,
        scalar_act="silu",
        tensor_act="sigmoid",
        edge_ace_hidden=None,
        nonlinear="gate",
        resnet_type="BB",
        resnet_linear_type="aware",
        scatter_norm="cutoff",
        mlp_type="mlp_v2",
    )


def layer_inputs(layer, nodes, edges, *, dtype, device):
    def rand(n, width):
        return torch.randn(n, width, dtype=dtype, device=device).requires_grad_()

    x = rand(nodes, layer.irreps_in.dim)
    # Deliberately not unit-vector harmonics; use *these* attributes verbatim.
    y = rand(edges, layer.irreps_sh.dim)
    r = rand(edges, layer.edge_feats_channel)
    c = rand(edges, 1)
    species = torch.arange(nodes, device=device) % layer.num_elements
    idx = torch.randint(nodes - 1, (2, edges), device=device)
    batch = torch.zeros_like(species)
    graph = SimpleNamespace(lmp_natoms=(nodes, 0), lmp_data=None)
    args = (x, species, species, r, r, y, idx, c, None, None, None, None, batch, graph)
    return args, (x, y, r, c)


@pytest.mark.parametrize("interaction", [O3IctdInteraction, O2CgtpInteraction])
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cgtp_owned_layer_cut(
    monkeypatch, double_precision, interaction, streamed, device
):
    """Same first-layer halo senders; change only the allocation at the message cut."""
    from equivariantx.co3 import CartesianHarmonics
    from tace.utils.env import ACCELERATION_ENV

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    for name in ACCELERATION_ENV.values():
        monkeypatch.setenv(name, "0")
    monkeypatch.setenv("TACE_USE_EQX", str(int(streamed)))
    layer = ordinary_layer(2, 2, interaction=interaction, layer=0).to(device)
    assert layer.use_eqx == streamed
    owners, senders, edges = 3, 5, 9
    args, (x, _, radial, cutoff) = layer_inputs(
        layer, senders, edges, dtype=torch.float64, device=device
    )
    args = list(args)
    vectors = torch.randn(edges, 3, device=device, requires_grad=True)
    lengths = vectors.norm(dim=-1, keepdim=True)
    if interaction is O3IctdInteraction:
        args[5] = CartesianHarmonics(layer.irreps_sh, True, "component").to(x)(vectors)
    args[6] = torch.tensor(
        [[4, 3, 1, 4, 0, 1, 2, 3, 4], [0, 1, 1, 0, 1, 0, 0, 1, 0]], device=device
    )
    halo = SimpleNamespace(
        edge_vector=vectors,
        edge_length=lengths,
        lmp_natoms=(owners, senders - owners),
        lmp_data=object(),
    )
    args[-1] = halo
    ordinary = SimpleNamespace(**(vars(halo) | {"lmp_data": None}))
    message = layer._compute_messages
    inputs = (x, vectors, radial, cutoff, *layer.parameters())

    def run(compact):
        def observed(*a, graph, **kw):
            y = message(*a, graph=graph if compact else ordinary, **kw)
            assert y.shape[0] == (owners if compact else senders)
            assert y.untyped_storage().nbytes() == y.numel() * y.element_size()
            return y

        monkeypatch.setattr(layer, "_compute_messages", observed)
        output, residual = layer(*args)
        assert output.shape[0] == owners and residual is None
        loss = output.sin().sum()
        first = torch.autograd.grad(loss, inputs, create_graph=True)
        assert first[0][owners:].abs().sum() > 0
        mixed = torch.autograd.grad(
            first[1].square().sum(), inputs, allow_unused=True, retain_graph=True
        )
        return output, residual, first, mixed

    torch.testing.assert_close(run(True), run(False), atol=1e-8, rtol=1e-8)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_layer_with_gate_cutoff_and_residual(monkeypatch, backend):
    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    device = "cpu" if backend == "torch" else "cuda:0"
    torch.set_default_dtype(torch.float64)
    layer = ordinary_layer().to(device)
    apply, reason = interaction_rule(layer, backend=backend, task="training")
    assert apply is not None, reason
    fast = apply(shared_parameter_copy(layer))
    again, detail = interaction_rule(fast, backend=backend, task="training")
    assert again is None and "already uses" in detail
    assert tuple(fast.state_dict()) == tuple(layer.state_dict())
    assert all(
        a is b for a, b in zip(layer.parameters(), fast.parameters(), strict=True)
    )
    args, inputs = layer_inputs(layer, 9, 35, dtype=torch.float64, device=device)
    operands = (*inputs, *layer.parameters())

    def run(model):
        out, residual = model(*args)
        e = out.sin().sum() + residual.cos().sum()
        first = torch.autograd.grad(e, operands, create_graph=True)
        mixed = torch.autograd.grad(
            sum(a.square().sum() for a in first[:4]), operands, allow_unused=True
        )
        return out, residual, first, mixed

    torch.testing.assert_close(run(fast), run(layer), rtol=1e-8, atol=1e-8)


def test_preserves_existing_streaming_eqx(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "1")
    layer = ordinary_layer()
    for backend in ("torch", "cuda"):
        apply, reason = interaction_rule(layer, backend=backend, task="inference")
        assert apply is None and "already-enabled streamed EQX" in reason


def test_preserves_customized_tensor_product(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    layer = ordinary_layer()
    # Hooks are part of the native computation and may modify the result.
    layer.rejector.tp.register_forward_hook(lambda mod, args, y: y * 0.5)
    apply, reason = interaction_rule(layer, backend="torch", task="training")
    assert apply is None and "customized" in reason


def test_rejector_on_empty_graph(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    layer = ordinary_layer().double()
    apply, _ = interaction_rule(layer, backend="torch", task="inference")
    fast = apply(shared_parameter_copy(layer))
    args, _ = layer_inputs(
        layer, 4, 0, dtype=next(layer.parameters()).dtype, device="cpu"
    )
    torch.testing.assert_close(fast(*args), layer(*args))


def test_warm_e3nn_script_cache_is_not_an_instance_override(monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    layer = ordinary_layer().double()
    tp = layer.rejector.tp
    kernel = tp._compiled_main_left_right
    if not isinstance(kernel, torch.jit.ScriptModule):
        kernel = torch.jit.script(kernel)
    # Loading is how e3nn's CodeGenMixin restores its saved TorchScript program.
    image = io.BytesIO()
    torch.jit.save(kernel, image)
    program = image.getvalue()
    tp._compiled_main_left_right = loaded = torch.jit.load(io.BytesIO(program))
    assert "forward" not in vars(loaded)
    args, _ = layer_inputs(layer, 7, 19, dtype=torch.float64, device="cpu")

    def matched():
        return interaction_rule(layer, backend="torch", task="training")[0] is not None

    assert matched() and "forward" not in vars(loaded)  # inspecting is read-only
    expected = layer(*args)
    assert isinstance(vars(loaded)["forward"], torch.ScriptMethod)
    assert matched() and not customized_codegen(layer)
    lower, _ = interaction_rule(layer, backend="torch", task="training")
    fast = lower(shared_parameter_copy(layer))
    torch.testing.assert_close(fast(*args), expected)

    # The exact same code bound to another module is a genuine override.
    other = torch.jit.load(io.BytesIO(program))
    loaded.forward = other.forward
    assert not matched() and customized_codegen(layer)
    loaded.forward = lambda *args: other(*args) * 2
    assert not matched() and customized_codegen(layer)


@pytest.mark.parametrize(
    "register",
    ["register_full_backward_pre_hook", "register_full_backward_hook"],
)
def test_backward_hooks_stay_on_the_native_tensor_product(monkeypatch, register):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    layer = ordinary_layer().double()
    tp = layer.rejector.tp
    seen = []
    with getattr(tp, register)(lambda *args: seen.append(True)):
        assert custom_forward(layer, recursive=True)
        # A normal hook on TensorProduct itself does not live in __codegen__.
        assert not customized_codegen(layer)
        lower, reason = interaction_rule(layer, backend="torch", task="training")
        assert lower is None and "customized" in reason
        args, _ = layer_inputs(layer, 7, 19, dtype=torch.float64, device="cpu")
        layer(*args)[0].sum().backward()
        assert seen == [True]
    # FX generated children are pickled by e3nn; Python hook closures can
    # make the entire adapter uncopyable even when this region remains native.
    kernel = torch.fx.symbolic_trace(torch.nn.Identity())
    tp._compiled_main_left_right = kernel
    with getattr(kernel, register)(lambda *args: None):
        assert customized_codegen(layer)
    assert not customized_codegen(layer)


@pytest.mark.parametrize(
    "register",
    [
        "register_module_forward_pre_hook",
        "register_module_forward_hook",
        "register_module_full_backward_pre_hook",
        "register_module_full_backward_hook",
    ],
)
def test_global_hooks_observe_native_module_boundaries(monkeypatch, register):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    layer = ordinary_layer()
    hooks = torch.nn.modules.module
    # Removing the last global backward hook does not reset PyTorch's
    # process-wide legacy-vs-full choice; restore it after this test.
    monkeypatch.setattr(hooks, "_global_is_full_backward_hook", None)
    with getattr(hooks, register)(lambda *args: None):
        assert not customized_codegen(layer)
        lower, reason = interaction_rule(layer, backend="torch", task="training")
        assert lower is None and "customized" in reason


def test_only_canonical_component_harmonics_match_the_fused_polynomial():
    from e3nn.o3._spherical_harmonics import _spherical_harmonics

    irreps = o3.Irreps.spherical_harmonics(2)
    basis = o3.SphericalHarmonics(irreps, False, "component")
    assert canonical_component_harmonics(basis, 2)
    assert not canonical_component_harmonics(basis, 1)
    assert not canonical_component_harmonics(
        o3.SphericalHarmonics([0, 2], False, "component"), 2
    )
    assert not canonical_component_harmonics(
        o3.SphericalHarmonics(irreps, True, "component"), 2
    )
    assert not canonical_component_harmonics(
        o3.SphericalHarmonics(irreps, False, "norm"), 2
    )
    if hasattr(basis, "sph_func"):
        for evaluator in (
            _spherical_harmonics,
            torch.jit.script(_spherical_harmonics),
            torch.compile(_spherical_harmonics, fullgraph=True),
        ):
            basis.sph_func = evaluator
            assert canonical_component_harmonics(basis, 2)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_ordinary_representation_keeps_custom_angular_evaluator(monkeypatch, backend):
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        graph,
        lower_model,
        make_model,
    )

    if backend == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    device = "cpu" if backend == "torch" else "cuda"
    source = make_model().to(device=device, dtype=torch.float64).eval()
    rep = source.readout_fn.representation
    basis = rep.o3_angular_basis
    if not hasattr(basis, "sph_func"):
        pytest.skip("older e3nn has no replaceable evaluator")
    assert canonical_geometry(rep)
    evaluate = basis.sph_func
    calls = []

    @wraps(evaluate)
    def scaled(*args):
        calls.append(True)
        return 0.5 * evaluate(*args)

    basis.sph_func = scaled
    assert not canonical_geometry(rep)
    data = graph(device)
    expected = source(data)
    scheduled = lower_model(source, backend=backend)
    assert not any(
        isinstance(layer, StreamedInteraction)
        for layer in scheduled.readout_fn.representation.interactions
    )
    calls.clear()
    got = scheduled(data)
    assert calls
    for name in EFS:
        torch.testing.assert_close(got[name], expected[name], atol=1e-8, rtol=1e-6)


def _wigner6j_model():
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.tace import SphericalTACE

    def host(**cfg):
        cfg.update(num_channel=2, Lmax=1, lmax=1, mmax=1)
        cfg["target_property"].append("noncollinear_magnetic_forces")
        cfg["fidelity"]["magnetic_scale"] = 2.0
        cfg["atomic_basis"]["type"] = "w6j_mag"
        cfg["angular_basis"]["magnetic_Lmax"] = 1
        cfg["radial_basis"]["apply_cutoff"] = False
        cfg["edge_embedding"]["type"] = "linear"
        cfg["edge_update"]["type"] = "element2"
        return SphericalTACE(**cfg)

    return make_model(host)


def test_wigner6j_native_provider_and_customization(monkeypatch):
    from equivariantx_foundation.models.tace.wigner6j import (
        host_wigner6j_provider,
        wigner6j_rule,
    )

    monkeypatch.setenv("TACE_USE_EQX", "0")
    layer = _wigner6j_model().readout_fn.representation.interactions[0]
    assert host_wigner6j_provider(layer) == "eqx_torch"
    assert wigner6j_rule(layer, backend="torch", task="training")[0] is None
    assert wigner6j_rule(layer, backend="cuda", task="training")[0] is None
    with layer.rejector.eqx_tp.schedule.register_forward_hook(lambda *_: None):
        assert host_wigner6j_provider(layer) is None
    monkeypatch.setattr(layer.rejector.eqx_tp.schedule, "evaluate", lambda *_: None)
    assert host_wigner6j_provider(layer) is None
    assert wigner6j_rule(layer, backend="cuda", task="training")[0] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Wigner-6j provider")
def test_magnetic_wigner6j_cuda_region_and_mixed_gradients(monkeypatch):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.planner import plan
    from equivariantx_foundation.models.tace.wigner6j import host_wigner6j_provider
    from equivariantx_foundation.tests.test_tace_plan import EFS, graph

    torch.set_default_dtype(torch.float64)
    torch.manual_seed(179)
    monkeypatch.setenv("TACE_USE_EQX", "0")
    host = _wigner6j_model().cuda().train()
    layers = host.readout_fn.representation.interactions
    with monkeypatch.context() as patch:
        patch.setenv("TACE_USE_EQX", "1")
        assert all(host_wigner6j_provider(layer) == "eqx_cuda" for layer in layers)
        assert [
            (r.provenance, r.provider)
            for r in plan(host, task="training", backend="cuda").regions
            if r.path in ("interactions.0", "interactions.1")
        ] == [("host_selected", "eqx_cuda")] * 2
    optimized = optimize(host, family="tace", task="training", backend="cuda")
    lowered = optimized.readout_fn.representation.interactions
    assert all(layer.rejector.selected_backend == "cuda" for layer in lowered)
    assert list(optimized.state_dict()) == list(host.state_dict())
    assert all(
        a is b for a, b in zip(host.parameters(), optimized.parameters(), strict=True)
    )
    data = graph("cuda")
    mag = "noncollinear_magnetic_forces"
    data["initial_noncollinear_magmoms"] = torch.randn(6, 3, device="cuda") / 2
    keys = (*EFS, mag)

    def run(model):
        out = model(dict(data))
        forces = out[mag]
        assert forces.abs().max() > 0
        # Isolate the spin-force derivative: the energy part of the combined
        # loss must not conceal a disconnected magnetic weight table.
        spin_grad = torch.autograd.grad(
            forces.square().mean(),
            tuple(
                p
                for name, p in model.named_parameters()
                if ".magnetic_edge_info." in name
            ),
            retain_graph=True,
        )
        assert any(g.count_nonzero() for g in spin_grad)
        loss = sum(out[k].square().mean() for k in keys)
        return (
            tuple(out[k] for k in keys),
            spin_grad,
            torch.autograd.grad(loss, tuple(model.parameters()), allow_unused=True),
        )

    expected = run(host)
    # Verify dispatch to the actual EQX CUDA implementation, not just a receipt.
    from torch.utils._python_dispatch import TorchDispatchMode

    from equivariantx.ops.cuda.o3.wigner6j import convolution

    calls, native_ops = [], set()

    class Record(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func.namespace == "eqx":
                native_ops.add(str(func))
            return func(*args, **(kwargs or {}))

    evaluate = convolution.evaluate

    def count(op, *args):
        calls.append(op)
        with Record():
            return evaluate(op, *args)

    monkeypatch.setattr(convolution, "evaluate", count)
    torch.testing.assert_close(run(optimized), expected, atol=2e-9, rtol=1e-7)
    assert calls == [layer.rejector.eqx_tp for layer in lowered]
    assert native_ops & {"eqx.sparse_bilinear.default", "eqx.o3_contraction.default"}
    torch_model = optimize(optimized, family="tace", task="training", backend="torch")
    assert all(
        layer.rejector.selected_backend == "torch"
        for layer in torch_model.readout_fn.representation.interactions
    )
    calls.clear()
    torch.testing.assert_close(run(torch_model), expected, atol=2e-9, rtol=1e-7)
    assert not calls
    from tace.models.time_reversal import supports_time_reversal

    if supports_time_reversal():
        assert any(
            ir.t == -1 for layer in lowered for _, ir in layer.rejector.irreps_in3
        )
        reversed_data = dict(data)
        reversed_data["initial_noncollinear_magmoms"] = -data[
            "initial_noncollinear_magmoms"
        ]
        reversed_out = optimized(reversed_data)
        for k, value in zip(keys, expected[0], strict=True):
            torch.testing.assert_close(
                reversed_out[k], -value if k == mag else value, atol=2e-9, rtol=1e-7
            )

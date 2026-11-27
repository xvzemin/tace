"""Regression checks for local O(2) execution on non-release TACE interactions."""

import pytest
import torch
from e3nn import o3

from equivariantx.o2 import WignerD
from equivariantx_foundation.models.tace.foundation import shared_parameter_copy
from equivariantx_foundation.models.tace.local import local_rule
from equivariantx_foundation.tests.test_interaction_schedule import layer_inputs
from tace.models.spherical.inter import UuO2Interaction, UvO2Interaction


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_uv_node_gate_reuses_live_host_layout_and_preserves_unsupported_calls(
    device, monkeypatch, request
):
    from unittest.mock import patch

    from e3nn.nn import Gate

    from equivariantx.ops.common import gate as provider
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.models.tace.local import _NodeGate, _uv_gate_expression
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.tace import SphericalTACE

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA")
    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    precision = torch.get_float32_matmul_precision()
    torch.set_float32_matmul_precision("highest")
    request.addfinalizer(lambda: torch.set_float32_matmul_precision(precision))

    def build(**cfg):
        cfg.update(num_channel=64, Lmax=2, lmax=3, mmax=3, parity=None)
        cfg["atomic_basis"].update(
            type="uv_o2",
            scalar_act=["silu", "tanh"],
            tensor_act="sigmoid",
            gate_m0=False,
        )
        cfg["radial_basis"]["apply_cutoff"] = False
        return SphericalTACE(**cfg)

    source = make_model(build).float().to(device)
    layer = source.readout_fn.representation.interactions[0]
    original = layer.nonlinearity
    # Use the *live* e3nn second-moment gain rather than reconstructing it
    # from the name of the activation or assuming a default initial value.
    original.act_scalars.acts[0].cst *= 0.7
    assert _uv_gate_expression(layer) is not None
    with original.act_gates.register_forward_hook(lambda *args: None):
        assert _uv_gate_expression(layer) is None
    with monkeypatch.context() as mp:
        mp.setattr(original.act_scalars, "paths", [])
        assert _uv_gate_expression(layer) is None
    with monkeypatch.context() as mp:
        mp.setitem(original.sc.cut._modules, "_compiled_forward", torch.nn.Identity())
        assert _uv_gate_expression(layer) is None
    for task in ("inference", "training"):
        matches = [
            r
            for r in plan(source, task=task).regions
            if "CUDA spherical node gate" in r.reason
        ]
        assert len(matches) == (2 if device == "cuda" else 0)
    if device == "cpu":
        return
    converted = optimize(source, family="tace")
    gate = converted.readout_fn.representation.interactions[0].nonlinearity
    assert type(original) is Gate and type(gate) is _NodeGate
    torch.testing.assert_close(gate.state_dict(), original.state_dict(), rtol=0, atol=0)
    x = torch.randn(
        11, gate.irreps_in.dim, device=device, dtype=torch.float32, requires_grad=True
    )
    seed = torch.randn(11, gate.irreps_out.dim, device=device)

    def evaluate(m):
        y = m(x)
        (dx,) = torch.autograd.grad((y * seed).sum(), x, create_graph=True)
        (d2,) = torch.autograd.grad(dx.square().sum(), x)
        return y, dx, d2

    with patch.object(provider, "gate", wraps=provider.gate) as call:
        torch.testing.assert_close(
            evaluate(gate), evaluate(original), rtol=3e-5, atol=5e-5
        )
        assert call.call_count == 1
        # A newly registered hook still observes the host's actual callable.
        with gate.act_gates.register_forward_hook(lambda m, args, value: value * 0.5):
            altered = gate(x)
        assert not torch.equal(altered, original(x))
        assert call.call_count == 1
        with monkeypatch.context() as mp:
            mp.setattr(gate.act_gates.acts[0], "cst", 0.25)
            torch.testing.assert_close(gate(x), Gate.forward(gate, x))
        assert call.call_count == 1
    restored = optimize(converted, family="tace", backend="torch")
    assert type(restored.readout_fn.representation.interactions[0].nonlinearity) is Gate
    torch.testing.assert_close(
        gate.cpu()(x.detach().cpu()), original.cpu()(x.detach().cpu()), rtol=0, atol=0
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed-precision Uu CUDA")
def test_two_uu_layers_only_share_pristine_mixed_precision_direction_frames(
    monkeypatch,
):
    from types import SimpleNamespace

    from equivariantx_foundation import optimize
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV

    for key in ACCELERATION_ENV.values():
        monkeypatch.setenv(key, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")

    def build(**cfg):
        cfg.update(num_channel=64, Lmax=2, lmax=3, mmax=3)
        cfg["atomic_basis"]["type"] = "uu_o2"
        cfg["radial_basis"]["apply_cutoff"] = False
        return SphericalTACE(**cfg)

    host = make_model(build).float().cuda()
    scheduled = optimize(host, family="tace", regional_only=True)
    rep = scheduled.readout_fn.representation
    assert not hasattr(host.readout_fn.representation, "_shared_direction_frame")
    vec = torch.randn(11, 3, device="cuda", dtype=torch.float64, requires_grad=True)
    graph = SimpleNamespace(edge_vector=vec, lmp=False, lmp_data=None)
    frame, inverse = rep._spatial_frames({}, graph)
    assert frame is inverse and frame.shape == (11, 84)
    gradient = torch.autograd.grad(
        frame.square().mul(frame).sum(), vec, create_graph=True
    )[0]
    assert torch.autograd.grad(gradient.sum(), vec)[0].isfinite().all()
    torch.testing.assert_close(
        rep._spatial_frames({"edge_pair_rows": object()}, graph)[0], frame
    )
    assert rep._spatial_frames(
        {}, SimpleNamespace(edge_vector=vec.float(), lmp=False, lmp_data=None)
    ) == (None, None)
    assert rep._spatial_frames(
        {}, SimpleNamespace(edge_vector=vec, lmp=True, lmp_data=object())
    ) == (None, None)

    second = rep.interactions[-1].rejector.eqx_tp
    from equivariantx.ops.common.cuda.graph import retain_constants

    with retain_constants({}):
        assert rep._spatial_frames({}, graph) == (None, None)
    with rep.node_embedding.register_forward_pre_hook(lambda *args: None):
        assert rep._spatial_frames({}, graph) == (None, None)
    with second.register_forward_hook(lambda *args: None):
        assert rep._spatial_frames({}, graph) == (None, None)
    with monkeypatch.context() as patch:
        patch.setattr(second.frame, "forward_packed", lambda *args: None)
        assert rep._spatial_frames({}, graph) == (None, None)
    restored = optimize(scheduled, family="tace", backend="torch", regional_only=True)
    assert not hasattr(restored.readout_fn.representation, "_shared_direction_frame")
    second.frame.cg_2.add_(0.1)
    assert rep._spatial_frames({}, graph) == (None, None)
    scheduled.double()
    assert not any(
        hasattr(l.rejector, "_foundation_uu_tables") for l in rep.interactions
    )


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_aligned_automatic_precomputed_scope(device, monkeypatch):
    from equivariantx_foundation.models.tace import plan
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")

    def host(**cfg):
        cfg.update(
            num_channel=64,
            lmax=3,
            mmax=3,
            Lmax=2,
            parity=None,
            target_property=["energy", "forces"],
        )
        cfg["atomic_basis"]["type"] = "o2_cgtp"
        cfg["product_basis"].update(type="cgtp", correlation=3)
        return SphericalTACE(**cfg)

    model = make_model(host).float().to(device)
    layers = model.readout_fn.representation.interactions

    def choices(task="training", backend=None):
        return [
            r
            for r in plan(model, task=task, backend=backend).regions
            if r.path in ("interactions.0", "interactions.1")
        ]

    precision = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        for task in ("inference", "training"):
            receipts = choices(task)
            if device == "cpu":
                assert all(
                    r.provider == "eqx_torch" and r.lower is None for r in receipts
                )
            else:
                assert all(
                    r.provider == "eqx_cuda" and r.lower is not None for r in receipts
                )
        if device == "cpu":
            return
        with layers[0].rejector.tp.convolution.register_forward_hook(
            lambda *args: None
        ):
            assert not any(r.lower for r in choices())
        op = layers[0].rejector.tp.convolution
        with monkeypatch.context() as mp:
            mp.setitem(op._buffers, "cg_0", op.cg_0 + 1)
            assert not any(r.lower for r in choices())
        with monkeypatch.context() as mp:
            mp.setitem(op.frame._buffers, "cg_2", op.frame.cg_2 + 1)
            assert not any(r.lower for r in choices())
        with monkeypatch.context() as mp:
            mp.setattr(layers[0].rejector.tp.convolution, "method", "baseline")
            assert not any(r.lower for r in choices())
            assert all(r.lower for r in choices(backend="cuda"))
        with monkeypatch.context() as mp:
            mp.setenv("TACE_MEMORY_FIRST", "0")
            assert not any(r.lower for r in choices())
        with monkeypatch.context() as mp:
            mp.setenv("TACE_USE_EQX", "1")
            assert all(r.lower is None and r.provider == "eqx_cuda" for r in choices())
        with monkeypatch.context() as mp:
            mp.setattr(layers[1], "mmax", 2)
            assert not any(r.lower for r in choices())
        torch.set_float32_matmul_precision("high")
        assert not any(r.lower for r in choices())
        torch.set_float32_matmul_precision("highest")
        model.double()
        assert not any(r.lower for r in choices())
    finally:
        torch.set_float32_matmul_precision(precision)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="native aligned-CGTP providers"
            ),
        ),
    ],
)
def test_aligned_cgtp_host_and_explicit_providers(
    device, monkeypatch, double_precision
):
    from unittest.mock import patch

    from equivariantx.ops.cuda.co2 import spherical as cuda_coupling
    from equivariantx.ops.cuda.o2.aligned import convolution as cuda_kernel
    from equivariantx.ops.torch.o2.aligned import convolution as torch_kernel
    from equivariantx_foundation.models.tace.planner import optimize_model, plan
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        efs_loss,
        graph,
        make_model,
    )
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV, EQX_KERNELS

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("TACE_USE_EQX", "1")
    monkeypatch.setitem(EQX_KERNELS, "product", False)
    monkeypatch.setitem(EQX_KERNELS, "linear", False)
    monkeypatch.setenv("TACE_MEMORY_FIRST", "0")

    def aligned(**cfg):
        cfg["atomic_basis"]["type"] = "o2_cgtp"
        return SphericalTACE(**cfg)

    model = make_model(aligned).to(device).double().train()
    source = model.readout_fn.representation.interactions
    layers = len(source)
    for layer in source:
        # Avoid hardware/timing-dependent algorithm autotuning in the test.
        layer.rejector.eqx_tp.method = layer.rejector.tp.convolution.method = "baseline"

    def regions(m, backend=None):
        return [
            r
            for r in plan(m, backend=backend, regional_only=True).regions
            if r.path in {f"interactions.{i}" for i in range(layers)}
        ]

    def run(m, backend):
        kernel = torch_kernel if backend == "torch" else cuda_kernel
        with (
            patch.object(kernel, "transverse", wraps=kernel.transverse) as call,
            patch.object(
                cuda_coupling, "forward", wraps=cuda_coupling.forward
            ) as nested_cuda,
        ):
            prediction = m(graph(device))
            assert call.call_count == layers
            if backend == "torch":
                assert not nested_cuda.called
        return tuple(prediction[name] for name in EFS), torch.autograd.grad(
            efs_loss(prediction), tuple(m.parameters()), allow_unused=True
        )

    backend = "cuda" if device == "cuda" else "torch"
    host = regions(model)
    assert len(host) == layers
    assert all(
        (r.provenance, r.provider, r.lower) == ("host_selected", f"eqx_{backend}", None)
        for r in host
    )
    # A retained hook or customized precomputed call must not acquire a receipt.
    with source[0].rejector.eqx_tp.register_forward_hook(lambda *_: None):
        assert regions(model, "torch")[0].provider is None
    with monkeypatch.context() as mp:
        mp.setattr(source[0].rejector.tp, "forward_scatter", lambda *a, **kw: None)
        assert regions(model, "torch")[0].provider is None
    expected = run(model, backend)
    if device == "cpu":
        monkeypatch.setenv("TACE_USE_EQX", "0")
        assert all(r.provider == "eqx_torch" for r in regions(model))
        torch.testing.assert_close(run(model, "torch"), expected)
        return

    with monkeypatch.context() as mp:
        # The Torch convolution can still delegate its angular subprogram to
        # CUDA on this mixed host. Don't advertise a uniform provider for it.
        mp.setattr(source[0].rejector.eqx_tp, "backend", "torch")
        mixed = regions(model)[0]
        assert mixed.provider is None and mixed.lower is None
        with patch.object(
            cuda_coupling, "forward", wraps=cuda_coupling.forward
        ) as nested_cuda:
            model(graph(device))
        assert nested_cuda.called
    assert all(r.lower is not None for r in regions(model, "torch"))
    torch_route = optimize_model(model, backend="torch", regional_only=True)
    cuda_route = optimize_model(torch_route, backend="cuda", regional_only=True)
    # Checkpoint names/values, live Parameters, and the original fast path survive
    # both retargets. Toggling the host flag still selects its native radial path.
    expected_state = model.state_dict()
    for route in (torch_route, cuda_route):
        assert route.state_dict().keys() == expected_state.keys()
        assert all(
            p is q for p, q in zip(route.parameters(), model.parameters(), strict=True)
        )
        assert all(
            torch.equal(v, expected_state[k]) for k, v in route.state_dict().items()
        )
    for flag in ("0", "1"):
        monkeypatch.setenv("TACE_USE_EQX", flag)
        torch.testing.assert_close(
            run(torch_route, "torch"), expected, atol=2e-8, rtol=2e-7
        )
        torch.testing.assert_close(
            run(cuda_route, "cuda"), expected, atol=2e-8, rtol=2e-7
        )
    assert all(layer.rejector.eqx_tp.backend == "cuda" for layer in source)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="magnetic Uv CUDA stream")
@pytest.mark.parametrize("rank,explicit_time", [(1, False), (3, False), (2, True)])
def test_full_order_magnetic_frames_and_spin_gradients(
    monkeypatch, double_precision, rank, explicit_time
):
    from contextlib import ExitStack
    from copy import deepcopy

    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.magnetic import (
        _MagneticUvFrames,
        magnetic_rule,
    )
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        efs_loss,
        graph,
        physics_model,
    )
    from tace.models.adapter import TensorModel
    from tace.models.spherical.tace import SphericalTACE
    from tace.models.time_reversal import supports_time_reversal
    from tace.utils.env import ACCELERATION_ENV

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    if explicit_time and not supports_time_reversal():
        pytest.skip("Requires the explicit time irreps of T-e3nn")
    moment, force = "initial_noncollinear_magmoms", "noncollinear_magnetic_forces"
    cfg = deepcopy(physics_model(force).readout_fn.model_config)
    cfg.update(lmax=rank, Lmax=rank, mmax=rank)
    cfg["angular_basis"]["magnetic_Lmax"] = rank
    host = TensorModel(SphericalTACE(**cfg)).double().cuda().train()
    layers = host.readout_fn.representation.interactions
    if explicit_time:
        assert all(layer.rejector.use_time_reversal for layer in layers)
        for layer in layers:
            assert magnetic_rule(layer, backend="cuda", task="training")[0] is None
        host.float()
        gate = layers[-1].rejector.nonlinearity.act_gates
        odd = next(
            a
            for (ir, _), a in zip(gate.irreps_in, gate.acts, strict=True)
            if ir.t == -1
        )
        with monkeypatch.context() as mp:
            # A time-odd scalar cannot use a sigmoid in place of its odd activation.
            mp.setitem(odd._modules, "f", torch.nn.Sigmoid())
            assert magnetic_rule(layers[-1], backend="cuda", task="training")[0] is None
    elif any(layer.rejector.use_time_reversal for layer in layers):
        # Some e3nn installations support explicit time-odd irreps. Those
        # gates are a distinct program and aren't covered by the degree sweep.
        for layer in layers:
            if layer.rejector.use_time_reversal:
                assert magnetic_rule(layer, backend="cuda", task="training")[0] is None
        return
    conv = layers[-1].rejector
    with monkeypatch.context() as mp:
        mp.setenv("TACE_MEMORY_FIRST", "0")
        assert magnetic_rule(layers[-1], backend="cuda", task="training")[0] is None
    for attr, value in (("use_time_reversal", not explicit_time), ("mmax", rank - 1)):
        with monkeypatch.context() as mp:
            mp.setattr(conv, attr, value)
            assert magnetic_rule(layers[-1], backend="cuda", task="training")[0] is None
    with monkeypatch.context() as mp:
        mp.setattr(conv.eqx_tp, "degree", rank + 1)
        assert magnetic_rule(layers[-1], backend="cuda", task="training")[0] is None
    model = optimize(
        host, family="tace", backend="cuda", task="training", regional_only=True
    )
    params = tuple(host.parameters())
    assert all(p is q for p, q in zip(params, model.parameters(), strict=True))
    assert model.state_dict().keys() == host.state_dict().keys()
    blocks = model.readout_fn.representation.interactions
    assert all(type(m.rejector) is _MagneticUvFrames for m in blocks)
    calls = []
    data = graph("cuda")
    data["node_type"].zero_()
    spin = torch.randn_like(data["positions"])

    def outputs_and_grads(m, moment_input):
        prediction = m({**data, moment: moment_input})
        target = torch.linspace(-0.2, 0.4, spin.numel(), device=spin.device).reshape_as(
            spin
        )
        loss = efs_loss(prediction) + (prediction[force] - target).square().mean()
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        return tuple(prediction[k].detach() for k in (*EFS, force)), grads

    # On stock e3nn no time-parity labels exist: compare against the native
    # model at both spin orientations, do not assume time-reversal symmetry.
    previous = None
    for moments in (spin, -spin):
        expected = outputs_and_grads(host, moments)
        assert expected[0][-1].abs().max() > 1e-10
        calls.clear()
        with ExitStack() as stack:
            for i, m in enumerate(blocks):
                stack.enter_context(
                    m.rejector.eqx_tp.register_forward_hook(
                        lambda _m, _args, _result, idx=i: calls.append(idx)
                    )
                )
            actual = outputs_and_grads(model, moments)
        assert calls == list(range(len(blocks)))
        tol = (
            dict(rtol=4e-4, atol=2e-5) if explicit_time else dict(rtol=3e-7, atol=2e-9)
        )
        torch.testing.assert_close(actual, expected, **tol)
        if explicit_time and previous is not None:
            torch.testing.assert_close(
                actual[0], (*previous[:-1], -previous[-1]), **tol
            )
        previous = actual[0]
    if explicit_time:
        # Spins and their magnetic forces are axial; positions and forces are polar.
        energy, forces, stress, magnetic_force = previous
        for R in (
            o3.rand_matrix(dtype=spin.dtype, device=spin.device),
            -torch.eye(3).to(spin),
        ):
            det = R.det()
            transformed = dict(
                data, positions=data["positions"] @ R.T, lattice=data["lattice"] @ R.T
            )
            transformed[moment] = det * (-spin @ R.T)
            expected = (
                energy,
                forces @ R.T,
                R @ stress @ R.T,
                det * (magnetic_force @ R.T),
            )
            for route in (host, model):
                y = route(transformed)
                torch.testing.assert_close(
                    tuple(y[k] for k in (*EFS, force)), expected, **tol
                )
        # Retargeting a selected model's dtype may not silently widen this
        # FP32-only policy. The native FP64 magnetic contraction remains valid.
        host.double()
        model.double()
        with ExitStack() as stack:
            for layer in blocks:
                stack.enter_context(
                    layer.rejector.eqx_tp.register_forward_hook(
                        lambda *_: pytest.fail("used FP32-only magnetic route in FP64")
                    )
                )
            torch.testing.assert_close(
                outputs_and_grads(model, spin),
                outputs_and_grads(host, spin),
                rtol=3e-7,
                atol=2e-9,
            )


def _scalar_spin_model(channels=5):
    from equivariantx_foundation.tests.test_tace_plan import EFS, make_model
    from tace.models.spherical.tace import SphericalTACE

    def host(**cfg):
        cfg.update(
            num_layers=3,
            num_channel=channels,
            target_property=[*EFS, "noncollinear_magnetic_forces"],
        )
        cfg["statistics"][0].update(
            atomic_numbers=[7, 24], atomic_energy={7: 0.0, 24: 0.0}
        )
        cfg["fidelity"]["magnetic_scale"] = {7: 0.3, 24: 3.5}
        cfg["atomic_basis"]["type"] = "uv_o2_mag"
        cfg["node_update"]["magnetic_type"] = "element"
        cfg["angular_basis"].update(magnetic_Lmax=2, use_spin_orbit_coupling=False)
        cfg["radial_basis"]["apply_cutoff"] = False
        cfg["readout_emlp"]["use_one_body_magmoms"] = True
        return SphericalTACE(**cfg)

    return make_model(host).double().train()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA execution policy")
def test_nosoc_automatic_memory_policy(monkeypatch):
    from equivariantx_foundation.models.tace.magnetic import magnetic_rule

    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    model = _scalar_spin_model(48).cuda().float()
    layers = model.readout_fn.representation.interactions

    def selected(automatic):
        return [
            magnetic_rule(layer, backend="cuda", task=task, automatic=automatic)[0]
            is not None
            for layer in layers
            for task in ("inference", "training")
        ]

    assert all(selected(True))
    monkeypatch.setenv("TACE_MEMORY_FIRST", "0")
    assert not any(selected(True))
    assert all(selected(False))
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    model.double()
    assert not any(selected(True))
    assert all(selected(False))


def test_nosoc_keeps_packed_frames_and_empty_edges_native(
    monkeypatch, double_precision
):
    from equivariantx_foundation.models.tace import magnetic
    from equivariantx_foundation.tests.test_tace_plan import graph

    monkeypatch.setenv("TACE_USE_EQX", "0")
    host = _scalar_spin_model()
    layer = host.readout_fn.representation.interactions[0]
    sample = []
    data = graph("cpu")
    with layer.rejector.register_forward_pre_hook(lambda _, args: sample.extend(args)):
        host(
            {
                **data,
                "initial_noncollinear_magmoms": torch.randn_like(data["positions"]),
            }
        )
    x, attrs, weights, index, frame, inverse, radial, cutoff = sample
    fast = magnetic._lower(shared_parameter_copy(layer)).rejector
    packed = torch.cat([torch.eye(2 * l + 1).flatten() for l in range(3)]).expand(
        index.size(1), -1
    )
    # Exercise the noSOC ABI guards even on a CPU-only test runner; this
    # does not emulate a CUDA kernel. The actual CUDA route is tested below.
    monkeypatch.setattr(
        magnetic, "_eligible_inputs", lambda *args: bool(args[-1].size(1))
    )
    monkeypatch.setattr(
        magnetic,
        "uv_convolution",
        lambda *_a, **_kw: pytest.fail("unexpected lowering"),
    )
    for args in (
        (x, attrs, weights, index, packed, packed, radial, cutoff),
        (
            x,
            attrs[:0],
            weights[:0],
            index[:, :0],
            frame[:0],
            inverse[:0],
            radial[:0],
            cutoff[:0],
        ),
    ):
        torch.testing.assert_close(fast(*args), layer.rejector(*args))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="scalar-spin CUDA reduction"
            ),
        ),
    ],
)
def test_nosoc_scalar_spin_frames_and_independent_rotations(
    device, monkeypatch, double_precision
):
    from unittest.mock import patch

    from equivariantx.ops.cuda.o2.uv import convolution as uv_cuda
    from equivariantx_foundation.models.tace.magnetic import (
        _MagneticUvFrames,
        _scalar_spin_contract,
        magnetic_rule,
    )
    from equivariantx_foundation.models.tace.planner import optimize_model, plan
    from equivariantx_foundation.tests.test_tace_plan import EFS, efs_loss, graph
    from tace.utils.env import ACCELERATION_ENV

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    host = _scalar_spin_model().to(device)
    layers = host.readout_fn.representation.interactions
    assert [layer.rejector.mmax for layer in layers] == [0, 2, 2]
    assert all(_scalar_spin_contract(layer, layer.rejector) for layer in layers)
    assert all(
        r.lower is None
        for r in plan(host, regional_only=True).regions
        if r.path in {f"interactions.{i}" for i in range(len(layers))}
    )
    if device == "cuda":
        for layer in layers:
            conv = layer.rejector
            with monkeypatch.context() as mp:
                mp.setattr(conv.eqx_tp, "degree", 3)
                assert magnetic_rule(layer, backend="cuda", task="training")[0] is None
            with monkeypatch.context() as mp:
                mp.setattr(conv, "use_lora", True, raising=False)
                assert magnetic_rule(layer, backend="cuda", task="training")[0] is None
            with conv.linear_up.register_forward_hook(lambda *_: None):
                assert magnetic_rule(layer, backend="cuda", task="training")[0] is None
        model = optimize_model(
            host, backend="cuda", task="training", regional_only=True
        )
        assert all(
            type(l.rejector) is _MagneticUvFrames
            for l in model.readout_fn.representation.interactions
        )
        assert all(
            a is b for a, b in zip(model.parameters(), host.parameters(), strict=True)
        )
        assert model.state_dict().keys() == host.state_dict().keys()
    else:
        model = host

    moment, mf = "initial_noncollinear_magmoms", "noncollinear_magnetic_forces"
    data = graph(device)
    # Unequal elements, two magnetic scales, non-collinear spins, an exactly
    # zero spin, and points on both sides of the radial saturation boundary.
    spin = data["positions"].new_tensor(
        [
            [0, 0, 0],
            [0, 3.49, 0.1],
            [0, 0, 3.51],
            [0.20, -0.08, 0.11],
            [0.17, 0.33, -0.7],
            [0.6, -0.4, -0.3],
        ]
    )

    def derivatives(m):
        s = spin.detach().requires_grad_()
        linear = m.readout_fn.representation.interactions[0].magnetic_linear
        external = []
        with linear.register_forward_pre_hook(lambda _, args: external.extend(args)):
            y = m({**data, moment: s})
        trial = torch.linspace(
            -0.2, 0.3, s.numel(), device=s.device, dtype=s.dtype
        ).view_as(s)
        loss = efs_loss(y) + (y[mf] - trial).square().mean()
        gradients = torch.autograd.grad(
            loss, (s, *m.parameters(), *external), allow_unused=True
        )
        assert gradients[0] is not None and gradients[0].isfinite().all()
        raw_spin, endpoint_weights = gradients[-2:]
        assert raw_spin.shape[1] == 3 and (raw_spin.abs().sum(0) > 0).all()
        assert endpoint_weights.abs().max() > 0
        for name in (
            "source_magnetic_info",
            "target_magnetic_info",
            "edge_info",
            "linear_up",
            "linear_down",
        ):
            assert any(
                g is not None and g.abs().max() > 0
                for (key, _), g in zip(
                    m.named_parameters(), gradients[1:-2], strict=True
                )
                if name in key
            )
        return tuple(y[k].detach() for k in (*EFS, mf)), gradients

    expected, gradients = derivatives(host)
    with patch.object(uv_cuda, "edge_program", wraps=uv_cuda.edge_program) as kernel:
        actual, mixed = derivatives(model)
        assert kernel.call_count == (3 if device == "cuda" else 0)
    close = dict(rtol=3e-7, atol=3e-9)
    torch.testing.assert_close((actual, mixed), (expected, gradients), **close)
    energy, forces, stress, magnetic_force = actual
    eye = torch.eye(3).to(spin)
    rotation = o3.rand_matrix(dtype=spin.dtype, device=spin.device)
    reflection = torch.diag(spin.new_tensor([-1, 1, 1]))
    for spatial, magnetic in (
        (rotation, eye),
        (eye, rotation),
        (reflection, eye),
        (eye, -eye),
    ):
        transformed = {
            **data,
            "positions": data["positions"] @ spatial.T,
            "lattice": data["lattice"] @ spatial.T,
            moment: spin @ magnetic.T,
        }
        y = model(transformed)
        torch.testing.assert_close(
            tuple(y[k] for k in (*EFS, mf)),
            (
                energy,
                forces @ spatial.T,
                spatial @ stress @ spatial.T,
                magnetic_force @ magnetic.T,
            ),
            **close,
        )


def local_layer(kind, width, lmax=2, mmax=1):
    cls = UuO2Interaction if kind == "uu" else UvO2Interaction
    return cls(
        layer=1,
        num_layers=3,
        num_elements=3,
        avg_num_neighbors=6,
        mmax=mmax,
        Lmax=lmax,
        lmax=lmax,
        correlation=[2] * 3,
        num_channel=width,
        edge_feats_channel=5,
        num_radial_basis=5,
        target_irreps=o3.Irreps("0e"),
        radial_mlp=[8],
        radial_bias=True,
        irreps_in=o3.Irreps([(width, (l, (-1) ** l)) for l in range(lmax + 1)]),
        scalar_act="silu",
        tensor_act="sigmoid",
        edge_ace_hidden=None,
        nonlinear="gate",
        edge_nonlinear="gate",
        resnet_type="BB",
        resnet_linear_type="aware",
        scatter_norm="cutoff",
        mlp_type="mlp_v2",
    )


def frame_inputs(layer, nodes, edges, *, device, dtype):
    args, xs = layer_inputs(layer, nodes, edges, device=device, dtype=dtype)
    args = list(args)
    vec = (torch.randn(edges, 3, device=device, dtype=dtype) + 0.1).requires_grad_()
    args[-1].edge_vector = vec
    args[-1].edge_length = vec.norm(dim=-1, keepdim=True)
    frame = WignerD(mmax=layer.mmax, lmax=layer.lmax).to(vec)
    args[8], args[9] = frame(vec)
    return args, (xs[0], xs[2], xs[3], vec)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="native CUDA reductions")
@pytest.mark.parametrize(
    "kind,owners", [("uu", None), ("uv", None), ("uu", 4), ("uv", 4)]
)
def test_local_derivatives_and_independent_inverse(kind, owners, monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    torch.set_default_dtype(torch.float64)
    layer = local_layer(kind, 4).cuda().double()
    op, reason = local_rule(layer, backend="cuda", task="training")
    assert op is not None, reason
    fast = op(shared_parameter_copy(layer))
    again, _ = local_rule(fast, backend="cuda", task="inference")
    restore, _ = local_rule(fast, backend="torch", task="inference")
    assert again is None and restore is not None
    assert type(restore(shared_parameter_copy(fast)).rejector) is type(layer.rejector)
    assert tuple(fast.state_dict()) == tuple(layer.state_dict())
    assert all(
        a is b for a, b in zip(layer.parameters(), fast.parameters(), strict=True)
    )
    args, xs = frame_inputs(layer, 8, 29, device="cuda", dtype=torch.float64)
    # The inverse frame is an explicit input, not necessarily the transpose.
    delta = torch.randn_like(args[9]).requires_grad_()
    args[9] = args[9] * (1 + 0.03 * delta)
    if owners is not None:
        args[6][1].remainder_(owners)
        args[-1].lmp_natoms = (owners, args[0].shape[0] - owners)
    params = (*xs, delta, *(layer if owners is None else layer.edge_info).parameters())

    def evaluate(model):
        if owners is None:
            y, sc = model(*args)
            loss = y.sin().sum() + sc.cos().sum()
        else:
            # Compare the Foundation receiver scatter to the unmodified full
            # native scatter, including independently supplied inverse frames.
            graph = args[-1]
            graph.lmp_data = object() if model is fast else None
            y = model._compute_messages(args[0], *args[2:10], graph=graph)
            assert y.shape[0] == (owners if model is fast else args[0].shape[0])
            y, sc = y[:owners], None
            loss = y.sin().sum()
        first = torch.autograd.grad(loss, params, create_graph=True, retain_graph=True)
        second = torch.autograd.grad(
            sum(g.square().sum() for g in first[:4]),
            params,
            allow_unused=True,
            retain_graph=True,
        )
        return y, sc, first, second

    torch.testing.assert_close(evaluate(layer), evaluate(fast), rtol=1e-8, atol=1e-8)


@pytest.mark.parametrize("kind", ["uu", "uv"])
def test_local_rules_preserve_existing_providers_and_low_degrees(kind, monkeypatch):
    monkeypatch.setenv("TACE_USE_EQX", "0")
    layer = local_layer(kind, 4, lmax=1)
    rule, reason = local_rule(layer, backend="cuda", task="training")
    assert rule is None and "degree range" in reason
    monkeypatch.setenv("TACE_USE_EQX", "1")
    rule, reason = local_rule(layer, backend="cuda", task="inference")
    assert rule is None and "streamed EQX" in reason


@pytest.mark.skipif(not torch.cuda.is_available(), reason="canonical CUDA stream")
@pytest.mark.parametrize("kind", ["uu_o2", "uv_o2"])
def test_canonical_local_stream_respects_basis_and_force_training(kind, monkeypatch):
    from equivariantx_foundation.models.tace.local import (
        configure_local_geometry,
        local_geometry,
    )
    from equivariantx_foundation.tests.test_tace_plan import efs_loss, graph, make_model
    from tace.models.spherical.tace import SphericalTACE

    monkeypatch.setenv("TACE_USE_EQX", "0")
    torch.set_default_dtype(torch.float64)

    def build(**cfg):
        cfg["mmax"] = 1
        cfg["atomic_basis"]["type"] = kind
        cfg["radial_basis"]["apply_cutoff"] = False
        return SphericalTACE(**cfg)

    src = make_model(build).double().cuda().train()
    rep = src.readout_fn.representation
    assert local_geometry(rep)
    with rep.o2_angular_basis.register_forward_hook(lambda *args: None):
        assert not local_geometry(rep)
    # Do not bypass custom packed or dense geometry even on the right class.
    with monkeypatch.context() as patch:
        patch.setattr(rep.o2_angular_basis, "forward_packed", lambda x: x)
        assert not local_geometry(rep)
    fast = shared_parameter_copy(src)
    dest = fast.readout_fn.representation
    for block in dest.interactions:
        apply, reason = local_rule(
            block, backend="cuda", task="training", canonical=True
        )
        assert apply is not None, reason
        apply(block)
    configure_local_geometry(dest)
    assert (
        dest.o2_angular_basis.backend == "cuda"
        and rep.o2_angular_basis.backend == "torch"
    )
    assert not rep.interactions[0].use_eqx and dest.interactions[0].use_eqx
    params = tuple(src.parameters())

    def evaluate(model):
        out = model(graph("cuda"))
        return tuple(
            out[k] for k in ("energy", "forces", "stress")
        ), torch.autograd.grad(efs_loss(out), params, allow_unused=True)

    torch.testing.assert_close(evaluate(fast), evaluate(src), atol=2e-7, rtol=2e-7)
    for block in dest.interactions:
        restore, _ = local_rule(block, backend="torch", task="training", canonical=True)
        restore(block)
    configure_local_geometry(dest)
    assert dest.o2_angular_basis.backend == "torch"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed canonical CUDA stream")
def test_pure_o2_mixed_stack_shares_only_the_measured_training_frame(monkeypatch):
    from contextlib import ExitStack

    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.local import (
        _SharedUvFrameUuConv,
        _StreamUu,
    )
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        efs_loss,
        graph,
        make_model,
    )
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    torch.manual_seed(172)

    def build(**cfg):
        cfg.update(num_channel=32, num_layers=3)
        cfg["atomic_basis"]["type"] = ["uu_o2", "uv_o2", "uu_o2"]
        cfg["radial_basis"]["apply_cutoff"] = False
        return SphericalTACE(**cfg)

    host = make_model(build).float().train().cuda()

    def plan(source, backend="cuda", task="training"):
        return optimize(
            source, family="tace", backend=backend, task=task, regional_only=True
        )

    def uses_shared(m):
        return [
            type(l.rejector) is _SharedUvFrameUuConv
            for l in m.readout_fn.representation.interactions
        ]

    fast = plan(host)
    assert uses_shared(fast) == [True, False, True]
    automatic = plan(host, backend=None)
    assert not any(uses_shared(automatic))
    assert not any(uses_shared(plan(host, task="inference")))
    with monkeypatch.context() as mp:
        mp.setenv("TACE_MEMORY_FIRST", "0")
        assert not any(uses_shared(plan(host)))
    basis = host.readout_fn.representation.o2_angular_basis
    with monkeypatch.context() as mp:
        mp.setitem(basis._buffers, "inverse_scale", basis.inverse_scale * 1.75)
        assert not any(uses_shared(plan(host)))
    params = tuple(host.parameters())
    assert all(a is b for a, b in zip(fast.parameters(), params, strict=True))
    assert tuple(fast.state_dict()) == tuple(host.state_dict())
    sample = graph("cuda")

    def evaluate(model):
        y = model(dict(sample))
        return tuple(y[k] for k in EFS), torch.autograd.grad(
            efs_loss(y), params, allow_unused=True
        )

    reference = evaluate(host)

    def audit(model, shared, extra_frames=0):
        rep = model.readout_fn.representation
        constructors, frames, directions = [], [], []
        compute = WignerD.forward_packed

        def count(frame, *args, **kwargs):
            constructors.append(frame)
            return compute(frame, *args, **kwargs)

        with ExitStack() as stack, monkeypatch.context() as mp:
            mp.setattr(WignerD, "forward_packed", count)
            for layer in rep.interactions:
                uu = type(layer) is _StreamUu

                def check(_op, args, kwargs, uu=uu):
                    frames.append(args[3] if uu else args[4])
                    if uu:
                        directions.append(kwargs.get("vectors"))

                stack.enter_context(
                    layer.rejector.eqx_tp.register_forward_pre_hook(
                        check, with_kwargs=True
                    )
                )
            actual = evaluate(model)
        assert constructors.count(rep.o2_angular_basis) == 1
        assert len(constructors) == 1 + extra_frames
        assert len(frames) == 3 and all(f is frames[0] for f in frames)
        assert len(directions) == 2 and all((d is None) == shared for d in directions)
        return actual

    torch.testing.assert_close(audit(fast, True), reference, rtol=4e-4, atol=3e-5)
    # In the standard mixed precision (FP64 directions, FP32 channels), the
    # automatic Uu path reconstructs a packed frame internally for each Uu.
    torch.testing.assert_close(
        audit(automatic, False, extra_frames=2), reference, rtol=4e-4, atol=3e-5
    )
    restored = plan(fast, backend="torch")
    assert not any(uses_shared(restored))
    assert all(
        type(m) is UuO2Interaction
        for m in restored.readout_fn.representation.interactions[::2]
    )
    # A later precision change does not silently widen the FP32-only policy.
    host.double()
    fast.double()
    torch.testing.assert_close(audit(fast, False), evaluate(host), rtol=3e-6, atol=1e-8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="mixed canonical CUDA stream")
@pytest.mark.parametrize(
    "kinds,node_kind,dtype",
    [
        (("cgtp", "uu_o2"), "linear", torch.float32),
        (("uv_o2", "cgtp"), "linear", torch.float64),
        (("cgtp", "uv_o2", "uu_o2", "cgtp"), "spherical_tensor", torch.float64),
    ],
)
def test_mixed_angular_consumers(kinds, node_kind, dtype, monkeypatch):
    from contextlib import ExitStack
    from unittest.mock import patch

    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.interactions import (
        _SharedHarmonicInteraction,
        canonical_geometry,
    )
    from equivariantx_foundation.models.tace.local import (
        _StreamUu,
        _StreamUv,
        local_geometry,
    )
    from equivariantx_foundation.tests.test_tace_plan import (
        EFS,
        efs_loss,
        graph,
        make_model,
    )
    from tace.models.spherical.tace import SphericalTACE

    monkeypatch.setenv("TACE_USE_EQX", "0")
    torch.manual_seed(123)

    def build(**cfg):
        cfg.update(mmax=1, num_layers=len(kinds))
        cfg["atomic_basis"]["type"] = list(kinds)
        cfg["node_embedding"]["type"] = node_kind
        cfg["radial_basis"]["apply_cutoff"] = False
        return SphericalTACE(**cfg)

    src = make_model(build).to(dtype=dtype, device="cuda").train()
    rep = src.readout_fn.representation

    def check(expected):
        assert canonical_geometry(rep) is expected
        assert local_geometry(rep) is expected

    check(True)
    with rep.node_embedding.register_forward_pre_hook(lambda *a: None):
        check(False)
    with monkeypatch.context() as mp:
        mp.setattr(rep.o3_angular_basis, "normalization", "norm")
        check(False)
    with monkeypatch.context() as mp:
        mp.setitem(
            rep.o2_angular_basis._buffers,
            "inverse_scale",
            1.5 * rep.o2_angular_basis.inverse_scale,
        )
        check(False)
    with monkeypatch.context() as mp:
        mp.setattr(rep, "_can_pack_wigner", False)
        check(False)

    params = tuple(src.parameters())

    def evaluate(model):
        result = model(graph("cuda"))
        return tuple(result[k] for k in EFS), torch.autograd.grad(
            efs_loss(result), params, allow_unused=True
        )

    reference = evaluate(src)
    model = optimize(src, family="tace", backend="cuda", task="training")
    assert all(x is y for x, y in zip(params, model.parameters(), strict=True))
    assert tuple(model.state_dict()) == tuple(src.state_dict())
    dest = model.readout_fn.representation
    classes = {
        "cgtp": _SharedHarmonicInteraction,
        "uu_o2": _StreamUu,
        "uv_o2": _StreamUv,
    }
    assert [type(m) for m in dest.interactions] == [classes[k] for k in kinds]

    # Prove a single evaluation of each needed basis and actual execution of
    # every CUDA kernel. In particular FP64 O3 must not resynthesize harmonics
    # from per-layer vectors. No Uv means Uu can skip the dense frame entirely.
    with ExitStack() as stack:
        sh_calls, hits, frames = [], [], []
        stack.callback(
            dest.o3_angular_basis.register_forward_hook(
                lambda *a: sh_calls.append(1)
            ).remove
        )
        packed = stack.enter_context(
            patch.object(
                dest.o2_angular_basis,
                "forward_packed",
                wraps=dest.o2_angular_basis.forward_packed,
            )
        )
        for k, m in zip(kinds, dest.interactions, strict=True):

            def check_kernel(mod, args, kwargs, kind=k):
                hits.append(kind)
                if kind == "cgtp":
                    assert kwargs.get("vectors") is None and args[1] is not None
                elif "uv_o2" in kinds:
                    assert kwargs.get("vectors") is None
                    frames.append(args[3] if kind == "uu_o2" else args[4])

            stack.callback(
                m.rejector.eqx_tp.register_forward_pre_hook(
                    check_kernel, with_kwargs=True
                ).remove
            )
        fast = evaluate(model)
        assert hits == list(kinds) and sh_calls == [1]
        assert packed.call_count == int("uv_o2" in kinds)
        assert all(frame is frames[0] for frame in frames)
    tol = 3e-4 if dtype == torch.float32 else 3e-7
    torch.testing.assert_close(fast, reference, atol=tol, rtol=tol)
    if "uu_o2" in kinds and "uv_o2" in kinds:
        uu = kinds.index("uu_o2")
        assert rep.interactions[uu].rejector.eqx_tp.with_transverse
        assert not dest.interactions[uu].rejector.eqx_tp.with_transverse
        restored = optimize(model, family="tace", backend="torch", task="training")
        original_uu = restored.readout_fn.representation.interactions[uu]
        assert type(original_uu) is UuO2Interaction
        assert original_uu.rejector.eqx_tp.with_transverse
    if kinds == ("uv_o2", "cgtp"):
        # A real user hook changes the harmonics. Keep its output, the dense
        # rotations, and every parameter derivative on the non-streaming path.
        calls = []

        def scaled(module, args, out):
            calls.append(True)
            return out * 0.4

        with rep.o3_angular_basis.register_forward_hook(scaled):
            custom = optimize(src, family="tace", backend="cuda", task="training")
            assert not any(
                isinstance(m, tuple(classes.values()))
                for m in custom.readout_fn.representation.interactions
            )
            torch.testing.assert_close(
                evaluate(custom), evaluate(src), atol=tol, rtol=tol
            )
            assert len(calls) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA Graph training")
@pytest.mark.parametrize("kinds", [("cgtp", "uu_o2"), ("uv_o2", "cgtp", "uu_o2")])
def test_mixed_spherical_training_bucket(kinds, monkeypatch):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace import training
    from equivariantx_foundation.tests.test_regional_aoti import _cases
    from equivariantx_foundation.tests.test_tace_plan import make_model
    from tace.models.spherical.tace import SphericalTACE
    from tace.utils.env import ACCELERATION_ENV

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    torch.set_default_dtype(torch.float32)

    def build(**cfg):
        cfg.update(mmax=1, num_layers=len(kinds))
        cfg["atomic_basis"]["type"] = list(kinds)
        cfg["radial_basis"]["apply_cutoff"] = False
        return SphericalTACE(**cfg)

    host, eager = (make_model(build).cuda().train() for _ in range(2))
    eager.load_state_dict(host.state_dict())
    opt, ref_opt = (torch.optim.SGD(x.parameters(), lr=0.02) for x in (host, eager))
    model = optimize(host, family="tace", backend="cuda", task="training")
    rep = model.readout_fn.representation
    assert training.supports_training_buckets(model)
    with monkeypatch.context() as mp:
        mp.setitem(
            rep.interactions._modules,
            "0",
            host.readout_fn.representation.interactions[0],
        )
        assert not training.supports_training_buckets(model)
    with monkeypatch.context() as mp:
        mp.setattr(rep.o3_angular_basis, "normalization", "norm")
        assert not training.supports_training_buckets(model)

    batches = _cases()
    for i, data in enumerate(batches):
        data["energy"] = data["positions"].new_tensor([0.23, -0.31]) + 0.1 * i
        data["forces"] = 0.2 * data["positions"].sin()
        data["stress"] = 0.03 * data["lattice"]
    cap = dict(max_atoms=6, max_edges=12)
    pack = training._pack

    def active_pack(data, *args, **kwargs):
        x = pack(data, *args, **kwargs)
        x["lattice"][-1] = torch.diag(x["lattice"].new_tensor([0.8, 1.1, 1.3]))
        x["energy"][-1] = 100
        x["forces"][len(data["positions"]) :] = 20
        x["stress"][-1] = -30
        return x

    monkeypatch.setattr(training, "_pack", active_pack)
    bucket = training.BucketedEFSTrainingStep(model, opt, batches[0], **cap)
    with torch.enable_grad():
        cold, hot = (
            model(fn(batches[0], *bucket.cap, edge_layout=bucket.edge_layout))[
                "energy"
            ][-1].detach()
            for fn in (pack, active_pack)
        )
    assert (cold - hot).abs() > 1e-12
    graph = bucket.step.graph
    for data in batches:
        ref_opt.zero_grad(set_to_none=True)
        pred = eager(dict(data))
        loss = ((pred["energy"] - data["energy"]) / data["ptr"].diff()).square().mean()
        for k in ("forces", "stress"):
            loss = loss + (pred[k] - data[k]).square().mean()
        loss.backward()
        ref_opt.step()
        out = bucket(data)
        assert bucket.step.graph is graph
        for key, expected in (
            ("loss", loss),
            *((k, pred[k]) for k in ("energy", "forces", "stress")),
        ):
            torch.testing.assert_close(
                out[key], expected, atol=3e-5, rtol=4e-4, msg=key
            )
        for (name, p), (ref_name, q) in zip(
            model.named_parameters(), eager.named_parameters(), strict=True
        ):
            assert name == ref_name and (p.grad is None) == (q.grad is None)
            if p.grad is not None:
                torch.testing.assert_close(
                    p.grad, q.grad, atol=3e-5, rtol=4e-4, msg=name
                )
            torch.testing.assert_close(p, q, atol=2e-6, rtol=3e-5, msg=name)
        for layer in (*rep.interactions, *rep.products):
            assert any(
                p.grad is not None and p.grad.numel() and p.grad.abs().max() > 0
                for p in layer.parameters()
            )


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="host-frame O3 CUDA radial stream"
)
def test_mixed_rotary_o3_stream_retarget_and_force_training(
    monkeypatch, double_precision
):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.interactions import (
        _HostFrameRadialInteraction,
    )
    from equivariantx_foundation.models.tace.tece_oam_rra.regional import (
        StreamedRotaryInteraction,
    )
    from equivariantx_foundation.tests.test_tace_plan import EFS, efs_loss, graph
    from equivariantx_foundation.tests.test_tece import _rotary_model
    from tace.models.spherical.inter import O3CgtpInteraction
    from tace.models.spherical.tece_oam_rra import UvSO2Interaction

    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    host = _rotary_model().cuda().double().train()
    params = dict(host.named_parameters(remove_duplicate=False))

    def select(model, backend):
        return optimize(
            model, family="tace", backend=backend, task="training", regional_only=True
        )

    def evaluate(model):
        result = model(graph("cuda"))
        grads = torch.autograd.grad(
            efs_loss(result), tuple(model.parameters()), allow_unused=True
        )
        return tuple(result[k].detach() for k in EFS), grads

    ref = evaluate(host)
    fast = select(host, "cuda")
    portable = select(fast, "torch")
    again = select(portable, "cuda")
    for model, backend in ((fast, "cuda"), (portable, "torch"), (again, "cuda")):
        assert select(model, backend) is model
        active = dict(model.named_parameters(remove_duplicate=False))
        assert list(active) == list(params) and all(
            active[n] is p for n, p in params.items()
        )
        assert tuple(model.state_dict()) == tuple(host.state_dict())
        rep = model.readout_fn.representation
        first, second = rep.interactions
        assert not rep.use_packed_wigner and not first.use_eqx
        assert type(first) is (
            _HostFrameRadialInteraction if backend == "cuda" else O3CgtpInteraction
        )
        assert type(second) is (
            StreamedRotaryInteraction if backend == "cuda" else UvSO2Interaction
        )
        op = first.rejector.eqx_tp
        assert op.backend == backend
        received = []

        def check(module, args, kwargs):
            # A host-produced harmonic tensor enters both backends. Only the
            # CUDA schedule receives the radial MLP itself rather than the
            # materialized matrix of per-edge coefficients.
            received.append(
                (
                    args[1] is not None,
                    kwargs.get("vectors"),
                    kwargs.get("radial_network") is not None,
                )
            )

        with (
            op.register_forward_pre_hook(check, with_kwargs=True),
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU]
            ) as prof,
        ):
            actual = evaluate(model)
        assert received == [(True, None, backend == "cuda")]
        calls = {event.key: event.count for event in prof.key_averages()}
        assert calls.get("eqx_foundation::tece_interaction", 0) == int(
            backend == "cuda"
        )
        if backend == "cuda":
            assert calls.get("eqx::edge_program", 0) > 0
        torch.testing.assert_close(actual, ref, rtol=1e-7, atol=5e-8)

    # The host accepts other gather orders. A runtime producer replacement
    # must keep that meaning, even if shapes/dtypes still match the fast path.
    def reverse_gathers(module, args, parts):
        edge, (target, dst), (source, src) = parts
        return edge, (target, src), (source, dst)

    rep = again.readout_fn.representation
    networks = []
    with (
        rep.edge_updates[0].register_forward_hook(reverse_gathers),
        host.readout_fn.representation.edge_updates[0].register_forward_hook(
            reverse_gathers
        ),
        rep.interactions[0].rejector.eqx_tp.register_forward_pre_hook(
            lambda m, args, kwargs: networks.append(kwargs.get("radial_network")),
            with_kwargs=True,
        ),
    ):
        torch.testing.assert_close(
            evaluate(again), evaluate(host), rtol=1e-7, atol=5e-8
        )
    assert networks == [None]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="host-frame O3 CUDA admission"
)
def test_mixed_rotary_o3_respects_noncanonical_host(monkeypatch):
    from equivariantx_foundation import optimize
    from equivariantx_foundation.models.tace.interactions import (
        _HostFrameRadialInteraction,
    )
    from equivariantx_foundation.tests.test_tece import _rotary_model

    monkeypatch.setenv("TACE_USE_EQX", "0")
    monkeypatch.setenv("TACE_MEMORY_FIRST", "1")
    host = _rotary_model().cuda().float().train()
    rep = host.readout_fn.representation

    def uses_stream(backend="cuda"):
        model = optimize(
            host, family="tace", task="training", regional_only=True, backend=backend
        )
        return (
            type(model.readout_fn.representation.interactions[0])
            is _HostFrameRadialInteraction
        )

    assert uses_stream() and not uses_stream(None)
    with monkeypatch.context() as m:
        m.setattr(rep, "reciprocal_radial", True)
        assert not uses_stream()
    with rep.edge_updates[0].register_forward_hook(lambda *args: None):
        assert not uses_stream()
    with monkeypatch.context() as m:
        m.setattr(rep.o3_angular_basis, "normalization", "norm")
        assert not uses_stream()
    with monkeypatch.context() as m:
        m.setenv("TACE_MEMORY_FIRST", "0")
        assert not uses_stream()

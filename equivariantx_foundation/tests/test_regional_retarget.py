"""Changing providers on a live, already-planned ordinary TACE representation."""

from contextlib import ExitStack
from unittest.mock import patch

import pytest
import torch

from equivariantx_foundation.models.tace.planner import optimize_model, plan
from equivariantx_foundation.tests.test_tace_plan import EFS, graph, make_model
from tace.models.spherical.tace import SphericalTACE
from tace.utils.env import ACCELERATION_ENV


def test_untracked_lowered_block_requires_original_source(monkeypatch):
    from equivariantx_foundation.tests.test_tace_plan import lower_model

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    model = lower_model(make_model(), backend="torch")
    del model.readout_fn.representation.products[0]._foundation_regional_provider
    with pytest.raises(
        ValueError, match="identify the installed provider at products.0"
    ):
        lower_model(model, backend="torch")


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="switch providers on the same physical GPU"
)
@pytest.mark.parametrize("kind", ["cgtp", "uu_o2"])
def test_switches_installed_providers_without_replacing_user_state(kind, monkeypatch):
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    torch.set_default_dtype(torch.float64)

    def build(**cfg):
        cfg["atomic_basis"]["type"] = kind
        cfg["radial_basis"]["apply_cutoff"] = False
        return SphericalTACE(**cfg)

    source = make_model(build).cuda().double().eval()
    model = optimize_model(source, backend="cuda")
    rep = model.readout_fn.representation
    assert optimize_model(model, backend="cuda") is model
    # An optimizer update, user-owned metadata, and a callback OUTSIDE a
    # replaced kernel must survive a complete round trip.
    with torch.no_grad():
        next(model.parameters()).add_(0.003)
    rep.interactions[1].residual_scale *= 0.8
    rep.run_note = "post-conversion annotation"
    observed = []
    model.readout_fn.register_forward_hook(lambda *_: observed.append(True))
    expected = model(graph("cuda"))
    # Never discard a callback that observes a provider-specific module.
    with rep.products[0].scheduled_contraction.register_forward_hook(lambda *_: None):
        assert optimize_model(model, backend="cuda") is model
        with pytest.raises(ValueError, match="Cannot retarget products.0"):
            optimize_model(model, backend="torch")
    private = rep.products[0].scheduled_contraction
    private.register_buffer("user_calibration", torch.ones(1, device="cuda"))
    with pytest.raises(ValueError, match="customized symmetric-contraction"):
        optimize_model(model, backend="torch")
    assert private.user_calibration.item() == 1
    del private.user_calibration
    torch_model = optimize_model(model, backend="torch")
    assert optimize_model(torch_model, backend="torch") is torch_model
    r2 = torch_model.readout_fn.representation
    assert (
        r2.run_note == rep.run_note
        and r2.interactions[1].residual_scale == rep.interactions[1].residual_scale
    )
    assert all(
        p is q
        for p, q in zip(torch_model.parameters(), model.parameters(), strict=True)
    )
    assert all(p.scheduled_contraction.backend == "torch" for p in r2.products)
    assert not any(
        m.use_eqx_linear for m in r2.modules() if hasattr(m, "use_eqx_linear")
    )
    counters = []

    def observe(*_):
        counters.append(True)

    with ExitStack() as hooks:
        for prod in r2.products:
            hooks.enter_context(
                prod.scheduled_contraction.register_forward_hook(observe)
            )
        for layer in r2.interactions:
            assert not layer.use_eqx
            if kind == "cgtp":
                assert layer.rejector.eqx_tp.backend == "torch"
                hooks.enter_context(
                    layer.rejector.eqx_tp.register_forward_hook(observe)
                )
            else:
                assert (
                    layer.rejector.local_frame_in.backend
                    == layer.rejector.local_frame_out.backend
                    == "torch"
                )
        if kind == "uu_o2":
            from equivariantx.ops.cuda.o2 import local_frame

            for method in ("to_local", "to_global", "gather", "scatter"):
                hooks.enter_context(
                    patch.object(
                        local_frame,
                        method,
                        side_effect=AssertionError("old CUDA frame used"),
                    )
                )
        actual = torch_model(graph("cuda"))
    assert len(counters) == (4 if kind == "cgtp" else 2)
    for key in EFS:
        torch.testing.assert_close(actual[key], expected[key], atol=1e-8, rtol=2e-7)
    back = optimize_model(torch_model, backend="cuda")
    assert optimize_model(back, backend="cuda") is back
    r3 = back.readout_fn.representation
    assert all(layer.use_eqx for layer in r3.interactions)
    assert all(p.scheduled_contraction.backend == "cuda" for p in r3.products)
    assert all(
        r.implementation == "cuda"
        for r in plan(back, backend="cuda").regions
        if r.scope == "representation"
    )
    again = back(graph("cuda"))
    for key in EFS:
        torch.testing.assert_close(again[key], expected[key], atol=1e-8, rtol=2e-7)
    assert len(observed) == 3


def _norm_model():
    from tace.utils.precision import torch_default_dtype

    def build(**cfg):
        cfg["layer_norm"]["pre_norm_type"] = "merge_rms_norm"
        cfg["resnet"]["type"] = "BAB"
        return SphericalTACE(**cfg)

    with torch_default_dtype(torch.float32):
        return make_model(build).cuda().float().train()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA RMS provider")
def test_merge_rms_norm_training_round_trip(monkeypatch):
    from equivariantx.ops.cuda.nn import layer_norm
    from equivariantx_foundation.tests.test_tace_plan import efs_loss

    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    torch.manual_seed(43)
    source = _norm_model()
    # The release saves analytic block weights as FP32, not Python doubles.
    norm1 = source.readout_fn.representation.interactions[1].norm1
    norm1.block_weights = tuple(
        torch.tensor(norm1.default_block_weights, dtype=torch.float32).tolist()
    )
    assert norm1.block_weights != norm1.default_block_weights
    opts = dict(task="training", regional_only=True)
    paths = {"interactions.1.norm1", "interactions.1.norm2"}

    def leaves(m, backend):
        return {
            r.path: r.implementation
            for r in plan(m, backend=backend, **opts).regions
            if r.path in paths
        }

    assert not leaves(source, None)
    assert not leaves(source, "torch")
    assert leaves(source, "cuda") == dict.fromkeys(paths, "cuda")
    cuda = optimize_model(source, backend="cuda", **opts)
    assert optimize_model(cuda, backend="cuda", **opts) is cuda
    assert leaves(cuda, None) == dict.fromkeys(paths, "cuda")
    back = optimize_model(cuda, backend="torch", **opts)
    assert leaves(back, None) == dict.fromkeys(paths, "torch")
    assert leaves(back, "cuda") == dict.fromkeys(paths, "cuda")
    original = dict(source.named_parameters())

    def run(m):
        data = {
            k: v.float() if v.is_floating_point() else v
            for k, v in graph("cuda").items()
        }
        out = m(data)
        grads = torch.autograd.grad(
            efs_loss(out), tuple(original.values()), allow_unused=True
        )
        return {k: out[k].detach() for k in EFS}, grads

    outputs, _ = run(source)
    source_state = source.state_dict()
    for m, backend in ((cuda, "cuda"), (back, "torch")):
        params = dict(m.named_parameters())
        assert tuple(params) == tuple(original)
        assert all(p is original[n] for n, p in params.items())
        state = m.state_dict()
        assert tuple(state) == tuple(source_state)
        assert all(torch.equal(v, source_state[n]) for n, v in state.items())
        for path in paths:
            assert (
                source.readout_fn.representation.get_submodule(path).backend == "torch"
            )
            assert m.readout_fn.representation.get_submodule(path).backend == backend
        # Isolate normalization in the mixed gradients: the unrelated TP/ACE
        # regional providers need not perform their FP32 reductions in the
        # same order as the native model.
        with monkeypatch.context() as mp:
            for path in paths:
                mp.setattr(
                    m.readout_fn.representation.get_submodule(path), "backend", "torch"
                )
            _, grads = run(m)
        with patch.object(
            layer_norm, "normalize", wraps=layer_norm.normalize
        ) as kernel:
            got, dg = run(m)
        assert kernel.call_count == (len(paths) if backend == "cuda" else 0)
        for k in EFS:
            torch.testing.assert_close(got[k], outputs[k], atol=4e-5, rtol=1e-4)
        for n, x, y in zip(original, dg, grads, strict=True):
            assert (x is None) == (y is None), n
            if x is not None:
                torch.testing.assert_close(
                    x, y, atol=6e-5, rtol=3e-4, msg=lambda s: f"{n}: {s}"
                )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="explicit CUDA RMS provider")
def test_merge_rms_norm_leaves_host_ownership_and_custom_behavior(monkeypatch):
    for flag in ACCELERATION_ENV.values():
        monkeypatch.setenv(flag, "0")
    source = _norm_model()
    layer = source.readout_fn.representation.interactions[1]

    def paths(backend):
        return {
            r.path for r in plan(source, backend=backend, regional_only=True).regions
        }

    leaf = "interactions.1.norm1"
    assert leaf in paths("cuda")
    with layer.norm1.register_forward_hook(lambda *_: None):
        assert leaf not in paths("cuda")
    with monkeypatch.context() as mp:
        mp.setattr(layer.norm1, "inverse_rms", lambda x: x[:, :1])
        assert leaf not in paths("cuda")
    with monkeypatch.context() as mp:
        mp.setattr(layer.norm1, "backend", "cuda")
        assert leaf not in paths("cuda") and leaf not in paths("torch")
        back = optimize_model(source, backend="torch", regional_only=True)
        assert back.readout_fn.representation.interactions[1].norm1.backend == "cuda"
    with monkeypatch.context() as mp:
        mp.setenv("TACE_USE_EQX", "1")
        assert leaf not in paths("cuda")
    with monkeypatch.context() as mp:
        mp.setattr(layer.norm1, "block_weights", tuple(1.0 for _ in layer.norm1.widths))
        assert leaf not in paths("cuda")
    layer.norm1.double()
    assert leaf not in paths("cuda")

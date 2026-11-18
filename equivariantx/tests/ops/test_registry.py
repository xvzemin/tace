"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx.ops import config, get_kernel, register_kernel, resolve_backend
from equivariantx.ops.registry import IMPLEMENTATIONS, REFERENCES


def test_registered_providers_are_paired():
    names = {name for name, _ in IMPLEMENTATIONS}
    assert all(
        (name, backend) in IMPLEMENTATIONS
        for name in names
        for backend in ("torch", "cuda")
    )
    assert names == set(REFERENCES)


def test_geometry_providers_have_independent_references(monkeypatch, double_precision):
    from e3nn import o3

    device = "cuda" if torch.cuda.is_available() else "cpu"

    def rand(*shape):
        return torch.randn(*shape, device=device, dtype=torch.float64)

    e = 4
    q, k = rand(e, 4, 6), rand(e, 4, 6)
    phase, score = rand(e, 2, 2, 2), rand(e, 2)
    cg = o3.wigner_3j(1, 1, 2, dtype=torch.float64, device=device)
    x = rand(e, 3)
    quat = rand(e, 4) / 3
    cases = [
        (
            "alignment_cuda",
            (repr(key), [x] if key != "direction_gradient" else [x, rand(e, 3)]),
        )
        for key in (None, "quaternion", "normalize", "direction_gradient")
    ]
    cases += [
        ("rotation", (cg, i, [rand(e, 9), rand(e, 9), rand(e, 25)])) for i in range(3)
    ]
    cases += [("packed_wigner", (x, 2, [cg]))]
    cases += [
        (
            "quaternion_polynomial",
            (2, trans, [quat, rand(e, 4)] + ([rand(e, 35)] if trans else [])),
        )
        for trans in (False, True)
    ]
    cases += [
        (
            "rotary_product",
            (q, k, phase, score if role != 3 else score[:0], 1, 1, 2, role),
        )
        for role in range(4)
    ]
    cases += [
        (
            "frame_rotation",
            (q, rand(e, 3, 4), rand(e, 3, 6), repr(((1, 1, 2, 2), (1, 2, 1))), role),
        )
        for role in range(3)
    ]
    for name, args in cases:
        results = []
        for ref in (False, True):
            monkeypatch.setattr(config, "TORCH_REFERENCE", ref)
            results.append(get_kernel(name, "torch")(*args))
        torch.testing.assert_close(*results, rtol=1e-10, atol=1e-11)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
@pytest.mark.parametrize("product", ["GauntTensorProduct", "MatrixTensorProduct"])
def test_factorized_cuda_reference_isolation(product, monkeypatch):
    from unittest.mock import patch

    from equivariantx import o3
    from equivariantx.ops.cuda.o3 import tensor_product as kernel

    layer = (
        getattr(o3, product)(
            "2x1o",
            "2x1o",
            "2x0e+2x1e+2x2e",
            backend="cuda",
            internal_weights=True,
            shared_weights=True,
        )
        .cuda()
        .double()
    )
    x, y = torch.randn(2, 3, 6, device="cuda", dtype=torch.float64)
    with patch.object(kernel, "angular", wraps=kernel.angular) as angular:
        values = []
        for reference in (False, True):
            monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
            angular.reset_mock()
            values.append(layer(x, y))
            assert angular.called
    torch.testing.assert_close(*values)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_backend_selection(monkeypatch, backend):
    monkeypatch.setattr(config, "DEFAULT_BACKEND", backend)

    def probe():
        raise AssertionError("Backend selection must not probe CUDA at runtime.")

    monkeypatch.setattr(torch.cuda, "is_available", probe)
    assert resolve_backend() == backend
    monkeypatch.setitem(IMPLEMENTATIONS, ("selection_test", backend), torch.sin)
    assert get_kernel("selection_test") is get_kernel("selection_test", backend)
    assert resolve_backend(None, "cpu") == "torch"
    assert resolve_backend(None, "cuda:0") == backend
    assert resolve_backend("torch") == "torch"
    assert resolve_backend("cuda") == "cuda"
    with pytest.raises(ValueError, match="backend"):
        resolve_backend("invalid")


@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("space", ["o3", "co3", "o2", "co2"])
def test_default_tensor_product_backend(monkeypatch, backend, space):
    import equivariantx as eqx

    monkeypatch.setattr(config, "DEFAULT_BACKEND", backend)
    group = getattr(eqx, space)
    irrep = "1o" if space.endswith("3") else "1m"
    args = (f"2x{irrep}", f"2x{irrep}", "2x0e", [(0, 0, 0, "uuu", True)])
    automatic = group.TensorProduct(*args)
    explicit = group.TensorProduct(*args, backend="torch")
    explicit.load_state_dict(automatic.state_dict())
    assert automatic.backend == backend
    assert explicit.backend == "torch"
    # CPU inputs must not require CUDA even on a CUDA-capable host.
    x = torch.randn(3, automatic.irreps_in1.dim, requires_grad=True)
    y = torch.randn(3, automatic.irreps_in2.dim, requires_grad=True)
    actual, expected = automatic(x, y), explicit(x, y)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(
        torch.autograd.grad(actual.square().sum(), (x, y)),
        torch.autograd.grad(expected.square().sum(), (x, y)),
    )


@pytest.mark.parametrize("reference", [False, True])
def test_dispatch_rejects_reduced_storage(monkeypatch, reference):
    monkeypatch.setitem(IMPLEMENTATIONS, ("storage_test", "torch"), torch.sin)
    monkeypatch.setattr(config, "TORCH_REFERENCE", reference)
    with pytest.raises(TypeError, match="float32 or float64"):
        get_kernel("storage_test", "torch")(torch.ones(3, dtype=torch.bfloat16))


def test_compiled_reference_switch(monkeypatch):
    monkeypatch.setitem(IMPLEMENTATIONS, ("compiled_switch_test", "torch"), torch.sin)
    monkeypatch.setitem(REFERENCES, "compiled_switch_test", torch.cos)

    def function(value):
        return get_kernel("compiled_switch_test", "torch")(value)

    compiled = torch.compile(function, backend="eager", fullgraph=True)
    value = torch.randn(3)
    for enabled in (False, True, False):
        monkeypatch.setattr(config, "TORCH_REFERENCE", enabled)
        expected = torch.cos(value) if enabled else torch.sin(value)
        torch.testing.assert_close(compiled(value), expected)


def test_provider_extension(monkeypatch):
    monkeypatch.setitem(
        IMPLEMENTATIONS, ("test_extension", "future_device"), "torch:sin"
    )
    value = torch.randn(3)
    torch.testing.assert_close(
        get_kernel("test_extension", "future_device")(value), value.sin()
    )
    with pytest.raises(ValueError, match="already has"):
        register_kernel("test_extension", "future_device", torch.cos)


def test_reference_switch_changes_cached_dispatch(monkeypatch):
    monkeypatch.setitem(IMPLEMENTATIONS, ("switch_test", "torch"), torch.sin)
    monkeypatch.setitem(REFERENCES, "switch_test", torch.cos)
    value = torch.randn(4)
    for enabled in (False, True, False):
        monkeypatch.setattr(config, "TORCH_REFERENCE", enabled)
        expected = torch.cos(value) if enabled else torch.sin(value)
        torch.testing.assert_close(get_kernel("switch_test", "torch")(value), expected)


def test_execution_provider(monkeypatch):
    from equivariantx.ops import get_kernel, register_kernel
    from equivariantx.ops.registry import IMPLEMENTATIONS

    monkeypatch.setitem(IMPLEMENTATIONS, ("test_sine", "torch"), "torch:sin")
    value = torch.randn(3)
    torch.testing.assert_close(
        get_kernel("test_sine", "torch")(value), torch.sin(value)
    )
    with pytest.raises(ValueError, match="already has"):
        register_kernel("test_sine", "torch", torch.cos)
    with pytest.raises(NotImplementedError, match="does not support"):
        get_kernel("test_sine", "unknown")

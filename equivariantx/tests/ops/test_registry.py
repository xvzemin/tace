"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx.ops import config, get_kernel, register_kernel, resolve_backend
from equivariantx.ops.registry import IMPLEMENTATIONS, REFERENCES


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_backend_selection(monkeypatch, backend):
    monkeypatch.setattr(config, "DEFAULT_BACKEND", backend)

    def probe():
        raise AssertionError("Backend selection must not probe CUDA at runtime.")

    monkeypatch.setattr(torch.cuda, "is_available", probe)
    assert resolve_backend() == backend
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
        get_kernel("storage_test")(torch.ones(3, dtype=torch.bfloat16))


def test_compiled_reference_switch(monkeypatch):
    monkeypatch.setitem(IMPLEMENTATIONS, ("compiled_switch_test", "torch"), torch.sin)
    monkeypatch.setitem(REFERENCES, "compiled_switch_test", torch.cos)

    def function(value):
        return get_kernel("compiled_switch_test")(value)

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
        torch.testing.assert_close(get_kernel("switch_test")(value), expected)


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

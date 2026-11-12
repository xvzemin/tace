"""Tests for the corresponding EQX source module."""

import pytest
import torch

from equivariantx.ops import config, get_kernel, register_kernel
from equivariantx.ops.registry import IMPLEMENTATIONS, REFERENCES


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

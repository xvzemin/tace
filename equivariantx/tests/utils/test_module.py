"""Tests for the corresponding EQX source module."""

import pytest
import torch
from equivariantx.utils.module import convert_modules, copy_model, default_dtype


def test_default_dtype():
    previous = torch.get_default_dtype()
    with default_dtype(torch.float64):
        assert torch.get_default_dtype() == torch.float64
        with pytest.raises(RuntimeError), default_dtype(torch.float32):
            assert torch.get_default_dtype() == torch.float32
            raise RuntimeError("construction failed")
        assert torch.get_default_dtype() == torch.float64
    assert torch.get_default_dtype() == previous


def test_copy_model():
    model = torch.nn.Linear(3, 2)
    model.register_buffer("scale", torch.ones(2))
    copied = copy_model(model)
    for name, value in model.state_dict().items():
        other = copied.state_dict()[name]
        torch.testing.assert_close(other, value)
        assert other.data_ptr() != value.data_ptr()


@pytest.mark.parametrize("inplace", [False, True])
def test_convert_modules(inplace):
    shared = torch.nn.Linear(3, 3)
    model = torch.nn.Sequential(shared, shared)
    calls = []

    def factory(module):
        calls.append(module)
        if isinstance(module, torch.nn.Linear):
            return torch.nn.Sequential(module, torch.nn.Identity())
        return None

    converted = convert_modules(model, factory, inplace=inplace)
    assert (converted is model) == inplace
    assert len(calls) == 2
    assert converted[0] is converted[1]
    assert isinstance(converted[0], torch.nn.Sequential)
    torch.testing.assert_close(converted[0][0].weight, shared.weight)
    if inplace:
        assert converted[0][0] is shared
    else:
        assert model[0] is model[1] is shared
        assert converted[0][0] is not shared

    replacement = torch.nn.Identity()
    assert convert_modules(model, lambda module: replacement) is replacement

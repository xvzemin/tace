"""Module utilities and executable documentation examples."""

import re
import textwrap
from pathlib import Path

import pytest
import torch

from eqx.utils import convert_modules, copy_model, default_dtype


@pytest.mark.parametrize("page", ["installation", "api/o2", "api/tools"])
def test_documentation_examples(page):
    path = Path(__file__).resolve().parents[1] / "docs/source" / f"{page}.rst"
    blocks = re.findall(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*|)\n)+)",
        path.read_text(),
        re.MULTILINE,
    )
    assert blocks
    namespace = {"__name__": "__main__"}
    with default_dtype(torch.float64), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        for block in blocks:
            exec(compile(textwrap.dedent(block), str(path), "exec"), namespace)


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

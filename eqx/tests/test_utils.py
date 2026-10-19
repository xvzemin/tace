"""Module utilities and executable documentation examples."""

import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

from eqx.utils import convert_modules, copy_model, default_dtype

DOCS = Path(__file__).resolve().parents[1] / "docs/source"


@pytest.mark.parametrize("allow_slice", [False, True])
def test_import(tmp_path, allow_slice):
    root = Path(__file__).resolve().parents[1]
    names = sorted(
        {
            name
            for path in root.rglob("*.py")
            for name in re.findall(
                r'@torch\.library\.custom_op\(\s*"(eqx::[^"]+)"',
                path.read_text(encoding="utf-8"),
            )
        }
    )
    assert names
    script = f"""
import sys
import torch

sys.modules["torch.utils.cpp_extension"] = None
if {allow_slice}:
    torch.serialization.add_safe_globals([slice])
safe_globals = set(torch.serialization.get_safe_globals())
load = torch.load
import eqx
from eqx import o2

assert safe_globals.issubset(torch.serialization.get_safe_globals())
assert (slice in torch.serialization.get_safe_globals()) == (slice in safe_globals)
assert torch.load is load
for name in {names!r}:
    assert torch._C._dispatch_find_schema_or_throw(name, "").schema().name == name
linear = o2.Linear("2x0e+2x1m", "2x0e+2x1m")
linear(linear.irreps_in.randn(3, -1)).sum().backward()
frame = o2.LocalFrame("2x0e+2x1o")
rotation = o2.WignerD(mmax=1, lmax=1, method="recursive")
d, di = rotation(torch.randn(3, 3))
x = torch.randn(3, frame.input_dim)
frame.to_global(frame.to_local(x, d), di)
assert set(eqx.__all__).issubset(dir(eqx))
assert not torch.cuda.is_initialized()
assert "tace" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=root.parent,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "path",
    [
        path
        for path in sorted(DOCS.rglob("*.rst"))
        if ".. code-block:: python" in path.read_text(encoding="utf-8")
    ],
    ids=lambda path: path.relative_to(DOCS).as_posix(),
)
def test_documentation_examples(path):
    blocks = re.findall(
        r"^\.\. code-block:: python\n\n((?:(?:   [^\n]*|)\n)+)",
        path.read_text(encoding="utf-8"),
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

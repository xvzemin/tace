"""Tests for the corresponding EQX source module."""

import equivariantx
import doctest
import os
import re
import subprocess
import sys
from importlib import import_module
from pathlib import Path
import pytest
import torch
from equivariantx.utils.module import default_dtype

ROOT = Path(equivariantx.__file__).resolve().parent
from equivariantx.tests.utils.testing import has_documentation_examples


@pytest.mark.parametrize("allow_slice", [False, True])
def test_import(tmp_path, allow_slice):
    names = sorted(
        {
            name
            for path in ROOT.rglob("*.py")
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
sys.modules["torch_geometric"] = None
sys.modules["torch_scatter"] = None
if {allow_slice}:
    torch.serialization.add_safe_globals([slice])
safe_globals = set(torch.serialization.get_safe_globals())
load = torch.load
import equivariantx as eqx
from equivariantx import o2

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
assert not any(name.startswith("equivariantx.ops.kernels.cuda") for name in sys.modules)
assert not any(name.startswith("equivariantx.models.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT.parent,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "", "MPLCONFIGDIR": str(tmp_path)},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optional_dependencies(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    script = r"""
import sys
import torch

sys.modules.update({name: None for name in ("torch_geometric", "torch_scatter")})
from equivariantx import conv as eqx_conv
from equivariantx import o2

torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
device = sys.argv[1]
tp = o2.O3TensorProduct(
    "2x1o", "1o", "2x0e+2x1e+2x2e",
    [(0, 0, i, "uvu", True) for i in range(3)],
    internal_weights=False, shared_weights=False,
).to(device)
frame = o2.WignerD(2, 2, method="recursive").to(device)
x = torch.randn(3, 6, device=device, requires_grad=True)
vectors = torch.randn(4, 3, device=device, requires_grad=True)
weights = torch.randn(4, tp.weight_numel, device=device, requires_grad=True)
edges = torch.tensor([[0, 1, 2, 0], [1, 2, 0, 2]], device=device)
d, di = frame(vectors)
expected = x.new_zeros(3, tp.irreps_out.dim).index_add(
    0, edges[1], tp(x[edges[0]], d, di, weights)
)
packed = frame.forward_packed(vectors)
actual = tp.forward_scatter(x, edges, packed, weights)
torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
inputs = x, vectors, weights
for _ in range(2):
    a = torch.autograd.grad(actual.square().sum(), inputs, create_graph=True)
    b = torch.autograd.grad(expected.square().sum(), inputs, create_graph=True)
    for left, right in zip(a, b):
        torch.testing.assert_close(left, right, atol=1e-8, rtol=1e-8)
    actual = torch.cat([value.flatten() for value in a])
    expected = torch.cat([value.flatten() for value in b])
assert "equivariantx.ops.kernels.cuda.o2_o3.convolution" not in sys.modules
from equivariantx.ops.kernels.cuda import runtime as cuda
assert not cuda.runtime.cache_info().currsize
assert "tace" not in sys.modules
accelerated = eqx_conv.O2O3TensorProductConv(tp, backend="cuda").to(device)
arguments = (
    x, weights, x.new_empty(0, tp.weight_numel), packed,
    x.new_ones(1, tp.num_harmonics), edges, 3,
)
if device == "cpu":
    torch.testing.assert_close(accelerated(*arguments), tp.forward_scatter(x, edges, packed, weights))
else:
    import torch.utils.cpp_extension
    cuda_home = torch.utils.cpp_extension.CUDA_HOME
    torch.utils.cpp_extension.CUDA_HOME = None
    try:
        accelerated(*arguments)
    except ImportError as error:
        assert "CUDA_HOME" in str(error)
    else:
        raise AssertionError("CUDA execution must report the missing toolkit")
    finally:
        torch.utils.cpp_extension.CUDA_HOME = cuda_home
    quaternion_frame = o2.WignerD(2, 2).to(device)
    for actual, expected in zip(quaternion_frame(vectors), (d, di)):
        torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    assert "tace" not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", script, device],
        cwd=ROOT.parent,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize(
    "path",
    [
        path
        for path in sorted(ROOT.rglob("*.py"))
        if "tests" not in path.relative_to(ROOT).parts
        and has_documentation_examples(path)
    ],
    ids=lambda path: path.relative_to(ROOT).as_posix(),
)
def test_documentation_examples(path):
    module = import_module(
        ".".join(path.relative_to(ROOT.parent).with_suffix("").parts)
    )
    with default_dtype(torch.float64), torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        result = doctest.testmod(module, raise_on_error=True)
    assert result.attempted > 0

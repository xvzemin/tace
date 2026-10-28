"""Module utilities, kernel runtime, and executable documentation examples."""

import doctest
import logging
import os
import re
import runpy
import subprocess
import sys
from importlib import import_module
from pathlib import Path
from unittest.mock import MagicMock, Mock

import pytest
import torch

import eqx.backends.cuda.runtime as cuda
from eqx.utils.module import convert_modules, copy_model, default_dtype

ROOT = Path(__file__).resolve().parents[1]


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
assert not any(name.startswith("eqx.backends.cuda") for name in sys.modules)
assert not any(name.startswith("eqx.models.") for name in sys.modules)
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


def test_execution_provider(monkeypatch):
    from eqx.backends import get_kernel, register_kernel
    from eqx.backends.registry import IMPLEMENTATIONS

    monkeypatch.setitem(IMPLEMENTATIONS, ("test_sine", "torch"), "torch:sin")
    get_kernel.cache_clear()
    assert get_kernel("test_sine", "torch") is torch.sin
    with pytest.raises(ValueError, match="already has"):
        register_kernel("test_sine", "torch", torch.cos)
    with pytest.raises(NotImplementedError, match="does not support"):
        get_kernel("test_sine", "unknown")
    get_kernel.cache_clear()


def test_tensor_product_exports():
    from e3nn import o3 as e3nn_o3

    from eqx import conv, o3

    for name in (
        "TensorProduct",
        "FullyConnectedTensorProduct",
        "FullTensorProduct",
        "ElementwiseTensorProduct",
    ):
        assert getattr(o3, name) is getattr(o3.cgtp, name)
        assert getattr(o3, name) is getattr(e3nn_o3, name)
    for name, module in (
        ("O3TensorProductConv", o3.cgtp),
        ("GauntTensorProductConv", o3.gtp),
        ("MatrixTensorProductConv", o3.mtp),
        ("Wigner6jTensorProductConv", o3.wigner6j),
    ):
        assert getattr(o3, name) is getattr(module, name)
        assert getattr(o3, name) is getattr(conv, name)


@pytest.mark.parametrize(
    "path",
    [
        path
        for path in sorted(ROOT.rglob("*.py"))
        if "tests" not in path.relative_to(ROOT).parts
        and ">>> " in path.read_text(encoding="utf-8")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_aoti_reload(tmp_path):
    from eqx.ops.layout import gather_sum

    class Model(torch.nn.Module):
        def forward(self, features, indices):
            return gather_sum([features, features], [indices, indices])

    inputs = (
        torch.randn(4, 3, device="cuda", dtype=torch.float64),
        torch.tensor([2, 0, 3, 1, 0], device="cuda"),
    )
    model = Model().eval()
    torch.testing.assert_close(model(*inputs), 2 * inputs[0][inputs[1]])
    exported = torch.export.export(
        model,
        inputs,
        dynamic_shapes={
            "features": {0: torch.export.Dim("nodes", min=2)},
            "indices": {0: torch.export.Dim("edges", min=1)},
        },
    )
    path = tmp_path / "model.pt2"
    torch._inductor.aoti_compile_and_package(exported, package_path=str(path))
    script = """
import sys
import torch
import eqx

assert "tace" not in sys.modules
model = torch._inductor.aoti_load_package(sys.argv[1])
for nodes, edges in ((4, 5), (7, 11)):
    features = torch.randn(nodes, 3, device="cuda", dtype=torch.float64)
    indices = torch.arange(edges, device="cuda") % nodes
    torch.testing.assert_close(model(features, indices), 2 * features[indices])
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(path)],
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("platform", ["linux", "win32"])
def test_runtime_linker_flags(monkeypatch, tmp_path, platform):
    import torch.utils.cpp_extension as extension

    namespace = runpy.run_path(cuda.__file__)
    load = Mock(return_value=object())
    dll_directory = MagicMock()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "add_dll_directory", dll_directory, raising=False)
    monkeypatch.setattr(extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(extension, "load", load)
    try:
        assert namespace["runtime"]() is load.return_value
        kwargs = load.call_args.kwargs
        if platform == "win32":
            dll_directory.assert_called_once_with(str(tmp_path / "bin"))
            assert kwargs["extra_cflags"] == ["/O2"]
            assert kwargs["extra_ldflags"] == [
                f"/LIBPATH:{tmp_path / 'lib' / 'x64'}",
                "nvrtc.lib",
                "cuda.lib",
            ]
        else:
            dll_directory.assert_not_called()
            assert kwargs["extra_cflags"] == ["-O3"]
            assert "-lnvrtc" in kwargs["extra_ldflags"]
            assert "-lcuda" in kwargs["extra_ldflags"]
    finally:
        namespace["_POOL"].shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("edges", [0, 17])
@pytest.mark.parametrize("first_indexed", [False, True])
def test_gather_sum(dtype, edges, first_indexed):
    from eqx.ops.layout import gather_sum

    indices = [torch.randint(5, (2 * edges,), device="cuda")[::2] for _ in range(3)]
    if not first_indexed:
        indices[0] = None
    inputs = [
        torch.randn(9, 2 * (edges if index is None else 5), device="cuda", dtype=dtype)
        .T[::2]
        .requires_grad_()
        for index in indices
    ]
    reference = sum(
        value if index is None else value.index_select(0, index)
        for value, index in zip(inputs, indices)
    )
    actual = gather_sum(inputs, indices)
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    outputs = [
        value.sin().sum() / max(value.numel(), 1) for value in (actual, reference)
    ]
    for _ in range(3):
        gradients = [
            torch.autograd.grad(output, inputs, create_graph=True, retain_graph=True)
            for output in outputs
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)
        outputs = [sum(value.square().sum() for value in grad) for grad in gradients]
    compiled = torch.compile(gather_sum, backend="aot_eager", fullgraph=True)
    torch.testing.assert_close(compiled(inputs, indices), reference, rtol=0, atol=0)


@pytest.mark.parametrize(
    "cpu_count,max_jobs,expected",
    [(8, None, 8), (None, None, 1), (64, None, 16), (8, "0", 1), (8, "3", 3)],
)
def test_compilation_workers(monkeypatch, cpu_count, max_jobs, expected):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: cpu_count)
    if max_jobs is None:
        monkeypatch.delenv("MAX_JOBS", raising=False)
    else:
        monkeypatch.setenv("MAX_JOBS", max_jobs)

    logger = logging.getLogger("filelock")
    monkeypatch.setattr(logger, "filters", list(logger.filters))
    namespace = runpy.run_path(cuda.__file__)
    pool = namespace["_POOL"]
    try:
        assert pool._max_workers == expected
        assert not pool._threads
        assert not hasattr(os, "sched_getaffinity")
    finally:
        pool.shutdown()


def test_contraction_source_reuses_polynomials():
    from eqx.backends.cuda.codegen import contraction_source

    cache, lines = {}, []
    value = contraction_source([(2.0, ("x", "y")), (3.0, ("z", "x"))], cache, lines)
    count = len(lines)
    assert (
        contraction_source(
            [(3.0, ("x", "z")), (1.0, ("y", "x")), (1.0, ("x", "y"))],
            cache,
            lines,
        )
        == value
    )
    assert (
        contraction_source([(-3.0, ("z", "x")), (-2.0, ("y", "x"))], cache, lines)
        == f"(-{value})"
    )
    assert len(lines) == count
    assert (
        contraction_source([(1.0, ("x", "y")), (-1.0, ("y", "x"))], cache, lines)
        == "T(0)"
    )
    assert contraction_source([(1.0, ("x",))], cache, lines) == "x"
    assert contraction_source([(2.0, ())], cache, lines) == "T(2)"
    assert contraction_source([(1e-20, ("x",))], cache, lines) != "T(0)"


def test_shared_metadata_cache():
    from eqx.utils.metadata import parse_metadata

    specification = (2, ((0, 1, "uvu", True, 0.5),))
    metadata = repr(specification)
    result = parse_metadata(metadata)
    assert result == specification
    assert parse_metadata(metadata) is result
    assert parse_metadata("None") is None
    with pytest.raises(ValueError):
        parse_metadata("tuple()")


def test_cuda_cache_does_not_log_lock_creation(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from eqx.backends.cuda.runtime import compile_binary

    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    caplog.set_level(logging.DEBUG, logger="filelock")
    calls = []
    compiler = SimpleNamespace(
        version=lambda: (1, 0),
        compile=lambda source, options: calls.append(source) or b"test binary",
    )
    for _ in range(2):
        assert compile_binary("test source", (), compiler) == b"test binary"
    assert calls == ["test source"]
    assert not [record for record in caplog.records if record.name == "filelock"]
    logger = logging.getLogger("filelock")
    logger.debug("Other lock: %s", str(tmp_path / "other.lock"))
    logger.warning("Cache warning: %s", str(tmp_path / "eqx/cuda/test.lock"))
    assert len([record for record in caplog.records if record.name == "filelock"]) == 2


def test_replay_tf32_cache():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    from eqx.ops.recompute import Replay

    previous = torch.backends.cuda.matmul.allow_tf32
    program = Replay(lambda values, create_graph: (values[0] @ values[1],))
    x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
    y = torch.randn_like(x)
    try:
        for enabled in (False, True, False):
            torch.backends.cuda.matmul.allow_tf32 = enabled
            torch.testing.assert_close(program(x, y)[0], x @ y)
        assert len(program.graphs) == 2
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous

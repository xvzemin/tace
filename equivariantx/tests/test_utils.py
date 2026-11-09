"""Shared modules, kernel runtime, and executable documentation examples."""

import doctest
import logging
import os
import re
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib import import_module
from pathlib import Path
from threading import Event
from unittest.mock import MagicMock, Mock

import pytest
import torch
from e3nn import o3

import equivariantx.backends.cuda.runtime as cuda
from equivariantx import co2, co3, o2
from equivariantx import o3 as eqx_o3
from equivariantx.nn.activation import PolynomialActivation
from equivariantx.o2.irreps import Irreps
from equivariantx.o2.o2grid import O2Grid
from equivariantx.o3.s2grid import S2Grid
from equivariantx.utils.module import convert_modules, copy_model, default_dtype

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SPACES = {2: o2, 3: eqx_o3}


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
assert not any(name.startswith("equivariantx.backends.cuda") for name in sys.modules)
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
assert "equivariantx.backends.cuda.o2_o3.convolution" not in sys.modules
from equivariantx.backends.cuda import runtime as cuda
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


def test_execution_provider(monkeypatch):
    from equivariantx.backends import get_kernel, register_kernel
    from equivariantx.backends.registry import IMPLEMENTATIONS

    monkeypatch.setitem(IMPLEMENTATIONS, ("test_sine", "torch"), "torch:sin")
    get_kernel.cache_clear()
    assert get_kernel("test_sine", "torch") is torch.sin
    with pytest.raises(ValueError, match="already has"):
        register_kernel("test_sine", "torch", torch.cos)
    with pytest.raises(NotImplementedError, match="does not support"):
        get_kernel("test_sine", "unknown")
    get_kernel.cache_clear()


@pytest.mark.parametrize("space,experts", [("o3", 1), ("o3", 2), ("co3", 1), ("o2", 1)])
@pytest.mark.parametrize("mode", ["single", "pair"])
@pytest.mark.parametrize("backend", ["torch", "cuda"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_element_linear_compile(
    space, experts, mode, backend, device, double_precision
):
    from torch.fx.experimental.proxy_tensor import make_fx

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    group = {"o3": eqx_o3, "co3": co3, "o2": o2}[space]
    linear_cls = o3.Linear if space == "o3" else group.Linear
    linear = linear_cls(
        "2x0e+0e+2x1m+2x2m" if space == "o2" else "2x0e+0e+2x1o+2x2e",
        "3x0e+2x1m+2m+3m" if space == "o2" else "3x0e+2x1o+1o+3o",
        internal_weights=False,
        shared_weights=False,
    ).to(device)
    cls = group.ElementLinear if experts == 1 else group.MoEElementLinear
    module = cls(
        linear,
        backend=backend,
        element_mode=mode,
        **({} if experts == 1 else {"num_experts": experts}),
    )
    reference = copy_model(module)
    reference.backend = "torch"
    reference.expand_weights = True
    compiled = torch.compile(module, backend="aot_eager", fullgraph=True, dynamic=True)
    shape = (3, 4) if mode == "pair" else (3,)
    weight = torch.randn(
        *shape, experts, linear.weight_numel, device=device, requires_grad=True
    )
    traced = None
    for count in (5, 9, 9, 0):
        features = torch.randn(
            module.input_dim, count, device=device
        ).T.requires_grad_()
        source = torch.randint(3, (count,), device=device)
        target = torch.randint(4, (count,), device=device) if mode == "pair" else None
        args = (features, weight, source, target)
        if traced is None:
            traced = make_fx(module, tracing_mode="symbolic")(*args)
        expected = reference(*args)
        for evaluate in (module, traced, compiled):
            actual = evaluate(*args)
            torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
            for a, b in zip(
                torch.autograd.grad(actual.square().sum(), (features, weight)),
                torch.autograd.grad(
                    expected.square().sum(), (features, weight), retain_graph=True
                ),
            ):
                torch.testing.assert_close(a, b, atol=2e-10, rtol=2e-10)


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
    from equivariantx.ops.layout import gather_sum

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
import equivariantx as eqx

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


@pytest.mark.parametrize(
    "platform,lib_dir", [("linux", "lib64"), ("linux", "lib"), ("win32", "lib/x64")]
)
def test_runtime_linker_flags(monkeypatch, tmp_path, platform, lib_dir):
    import torch.utils.cpp_extension as extension

    namespace = runpy.run_path(cuda.__file__)
    cache = tmp_path / "eqx_cuda_runtime"
    build_dir = cache / "build"
    build_dir.mkdir(parents=True)
    (cache / "lock").touch()
    (build_dir / "lock").touch()
    lib_dir = tmp_path / lib_dir
    lib_dir.mkdir(parents=True)
    load = Mock(return_value=object())
    dll_directory = MagicMock()
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(os, "add_dll_directory", dll_directory, raising=False)
    monkeypatch.setattr(extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(extension, "_get_build_directory", lambda *args: str(cache))
    monkeypatch.setattr(extension, "load", load)
    try:
        assert namespace["runtime"]() is load.return_value
        kwargs = load.call_args.kwargs
        assert kwargs["build_directory"] == str(build_dir)
        assert not (build_dir / "lock").exists()
        assert (cache / "lock").exists()
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
            assert f"-L{lib_dir}" in kwargs["extra_ldflags"]
            assert f"-Wl,-rpath,{lib_dir}" in kwargs["extra_ldflags"]
    finally:
        namespace["_POOL"].shutdown()


def test_runtime_build_lock(monkeypatch, tmp_path):
    import torch.utils.cpp_extension as extension

    namespace = runpy.run_path(cuda.__file__)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    started, loaded = Event(), Event()

    def build_directory(*args):
        started.set()
        return str(tmp_path)

    def load(**kwargs):
        assert not (build_dir / "lock").exists()
        loaded.set()

    monkeypatch.setattr(extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr(extension, "_get_build_directory", build_directory)
    monkeypatch.setattr(extension, "load", load)
    monkeypatch.setattr(os, "add_dll_directory", MagicMock(), raising=False)
    script = """
import sys
from pathlib import Path
from filelock import FileLock
with FileLock(sys.argv[1] + ".lock"):
    Path(sys.argv[1], "lock").touch()
    print("locked", flush=True)
    sys.stdin.read()
"""
    try:
        with subprocess.Popen(
            [sys.executable, "-c", script, str(build_dir)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
        ) as owner:
            assert owner.stdout.readline().strip() == "locked"
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(namespace["runtime"])
                try:
                    assert started.wait(10)
                    assert not loaded.wait(0.1)
                    assert (build_dir / "lock").exists()
                finally:
                    owner.kill()
                    owner.wait(timeout=10)
                pending.result(timeout=10)
            assert loaded.is_set()
    finally:
        namespace["_POOL"].shutdown()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("edges", [0, 17])
@pytest.mark.parametrize("first_indexed", [False, True])
def test_gather_sum(dtype, edges, first_indexed):
    from equivariantx.ops.layout import gather_sum

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
    from equivariantx.backends.cuda.codegen import contraction_source

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
    from equivariantx.utils.metadata import compact_metadata, parse_metadata

    specification = (2, ((0, 1, "uvu", True, 0.5),))
    metadata = repr(specification)
    result = parse_metadata(metadata)
    assert result == specification
    assert parse_metadata(metadata) is result
    assert compact_metadata(metadata) == metadata
    large = repr(specification * 1000)
    packed = compact_metadata(large)
    assert len(packed) < len(large)
    assert compact_metadata(packed) == packed
    assert parse_metadata(packed) == specification * 1000
    assert parse_metadata(packed) is parse_metadata(packed)
    assert parse_metadata("None") is None
    with pytest.raises(ValueError):
        parse_metadata("tuple()")


def test_cuda_cache_does_not_log_lock_creation(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from equivariantx.backends.cuda.runtime import compile_binary

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
    from equivariantx.ops.recompute import Replay

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


@pytest.mark.parametrize("cls", [O2Grid, S2Grid])
def test_nonlinear_convergence(cls, double_precision, device):
    kwargs = {} if cls is O2Grid else {"quadrature": "gauss_legendre"}
    irreps = "0e+0o+2x1m+2x2m" if cls is O2Grid else 2
    grids = [cls(irreps, resolution, **kwargs).to(device) for resolution in (9, 25, 49)]
    x = torch.randn(2, grids[0].dim, device=device)
    outputs = [grid.from_grid(torch.nn.functional.silu(grid(x))) for grid in grids]
    coarse = (outputs[0] - outputs[2]).abs().max()
    fine = (outputs[1] - outputs[2]).abs().max()
    assert fine < coarse * 0.01


@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "tanh", "sigmoid", "softplus"])
def test_polynomial_activation(act, double_precision, device):
    polynomial = PolynomialActivation(act, dtype=torch.float32).to(
        device, torch.float64
    )
    reference = PolynomialActivation(act, dtype=torch.float64, device=device)
    torch.testing.assert_close(
        polynomial.coefficients, reference.coefficients, atol=0, rtol=0
    )
    function = getattr(torch.nn.functional, act)
    x = torch.linspace(-3, 3, 1001, device=device)
    error = (polynomial(x) - function(x)).abs().max()
    assert error < 0.12
    refined = PolynomialActivation(act, degree=16, device=device)
    assert (refined(x) - function(x)).abs().max() < error
    if act == "tanh":
        torch.testing.assert_close(polynomial(-x), -polynomial(x), atol=0, rtol=0)
    elif act == "sigmoid":
        torch.testing.assert_close(
            polynomial(-x), 1 - polynomial(x), atol=1e-15, rtol=1e-15
        )
    else:
        torch.testing.assert_close(
            polynomial(x) - polynomial(-x), x, atol=2e-14, rtol=2e-14
        )
    x = torch.randn(4, device=device, requires_grad=True)
    assert torch.autograd.gradcheck(polynomial, (x,))
    assert torch.autograd.gradgradcheck(polynomial, (x,))
    assert polynomial(x[:0]).shape == (0,)
    assert not polynomial.state_dict()


@pytest.mark.parametrize(
    "representation", ["1m+2m+3m+0e+1m+2m+3m+0o", "0e+0o+2x1m+2x2m+2x3m", "s2"]
)
@pytest.mark.parametrize("act", ["silu", "gelu", "relu", "tanh", "sigmoid", "softplus"])
def test_polynomial_grid_equivariance(representation, act, double_precision, device):
    polynomial = PolynomialActivation(act).to(device)
    resolution = (polynomial.degree + 1) * 3
    if representation != "s2":
        irreps = Irreps(representation)
        grid = O2Grid(irreps, resolution + 1).to(device)
        rotation = irreps.D_from_angle(
            0.741, reflected=True, dtype=torch.float64, device=device
        )
    else:
        grid = S2Grid(3, resolution, quadrature="gauss_legendre").to(device)
        rotation = (
            o3.Irreps.spherical_harmonics(3).D_from_matrix(-o3.rand_matrix()).to(device)
        )
    x = torch.randn(2, 4, grid.dim, device=device, requires_grad=True)
    # Also test polynomial extrapolation, without clipping grid values.
    for scale in (1, 5):
        actual = grid.from_grid(polynomial(grid(scale * x @ rotation.T)))
        expected = grid.from_grid(polynomial(grid(scale * x))) @ rotation.T
        normalizer = expected.detach().abs().amax().clamp_min(1)
        torch.testing.assert_close(
            actual / normalizer, expected / normalizer, atol=2e-13, rtol=2e-12
        )
        seed = torch.randn_like(actual)
        for _ in range(2):
            actual, expected = [
                torch.autograd.grad((y * seed).sum(), x, create_graph=True)[0]
                for y in (actual, expected)
            ]
            normalizer = expected.detach().abs().amax().clamp_min(1)
            torch.testing.assert_close(
                actual / normalizer, expected / normalizer, atol=2e-13, rtol=2e-12
            )


def test_polynomial_compile_and_clenshaw(double_precision, device):
    polynomial = PolynomialActivation(degree=12).to(device)
    x = torch.linspace(-5, 5, 61, device=device, requires_grad=True)
    z = x / polynomial.bound
    previous, current = torch.ones_like(z), z
    expected = (
        polynomial.coefficients[0] * previous + polynomial.coefficients[1] * current
    )
    for i in range(2, polynomial.degree + 1):
        previous, current = current, 2 * z * current - previous
        expected = expected + polynomial.coefficients[i] * current
    torch.testing.assert_close(polynomial(x), expected, atol=2e-12, rtol=2e-12)
    compiled = torch.compile(
        polynomial, backend="aot_eager", fullgraph=True, dynamic=True
    )
    actual = compiled(x)
    torch.testing.assert_close(actual, expected, atol=2e-12, rtol=2e-12)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), x)[0],
        torch.autograd.grad(expected.sum(), x)[0],
        atol=2e-12,
        rtol=2e-12,
    )
    assert torch.isfinite(polynomial(torch.tensor([10.0], device=device))).all()
    assert not torch.allclose(
        polynomial(x.detach()), polynomial(x.detach().clamp(-3, 3))
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"act": "invalid"},
        {"degree": 0},
        {"degree": 1.5},
        {"bound": 0},
        {"bound": float("nan")},
    ],
)
def test_polynomial_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        PolynomialActivation(**kwargs)


@pytest.mark.parametrize(
    "rank,d", [(n, d) for d in (2, 3) for n in range(5)] + [(9, 2), (6, 3)]
)
def test_ictd_basis(rank, d, double_precision):
    module = SPACES[d].ICTD(rank, device=DEVICE)
    matrix = module.change_of_basis
    identity = torch.eye(d**rank, device=DEVICE)
    torch.testing.assert_close(matrix.T @ matrix, identity, atol=3e-14, rtol=3e-14)
    torch.testing.assert_close(matrix @ matrix.T, identity, atol=3e-14, rtol=3e-14)
    assert module.irreps_out.dim == module.dim == d**rank
    assert len(module.paths) == len(module.irreps_out)
    assert all(len(path) == rank + 1 for path in module.paths)
    basis = co3.path_matrix(rank) if d == 3 else co2.path_matrix(rank)
    torch.testing.assert_close(
        module.path_matrix(0), basis.to(DEVICE), atol=3e-15, rtol=3e-15
    )
    x = torch.randn(2, d**rank, 3, device=DEVICE).transpose(-1, -2)
    h = module(x)
    torch.testing.assert_close(module.inverse(h), x, atol=3e-14, rtol=3e-14)
    assert module(x[:0]).shape == x[:0].shape
    assert module.inverse(h[:0]).shape == h[:0].shape
    assert not tuple(module.parameters())


@pytest.mark.parametrize("d", [2, 3])
def test_ictd_projectors(d, double_precision):
    module = SPACES[d].ICTD(3, device=DEVICE)
    x = torch.randn(5, d**3, device=DEVICE)
    projections = [module.project(x, i) for i in range(len(module.paths))]
    torch.testing.assert_close(sum(projections), x, atol=3e-14, rtol=3e-14)
    for i, projected in enumerate(projections):
        torch.testing.assert_close(module.project(projected, i), projected)
        for j in range(i):
            torch.testing.assert_close(
                module.project(projected, j), torch.zeros_like(x), atol=3e-14, rtol=0
            )
    assert module.project(x[:0], 0).shape == x[:0].shape


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("reflected", [False, True])
def test_ictd_equivariance(rank, d, reflected, double_precision):
    module = SPACES[d].ICTD(rank, device=DEVICE)
    if d == 3:
        rotation = o3.rand_matrix(device=DEVICE) * (-1 if reflected else 1)
        output_rotation = module.irreps_out.D_from_matrix(rotation.cpu()).to(DEVICE)
    else:
        angle = torch.rand((), device=DEVICE)
        rotation = o2.Irrep("1m").D_from_angle(angle, reflected)
        output_rotation = module.irreps_out.D_from_angle(angle, reflected)
    cartesian_rotation = torch.ones(1, 1, device=DEVICE)
    for _ in range(rank):
        cartesian_rotation = torch.kron(cartesian_rotation, rotation.contiguous())
    x = torch.randn(3, d**rank, device=DEVICE)
    torch.testing.assert_close(
        module(x @ cartesian_rotation.T),
        module(x) @ output_rotation.T,
        atol=3e-12,
        rtol=3e-12,
    )


def test_ictd_rank_two_decomposition(double_precision):
    for d in (2, 3):
        module = SPACES[d].ICTD(2, device=DEVICE)
        x = torch.randn(4, d, d, device=DEVICE)
        transpose = x.transpose(-1, -2)
        trace = x.diagonal(dim1=-2, dim2=-1).sum(-1)
        isotropic = trace[:, None, None] * torch.eye(d, device=DEVICE) / d
        expected = {
            "stf": (x + transpose) / 2 - isotropic,
            "antisymmetric": (x - transpose) / 2,
            "scalar": isotropic,
        }
        for index, path in enumerate(module.paths):
            ir = path[-1]
            order = ir.l if d == 3 else ir.m
            kind = (
                "stf"
                if order == 2
                else "scalar"
                if order == 0 and ir.p == 1
                else "antisymmetric"
            )
            torch.testing.assert_close(
                module.project(x.flatten(-2), index).reshape_as(x),
                expected[kind],
                atol=2e-14,
                rtol=2e-14,
            )
    assert o2.ICTD(2).irreps_out == o2.Irreps("1x2m+1x0e+1x0o")
    assert eqx_o3.ICTD(2).irreps_out == o3.Irreps("1x2e+1x1e+1x0e")


@pytest.mark.parametrize("d", [2, 3])
def test_ictd_precision_and_derivatives(d):
    module = SPACES[d].ICTD(3, dtype=torch.float32, device=DEVICE).double()
    reference = SPACES[d].ICTD(3, dtype=torch.float64, device=DEVICE)
    torch.testing.assert_close(
        module.change_of_basis, reference.change_of_basis, atol=0, rtol=0
    )
    x = torch.randn(2, d**3, dtype=torch.float64, device=DEVICE, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: module(x).sin(), (x,), fast_mode=True)
    assert torch.autograd.gradgradcheck(lambda x: module(x).sin(), (x,), fast_mode=True)
    compiled = torch.compile(module, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(x), module(x))


@pytest.mark.parametrize("d", [2, 3])
@pytest.mark.parametrize("rank", [-1, 1.5])
def test_ictd_invalid_rank(rank, d):
    with pytest.raises(ValueError):
        SPACES[d].ICTD(rank)
    with pytest.raises(ValueError):
        list(SPACES[d].path_matrices(rank))

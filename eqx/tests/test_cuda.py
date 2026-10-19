"""Kernel code generation, caching, and platform-independent initialization."""

import logging
import os
import runpy
import sys
from unittest.mock import MagicMock, Mock

import pytest
import torch

from eqx.kernels import cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "mmax,basis_change,packed", [(0, True, True), (1, False, False), (3, True, True)]
)
def test_indexed_frame(dtype, mmax, basis_change, packed):
    from eqx import o2

    frame = o2.LocalFrame(
        "2x0e+3x1o+2x1e+1x3o+2x1o", mmax, basis_change=basis_change
    ).to("cuda", dtype)
    rotation = o2.WignerD(4, 4, method="recursive").to("cuda", dtype)
    for edges in (9, 0):
        nodes = torch.randn(
            6, frame.input_dim, device="cuda", dtype=dtype, requires_grad=True
        )
        vectors = torch.randn(edges, 3, device="cuda", dtype=dtype, requires_grad=True)
        index = torch.arange(edges, device="cuda") % 5
        wigner, inverse = rotation(vectors)
        if packed:
            wigner = inverse = rotation.forward_packed(vectors)
        for inverse_map in (False, True):
            values = (
                torch.randn(
                    edges,
                    frame.output_dim,
                    device="cuda",
                    dtype=dtype,
                    requires_grad=True,
                )
                if inverse_map
                else nodes
            )
            outputs = [
                frame.scatter(values, inverse, index, 6, backend=backend)
                if inverse_map
                else frame.gather(values, wigner, index, backend=backend)
                for backend in ("cuda", "torch")
            ]
            for _ in range(3):
                torch.testing.assert_close(
                    *outputs,
                    rtol=3e-4 if dtype == torch.float32 else 1e-9,
                    atol=3e-5 if dtype == torch.float32 else 1e-9,
                )
                gradients = [
                    torch.autograd.grad(
                        value.sin().sum(),
                        (values, vectors),
                        create_graph=True,
                        retain_graph=True,
                    )
                    for value in outputs
                ]
                outputs = [
                    torch.cat([g.flatten() for g in grads]) / 10 for grads in gradients
                ]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fused_o2_gate(dtype):
    from eqx import o2

    args = (
        "2x0ee+1x0oe",
        [torch.nn.SiLU(), torch.tanh],
        "2x0ee+3x0oo",
        [torch.sigmoid, torch.tanh],
        "1x0oe+2x1mo+2x2me",
    )
    reference = o2.Gate(*args).to("cuda", dtype)
    fused = o2.Gate(*args, backend="cuda").to("cuda", dtype)
    for batch in ((), (2, 3), (0,)):
        features = torch.randn(
            *batch, fused.irreps_in.dim, device="cuda", dtype=dtype, requires_grad=True
        )
        outputs = [module(features) for module in (fused, reference)]
        for _ in range(3):
            torch.testing.assert_close(*outputs)
            outputs = [
                torch.autograd.grad(value.sin().sum(), features, create_graph=True)[0]
                for value in outputs
            ]
        for reflected, time_reversal in ((True, False), (False, True), (True, True)):
            angle = features.new_tensor(0.37)

            def transform(irreps):
                return irreps.D_from_angle(
                    angle, reflected=reflected, time_reversal=time_reversal
                )

            torch.testing.assert_close(
                fused(features @ transform(fused.irreps_in).T),
                fused(features) @ transform(fused.irreps_out).T,
            )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_o2_pipeline_compile(double_precision):
    from eqx import o2

    frame = o2.LocalFrame("2x0e+2x1o+2x1e", mmax=1).cuda()
    args = ("2x0e+2x0o", [torch.nn.SiLU(), torch.tanh], "4x0e", [torch.sigmoid], "4x1m")
    gates = [o2.Gate(*args, backend=backend).cuda() for backend in ("torch", "cuda")]
    up = o2.Linear(frame.irreps_out, gates[0].irreps_in).cuda()
    down = o2.Linear(gates[0].irreps_out, frame.irreps_out).cuda()

    def evaluate(nodes, wigner, index, backend):
        local = frame.gather(nodes, wigner, index[0], backend=backend)
        local = down(gates[backend == "cuda"](up(local)))
        return frame.scatter(local, wigner, index[1], nodes.size(0), backend=backend)

    torch.compiler.reset()
    compiled = torch.compile(
        evaluate, backend="aot_eager", fullgraph=True, dynamic=True
    )
    for edges in (7, 11, 0):
        nodes = torch.randn(5, frame.input_dim, device="cuda", requires_grad=True)
        wigner = torch.randn(edges, 10, device="cuda", requires_grad=True)
        index = torch.randint(5, (2, edges), device="cuda")
        outputs = [
            compiled(nodes, wigner, index, "cuda"),
            evaluate(nodes, wigner, index, "torch"),
        ]
        torch.testing.assert_close(*outputs)
        inputs = (nodes, wigner, up.weight, down.weight)
        gradients = [
            torch.autograd.grad(value.square().sum(), inputs) for value in outputs
        ]
        for a, b in zip(*gradients):
            torch.testing.assert_close(a, b)

    vectors = torch.randn(7, 3, device="cuda", requires_grad=True)
    rotation = o2.WignerD(1, 1, method="recursive").cuda()
    nodes = torch.randn(5, frame.input_dim, device="cuda", requires_grad=True)
    index = torch.randint(5, (2, 7), device="cuda")
    losses = []
    for backend in ("cuda", "torch"):
        output = evaluate(nodes, rotation.forward_packed(vectors), index, backend)
        forces = -torch.autograd.grad(
            output.square().mean(), vectors, create_graph=True
        )[0]
        losses.append(forces.square().mean())
    gradients = [
        torch.autograd.grad(loss, (nodes, up.weight, down.weight)) for loss in losses
    ]
    for a, b in zip(*gradients):
        torch.testing.assert_close(a, b, rtol=1e-8, atol=1e-9)


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
    from eqx.kernels.layout import gather_sum

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
    from eqx.kernels.codegen import contraction_source

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
    from eqx.conv import program
    from eqx.utils.metadata import parse_metadata

    assert program.parse_metadata is parse_metadata
    specification = (2, ((0, 1, "uvu", True, 0.5),))
    metadata = repr(specification)
    result = parse_metadata(metadata)
    assert result == specification
    assert parse_metadata(metadata) is result
    assert parse_metadata.cache_info().maxsize == 256
    assert parse_metadata("None") is None
    with pytest.raises(ValueError):
        parse_metadata("tuple()")


def test_cuda_cache_does_not_log_lock_creation(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from eqx.kernels.cuda import compile_binary

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
    from eqx.kernels.recompute import Replay

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

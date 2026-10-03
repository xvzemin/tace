"""Kernel code generation, caching, and platform-independent initialization."""

import logging
import os
import runpy

import pytest
import torch

from eqx.kernels import cuda


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

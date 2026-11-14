"""Tests for the corresponding EQX source module."""

import logging
import os
import runpy
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from threading import Event
from unittest.mock import MagicMock, Mock
import pytest
import equivariantx.ops.common.cuda.runtime as cuda


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


def test_cuda_cache_does_not_log_lock_creation(tmp_path, monkeypatch, caplog):
    from types import SimpleNamespace

    from equivariantx.ops.common.cuda.runtime import compile_binary

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

"""TACE imports without Linux CPU affinity or CUDA extensions."""

import os
import subprocess
import sys
from pathlib import Path


def test_import_without_cpu_affinity_or_cuda(tmp_path):
    script = """
import os
import sys

if hasattr(os, "sched_getaffinity"):
    del os.sched_getaffinity
sys.modules["torch.utils.cpp_extension"] = None

from tace.interface.ase import TACEAseCalc
from tace.lightning import load_tace
from tace.scripts.train import main
from eqx.kernels import cuda

assert not hasattr(os, "sched_getaffinity")
assert not cuda._POOL._threads
assert not cuda.runtime.cache_info().currsize
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "",
            "MPLCONFIGDIR": str(tmp_path),
        },
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

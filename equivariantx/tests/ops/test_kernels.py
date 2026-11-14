"""Compatibility with internal class paths saved in older checkpoints."""

import subprocess
import sys


def test_lazy_kernel_aliases():
    script = """
import importlib
import sys
import equivariantx
assert not any(name.startswith('equivariantx.ops.common.cuda') for name in sys.modules)
for name in ('torch.tensor_product', 'torch.spherical_product', 'cuda.runtime'):
    canonical = importlib.import_module('equivariantx.ops.common.' + name)
    legacy = importlib.import_module('equivariantx.ops.kernels.' + name)
    assert legacy is canonical
assert not any(name.startswith('eqx_cuda_runtime') for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr

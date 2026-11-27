import os
from importlib import import_module
from pathlib import Path

import pytest
import torch


@pytest.fixture
def subprocess_package_env():
    """Resolve the packages under test when a fresh interpreter uses a different cwd."""

    def for_packages(*names):
        roots = dict.fromkeys(
            str(Path(import_module(name).__file__).resolve().parent.parent)
            for name in names
        )
        # Source checkouts need not also be installed in the interpreter, and
        # an older installed package is not a substitute for the tested code.
        paths = [
            *roots,
            *(
                os.path.abspath(p)
                for p in os.environ.get("PYTHONPATH", "").split(os.pathsep)
                if p
            ),
        ]
        return {**os.environ, "PYTHONPATH": os.pathsep.join(paths)}

    return for_packages


@pytest.fixture(autouse=True)
def numerical_state():
    dtype = torch.get_default_dtype()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        try:
            yield
        finally:
            torch.set_default_dtype(dtype)


@pytest.fixture
def double_precision():
    torch.set_default_dtype(torch.float64)
    torch.manual_seed(12)

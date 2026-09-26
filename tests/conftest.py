"""Shared numerical-test settings and isolation of process-wide state."""

import pytest
import torch

import eqx  # Load packaged constants before importing legacy e3nn modules.
from tace.utils.env import ACCELERATION_ENV


@pytest.fixture(scope="session", autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def isolated_state(monkeypatch):
    dtype = torch.get_default_dtype()
    for name in ACCELERATION_ENV.values():
        monkeypatch.delenv(name, raising=False)
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

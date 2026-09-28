"""Numerical-test settings for EQX."""

import pytest
import torch

import eqx  # noqa: F401 - Load constants before importing legacy e3nn modules.


@pytest.fixture(scope="session", autouse=True)
def torch_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(min(previous, 4))
    yield
    torch.set_num_threads(previous)


@pytest.fixture(autouse=True)
def isolated_state():
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


@pytest.fixture
def wigner_3j():
    """Skip reference degrees absent from the installed e3nn CG table."""
    from e3nn import o3

    def coefficients(*args, **kwargs):
        try:
            return o3.wigner_3j(*args, **kwargs)
        except NotImplementedError as error:
            pytest.skip(str(error))

    return coefficients

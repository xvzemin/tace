"""Irreducible tensor decomposition tests."""

import pytest
import torch

from equivariantx.tests.utils.testing import (
    check_ictd_basis,
    check_ictd_equivariance,
    check_ictd_invalid_rank,
    check_ictd_method_precision,
    check_ictd_precision_and_derivatives,
    check_ictd_projectors,
)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_reference_preserves_cuda_provider(device, monkeypatch, double_precision):
    from unittest.mock import patch

    from equivariantx import o2
    from equivariantx.ops import config
    from equivariantx.ops.cuda.o2 import ictd as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module, reference = o2.ICTD(3, backend="cuda").to(device), o2.ICTD(3).to(device)
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        x = torch.randn(count, 8, device=device)
        for name, args in (("forward", (x,)), ("inverse", (x,)), ("project", (x, 0))):
            with patch.object(kernel, name, wraps=getattr(kernel, name)) as selected:
                actual = getattr(module, name)(*args)
            assert selected.call_count == (device == "cuda")
            torch.testing.assert_close(actual, getattr(reference, name)(*args))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    check_ictd_method_precision(2, device)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 9])
def test_ictd_basis(rank, double_precision):
    check_ictd_basis(rank, 2, double_precision)


def test_ictd_projectors(double_precision):
    check_ictd_projectors(2, double_precision)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("reflected", [False, True])
def test_ictd_equivariance(rank, reflected, double_precision):
    check_ictd_equivariance(rank, 2, reflected, double_precision)


def test_ictd_precision_and_derivatives():
    check_ictd_precision_and_derivatives(2)


@pytest.mark.parametrize("rank", [-1, 1.5])
def test_ictd_invalid_rank(rank):
    check_ictd_invalid_rank(rank, 2)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_sparse_backend(backend, device, double_precision, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_sparse_backend

    check_ictd_sparse_backend(2, backend, device, monkeypatch)


def test_lazy_reference_compile(device, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_lazy_compile

    check_ictd_lazy_compile(2, device, monkeypatch)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_path_cache(backend, device, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_path_cache

    check_ictd_path_cache(2, backend, device, monkeypatch)

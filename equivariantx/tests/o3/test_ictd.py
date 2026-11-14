"""Irreducible tensor decomposition tests."""

import pytest
import torch
from e3nn import o3

from equivariantx import o2
from equivariantx import o3 as eqx_o3
from equivariantx.tests.utils.testing import (
    DEVICE,
    SPACES,
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

    from equivariantx.ops import config
    from equivariantx.ops.cuda.o3 import ictd as kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    module, reference = (
        eqx_o3.ICTD(3, backend="cuda").to(device),
        eqx_o3.ICTD(3).to(device),
    )
    monkeypatch.setattr(config, "TORCH_REFERENCE", True)
    for count in (3, 0):
        x = torch.randn(count, 27, device=device)
        for name, args in (("forward", (x,)), ("inverse", (x,)), ("project", (x, 0))):
            with patch.object(kernel, name, wraps=getattr(kernel, name)) as selected:
                actual = getattr(module, name)(*args)
            assert selected.call_count == (device == "cuda")
            torch.testing.assert_close(actual, getattr(reference, name)(*args))
    assert module.backend == "cuda" and config.TORCH_REFERENCE is True


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_public_method_precision(device, double_precision):
    check_ictd_method_precision(3, device)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4, 6])
def test_ictd_basis(rank, double_precision):
    check_ictd_basis(rank, 3, double_precision)


def test_ictd_projectors(double_precision):
    check_ictd_projectors(3, double_precision)


@pytest.mark.parametrize("rank", [0, 1, 2, 3, 4])
@pytest.mark.parametrize("reflected", [False, True])
def test_ictd_equivariance(rank, reflected, double_precision):
    check_ictd_equivariance(rank, 3, reflected, double_precision)


def test_ictd_precision_and_derivatives():
    check_ictd_precision_and_derivatives(3)


@pytest.mark.parametrize("rank", [-1, 1.5])
def test_ictd_invalid_rank(rank):
    check_ictd_invalid_rank(rank, 3)


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


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_sparse_backend(backend, device, double_precision, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_sparse_backend

    check_ictd_sparse_backend(3, backend, device, monkeypatch)


def test_lazy_reference_compile(device, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_lazy_compile

    check_ictd_lazy_compile(3, device, monkeypatch)


@pytest.mark.parametrize("backend", ["torch", "cuda"])
def test_path_cache(backend, device, monkeypatch):
    from equivariantx.tests.utils.testing import check_ictd_path_cache

    check_ictd_path_cache(3, backend, device, monkeypatch)

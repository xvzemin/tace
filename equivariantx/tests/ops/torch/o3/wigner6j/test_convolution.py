"""Numerical derivatives and standalone compilation for the matching kernel."""

import pytest
import torch

from equivariantx.ops import config
from equivariantx.tests.o3.wigner6j.testing import (
    check_derivatives,
    check_dynamic_convolution,
    check_package,
    problem,
)


def test_large_standard_cpu_no_cuda_setup(monkeypatch):
    import sys

    from equivariantx.o3.wigner6j import Wigner6jTensorProductConv

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda, "_lazy_init", lambda: pytest.fail("unexpected CUDA initialization")
    )
    monkeypatch.setitem(sys.modules, "equivariantx.ops.common.cuda.runtime", None)
    monkeypatch.setitem(
        sys.modules, "equivariantx.ops.cuda.o3.wigner6j.convolution", None
    )
    op = Wigner6jTensorProductConv(
        "2x1o",
        "1o",
        "1e",
        "2x1e",
        "2x1e",
        [(0, 0, 0, "uvu", True)],
        [(0, 0, 0, "uvu", True)] * 32,
        shared_weights=False,
        internal_weights=False,
        backend="torch",
        recouple=False,
    ).double()
    assert not hasattr(op, "standard12") and not hasattr(op, "standard123")
    assert not any(name.startswith("standard") for name in op.state_dict())
    vals = [
        torch.randn(n, w, dtype=torch.float64)
        for n, w in ((3, 6), (7, 3), (3, 3), (7, 2), (7, 64))
    ]
    edges = torch.randint(3, (2, 7))
    args = (*vals[:3], edges, *vals[3:])
    torch.testing.assert_close(
        op(*args), op.forward_reference(*args), rtol=5e-10, atol=3e-10
    )


@pytest.mark.parametrize("adjoint", [False, True])
@pytest.mark.parametrize("recouple", [False, True])
def test_fifth_derivatives_and_transforms(monkeypatch, recouple, adjoint):
    monkeypatch.setattr(config, "TORCH_WIGNER6J_ADJOINT", adjoint)
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_derivatives("convolution", "torch", "cpu")


@pytest.mark.parametrize("recouple", [None, True])
def test_joint_training_package(tmp_path, monkeypatch, recouple):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_package("convolution", tmp_path, "cpu")


def test_dynamic_inductor(monkeypatch):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    layer, values, index, _ = problem("convolution", "torch", "cpu")
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    for count, nodes in ((4, 3), (7, 5), (0, 2)):
        _, values, index, _ = problem("convolution", "torch", "cpu", count, nodes)
        x1, x2, x3, w1, w2 = values
        actual = compiled(x1, x2, x3, index, w1, w2, nodes + 1)
        expected = layer.forward_reference(x1, x2, x3, index, w1, w2, nodes + 1)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("shared", [False, True])
def test_unbacked_joint_aoti(tmp_path, monkeypatch, shared):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", None)
    check_dynamic_convolution(tmp_path, "torch", "cpu", shared)


@pytest.mark.parametrize("recouple", [False, True])
def test_shared_scalar_derivatives(monkeypatch, recouple):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    check_derivatives(
        "convolution", "torch", "cpu", shared=True, multiplicity=1, edge_degree=0
    )


@pytest.mark.parametrize("recouple", [None, True])
def test_shared_scalar_package(tmp_path, monkeypatch, recouple):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    check_package(
        "convolution", tmp_path, "cpu", shared=True, multiplicity=1, edge_degree=0
    )

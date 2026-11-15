"""Numerical derivatives and standalone compilation for the matching kernel."""

import pytest
import torch

from equivariantx.o3.wigner6j import Wigner6jTensorProductConv
from equivariantx.ops import config
from equivariantx.tests.o3.wigner6j.testing import (
    check_derivatives,
    check_dynamic_convolution,
    check_package,
    problem,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native Wigner-6j requires CUDA"
)


def test_fused_node_reduction(monkeypatch):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", True)
    monkeypatch.setattr(config, "WIGNER6J_INLINE", False)
    layer, values, index, _ = problem("convolution", "cuda", "cuda")

    def materialized(*args, **kwargs):
        raise AssertionError(
            "The fused convolution must not materialize edge messages."
        )

    monkeypatch.setattr(layer.schedule, "evaluate", materialized)
    x1, x2, x3, w1, w2 = values
    actual = layer(x1, x2, x3, index, w1, w2)
    expected = layer.forward_reference(x1, x2, x3, index, w1, w2)
    torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


def test_compensated_node_reduction():
    from equivariantx.utils.module import default_dtype

    dtype = torch.float32
    with default_dtype(dtype):
        layer = Wigner6jTensorProductConv(
            "0e",
            "0e",
            "0e",
            "0e",
            "0e",
            [(0, 0, 0, "uvu", False)],
            [(0, 0, 0, "uvu", False)],
            backend="cuda",
            recouple=True,
        ).cuda()
    large = 4 / torch.finfo(dtype).eps
    attrs = torch.tensor([large, 1, 1, 1, 1, 1, 1, -large], device="cuda", dtype=dtype)
    attrs = attrs.repeat(128).reshape(-1, 1).requires_grad_()
    nodes = torch.ones(1, 1, device="cuda", dtype=dtype, requires_grad=True)
    edges = torch.zeros(2, attrs.shape[0], device="cuda", dtype=torch.long)
    result = layer(nodes, attrs, nodes, edges)
    torch.testing.assert_close(result, torch.full_like(result, 768), atol=0, rtol=0)
    torch.testing.assert_close(
        torch.autograd.grad(result.sum(), attrs)[0], torch.ones_like(attrs)
    )


@pytest.mark.parametrize("recouple", [False, True])
def test_fifth_derivatives_and_transforms(monkeypatch, recouple):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_derivatives("convolution", "cuda", "cuda")


@pytest.mark.parametrize("recouple", [None, True])
def test_joint_training_package(tmp_path, monkeypatch, recouple):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_package("convolution", tmp_path, "cuda")


def test_dynamic_inductor(monkeypatch):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    layer, values, index, _ = problem("convolution", "cuda", "cuda")
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    for count, nodes in ((4, 3), (7, 5), (0, 2)):
        _, values, index, _ = problem("convolution", "cuda", "cuda", count, nodes)
        x1, x2, x3, w1, w2 = values
        actual = compiled(x1, x2, x3, index, w1, w2, nodes + 1)
        expected = layer.forward_reference(x1, x2, x3, index, w1, w2, nodes + 1)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


@pytest.mark.parametrize("shared", [False, True])
def test_unbacked_joint_aoti(tmp_path, monkeypatch, shared):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", None)
    check_dynamic_convolution(tmp_path, "cuda", "cuda", shared)


@pytest.mark.parametrize("recouple", [False, True])
def test_shared_scalar_derivatives(monkeypatch, recouple):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    check_derivatives(
        "convolution", "cuda", "cuda", shared=True, multiplicity=1, edge_degree=0
    )


@pytest.mark.parametrize("recouple", [None, True])
def test_shared_scalar_package(tmp_path, monkeypatch, recouple):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    check_package(
        "convolution", tmp_path, "cuda", shared=True, multiplicity=1, edge_degree=0
    )

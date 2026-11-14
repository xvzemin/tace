"""Numerical derivatives and standalone compilation for the matching kernel."""

import pytest
import torch

from equivariantx.ops import config
from equivariantx.tests.o3.wigner6j.testing import (
    check_derivatives,
    check_package,
    problem,
)

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="Native Wigner-6j requires CUDA"
)


@pytest.mark.parametrize("recouple", [False, True])
def test_fifth_derivatives_and_transforms(monkeypatch, recouple):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_derivatives("tensor_product", "cuda", "cuda")


@pytest.mark.parametrize("recouple", [None, True])
def test_joint_training_package(tmp_path, monkeypatch, recouple):
    monkeypatch.setattr(config, "WIGNER6J_RECOUPLE", recouple)
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    check_package("tensor_product", tmp_path, "cuda")


def test_dynamic_inductor(monkeypatch):
    monkeypatch.setattr(config, "TORCH_WIGNER6J", True)
    layer, values, index, _ = problem("tensor_product", "cuda", "cuda")
    compiled = torch.compile(layer, fullgraph=True, dynamic=True)
    for count, nodes in ((4, 3), (7, 5), (0, 2)):
        _, values, index, _ = problem("tensor_product", "cuda", "cuda", count, nodes)
        x1, x2, x3, w1, w2 = values
        actual = compiled(*values)
        expected = layer.forward_reference(*values)
        torch.testing.assert_close(actual, expected, atol=3e-10, rtol=3e-10)


def test_native_angular_dispatch():
    from torch.fx.experimental.proxy_tensor import make_fx

    layer, values, _, _ = problem("tensor_product", "cuda", "cuda")
    graph = make_fx(layer)(*values)
    assert any(
        node.target == torch.ops.eqx.sparse_bilinear.default
        for node in graph.graph.nodes
    )

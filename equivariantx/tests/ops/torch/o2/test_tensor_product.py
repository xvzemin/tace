"""CPU diagonal schedules through the public tensor-product module."""

import pytest
import torch

from equivariantx.tests.ops.common.test_channel_product import (
    check_diagonal_boundaries,
    check_diagonal_broadcast,
    check_diagonal_dynamic,
    check_diagonal_product,
    check_diagonal_transforms,
)


@pytest.mark.parametrize("shared", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nodes", [0, 7])
@pytest.mark.parametrize("mode", ["independent", "self", "frozen_left"])
def test_diagonal_product(shared, dtype, nodes, mode, monkeypatch):
    check_diagonal_product("o2", shared, dtype, nodes, mode, monkeypatch)


def test_transforms(monkeypatch):
    check_diagonal_transforms("o2", monkeypatch)


@pytest.mark.parametrize("shared", [False, True])
def test_broadcast(shared, monkeypatch):
    check_diagonal_broadcast("o2", shared, monkeypatch)


def test_boundaries(monkeypatch):
    check_diagonal_boundaries("o2", monkeypatch)


@pytest.mark.parametrize("shared", [False, True])
def test_dynamic_batches(shared, monkeypatch):
    check_diagonal_dynamic("o2", shared, monkeypatch)


def test_small_shared_u1u(monkeypatch):
    from torch.fx.experimental.proxy_tensor import make_fx

    from equivariantx import o2
    from equivariantx.ops.common.torch import channel_product

    layer = o2.TensorProduct(
        "1024x1m",
        "1x1m",
        "1024x2m",
        [(0, 0, 0, "u1u", True)],
        internal_weights=False,
        shared_weights=True,
    ).double()
    x = torch.randn(7, 2048, dtype=torch.float64)
    y = torch.randn(7, 2, dtype=torch.float64)
    weight = torch.randn(1024, dtype=torch.float64)

    def reject(*args):
        raise AssertionError("selected CPU slices")

    monkeypatch.setattr(channel_product, "tensor_product", reject)
    graph = make_fx(layer)(x, y, weight)
    torch.testing.assert_close(graph(x, y, weight), layer(x, y, weight))
    with pytest.raises(AssertionError, match="selected CPU slices"):
        make_fx(layer)(x[:1], torch.randn(32, 2, dtype=torch.float64), weight)

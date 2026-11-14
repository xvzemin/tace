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
    check_diagonal_product("co2", shared, dtype, nodes, mode, monkeypatch)


def test_transforms(monkeypatch):
    check_diagonal_transforms("co2", monkeypatch)


@pytest.mark.parametrize("shared", [False, True])
def test_broadcast(shared, monkeypatch):
    check_diagonal_broadcast("co2", shared, monkeypatch)


def test_boundaries(monkeypatch):
    check_diagonal_boundaries("co2", monkeypatch)


@pytest.mark.parametrize("shared", [False, True])
def test_dynamic_batches(shared, monkeypatch):
    check_diagonal_dynamic("co2", shared, monkeypatch)

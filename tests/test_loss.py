import inspect

import pytest
import torch

from tace.utils.loss.dens import (
    huber_dens_forces,
    l2mae_dens_forces,
    mae_dens_forces,
    mse_dens_forces,
)
from tace.utils.loss.huber_fn import huber_forces
from tace.utils.loss.mse_fn import LOSS_FN, mse_forces
from tace.utils.loss.normal import NormalLoss
from tace.utils.loss.registry import ensure_loss_functions_registered
from tace.utils.loss.uncertainty import UncertaintyLoss


def _dens_inputs():
    pred = {
        "forces": torch.tensor([[2.0, 0.0, 0.0], [7.0, 0.0, 0.0]]),
        "noise_vec": torch.tensor([[8.0, 0.0, 0.0], [3.0, 0.0, 0.0]]),
    }
    label = {
        "forces": torch.zeros(2, 3),
        "noise_vec": torch.zeros(2, 3),
        "noise_mask": torch.tensor([False, True]),
        "batch": torch.zeros(2, dtype=torch.long),
        "entropy": torch.ones(1),
        "forces_weight": torch.ones(1),
    }
    return pred, label


def _element_weight_inputs():
    pred = {"forces": torch.tensor([[1.0, 0.0, 0.0]] * 3)}
    label = {
        "forces": torch.zeros(3, 3),
        "node_attrs": torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]
        ),
        "batch": torch.zeros(3, dtype=torch.long),
        "entropy": torch.ones(1),
        "forces_weight": torch.ones(1),
    }
    return pred, label


@pytest.mark.parametrize(
    ("loss_fn", "forces_loss", "dens_loss"),
    [
        (mse_dens_forces, 4.0 / 6.0, 9.0 / 6.0),
        (mae_dens_forces, 2.0 / 6.0, 3.0 / 6.0),
        (l2mae_dens_forces, 2.0 / 2.0, 3.0 / 2.0),
    ],
)
def test_dens_loss_ratio_scales_loss_linearly(loss_fn, forces_loss, dens_loss):
    pred, label = _dens_inputs()
    ratio = 0.25
    actual = loss_fn(pred, label, dens_loss_ratio=ratio)
    expected = forces_loss + ratio * dens_loss
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_huber_dens_loss_ratio_scales_loss_linearly():
    pred, label = _dens_inputs()
    ratio = 0.25
    actual = huber_dens_forces(
        pred,
        label,
        huber_delta=1.0,
        dens_loss_ratio=ratio,
    )
    expected = (1.5 + ratio * 2.5) / 6.0
    torch.testing.assert_close(actual, torch.tensor(expected))


def test_loss_function_kwargs_are_forwarded():
    pred, label = _dens_inputs()
    loss = NormalLoss(
        loss_property=["forces"],
        loss_function_name=["mse_dens_forces"],
        loss_property_weights=[1.0],
        loss_function_kwargs=[{"dens_loss_ratio": 0.25}],
    )
    expected = mse_dens_forces(pred, label, dens_loss_ratio=0.25)
    torch.testing.assert_close(loss(pred, label), expected)


def test_per_atom_element_weights_follow_node_attrs_order():
    pred, label = _element_weight_inputs()
    actual = mse_forces(pred, label, element_weights=[2.0, 4.0])
    torch.testing.assert_close(actual, torch.tensor(8.0 / 9.0))


def test_default_element_weights_are_one():
    pred, label = _element_weight_inputs()
    default = mse_forces(pred, label)
    explicit = mse_forces(pred, label, element_weights=[1.0, 1.0])
    torch.testing.assert_close(default, explicit)


def test_huber_element_weights_scale_loss_not_residual():
    pred, label = _element_weight_inputs()
    actual = huber_forces(
        pred,
        label,
        huber_delta=1.0,
        element_weights=[2.0, 4.0],
    )
    torch.testing.assert_close(actual, torch.tensor(4.0 / 9.0))


def test_element_weights_must_cover_all_node_attrs():
    pred, label = _element_weight_inputs()
    with pytest.raises(ValueError, match="expected 2, got 1"):
        mse_forces(pred, label, element_weights=[1.0])


def test_uncertainty_loss_forwards_loss_function_kwargs():
    pred, label = _dens_inputs()
    loss = UncertaintyLoss(
        loss_property=["forces"],
        loss_function_name=["mse_dens_forces"],
        loss_property_weights=[0.5],
        loss_function_kwargs=[{"dens_loss_ratio": 0.25}],
    )
    expected = 0.5 * mse_dens_forces(pred, label, dens_loss_ratio=0.25)
    torch.testing.assert_close(loss(pred, label), expected)


def test_loss_function_kwargs_are_validated_at_construction():
    with pytest.raises(ValueError, match="huber_delta"):
        NormalLoss(
            loss_property=["forces"],
            loss_function_name=["mse_forces"],
            loss_property_weights=[1.0],
            loss_function_kwargs=[{"huber_delta": 0.1}],
        )


def test_only_huber_losses_accept_huber_delta():
    ensure_loss_functions_registered()
    for name, loss_fn in LOSS_FN.items():
        has_huber_delta = "huber_delta" in inspect.signature(loss_fn).parameters
        assert has_huber_delta == ("huber" in name)

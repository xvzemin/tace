import pytest
import torch

from tace.dataset.statistics import balanced_element_weights
from tace.utils.loss.dens import (
    add_gaussian_noise_to_position,
    huber_dens_forces,
    l2mae_dens_forces,
    mae_dens_forces,
    mse_dens_forces,
)
from tace.utils.loss.huber_fn import huber_forces
from tace.utils.loss.mse_fn import mse_forces
from tace.utils.loss.normal import NormalLoss
from tace.utils.loss.uncertainty import UncertaintyLoss


def test_dens_masks_graphs_by_summed_forces():
    batch = {
        "positions": torch.zeros(3, 3),
        "node_attrs": torch.ones(3, 1),
        "ptr": torch.tensor([0, 2, 3]),
        "batch": torch.tensor([0, 0, 1]),
        "forces": torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
    }
    result = add_gaussian_noise_to_position(
        batch,
        prob=1.0,
        corrupt_ratio=None,
        strict_max_ratio=None,
        max_forces_norm=None,
        max_mean_forces_norm=1.0,
    )
    torch.testing.assert_close(result["dens_batch_mask"], torch.tensor([True, False]))
    torch.testing.assert_close(result["noise_vec"][2], torch.zeros(3))


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
    ("loss_fn", "kwargs", "forces_loss", "dens_loss"),
    [
        (mse_dens_forces, {}, 4.0 / 6.0, 9.0 / 6.0),
        (mae_dens_forces, {}, 2.0 / 6.0, 3.0 / 6.0),
        (l2mae_dens_forces, {}, 2.0 / 2.0, 3.0 / 2.0),
        (huber_dens_forces, {"huber_delta": 1.0}, 1.5 / 6.0, 2.5 / 6.0),
    ],
)
def test_dens_loss_ratio_scales_loss_linearly(
    loss_fn,
    kwargs,
    forces_loss,
    dens_loss,
):
    pred, label = _dens_inputs()
    ratio = 0.25
    actual = loss_fn(pred, label, dens_loss_ratio=ratio, **kwargs)
    expected = forces_loss + ratio * dens_loss
    torch.testing.assert_close(actual, torch.tensor(expected))


@pytest.mark.parametrize(
    ("loss_cls", "loss_weight"),
    [(NormalLoss, 1.0), (UncertaintyLoss, 0.5)],
)
def test_loss_function_kwargs_are_forwarded(loss_cls, loss_weight):
    pred, label = _dens_inputs()
    loss = loss_cls(
        loss_property=["forces"],
        loss_function_name=["mse_dens_forces"],
        loss_property_weights=[loss_weight],
        loss_function_kwargs=[{"dens_loss_ratio": 0.25}],
    )
    expected = loss_weight * mse_dens_forces(pred, label, dens_loss_ratio=0.25)
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


def test_balanced_element_weights_have_weighted_mean_one():
    counts = torch.tensor([12.0, 12.0, 2.0, 1.0])
    mean_losses = torch.tensor([1.0, 4.0, 3.0, 2.0])
    frequencies = counts / counts.sum()
    for alpha in (0.0, 0.5, 1.0):
        weights = balanced_element_weights(counts, mean_losses, alpha=alpha)
        torch.testing.assert_close(
            torch.sum(frequencies * weights),
            torch.tensor(1.0, dtype=weights.dtype),
        )
        assert torch.all((0.25 <= weights) & (weights <= 4.0))
    torch.testing.assert_close(
        balanced_element_weights(counts, mean_losses, alpha=0.0),
        torch.ones(4, dtype=torch.float64),
    )

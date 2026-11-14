import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tace.dataset.statistics import balanced_element_weights
from tace.utils.loss.common import (
    has_valid_labels,
    prepare_tensor_loss_counts,
    tensor_loss,
)
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
from tace.utils.loss.registry import LOSS_FN
from tace.utils.loss.uncertainty import UncertaintyLoss
from tace.utils.metrics import MaskMAE, MaskRMSE
from tace.utils.optimizer.hybrid_muon import HybridMuonOptimizer


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("column", [False, True])
def test_u_shift_energy_convention(dtype, column):
    from ase import Atoms
    from torch_geometric.data import Batch, Data

    from tace.foundations.u_shift import apply_u_shift, apply_wtc_u_shift

    metals = [23, 24, 25, 26, 27, 28, 42, 74]
    offsets = [-2.271, -2.458, -1.602, -1.925, -2.004, -2.586, -4.895, -5.875]
    structures = [[z, ligand] for ligand in (8, 9) for z in metals]
    structures += [[z] for z in range(1, 90)] + [[26, 26, 8], [26, 74], [8, 9]]
    expected = offsets * 2 + [0.0] * 89 + [-3.85, 0.0, 0.0]
    graphs = [
        Data(
            atomic_numbers=torch.tensor(numbers),
            num_nodes=len(numbers),
            # The correction applies even when the metal has no O/F neighbor.
            edge_index=torch.empty(2, 0, dtype=torch.long),
        )
        for numbers in structures
    ]
    batch = Batch.from_data_list(graphs)
    energy = torch.full(
        (len(graphs), 1) if column else (len(graphs),), -10.0, dtype=dtype
    )
    reference = energy + energy.new_tensor(expected).reshape_as(energy)
    torch.testing.assert_close(apply_u_shift(batch, energy), reference)
    compiled = torch.compile(apply_u_shift, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(batch.to_dict(), energy), reference)
    assert (energy == -10.0).all()
    for graph, numbers, offset in zip(graphs, structures, expected):
        atoms = Atoms(numbers=numbers)
        shifted = apply_wtc_u_shift(atoms, -10.0)
        assert shifted == pytest.approx(-10.0 + offset)
        assert apply_wtc_u_shift(atoms, shifted, minus=True) == pytest.approx(-10.0)
        torch.testing.assert_close(apply_u_shift(graph, energy[0]), energy[0] + offset)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("dens", [False, True])
@pytest.mark.parametrize("prefix", ["train", "val", "test_0"])
def test_shared_step_preprocessing_preserves_raw_batch(
    monkeypatch, enabled, dens, prefix
):
    from torch_geometric.data import Batch, Data

    from tace.foundations.u_shift import shift_energy_labels
    from tace.lightning import lit_model

    monkeypatch.setenv("TACE_USE_U_SHIFT", "1" if enabled else "0")
    monkeypatch.setenv("TACE_USE_DENS", "1" if dens else "0")
    monkeypatch.setattr(
        lit_model,
        "add_gaussian_noise_to_position",
        lambda batch: add_gaussian_noise_to_position(
            batch,
            prob=1.0,
            corrupt_ratio=1.0,
            strict_max_ratio=None,
            max_forces_norm=None,
        ),
    )
    graphs = [
        Data(
            num_nodes=len(numbers),
            atomic_numbers=torch.tensor(numbers),
            energy=torch.tensor([-10.0], dtype=torch.float64),
            positions=torch.randn(len(numbers), 3, dtype=torch.float64),
            forces=torch.randn(len(numbers), 3, dtype=torch.float64),
            stress=torch.randn(1, 3, 3, dtype=torch.float64),
            energy_weight=torch.ones(1),
            forces_weight=torch.ones(1),
            stress_weight=torch.ones(1),
        )
        for numbers in ([26, 8], [26])
    ]
    batch = Batch.from_data_list(graphs)
    original = batch.to_dict()
    expected = batch.energy + batch.energy.new_tensor([-1.925 if enabled else 0, 0])
    parameter = torch.tensor(1.0, dtype=torch.float64, requires_grad=True)
    seen = []

    def process(labels):
        torch.testing.assert_close(labels.energy, expected)
        if dens and prefix == "train":
            assert labels.noise_mask.all()
            torch.testing.assert_close(
                labels.positions, original["positions"] + labels.noise_vec
            )
            assert not torch.equal(labels.positions, original["positions"])
        else:
            assert "noise_vec" not in labels and "noise_mask" not in labels
            assert labels.positions is original["positions"]
        for name in (
            "forces",
            "stress",
            "energy_weight",
            "forces_weight",
            "stress_weight",
        ):
            assert labels[name] is original[name]
        output = {"energy": parameter.expand_as(labels.energy)}
        return output, (output["energy"] - labels.energy).square().mean()

    def metrics_update(metrics, stage, output, labels, properties):
        assert stage == prefix
        torch.testing.assert_close(labels.energy, expected)
        seen.append(stage)

    monkeypatch.setattr(lit_model, "update_metrics", metrics_update)
    wrapper = SimpleNamespace(
        _process_batch=process,
        skip_controller=None,
        sync_dist=False,
        log=lambda *args, **kwargs: None,
        optimizers=lambda: SimpleNamespace(param_groups=[{"lr": 0.01}]),
        loss_property=["energy"],
        **{f"{prefix}_metrics": {}},
    )
    for index in range(2):
        loss = lit_model.LightningWrapperModel._shared_step(
            wrapper, batch, index, prefix
        )
        gradient = torch.autograd.grad(loss, parameter)[0]
        torch.testing.assert_close(gradient, 2 * (parameter - expected).mean())
        current = batch.to_dict()
        assert all(current[key] is value for key, value in original.items())
        assert (batch.energy == -10.0).all()
    assert seen == [prefix, prefix]
    labels = shift_energy_labels(batch.to_dict())
    torch.testing.assert_close(labels["energy"], expected)
    unlabeled = {"atomic_numbers": batch.atomic_numbers}
    assert shift_energy_labels(unlabeled) is unlabeled


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["mse", "mae", "huber", "l2mae"])
def test_loss_precision(kind, dtype):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    prediction = torch.tensor(
        [[1.0e8, 2.0, 3.0]], dtype=dtype, device=device, requires_grad=True
    )
    label = {
        "forces": torch.ones_like(prediction),
        "forces_weight": torch.ones(1, dtype=dtype, device=device),
        "entropy": torch.ones(1, dtype=dtype, device=device),
        "batch": torch.zeros(1, dtype=torch.long, device=device),
    }
    original = dict(label)
    error = prediction.double() - label["forces"].double()
    reference = {
        "mse": error.square().mean(),
        "mae": error.abs().mean(),
        "huber": torch.nn.functional.huber_loss(
            error, torch.zeros_like(error), delta=0.01
        ),
        "l2mae": torch.linalg.vector_norm(error, dim=-1).mean(),
    }[kind]
    with torch.autocast(device_type=device.type, dtype=torch.bfloat16):
        actual = LOSS_FN[f"{kind}_forces"]({"forces": prediction}, label)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, reference, atol=0, rtol=1e-14)
    gradient = torch.autograd.grad(actual, prediction, create_graph=True)[0]
    expected = torch.autograd.grad(reference, prediction, create_graph=True)[0]
    assert gradient.dtype == dtype
    torch.testing.assert_close(gradient, expected)
    if kind == "mse":
        second = torch.autograd.grad(gradient.sum(), prediction)[0]
        torch.testing.assert_close(second, torch.full_like(prediction, 2 / 3))
    assert all(label[key] is value for key, value in original.items())


@pytest.mark.parametrize("loss_type", [NormalLoss, UncertaintyLoss])
def test_loss_compiled_precision(loss_type):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = loss_type(["energy"], ["mse_energy"], [1.0]).to(device)
    prediction = torch.tensor(
        [1.0e20], dtype=torch.float32, device=device, requires_grad=True
    )
    label = {
        "energy": -prediction.detach(),
        "energy_weight": torch.ones(1, dtype=torch.float32, device=device),
        "entropy": torch.ones(1, dtype=torch.float32, device=device),
    }
    reference = (prediction.double() - label["energy"].double()).square().mean()
    compiled = torch.compile(loss, backend="aot_eager", fullgraph=True)
    actual = compiled({"energy": prediction}, label)
    assert actual.dtype == torch.float64 and torch.isfinite(actual)
    torch.testing.assert_close(actual, reference)
    actual.backward()
    torch.testing.assert_close(prediction.grad, 4 * prediction.detach())
    if loss_type is UncertaintyLoss:
        assert loss.log_sigmas["energy"].dtype == torch.float64
        torch.testing.assert_close(
            loss.log_sigmas["energy"].grad, 1 - reference.detach()
        )


def test_loss_coordinate_derivatives():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    positions = torch.ones(1, 3, dtype=torch.float32, device=device, requires_grad=True)
    weight = torch.tensor(2.0, dtype=torch.float32, device=device, requires_grad=True)
    forces = weight * positions.square().roll(1, dims=-1)
    loss = LOSS_FN["mse_direct_forces_curl"](
        {"direct_forces": forces}, {"positions": positions}
    )
    assert loss.dtype == torch.float64
    loss.backward()
    assert positions.grad is not None and torch.isfinite(positions.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad)


@pytest.mark.parametrize("loss", ["mse", "mae", "huber", "l2mae"])
@pytest.mark.parametrize("voigt", [False, True])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "key", ["stress", "virials", "direct_stress", "direct_virials"]
)
def test_tensor_loss_missing_labels(loss, voigt, key, prepared):
    reference = torch.arange(1.0, 10.0).reshape(1, 3, 3)
    target = torch.zeros_like(reference)
    pred = torch.cat([reference, 100 * reference]).requires_grad_()
    labels = {
        key: torch.cat([target, target]),
        f"{key}_weight": torch.tensor([1.0, 0.0]),
        "entropy": torch.ones(2),
    }
    actual = tensor_loss(
        {key: pred},
        prepare_tensor_loss_counts(labels, [key]) if prepared else labels,
        key,
        loss,
        voigt=voigt,
    )
    expected = tensor_loss(
        {key: reference}, {k: v[:1] for k, v in labels.items()}, key, loss, voigt=voigt
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all() and not pred.grad[1].any()
    labels[f"{key}_weight"].zero_()
    pred.grad = None
    empty = tensor_loss(
        {key: pred},
        prepare_tensor_loss_counts(labels, [key]) if prepared else labels,
        key,
        loss,
        voigt=voigt,
    )
    empty.backward()
    assert empty == 0 and not pred.grad.any()


@pytest.mark.parametrize("prepared", [False, True])
def test_tensor_loss_distributed_count(monkeypatch, prepared):
    from tace.utils.loss import common

    pred = {"stress": torch.ones(2, 3, 3, requires_grad=True)}
    label = {
        "stress": torch.zeros(2, 3, 3),
        "entropy": torch.ones(2),
        "stress_weight": torch.tensor([2.0, 0.0]),
    }
    torch.testing.assert_close(
        tensor_loss(pred, label, "stress", "mse"), torch.tensor(2.0)
    )
    monkeypatch.setattr(common.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(common.dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(
        common.dist, "all_reduce", lambda count, op: count.add_(2 if prepared else 18)
    )
    if prepared:
        label = prepare_tensor_loss_counts(label, ["stress"])
    torch.testing.assert_close(
        tensor_loss(pred, label, "stress", "mse"), torch.tensor(36.0 / 27.0)
    )


@pytest.mark.parametrize("metric_type", [MaskMAE, MaskRMSE])
def test_stress_metric_missing_labels(metric_type):
    metric = metric_type("stress", 1.0)
    target = torch.full((2, 3, 3), float("nan"))
    label = {"stress": target, "stress_weight": torch.zeros(2)}
    metric.update(torch.ones_like(target), target, label)
    assert torch.isnan(metric.compute())
    target[0] = 0
    label["stress_weight"][0] = 1
    metric.update(torch.ones_like(target), target, label)
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0, dtype=torch.float64))
    assert metric.count == 9


@pytest.mark.parametrize("metric_type", [MaskMAE, MaskRMSE])
@pytest.mark.parametrize(
    "key", ["energy", "forces", "stress", "edge_forces", "hessian"]
)
def test_masked_metric_counts(metric_type, key):
    shapes = {
        "energy": (3,),
        "forces": (5, 3),
        "stress": (3, 3, 3),
        "edge_forces": (4, 3),
        "hessian": (13, 3, 3),
    }
    labels = {
        f"{key}_weight": torch.tensor([2.0, 0.0, 0.0]),
        "batch": torch.tensor([0, 0, 1, 1, 1]),
        "ptr": torch.tensor([0, 2, 5, 5]),
        "edge_index": torch.tensor([[2, 0, 3, 1], [3, 1, 4, 0]]),
    }
    valid = {
        "energy": torch.tensor([True, False, False]),
        "forces": torch.tensor([True, False, False, False, False]),
        "stress": torch.tensor([True, False, False]),
        "edge_forces": torch.tensor([False, True, False, True]),
        "hessian": torch.tensor([True] * 4 + [False] * 9),
    }[key]
    if key == "forces":
        labels["noise_mask"] = torch.tensor([False, True, False, False, False])
    target = torch.zeros(shapes[key], dtype=torch.float64)
    prediction = torch.arange(target.numel(), dtype=torch.float64).reshape_as(target)
    # Missing and denoised labels may be non-finite and must not enter the sums.
    prediction[~valid] = float("inf")
    target[~valid] = float("nan")
    reference = prediction[valid] - target[valid]
    metric = metric_type(key, 1.0)
    for _ in range(2):
        metric.update(prediction, target, labels)
    expected = (
        reference.abs().mean()
        if metric_type is MaskMAE
        else reference.square().mean().sqrt()
    )
    torch.testing.assert_close(metric.compute(), expected)
    assert metric.count == 2 * reference.numel()


def test_training_metric_epoch_reductions():
    from torchmetrics import MetricCollection

    from tace.utils.metrics import build_metrics, update_metrics

    properties = ["energy", "forces", "stress"]
    labels = {
        "ptr": torch.tensor([0, 2, 5]),
        "batch": torch.tensor([0, 0, 1, 1, 1]),
        "entropy": torch.tensor([2.0, 0.5]),
        "energy": torch.zeros(2),
        "forces": torch.zeros(5, 3),
        "stress": torch.zeros(2, 3, 3),
        "energy_weight": torch.tensor([1.0, 0.0]),
        "forces_weight": torch.tensor([2.0, 0.0]),
        "stress_weight": torch.tensor([0.0, 1.0]),
        "noise_mask": torch.tensor([False, True, False, False, False]),
    }
    predictions = {
        "energy": torch.tensor([2.0, float("nan")]),
        "forces": torch.tensor([[1.0, 2.0, 3.0]] + [[float("nan")] * 3] * 4),
        "stress": torch.stack(
            (torch.full((3, 3), float("nan")), torch.full((3, 3), 2.0))
        ),
    }
    expected = {
        "train/energy_mae": 2000.0,
        "train/energy_rmse": 2000.0,
        "train/energy_per_atom_mae": 1000.0,
        "train/energy_per_atom_rmse": 1000.0,
        "train/forces_mae": 8000.0,
        "train/forces_rmse": (224.0 / 3) ** 0.5 * 1000,
        "train/stress_mae": 2000.0,
        "train/stress_rmse": 2000.0,
    }
    metrics = MetricCollection(build_metrics("train", properties))
    for _ in range(2):
        for _ in range(3):
            update_metrics(metrics, "train", predictions, labels, properties)
        actual = metrics.compute()
        assert actual.keys() == expected.keys()
        for key, value in expected.items():
            torch.testing.assert_close(
                actual[key], torch.tensor(value, dtype=torch.float64)
            )
        metrics.reset()


@pytest.mark.parametrize("masked", [False, True])
@pytest.mark.parametrize("squared", [False, True])
def test_metric_accumulation_precision(masked, squared):
    from tace.utils.metrics import MAE, RMSE

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metric_cls = (
        (MaskRMSE if squared else MaskMAE) if masked else (RMSE if squared else MAE)
    )
    metric = (metric_cls("energy", 1.0) if masked else metric_cls(1.0)).to(device)
    for value in (1.0e8, 1.0, 1.0):
        prediction = torch.tensor([value], dtype=torch.float32, device=device)
        target = torch.zeros(1, dtype=torch.float64, device=device)
        if masked:
            metric.update(
                prediction, target, {"energy_weight": torch.ones_like(target)}
            )
        else:
            metric.update(prediction, target)
    expected = torch.tensor([1.0e8, 1.0, 1.0], dtype=torch.float64, device=device)
    expected = expected.square().mean().sqrt() if squared else expected.mean()
    torch.testing.assert_close(metric.compute(), expected, atol=0, rtol=1e-15)
    assert metric.compute().dtype == torch.float64
    metric.reset()
    state = metric.sum_squared_error if squared else metric.sum_abs_error
    assert state.dtype == torch.float64 and not state.any()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("prepared", [False, True])
@pytest.mark.parametrize(
    "key, shape",
    [
        ("energy", (2,)),
        ("forces", (5, 3)),
        ("stress", (2, 3, 3)),
        ("direct_hessian", (13, 3, 3)),
    ],
)
def test_uncertainty_labels(key, shape, dtype, prepared):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss = UncertaintyLoss([key], [f"mse_{key}"], [1.0]).to(device, dtype)
    normal = NormalLoss([key], [f"mse_{key}"], [1.0])
    pred = {key: torch.ones(shape, device=device, dtype=dtype, requires_grad=True)}
    label = {
        key: torch.zeros(shape, device=device, dtype=dtype),
        "entropy": torch.ones(2, device=device, dtype=dtype),
        f"{key}_weight": torch.zeros(2, device=device, dtype=dtype),
        "batch": torch.tensor([0, 0, 1, 1, 1], device=device),
        "ptr": torch.tensor([0, 2, 5], device=device),
    }
    for present in (False, True):
        label[f"{key}_weight"][1] = float(present)
        loss.zero_grad()
        pred[key].grad = None
        reference = normal(pred, label).detach()
        value = loss(
            pred, prepare_tensor_loss_counts(label, [key]) if prepared else label
        )
        log_sigma = loss.log_sigmas[key]
        torch.testing.assert_close(value, reference + log_sigma * present)
        value.backward()
        torch.testing.assert_close(
            log_sigma.grad, (float(present) - reference).to(dtype)
        )
        assert torch.isfinite(pred[key].grad).all()
        if not present:
            assert not pred[key].grad.any()

    # A zero residual with valid labels must retain the uncertainty regularizer.
    label[key] = pred[key].detach().clone()
    loss.zero_grad()
    value = loss(pred, label)
    torch.testing.assert_close(value, loss.log_sigmas[key].double())
    value.backward()
    torch.testing.assert_close(
        loss.log_sigmas[key].grad, torch.ones_like(loss.log_sigmas[key])
    )


def test_valid_labels_weights_and_distributed_presence(monkeypatch):
    from tace.utils.loss import common

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert has_valid_labels(
        {
            "entropy": torch.tensor([1.0e-4], dtype=torch.float16, device=device),
            "energy_weight": torch.tensor([1.0e-4], dtype=torch.float16, device=device),
        },
        "energy",
    )
    for key, shape in (
        ("energy", (2,)),
        ("forces", (5, 3)),
        ("stress", (2, 3, 3)),
        ("direct_hessian", (13, 3, 3)),
    ):
        label = {
            key: torch.zeros(shape, device=device),
            "entropy": torch.tensor([1.0, 0.0], device=device),
            f"{key}_weight": torch.tensor([0.0, 1.0], device=device),
            "batch": torch.tensor([0, 0, 1, 1, 1], device=device),
            "ptr": torch.tensor([0, 2, 5], device=device),
        }
        assert not has_valid_labels(label, key)
        label["entropy"].fill_(1)
        assert has_valid_labels(label, key)
        label[f"{key}_weight"].zero_()
        assert not has_valid_labels(label, key)

    def all_reduce(present, op):
        assert present == 0
        assert op == common.dist.ReduceOp.MAX
        present.fill_(1)

    monkeypatch.setattr(common.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(common.dist, "all_reduce", all_reduce)
    assert has_valid_labels(label, key)


def test_dens_masks_graphs_by_summed_forces():
    batch = {
        "positions": torch.zeros(3, 3),
        "node_type": torch.zeros(3, dtype=torch.long),
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
        "node_type": torch.tensor([0, 1, 0]),
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
    torch.testing.assert_close(actual, torch.tensor(expected, dtype=torch.float64))


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


@pytest.mark.parametrize("loss_cls", [NormalLoss, UncertaintyLoss])
@pytest.mark.parametrize("kind", ["mse", "mae", "huber", "l2mae"])
def test_dens_evaluation_uses_ordinary_losses(loss_cls, kind):
    keys = ["forces", "direct_forces", "stress", "direct_stress"]
    functions = [
        f"{kind}_dens_forces",
        f"{kind}_dens_direct_forces",
        f"{kind}_dens_voigt_stress",
        f"{kind}_dens_direct_stress",
    ]
    kwargs = [{"dens_loss_ratio": 0.25}, {"dens_loss_ratio": 0.5}, {}, {}]
    if kind == "huber":
        kwargs = [dict(item, huber_delta=0.1) for item in kwargs]
    loss = loss_cls(keys, functions, [1.0] * 4, kwargs).eval()
    reference = loss_cls(
        keys,
        [name.replace("_dens_", "_") for name in functions],
        [1.0] * 4,
        [{k: v for k, v in item.items() if k != "dens_loss_ratio"} for item in kwargs],
    ).eval()
    labels = {"batch": torch.tensor([0, 0, 1]), "entropy": torch.tensor([1.0, 0.5])}
    predictions = {}
    for key in keys:
        shape = (3, 3) if "forces" in key else (2, 3, 3)
        predictions[key] = torch.randn(shape, requires_grad=True)
        labels[key] = torch.randn(shape)
        labels[f"{key}_weight"] = torch.tensor([1.0, 0.0])
    actual, expected = loss(predictions, labels), reference(predictions, labels)
    torch.testing.assert_close(actual, expected)
    gradients = torch.autograd.grad(actual, tuple(predictions.values()))
    expected_gradients = torch.autograd.grad(expected, tuple(predictions.values()))
    for gradient, expected_gradient in zip(gradients, expected_gradients):
        torch.testing.assert_close(gradient, expected_gradient)
    assert loss.loss_function_name == functions
    assert loss.loss_function_kwargs == kwargs


def test_per_atom_element_weights_follow_node_type_order():
    pred, label = _element_weight_inputs()
    actual = mse_forces(pred, label, element_weights=[2.0, 4.0])
    torch.testing.assert_close(actual, torch.tensor(8.0 / 9.0, dtype=torch.float64))


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
    torch.testing.assert_close(actual, torch.tensor(4.0 / 9.0, dtype=torch.float64))


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


@pytest.mark.parametrize(
    "optimizer_name", ["Adam", "AdamW", "SGD", "HybridMuonOptimizer"]
)
@pytest.mark.parametrize("matrix_weight", ["0", "1"])
@pytest.mark.parametrize("weight_decay", [None, 0.1])
def test_optimizer_weight_decay(optimizer_name, matrix_weight, weight_decay):
    from e3nn.nn import BatchNorm

    from tace.lightning.lit_model import LightningWrapperModel
    from tace.models.blocks import ScaleShift
    from tace.models.linear import e3nnLinear, enable_lora
    from tace.models.radial import (
        RadialBasis,
        ZBLBasis,
        j0SphericalBesselBasis,
        jnSphericalBesselBasis,
    )
    from tace.models.spherical.layer_norm import EquivariantMergeLayerNorm
    from tace.models.spherical.scatter_norm import DensityScatterNorm

    class TrainingModel(torch.nn.Module):
        configure_optimizers = LightningWrapperModel.configure_optimizers

        def __init__(self):
            super().__init__()
            self.cfg = {
                "optimizer": {
                    "_target_": (
                        "tace.utils.optimizer.HybridMuonOptimizer"
                        if optimizer_name == "HybridMuonOptimizer"
                        else f"torch.optim.{optimizer_name}"
                    ),
                    "lr": 0.01,
                }
            }
            if weight_decay is not None:
                self.cfg["optimizer"]["weight_decay"] = weight_decay
            self.linear = e3nnLinear(
                "2x0e+2x1o",
                "2x0e+2x1o",
                bias=True,
                use_matrix_weight=matrix_weight,
            )
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(3, 3), torch.nn.SiLU(), torch.nn.Linear(3, 2)
            )
            self.norm = EquivariantMergeLayerNorm([0, 1], 2)
            self.batch_norm = BatchNorm("2x0e+2x1o")
            self.density = DensityScatterNorm(2.0, 4, radial_bias=True)
            enable_lora(self.density.edge_density.mlp[0], r=2, freeze_base=False)
            self.radial = RadialBasis(trainable=True)
            self.j0 = j0SphericalBesselBasis(num_basis=2, trainable=True)
            self.j1 = jnSphericalBesselBasis(order=1, num_basis=2, trainable=True)
            self.scale_shift = ScaleShift(
                [1], [{1: 1.0}], [{1: 0.0}], True, True, all_atoms=True
            )
            self.zbl = ZBLBasis("c2poly", trainable=True)
            self.loss_fn = UncertaintyLoss(["energy"], ["mse_energy"], [1.0])
            self.activation = torch.nn.PReLU(2)
            self.temperature_logit = torch.nn.Parameter(torch.ones(2))
            self.bias = torch.nn.ParameterDict(
                {"energy": torch.nn.Parameter(torch.ones(2))}
            )
            self.biases = torch.nn.ParameterList([torch.nn.Parameter(torch.ones(2))])
            self.tied = torch.nn.Linear(3, 3, bias=False)
            self.embedding = torch.nn.Embedding(3, 3)
            self.embedding.weight = self.tied.weight
            self.alias = torch.nn.Linear(3, 3, bias=False)
            self.alias.weight = self.tied.weight
            self.frozen = torch.nn.Parameter(torch.ones(3), requires_grad=False)

    model = TrainingModel()
    optimizer = model.configure_optimizers()["optimizer"]
    expected_decay = weight_decay
    if expected_decay is None:
        expected_decay = 0.001 if optimizer_name == "HybridMuonOptimizer" else 0.0
    params = [p for group in optimizer.param_groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    decay = {
        id(p): group["weight_decay"]
        for group in optimizer.param_groups
        for p in group["params"]
    }
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        expected = name.startswith("linear.weight") or (
            name.startswith("mlp.") and name.endswith("weight")
        )
        assert decay[id(param)] == (expected_decay if expected else 0.0), name
        with torch.no_grad():
            param.fill_(1.0)
        param.grad = torch.zeros_like(param)
    optimizer.step()
    for param in params:
        if decay[id(param)]:
            assert torch.all(param < 1)
        else:
            torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)


@pytest.mark.parametrize("matrix_weight", ["0", "1"])
def test_optimizer_weight_decay_parameter_roles(matrix_weight):
    from e3nn import o3

    from tace.lightning.optimizer import (
        NO_WEIGHT_DECAY_NAMES,
        get_optimizer_param_groups,
    )
    from tace.models.linear import enable_lora, switch_e3nn_weight_layout
    from tace.models.spherical.edge import Element2EdgeUpdate
    from tace.models.spherical.node import (
        Element2NodeUpdate,
        ElementNodeUpdate,
        SphericalTensorNodeEmbedding,
    )
    from tace.models.spherical.readout import (
        DirectHessianReadout,
        EdgeReadOut,
        ElementScalarReadOut,
        ElementTensorReadOut,
        ScalarReadOut,
        TensorReadOut,
    )
    from tace.models.spherical.ue import (
        UniversalEquivariantEmbedding,
        UniversalInvariantEmbedding,
    )

    model = torch.nn.Module()
    # Names do not make continuous feature maps into lookups or output heads.
    for name in ("node_embedding", "source_embedding", "embedding", "energy_readout"):
        model.add_module(name, torch.nn.Linear(3, 4))
    model.initial = SphericalTensorNodeEmbedding(3, 4, 4, 1, 1, 2.0, bias=True)
    model.endpoints = Element2EdgeUpdate(0, 1, 3, 4, 4, 4, bias=True)
    model.node_update = ElementNodeUpdate(3, 4, 4, bias=True)
    model.node_update_pair = Element2NodeUpdate(3, 4, 4, bias=True)
    model.invariant = UniversalInvariantEmbedding(
        4, {"energy": {}, "fidelity_idx": {"num_embeddings": 3}}, bias=True
    )
    model.equivariant = UniversalEquivariantEmbedding(
        o3.Irreps("4x0e"),
        4,
        3,
        {"electric_field": {"normalizer": 1.0}},
        time_reversal=False,
    )
    model.lookup = torch.nn.Embedding(3, 4)
    output_maps = []
    for cls in (
        ScalarReadOut,
        ElementScalarReadOut,
        TensorReadOut,
        ElementTensorReadOut,
    ):
        for layer in (0, 1):
            head = cls(
                layer=layer,
                num_layers=2,
                hidden_channel=[4],
                bias=True,
                num_elements=3,
                num_fidelities=1,
                parity=True,
                irreps_in="4x0e+4x1o",
                irreps_out="0e" if issubclass(cls, ScalarReadOut) else "1o",
            )
            model.add_module(f"head_{cls.__name__}_{layer}", head)
            output_maps.append(head.linear2[-1] if head.last_layer else head.linear1[0])
    model.edge_head = EdgeReadOut("4x0e+4x1o", "4x0e+4x1o", "2x0e", bias=True)
    model.pair_head = DirectHessianReadout(
        "4x0e+4x1e+4x2e", "4x0e+4x1e+4x2e", 4, 1, bias=True
    )
    output_maps.extend(
        (
            model.edge_head.linear_nonlinearity,
            model.pair_head.node_readout,
            model.pair_head.edge_readout.linear_nonlinearity,
        )
    )
    lookup_maps = (model.initial.node_embedding,)
    switch_e3nn_weight_layout(model, "matrix" if matrix_weight == "1" else "flat")
    # LoRA remains regularized even when the base is a lookup or final output map.
    for module in (*output_maps, *lookup_maps):
        enable_lora(module, r=2, freeze_base=False)

    excluded = {id(p) for name, p in model.named_parameters() if name.endswith("bias")}
    for module in (*output_maps, *lookup_maps):
        weights = module.weight
        if not isinstance(weights, torch.nn.ParameterList):
            weights = (weights,)
        excluded.update(id(p) for p in weights)
    excluded.update(
        id(p)
        for module in model.modules()
        if isinstance(module, torch.nn.Embedding)
        for p in module.parameters()
    )

    # Inspect the same owners standalone, in the model, and through another wrapper.
    wrapper = torch.nn.Module()
    wrapper.block = model
    assert not any(
        "embedding" in pattern or "readout" in pattern
        for pattern in NO_WEIGHT_DECAY_NAMES
    )
    for inspected in (*model.children(), model, wrapper):
        groups = get_optimizer_param_groups(inspected, 0.1)
        params = [p for group in groups for p in group["params"]]
        assert len(params) == len({id(p) for p in params})
        decay = {
            id(p): group["weight_decay"] for group in groups for p in group["params"]
        }
        for name, param in inspected.named_parameters():
            assert decay[id(param)] == (0.0 if id(param) in excluded else 0.1), name

    optimizer = torch.optim.AdamW(get_optimizer_param_groups(model, 0.1), lr=0.01)
    for param in model.parameters():
        with torch.no_grad():
            param.fill_(1.0)
        param.grad = torch.zeros_like(param)
    optimizer.step()
    for name, param in model.named_parameters():
        expected = 1.0 if id(param) in excluded else 0.999
        torch.testing.assert_close(param, torch.full_like(param, expected), msg=name)


@pytest.mark.parametrize("matrix_weight", ["0", "1"])
@pytest.mark.parametrize("muon_mode", ["2d", "flat", "slice"])
def test_optimizer_semantic_muon_groups(matrix_weight, muon_mode):
    from e3nn import o3
    from e3nn.nn import FullyConnectedNet

    from tace.lightning.optimizer import get_optimizer_param_groups
    from tace.models.cartesian.prod import NonlinearIctdACE
    from tace.models.linear import (
        e3nnElementLinear,
        e3nnLinear,
        e3nnMoEElementLinear,
        enable_lora,
        mlpLinear,
        switch_e3nn_weight_layout,
        torchLinear,
    )
    from tace.models.spherical.layer_norm import EquivariantMergeLayerNorm
    from tace.models.spherical.node import (
        Element2NodeUpdate,
        ElementNodeUpdate,
        SphericalTensorNodeEmbedding,
    )
    from tace.models.spherical.prod import CgtpACE, MaceACE, NonlinearCgtpACE
    from tace.models.spherical.readout import EdgeReadOut, ScalarReadOut
    from tace.models.spherical.scatter_norm import DensityScatterNorm

    model = torch.nn.Module()
    model.dense = torch.nn.Linear(4, 6)
    model.torch_dense = torchLinear(4, 6)
    model.mlp_dense = mlpLinear(4, 6)
    model.fully_connected = FullyConnectedNet([4, 6, 3], torch.nn.functional.silu)
    model.equivariant = e3nnLinear(
        "4x0e+4x1o", "6x0e+6x1o", use_matrix_weight=matrix_weight
    )
    model.element = e3nnElementLinear(
        "4x0e+4x1o",
        "6x0e+6x1o",
        num_elements=3,
        use_matrix_weight=matrix_weight,
    )
    model.experts = e3nnMoEElementLinear(
        "4x0e+4x1o",
        "6x0e+6x1o",
        num_elements=3,
        num_experts=2,
        use_matrix_weight=matrix_weight,
    )
    model.singletons = torch.nn.ModuleList(
        [
            e3nnElementLinear(left, right, num_elements=3, use_matrix_weight="1")
            for left, right in (("1x0e", "4x0e"), ("4x0e", "1x0e"))
        ]
        + [
            e3nnMoEElementLinear(
                "2x0e",
                "8x0e",
                num_elements=3,
                num_experts=2,
                use_matrix_weight="1",
            )
        ]
    )
    model.lookup = torch.nn.Embedding(3, 4)
    model.norm = EquivariantMergeLayerNorm([0, 1], 4)
    model.router = torch.nn.Linear(4, 4)
    model.density = DensityScatterNorm(2.0, 4, radial_bias=True)
    model.adapter = enable_lora(mlpLinear(4, 6), r=2, freeze_base=False)
    model.initial = SphericalTensorNodeEmbedding(3, 4, 4, 1, 1, 2.0)
    model.node_update = ElementNodeUpdate(3, 4, 6)
    model.node_update_pair = Element2NodeUpdate(3, 4, 6)
    model.head = ScalarReadOut(
        layer=0,
        num_layers=1,
        hidden_channel=[4],
        bias=True,
        num_elements=3,
        num_fidelities=1,
        parity=True,
        irreps_in="4x0e",
        irreps_out="2x0e",
    )
    model.edge_head = EdgeReadOut("4x0e", "4x0e", "2x0e")
    # Neutral module names ensure exclusion follows the owning module type.
    for module in (
        model.head,
        model.edge_head,
        model.initial,
        model.node_update,
        model.node_update_pair,
    ):
        switch_e3nn_weight_layout(module, "matrix")
    product_kwargs = dict(
        layer=0,
        num_layers=1,
        num_elements=3,
        Lmax=1,
        lmax=1,
        num_channel=4,
        num_expert=None,
        num_channel_per_expert=None,
        target_irreps=o3.Irreps("0e+1o"),
        irreps_in=o3.Irreps("4x0e+4x1o"),
        correlation=[2],
        l1l2=None,
        bias=True,
        nonlinear=None,
        parity=True,
    )
    model.product = CgtpACE(**product_kwargs)
    model.mace = MaceACE(**product_kwargs)
    model.nonlinear_product = NonlinearCgtpACE(**product_kwargs)
    model.nonlinear_cartesian = NonlinearIctdACE(**product_kwargs)
    for module in (
        model.product,
        model.mace,
        model.nonlinear_product,
        model.nonlinear_cartesian,
    ):
        switch_e3nn_weight_layout(module, "matrix" if matrix_weight == "1" else "flat")
    model.unknown_coefficients = torch.nn.Parameter(torch.ones(3, 4, 6))
    model.frozen = torch.nn.Parameter(torch.ones(4, 6), requires_grad=False)

    groups = get_optimizer_param_groups(model, 0.1, use_muon=True, muon_mode=muon_mode)
    params = [p for group in groups for p in group["params"]]
    assert len(params) == len({id(p) for p in params})
    assert {id(p) for p in params} == {
        id(p) for p in model.parameters() if p.requires_grad
    }
    by_param = {id(p): group for group in groups for p in group["params"]}
    expected = {
        id(module.weight)
        for module in (model.dense, model.torch_dense, model.mlp_dense, model.adapter)
    }
    expected.update(id(p) for p in model.fully_connected.parameters())
    expected.update(
        id(module.weight)
        for module in model.initial.edge_info.modules()
        if isinstance(module, (torch.nn.Linear, mlpLinear, torchLinear))
    )
    for module in (
        model.head.linear2[0],
        model.edge_head.linear_down,
        model.node_update.embedding,
        model.node_update_pair.source_embedding,
        model.node_update_pair.target_embedding,
    ):
        expected.update(id(p) for p in module.weight)
    expected.add(id(model.mace.weight))
    for product in (model.nonlinear_product, model.nonlinear_cartesian):
        expected.add(id(product.gate[0].weight))
    density_weight = model.density.edge_density.mlp[0].weight
    expected.add(id(density_weight))
    assert by_param[id(density_weight)]["weight_decay"] == 0.0
    if matrix_weight == "1":
        for module in (model.equivariant, model.product.linear, model.mace.linear):
            expected.update(id(p) for p in module.weight)
        for module in (model.element, model.experts, *model.product.coefs):
            expected.update(id(p) for p in module.weight)
        for product in (model.nonlinear_product, model.nonlinear_cartesian):
            for module in (product.linear_up, *product.coefs, product.linear):
                expected.update(id(p) for p in module.weight)
    actual = {id(p) for group in groups if group["use_muon"] for p in group["params"]}
    names = {id(p): name for name, p in model.named_parameters()}
    assert {names[p] for p in actual} == {names[p] for p in expected}
    assert by_param[id(model.mace.weight)]["weight_decay"] == 0.1
    assert by_param[id(model.mace.weight)]["muon_mode"] == "slice"
    for name, param in model.named_parameters():
        if id(param) in expected and param.ndim > 2:
            assert by_param[id(param)]["muon_mode"] == "slice", name
    assert by_param[id(model.unknown_coefficients)]["weight_decay"] == 0.1

    ordinary_groups = get_optimizer_param_groups(model, 0.1)
    assert len(ordinary_groups) == 2
    assert all("use_muon" not in group for group in ordinary_groups)
    assert {
        id(p): group["weight_decay"]
        for group in ordinary_groups
        for p in group["params"]
    } == {key: group["weight_decay"] for key, group in by_param.items()}


def test_optimizer_eqx_parameter_roles(caplog):
    from e3nn import o3

    from equivariantx import co2, co3, o2
    from equivariantx import o3 as eqx_o3
    from equivariantx.models.mace.symmetric_contraction import UuuSymmetricContraction
    from tace.lightning.optimizer import get_optimizer_param_groups

    class Contraction(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.correlation = 2
            self.weights_max = torch.nn.Parameter(torch.randn(3, 2, 4))
            self.weights = torch.nn.ParameterList(
                [torch.nn.Parameter(torch.randn(3, 2, 4))]
            )

        def U_tensors(self, degree):
            return torch.ones(*([1] * (degree + 1)), 2)

    original = torch.nn.Module()
    original.irreps_in = original.irreps_out = o3.Irreps("4x0e")
    original.contractions = torch.nn.ModuleList([Contraction()])
    model = torch.nn.Module()
    model.product = UuuSymmetricContraction(original, backend="torch")
    model.norm = torch.nn.LayerNorm((4, 6))
    model.o3_linear = o3.Linear("4x0e+4x1o", "6x0e+6x1o", biases=True)
    model.o3_product = eqx_o3.TensorProduct(
        "4x1o", "4x1o", "4x0e", [(0, 0, 0, "uuu", True)]
    )
    for name, space, vector in (
        ("o2", o2, "1m"),
        ("co2", co2, "1m"),
        ("co3", co3, "1o"),
    ):
        model.add_module(
            f"{name}_linear",
            space.Linear(f"4x0e+4x{vector}", f"6x0e+6x{vector}", biases=True),
        )
        model.add_module(
            f"{name}_product",
            space.TensorProduct(
                f"4x{vector}", f"4x{vector}", "4x0e", [(0, 0, 0, "uuu", True)]
            ),
        )
    model.matrix_product = eqx_o3.MatrixTensorProduct("4x1o", "4x1o", "4x0e")
    model.gaunt_product = eqx_o3.GauntTensorProduct("4x1o", "4x1o", "4x0e")
    model.element = eqx_o3.ElementLinear(o3.Linear("4x0e", "6x0e"))
    model.circular_element = o2.ElementLinear(o2.Linear("4x0e", "6x0e"))

    groups = get_optimizer_param_groups(model, 0.1, use_muon=True)
    by_param = {id(p): group for group in groups for p in group["params"]}
    coefficients = {id(p) for p in model.product.parameters()}
    assert len(by_param) == len(list(model.parameters()))
    assert coefficients
    for name, param in model.named_parameters():
        assert by_param[id(param)]["use_muon"] == (id(param) in coefficients), name
        expected_decay = (
            0.0 if name.startswith("norm.") or name.endswith("bias") else 0.1
        )
        assert by_param[id(param)]["weight_decay"] == expected_decay, name
    ordinary = get_optimizer_param_groups(model, 0.1)
    assert {
        id(param): group["weight_decay"]
        for group in ordinary
        for param in group["params"]
    } == {key: group["weight_decay"] for key, group in by_param.items()}
    assert all(by_param[p]["muon_mode"] == "slice" for p in coefficients)
    assert "packed" in caplog.text.lower()


@pytest.mark.parametrize("callable_rules", [False, True])
def test_optimizer_external_parameter_rules(callable_rules, caplog):
    from tace.lightning.optimizer import (
        get_optimizer_param_groups,
        get_weight_decay_exclusions,
    )

    class Channels(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.matrix = torch.nn.Parameter(torch.randn(4, 6))
            self.independent = torch.nn.Parameter(torch.randn(3, 4, 6))
            self.singleton = torch.nn.Parameter(torch.randn(3, 4, 1))
            self.lookup = torch.nn.Parameter(torch.randn(3, 4))
            self.packed = torch.nn.Parameter(torch.randn(3, 24))
            self.gain = torch.nn.Parameter(torch.ones(4, 6))

    roles = {
        "matrix": "matrix",
        "independent": "batched_matrix",
        "singleton": "batched_matrix",
        "lookup": "adam",
        "packed": "packed",
        "gain": "adam",
    }
    model = Channels()
    model.unknown = torch.nn.Parameter(torch.ones(4, 6))
    weight_decay_rules = {
        f"{Channels.__module__}.{Channels.__qualname__}": (
            (lambda module: ("gain",)) if callable_rules else ("gain",)
        )
    }
    assert get_weight_decay_exclusions(model) == set()
    assert get_weight_decay_exclusions(model, weight_decay_rules) == {"gain"}
    assert get_weight_decay_exclusions(model) == set()
    parameter_rules = {Channels: (lambda module: roles) if callable_rules else roles}
    groups = get_optimizer_param_groups(
        model,
        0.1,
        use_muon=True,
        muon_mode="flat",
        parameter_rules=parameter_rules,
        weight_decay_rules=weight_decay_rules,
    )
    by_param = {id(p): group for group in groups for p in group["params"]}
    assert len(by_param) == len(list(model.parameters()))
    for name, param in model.named_parameters():
        expected = name in ("matrix", "independent")
        assert by_param[id(param)]["use_muon"] is expected, name
        assert by_param[id(param)]["weight_decay"] == (0.0 if name == "gain" else 0.1)
        if expected:
            assert by_param[id(param)]["muon_mode"] == "slice"
    ordinary = get_optimizer_param_groups(
        model, 0.1, weight_decay_rules=weight_decay_rules
    )
    assert {
        id(param): group["weight_decay"]
        for group in ordinary
        for param in group["params"]
    } == {key: group["weight_decay"] for key, group in by_param.items()}
    for patterns in ("gain", (1,), None):
        with pytest.raises(TypeError, match="sequence of globs"):
            get_weight_decay_exclusions(model, {Channels: lambda module: patterns})
    assert "unknown" in caplog.text
    optimizer = HybridMuonOptimizer(
        groups, lr=0.01, muon_mode="flat", flash_muon=False, magma_muon=False
    )
    for param in model.parameters():
        param.grad = torch.randn_like(param)
    optimizer.step()
    for name, param in model.named_parameters():
        assert ("momentum_buffer" in optimizer.state[param]) == (
            name in ("matrix", "independent")
        ), name


@pytest.mark.parametrize("excluded_first", [False, True])
@pytest.mark.parametrize("alias_kind", ["embedding", "readout", "unknown"])
def test_optimizer_shared_parameter_exclusions(excluded_first, alias_kind):
    from tace.lightning.optimizer import (
        get_optimizer_param_groups,
        get_weight_decay_exclusions,
    )

    model = torch.nn.Module()
    dense = torch.nn.Linear(4, 4, bias=False)
    if alias_kind == "embedding":
        excluded = torch.nn.Embedding(4, 4)
        excluded.weight = dense.weight
        excluded_name = "lookup"
    elif alias_kind == "readout":
        from tace.models.linear import switch_e3nn_weight_layout
        from tace.models.spherical.readout import ScalarReadOut

        excluded = ScalarReadOut(
            layer=0,
            num_layers=2,
            hidden_channel=[4],
            bias=False,
            num_elements=3,
            num_fidelities=1,
            parity=True,
            irreps_in="4x0e",
            irreps_out="4x0e",
        )
        switch_e3nn_weight_layout(excluded, "matrix")
        excluded.linear1[0].weight[0] = dense.weight
        excluded_name = "energy_readout"
    else:
        excluded = torch.nn.Module()
        excluded.coefficients = dense.weight
        excluded_name = "unknown"
    entries = [("hidden", dense), (excluded_name, excluded)]
    for name, module in reversed(entries) if excluded_first else entries:
        model.add_module(name, module)
    assert get_weight_decay_exclusions(model) == (
        set(dict(model.named_parameters(remove_duplicate=False)))
        if alias_kind != "unknown"
        else set()
    )
    groups = get_optimizer_param_groups(model, 0.1, use_muon=True)
    nonempty = [group for group in groups if group["params"]]
    assert len(nonempty) == 1
    assert len(nonempty[0]["params"]) == 1
    assert nonempty[0]["params"][0] is dense.weight
    assert nonempty[0]["use_muon"] is False
    assert nonempty[0]["weight_decay"] == (0.1 if alias_kind == "unknown" else 0.0)
    zero_decay = get_optimizer_param_groups(model, 0.0)
    assert [p for group in zero_decay for p in group["params"]] == [dense.weight]
    assert all(group["weight_decay"] == 0 for group in zero_decay)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_optimizer_without_triton(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    script = r"""
import importlib.abc
import importlib.util
import sys

class NoTriton(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "triton" or fullname.startswith("triton."):
            raise ModuleNotFoundError("Triton is not installed", name=fullname)

find_spec = importlib.util.find_spec
importlib.util.find_spec = lambda name, *args, **kwargs: (
    None if name == "triton" or name.startswith("triton.")
    else find_spec(name, *args, **kwargs)
)
sys.meta_path.insert(0, NoTriton())

import torch
import tace.utils.optimizer as optimizers
assert "tace.utils.optimizer.hybrid_muon" not in sys.modules
torch.set_num_threads(2)
device = sys.argv[1]
if device == "cpu":
    assert not torch.cuda.is_available()
for optimizer_type in (torch.optim.Adam, torch.optim.AdamW, optimizers.HybridMuonOptimizer):
    model = torch.nn.Linear(8, 4).to(device)
    optimizer = optimizer_type(model.parameters(), lr=1e-3)
    for step in range(4):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.randn(2, 8, device=device)).square().mean()
        loss.backward()
        optimizer.step()
        assert torch.isfinite(loss)
    if optimizer_type is optimizers.HybridMuonOptimizer:
        assert bool(optimizer._graphs) == (device == "cuda")
assert not any(name == "triton" or name.startswith("triton.") for name in sys.modules)
"""
    env = os.environ.copy()
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", script, device],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_hybrid_muon_constructor_defaults():
    from inspect import Parameter, signature

    expected = {
        "params": Parameter.empty,
        "lr": 5e-4,
        "momentum": 0.95,
        "weight_decay": 0.001,
        "adam_betas": (0.9, 0.95),
        "lr_adjust": 0.0,
        "lr_adjust_coeff": 0.18,
        "muon_mode": "slice",
        "named_parameters": None,
        "enable_gram": True,
        "flash_muon": True,
        "magma_muon": True,
        "use_foreach": None,
        "adam_patterns": (),
    }
    parameters = signature(HybridMuonOptimizer).parameters
    assert tuple(parameters) == tuple(expected)
    for name, default in expected.items():
        assert parameters[name].default == default
        assert parameters[name].kind is Parameter.POSITIONAL_OR_KEYWORD

    optimizer = HybridMuonOptimizer([torch.nn.Parameter(torch.ones(2, 3))])
    assert optimizer.defaults == {
        name: expected[name]
        for name in (
            "lr",
            "momentum",
            "weight_decay",
            "adam_betas",
            "lr_adjust",
            "lr_adjust_coeff",
            "muon_mode",
            "enable_gram",
            "magma_muon",
        )
    }
    assert "use_muon" not in optimizer.param_groups[0]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("use_foreach", [False, True])
def test_hybrid_muon_checkpoint(dtype, use_foreach):
    names = ("bias", "weight")
    params = [
        torch.nn.Parameter(torch.linspace(-0.3, 0.7, 3, dtype=dtype)),
        torch.nn.Parameter(torch.linspace(-0.8, 0.5, 6, dtype=dtype).reshape(2, 3)),
    ]
    optimizer = HybridMuonOptimizer(
        [{"params": params, "use_muon": False}],
        named_parameters=zip(names, params),
        use_foreach=use_foreach,
    )
    for step in range(4):
        for param in params:
            param.grad = torch.sin(param.detach() + step)
        optimizer.step()

    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = HybridMuonOptimizer(
        [{"params": restored_params, "use_muon": False}],
        named_parameters=zip(names, restored_params),
        use_foreach=use_foreach,
    )
    restored.load_state_dict(deepcopy(optimizer.state_dict()))
    for param, restored_param in zip(params, restored_params):
        for key in ("exp_avg", "exp_avg_sq"):
            assert restored.state[restored_param][key].dtype == torch.float32
            torch.testing.assert_close(
                restored.state[restored_param][key],
                optimizer.state[param][key],
                rtol=0,
                atol=0,
            )

    for step in range(4, 9):
        for index, (param, restored_param) in enumerate(zip(params, restored_params)):
            grad = None if index == 1 and step % 2 else torch.cos(param.detach() + step)
            param.grad = grad
            restored_param.grad = None if grad is None else grad.clone()
        optimizer.step()
        restored.step()
        for param, restored_param in zip(params, restored_params):
            torch.testing.assert_close(restored_param, param, rtol=0, atol=0)


def test_hybrid_muon_parameter_groups():
    optimizer = HybridMuonOptimizer([{"params": []}])
    optimizer.step()
    params = [torch.nn.Parameter(torch.ones(3)) for _ in range(2)]
    for param in params:
        optimizer.add_param_group({"params": [param]})
        param.grad = torch.ones_like(param)
        optimizer.step()
        assert torch.all(param < 1)
        assert "exp_avg" in optimizer.state[param]
    expected = [param.detach().clone() for param in params]
    optimizer.zero_grad(set_to_none=True)
    optimizer.step()
    for param, value in zip(params, expected):
        torch.testing.assert_close(param, value, rtol=0, atol=0)


@pytest.mark.parametrize("adam_patterns", [(), ("weight",)])
@pytest.mark.parametrize("weight_decay", [0.0, 0.2])
def test_hybrid_muon_zero_gradient_decay(adam_patterns, weight_decay):
    shapes = ((), (4,), (2, 3), (3, 1, 4), (3, 4, 1), (3, 2, 1, 4))
    params = [torch.nn.Parameter(torch.ones(shape)) for shape in shapes]
    names = [f"weight_{index}" for index in range(len(params))]
    optimizer = HybridMuonOptimizer(
        params,
        lr=0.1,
        weight_decay=weight_decay,
        named_parameters=zip(names, params),
        adam_patterns=adam_patterns,
        enable_gram=False,
        flash_muon=False,
        magma_muon=False,
    )
    for param in params:
        param.grad = torch.zeros_like(param)
    optimizer.step()
    for param in params:
        torch.testing.assert_close(
            param, torch.full_like(param, 1.0 - 0.1 * weight_decay)
        )
    for index in (0, 1, 3, 4, 5):
        assert "exp_avg" in optimizer.state[params[index]]
        assert "momentum_buffer" not in optimizer.state[params[index]]
    assert ("momentum_buffer" in optimizer.state[params[2]]) == (not adam_patterns)


def test_hybrid_muon_explicit_routes_and_patterns():
    names = ("bias", "adam_weight", "adamw_weight", "adapter.weight", "weight")
    params = [torch.nn.Parameter(torch.ones(2, 3)) for _ in names]
    optimizer = HybridMuonOptimizer(
        [{"params": params[:-1]}, {"params": params[-1:], "use_muon": False}],
        named_parameters=zip(names, params),
        adam_patterns=("adapter",),
        enable_gram=False,
        flash_muon=False,
        magma_muon=False,
    )
    for param in params:
        param.grad = torch.ones_like(param)
    optimizer.step()
    for index, param in enumerate(params):
        assert ("momentum_buffer" in optimizer.state[param]) == (index < 3)
        assert ("exp_avg" in optimizer.state[param]) == (index >= 3)


@pytest.mark.parametrize(
    "option,value",
    [
        ("lr", -0.1),
        ("lr", float("nan")),
        ("lr", float("inf")),
        ("weight_decay", -0.1),
        ("weight_decay", float("nan")),
        ("momentum", -0.1),
        ("momentum", 1.0),
        ("adam_betas", (0.9, 1.0)),
        ("adam_betas", (0.9,)),
        ("adam_betas", (float("nan"), 0.9)),
        ("lr_adjust", float("inf")),
        ("lr_adjust_coeff", -0.1),
        ("muon_mode", "unknown"),
    ],
)
def test_hybrid_muon_invalid_options_are_atomic(option, value):
    param = torch.nn.Parameter(torch.ones(2, 3))
    invalid = {option: value}
    with pytest.raises(ValueError):
        HybridMuonOptimizer([param], **invalid)
    optimizer = HybridMuonOptimizer(
        [param], named_parameters=[("weight", param)], adam_patterns=("weight",)
    )
    other = torch.nn.Parameter(torch.ones(2, 3))
    with pytest.raises(ValueError):
        optimizer.add_param_group({"params": [other], **invalid})
    assert len(optimizer.param_groups) == 1
    optimizer.add_param_group({"params": [other], "use_muon": False})
    optimizer.param_groups[-1].update(invalid)
    for current in (param, other):
        current.grad = torch.ones_like(current)
    with pytest.raises(ValueError):
        optimizer.step()
    for current in (param, other):
        torch.testing.assert_close(current, torch.ones_like(current), rtol=0, atol=0)
    assert not optimizer.state


def test_hybrid_muon_sparse_gradient_rejection_is_atomic():
    params = [torch.nn.Parameter(torch.ones(2, 3)) for _ in range(2)]
    optimizer = HybridMuonOptimizer(
        [{"params": params[:1], "use_muon": False}, {"params": params[1:]}]
    )
    params[0].grad = torch.ones_like(params[0])
    params[1].grad = torch.sparse_coo_tensor(
        [[0], [1]], [2.0], (2, 3), check_invariants=True
    )
    with pytest.raises(RuntimeError, match="[Ss]parse|strided"):
        optimizer.step()
    for param in params:
        torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_foreach", [False, True])
def test_hybrid_muon_cuda_graph(dtype, use_foreach):
    names = ("weight", "tensor_weight", "bias", "auxiliary_weight")
    params = [
        torch.nn.Parameter(torch.randn(shape, dtype=dtype, device="cuda"))
        for shape in ((4, 8), (2, 4, 8), (8,), (4, 8))
    ]
    eager_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    optimizer = HybridMuonOptimizer(
        [
            {"params": params[:2], "use_muon": True},
            {"params": params[2:], "use_muon": False},
        ],
        named_parameters=zip(names, params),
        use_foreach=use_foreach,
    )
    eager = HybridMuonOptimizer(
        [
            {"params": eager_params[:2], "use_muon": True},
            {"params": eager_params[2:], "use_muon": False},
        ],
        named_parameters=zip(names, eager_params),
        use_foreach=use_foreach,
    )
    eager._graph_enabled = False

    for step in range(26):
        for index, (param, eager_param) in enumerate(zip(params, eager_params)):
            grad = torch.sin(param.detach() + step)
            if step >= 5 and step % 2 and index in (1, 3):
                grad = None
            param.grad = grad
            eager_param.grad = None if grad is None else grad.clone()
        for current in (optimizer, eager):
            for group in current.param_groups:
                group["lr"] = 5e-4 / (1 + step // 4)
                if step == 20:
                    group.update(
                        momentum=0.9, weight_decay=0.005, adam_betas=(0.85, 0.97)
                    )
            current.step()
        for param, eager_param in zip(params, eager_params):
            torch.testing.assert_close(param, eager_param, rtol=2e-6, atol=2e-6)
        if step == 11:
            assert len(optimizer._graphs) == 2
            optimizer.load_state_dict(deepcopy(optimizer.state_dict()))
            assert not optimizer._graphs
        if step == 20:
            assert not optimizer._graphs
    assert len(optimizer._graphs) == 2

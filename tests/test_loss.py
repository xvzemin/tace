import os
import subprocess
import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from tace.dataset.statistics import balanced_element_weights
from tace.utils.loss.common import has_valid_labels, tensor_loss
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
from tace.utils.mask_metrics import MaskMAE, MaskRMSE
from tace.utils.optimizer.hybrid_muon import HybridMuonOptimizer


@pytest.mark.parametrize("loss", ["mse", "mae", "huber", "l2mae"])
@pytest.mark.parametrize("voigt", [False, True])
@pytest.mark.parametrize(
    "key", ["stress", "virials", "direct_stress", "direct_virials"]
)
def test_tensor_loss_missing_labels(loss, voigt, key):
    reference = torch.arange(1.0, 10.0).reshape(1, 3, 3)
    target = torch.zeros_like(reference)
    pred = torch.cat([reference, 100 * reference]).requires_grad_()
    labels = {
        key: torch.cat([target, target]),
        f"{key}_weight": torch.tensor([1.0, 0.0]),
        "entropy": torch.ones(2),
    }
    actual = tensor_loss({key: pred}, labels, key, loss, voigt=voigt)
    expected = tensor_loss(
        {key: reference}, {k: v[:1] for k, v in labels.items()}, key, loss, voigt=voigt
    )
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert torch.isfinite(pred.grad).all() and not pred.grad[1].any()
    labels[f"{key}_weight"].zero_()
    pred.grad = None
    empty = tensor_loss({key: pred}, labels, key, loss, voigt=voigt)
    empty.backward()
    assert empty == 0 and not pred.grad.any()


def test_tensor_loss_distributed_count(monkeypatch):
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
    monkeypatch.setattr(common.dist, "all_reduce", lambda count, op: count.add_(4))
    torch.testing.assert_close(
        tensor_loss(pred, label, "stress", "mse"), torch.tensor(36.0 / 13.0)
    )


@pytest.mark.parametrize("metric_type", [MaskMAE, MaskRMSE])
def test_stress_metric_missing_labels(metric_type):
    metric = metric_type("stress", 1.0)
    target = torch.zeros(2, 3, 3)
    label = {"stress": target, "stress_weight": torch.zeros(2)}
    metric.update(torch.ones_like(target), target, label)
    assert torch.isnan(metric.compute())
    target[0] = 0
    label["stress_weight"][0] = 1
    metric.update(torch.ones_like(target), target, label)
    torch.testing.assert_close(metric.compute(), torch.tensor(1.0))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "key, shape",
    [
        ("energy", (2,)),
        ("forces", (5, 3)),
        ("stress", (2, 3, 3)),
        ("direct_hessian", (13, 3, 3)),
    ],
)
def test_uncertainty_labels(key, shape, dtype):
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
        value = loss(pred, label)
        log_sigma = loss.log_sigmas[key]
        torch.testing.assert_close(value, reference + log_sigma * present)
        value.backward()
        torch.testing.assert_close(log_sigma.grad, float(present) - reference)
        assert torch.isfinite(pred[key].grad).all()
        if not present:
            assert not pred[key].grad.any()

    # A zero residual with valid labels must retain the uncertainty regularizer.
    label[key] = pred[key].detach().clone()
    loss.zero_grad()
    value = loss(pred, label)
    torch.testing.assert_close(value, loss.log_sigmas[key])
    value.backward()
    torch.testing.assert_close(loss.log_sigmas[key].grad, torch.ones_like(value))


def test_valid_labels_weights_and_distributed_presence(monkeypatch):
    from tace.utils.loss import common

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        "node_attrs": torch.tensor([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]),
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


@pytest.mark.parametrize("optimizer_name", ["Adam", "AdamW", "SGD"])
@pytest.mark.parametrize("matrix_weight", ["0", "1"])
def test_optimizer_weight_decay(optimizer_name, matrix_weight):
    from tace.lightning.lit_model import LightningWrapperModel
    from tace.models.blocks import ScaleShift
    from tace.models.linear import e3nnLinear
    from tace.models.radial import RadialBasis, ZBLBasis
    from tace.models.spherical.layer_norm import EquivariantMergeLayerNorm
    from tace.models.spherical.scatter_norm import DensityScatterNorm

    class TrainingModel(torch.nn.Module):
        configure_optimizers = LightningWrapperModel.configure_optimizers

        def __init__(self):
            super().__init__()
            self.cfg = {
                "optimizer": {
                    "_target_": f"torch.optim.{optimizer_name}",
                    "lr": 0.01,
                    "weight_decay": 0.1,
                }
            }
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
            self.density = DensityScatterNorm(2.0, 4, radial_bias=True)
            self.radial = RadialBasis(trainable=True)
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
        assert decay[id(param)] == (0.1 if expected else 0.0), name
        with torch.no_grad():
            param.fill_(1.0)
        param.grad = torch.zeros_like(param)
    optimizer.step()
    for param in params:
        if decay[id(param)]:
            assert torch.all(param < 1)
        else:
            torch.testing.assert_close(param, torch.ones_like(param), rtol=0, atol=0)


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
        assert not optimizer._use_flash
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("use_foreach", [False, True])
def test_hybrid_muon_checkpoint(dtype, use_foreach):
    names = ("bias", "adamw_weight")
    params = [
        torch.nn.Parameter(torch.linspace(-0.3, 0.7, 3, dtype=dtype)),
        torch.nn.Parameter(torch.linspace(-0.8, 0.5, 6, dtype=dtype).reshape(2, 3)),
    ]
    optimizer = HybridMuonOptimizer(
        params, named_parameters=zip(names, params), use_foreach=use_foreach
    )
    for step in range(4):
        for param in params:
            param.grad = torch.sin(param.detach() + step)
        optimizer.step()

    restored_params = [torch.nn.Parameter(p.detach().clone()) for p in params]
    restored = HybridMuonOptimizer(
        restored_params,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("use_foreach", [False, True])
def test_hybrid_muon_cuda_graph(dtype, use_foreach):
    names = ("weight", "tensor_weight", "bias", "adamw_weight")
    params = [
        torch.nn.Parameter(torch.randn(shape, dtype=dtype, device="cuda"))
        for shape in ((4, 8), (2, 4, 8), (8,), (4, 8))
    ]
    eager_params = [torch.nn.Parameter(param.detach().clone()) for param in params]
    optimizer = HybridMuonOptimizer(
        params, named_parameters=zip(names, params), use_foreach=use_foreach
    )
    eager = HybridMuonOptimizer(
        eager_params, named_parameters=zip(names, eager_params), use_foreach=use_foreach
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
            current.param_groups[0]["lr"] = 5e-4 / (1 + step // 4)
            if step == 20:
                current.param_groups[0].update(
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

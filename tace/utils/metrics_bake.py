################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# TODO, refacotor all metric to allow missing property

from typing import Dict, List

import torch
from torch import Tensor
from torchmetrics import Metric

from ..dataset.quantity import (
    MAE_PER_ATOM_PROPERTY,
    MAE_PROPERTY,
    RMSE_PER_ATOM_PROPERTY,
    RMSE_PROPERTY,
)

SCALE = 1000.0  # for example, metric units from eV to meV


def expand_dims_to(T: torch.Tensor, n_dim: int, dim: int = -1) -> torch.Tensor:
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


class MAE(Metric):
    def __init__(self, scale: float = SCALE):
        super().__init__()
        self.scale = scale
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        abs_error = torch.abs(preds - targets)
        self.sum_abs_error += abs_error.sum()
        self.count += targets.numel()

    def compute(self):
        return (self.sum_abs_error / self.count) * self.scale


class RMSE(Metric):
    def __init__(self, scale: float = SCALE):
        super().__init__()
        self.scale = scale
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor):
        squared_error = (preds - targets) ** 2
        self.sum_squared_error += squared_error.sum()
        self.count += targets.numel()

    def compute(self):
        return torch.sqrt(self.sum_squared_error / self.count) * self.scale


class PerAtomMAE(MAE):
    def update(self, preds: Tensor, targets: Tensor, ptr: Tensor):
        num_nodes = ptr[1:] - ptr[:-1]  # [B]
        num_nodes = expand_dims_to(num_nodes, preds.ndim, dim=-1)
        preds_per_atom = preds / num_nodes
        targets_per_atom = targets / num_nodes
        super().update(preds_per_atom, targets_per_atom)


class PerAtomRMSE(RMSE):
    def update(self, preds: Tensor, targets: Tensor, ptr: Tensor):
        num_nodes = ptr[1:] - ptr[:-1]  # [B]
        num_nodes = expand_dims_to(num_nodes, preds.ndim, dim=-1)
        preds_per_atom = preds / num_nodes
        targets_per_atom = targets / num_nodes
        super().update(preds_per_atom, targets_per_atom)


class PolarizationMetric(Metric):
    def __init__(
        self, metric_type: str = "mae", per_atom: bool = False, scale: float = SCALE
    ):
        """
        Args:
            metric_type: "mae" or "rmse"
            per_atom: whether to normalize by number of atoms
            scale: scaling factor
        """
        super().__init__()
        assert metric_type in ["mae", "rmse"], "metric_type must be 'mae' or 'rmse'"
        self.metric_type = metric_type
        self.per_atom = per_atom
        self.scale = scale

        if self.metric_type == "mae":
            self.add_state(
                "sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:  # rmse
            self.add_state(
                "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor, label: Dict[str, Tensor]):
        lattice = label["lattice"]
        error = preds - targets

        # Project to lattice basis
        error = torch.einsum("bi, bij -> bj", error, torch.linalg.inv(lattice))
        error = torch.remainder(error, 1.0)
        error = torch.where(error > 0.5, error - 1.0, error)
        error = torch.where(error < -0.5, error + 1.0, error)
        error = torch.einsum("bi, bij -> bj", error, lattice)

        # Normalize per atom if needed
        if self.per_atom:
            ptr = label["ptr"]
            num_atoms = (ptr[1:] - ptr[:-1]).reshape(-1, 1)
            error = error / num_atoms

        if self.metric_type == "mae":
            abs_error = torch.abs(error)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = error**2
            self.sum_squared_error += squared_error.sum()

        self.count += targets.numel()

    def compute(self):
        if self.metric_type == "mae":
            return self.sum_abs_error / self.count * self.scale
        else:  # rmse
            return torch.sqrt(self.sum_squared_error / self.count) * self.scale


class AbsFinalCollinearMagmomsMetric(Metric):
    def __init__(self, metric_type: str = "mae", scale: float = SCALE):
        """
        Args:
            metric_type: "mae" or "rmse"
            scale: scaling factor
        """
        super().__init__()
        assert metric_type in ["mae", "rmse"], "metric_type must be 'mae' or 'rmse'"
        self.metric_type = metric_type
        self.scale = scale

        if self.metric_type == "mae":
            self.add_state(
                "sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:  # rmse
            self.add_state(
                "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, label: Tensor):
        batch = label.batch
        total_weight = (label.entropy * label.abs_final_collinear_magmoms_weight)[batch]
        mask = total_weight != 0
        error = pred["abs_final_collinear_magmoms"] - label.abs_final_collinear_magmoms
        error = error * total_weight
        error = error[mask]

        if self.metric_type == "mae":
            abs_error = torch.abs(error)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = error**2
            self.sum_squared_error += squared_error.sum()
        self.count += error.numel()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.count.device)

        if self.metric_type == "mae":
            return self.sum_abs_error / self.count * self.scale
        else:  # rmse
            return torch.sqrt(self.sum_squared_error / self.count) * self.scale


class DirectForcesMetric(Metric):
    def __init__(self, metric_type: str = "mae", scale: float = SCALE):
        """
        Args:
            metric_type: "mae" or "rmse"
            scale: scaling factor
        """
        super().__init__()
        assert metric_type in ["mae", "rmse"], "metric_type must be 'mae' or 'rmse'"
        self.metric_type = metric_type
        self.scale = scale

        if self.metric_type == "mae":
            self.add_state(
                "sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:  # rmse
            self.add_state(
                "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, label: Tensor):
        batch = label["batch"]
        key = "direct_forces"
        total_weight = (label["entropy"] * label[key + "_weight"])[batch]
        mask = total_weight != 0
        error = pred[key] - label[key]
        error = error * total_weight.unsqueeze(-1)
        error = error[mask]

        if self.metric_type == "mae":
            abs_error = torch.abs(error)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = error**2
            self.sum_squared_error += squared_error.sum()
        self.count += error.numel()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.count.device)

        if self.metric_type == "mae":
            return self.sum_abs_error / self.count * self.scale
        else:  # rmse
            return torch.sqrt(self.sum_squared_error / self.count) * self.scale


class ForcesMetric(Metric):
    def __init__(self, metric_type: str = "mae", scale: float = SCALE):
        """
        Args:
            metric_type: "mae" or "rmse"
            scale: scaling factor
        """
        super().__init__()
        assert metric_type in ["mae", "rmse"], "metric_type must be 'mae' or 'rmse'"
        self.metric_type = metric_type
        self.scale = scale

        if self.metric_type == "mae":
            self.add_state(
                "sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )
        else:  # rmse
            self.add_state(
                "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
            )

        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, pred: Tensor, label: Tensor):
        batch = label["batch"]
        key = "forces"
        total_weight = (label["entropy"] * label[key + "_weight"])[batch]
        mask = total_weight != 0
        error = pred[key] - label[key]

        if "noise_mask" in label:
            noise_mask = label["noise_mask"].bool()
            mask = mask & (~noise_mask)

        error = error * total_weight.unsqueeze(-1)
        error = error[mask]

        if self.metric_type == "mae":
            abs_error = torch.abs(error)
            self.sum_abs_error += abs_error.sum()
        else:  # rmse
            squared_error = error**2
            self.sum_squared_error += squared_error.sum()
        self.count += error.numel()

    def compute(self):
        if self.count == 0:
            return torch.tensor(0.0, device=self.count.device)

        if self.metric_type == "mae":
            return self.sum_abs_error / self.count * self.scale
        else:  # rmse
            return torch.sqrt(self.sum_squared_error / self.count) * self.scale


def build_metrics(prefix: str, loss_property: List[str]) -> Dict[str, Metric]:
    metrics = {}

    def add_metrics(property_name):
        if property_name in MAE_PROPERTY:
            metrics[f"{prefix}/{property_name}_mae"] = MAE()
        if property_name in RMSE_PROPERTY:
            metrics[f"{prefix}/{property_name}_rmse"] = RMSE()
        if property_name in MAE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{property_name}_per_atom_mae"] = PerAtomMAE()
        if property_name in RMSE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{property_name}_per_atom_rmse"] = PerAtomRMSE()
        if property_name == "polarization":
            metrics[f"{prefix}/{property_name}_mae"] = PolarizationMetric("mae", False)
            metrics[f"{prefix}/{property_name}_rmse"] = PolarizationMetric(
                "rmse", False
            )
            metrics[f"{prefix}/{property_name}_per_atom_mae"] = PolarizationMetric(
                "mae", True
            )
            metrics[f"{prefix}/{property_name}_per_atom_rmse"] = PolarizationMetric(
                "rmse", True
            )
        if property_name == "abs_final_collinear_magmoms":
            metrics[f"{prefix}/{property_name}_mae"] = AbsFinalCollinearMagmomsMetric(
                "mae"
            )
            metrics[f"{prefix}/{property_name}_rmse"] = AbsFinalCollinearMagmomsMetric(
                "rmse"
            )
        if property_name == "direct_forces":
            metrics[f"{prefix}/{property_name}_mae"] = DirectForcesMetric("mae")
            metrics[f"{prefix}/{property_name}_rmse"] = DirectForcesMetric("rmse")
        if property_name == "forces":
            metrics[f"{prefix}/{property_name}_mae"] = ForcesMetric("mae")
            metrics[f"{prefix}/{property_name}_rmse"] = ForcesMetric("rmse")

    for p in loss_property:
        add_metrics(p)

    return metrics


def update_metrics(metrics, prefix, pred, label, loss_property):
    for p in loss_property:
        if p not in pred:
            continue

        ptr = label["ptr"]
        output_value = pred[p]
        batch_value = label[p]
        if p in MAE_PROPERTY:
            metrics[f"{prefix}/{p}_mae"](output_value, batch_value)
        if p in RMSE_PROPERTY:
            metrics[f"{prefix}/{p}_rmse"](output_value, batch_value)
        if p in MAE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{p}_per_atom_mae"](output_value, batch_value, ptr)
        if p in RMSE_PER_ATOM_PROPERTY:
            metrics[f"{prefix}/{p}_per_atom_rmse"](output_value, batch_value, ptr)
        if p == "polarization":
            metrics[f"{prefix}/{p}_mae"](output_value, batch_value, label)
            metrics[f"{prefix}/{p}_rmse"](output_value, batch_value, label)
            metrics[f"{prefix}/{p}_per_atom_mae"](output_value, batch_value, label)
            metrics[f"{prefix}/{p}_per_atom_rmse"](output_value, batch_value, label)
        if p == "abs_final_collinear_magmoms":
            metrics[f"{prefix}/{p}_mae"](pred, label)
            metrics[f"{prefix}/{p}_rmse"](pred, label)
        if p == "direct_forces":
            metrics[f"{prefix}/{p}_mae"](pred, label)
            metrics[f"{prefix}/{p}_rmse"](pred, label)
        if p == "forces":
            metrics[f"{prefix}/{p}_mae"](pred, label)
            metrics[f"{prefix}/{p}_rmse"](pred, label)

################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

# TODO, refacotor all metric to allow missing property

import logging
from typing import Dict, List

import torch
from torch import Tensor
from torchmetrics import Metric

from ..dataset.quantity import (
    MAE_PER_ATOM_PROPERTIES,
    MAE_PROPERTIES,
    PROPERTIES,
    RMSE_PER_ATOM_PROPERTIES,
    RMSE_PROPERTIES,
    SPECIAL_METRIC_PROPERTIES,
)
from .mask_metrics import (
    MaskMAE,
    MaskPerAtomMAE,
    MaskPerAtomRMSE,
    MaskRMSE,
    supports_weight_filter,
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


def use_weight_filter(property_name: str) -> bool:
    return (
        property_name in PROPERTIES
        and supports_weight_filter(property_name)
        and property_name not in SPECIAL_METRIC_PROPERTIES  # TODO
    )


def log_metric_choice(metric_name: str, property_name: str, use_mask: bool):
    if property_name in SPECIAL_METRIC_PROPERTIES:
        logging.debug(
            "Build metric %s: special metric, see code.",
            metric_name,
        )
        return
    if use_mask:
        logging.debug(
            "Build metric %s: entries with %s_weight == 0 are ignored.",
            metric_name,
            property_name,
        )
    else:
        logging.debug(
            "Build metric %s: entries with %s_weight == 0 are keep.",
            metric_name,
            property_name,
        )


def build_metrics(prefix: str, loss_property: List[str]) -> Dict[str, Metric]:
    metrics = {}

    def add_metrics(property_name):
        use_mask = use_weight_filter(property_name)

        # === General ===
        if property_name in MAE_PROPERTIES:
            metric_name = f"{prefix}/{property_name}_mae"
            metrics[metric_name] = MaskMAE(property_name, SCALE) if use_mask else MAE()
            log_metric_choice(metric_name, property_name, use_mask)
        if property_name in RMSE_PROPERTIES:
            metric_name = f"{prefix}/{property_name}_rmse"
            metrics[metric_name] = (
                MaskRMSE(property_name, SCALE) if use_mask else RMSE()
            )
            log_metric_choice(metric_name, property_name, use_mask)
        if property_name in MAE_PER_ATOM_PROPERTIES:
            metric_name = f"{prefix}/{property_name}_per_atom_mae"
            metrics[metric_name] = (
                MaskPerAtomMAE(property_name, SCALE) if use_mask else PerAtomMAE()
            )
            log_metric_choice(metric_name, property_name, use_mask)
        if property_name in RMSE_PER_ATOM_PROPERTIES:
            metric_name = f"{prefix}/{property_name}_per_atom_rmse"
            metrics[metric_name] = (
                MaskPerAtomRMSE(property_name, SCALE) if use_mask else PerAtomRMSE()
            )
            log_metric_choice(metric_name, property_name, use_mask)

        # === Special ===
        if property_name == "polarization":
            metric_name = f"{prefix}/{property_name}_mae"
            metrics[metric_name] = PolarizationMetric("mae", False)
            log_metric_choice(metric_name, property_name, False)
            metric_name = f"{prefix}/{property_name}_rmse"
            metrics[metric_name] = PolarizationMetric("rmse", False)
            log_metric_choice(metric_name, property_name, False)
            metric_name = f"{prefix}/{property_name}_per_atom_mae"
            metrics[metric_name] = PolarizationMetric("mae", True)
            log_metric_choice(metric_name, property_name, False)
            metric_name = f"{prefix}/{property_name}_per_atom_rmse"
            metrics[metric_name] = PolarizationMetric("rmse", True)
            log_metric_choice(metric_name, property_name, False)
        if property_name == "abs_final_collinear_magmoms":
            metric_name = f"{prefix}/{property_name}_mae"
            metrics[metric_name] = AbsFinalCollinearMagmomsMetric("mae")
            log_metric_choice(metric_name, property_name, use_mask)
            metric_name = f"{prefix}/{property_name}_rmse"
            metrics[metric_name] = AbsFinalCollinearMagmomsMetric("rmse")
            log_metric_choice(metric_name, property_name, use_mask)
        if property_name == "direct_forces":
            metric_name = f"{prefix}/{property_name}_mae"
            metrics[metric_name] = DirectForcesMetric("mae")
            log_metric_choice(metric_name, property_name, use_mask)
            metric_name = f"{prefix}/{property_name}_rmse"
            metrics[metric_name] = DirectForcesMetric("rmse")
            log_metric_choice(metric_name, property_name, use_mask)
        if property_name == "forces":
            metric_name = f"{prefix}/{property_name}_mae"
            metrics[metric_name] = ForcesMetric("mae")
            log_metric_choice(metric_name, property_name, use_mask)
            metric_name = f"{prefix}/{property_name}_rmse"
            metrics[metric_name] = ForcesMetric("rmse")
            log_metric_choice(metric_name, property_name, use_mask)

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
        weighted = use_weight_filter(p)
        if p in MAE_PROPERTIES:
            if weighted:
                metrics[f"{prefix}/{p}_mae"](output_value, batch_value, label)
            else:
                metrics[f"{prefix}/{p}_mae"](output_value, batch_value)
        if p in RMSE_PROPERTIES:
            if weighted:
                metrics[f"{prefix}/{p}_rmse"](output_value, batch_value, label)
            else:
                metrics[f"{prefix}/{p}_rmse"](output_value, batch_value)
        if p in MAE_PER_ATOM_PROPERTIES:
            if weighted:
                metrics[f"{prefix}/{p}_per_atom_mae"](
                    output_value, batch_value, ptr, label
                )
            else:
                metrics[f"{prefix}/{p}_per_atom_mae"](output_value, batch_value, ptr)
        if p in RMSE_PER_ATOM_PROPERTIES:
            if weighted:
                metrics[f"{prefix}/{p}_per_atom_rmse"](
                    output_value, batch_value, ptr, label
                )
            else:
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

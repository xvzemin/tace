################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict

import torch
from torch import Tensor
from torchmetrics import Metric

from ..dataset.quantity import PROPERTY


def expand_dims_to(T: Tensor, n_dim: int, dim: int = -1) -> Tensor:
    while T.ndim < n_dim:
        T = T.unsqueeze(dim)
    return T


def supports_weight_filter(property_name: str) -> bool:
    return PROPERTY[property_name]["scope"] in {
        "per-system",
        "per-atom",
        "per-edge",
        "per-pair",
    }


def _property_weight_mask(
    label: Dict[str, Tensor],
    property_name: str,
    value: Tensor,
) -> Tensor:
    scope = PROPERTY[property_name]["scope"]
    graph_mask = label[f"{property_name}_weight"].reshape(-1) != 0

    if scope == "per-system":
        mask = graph_mask
    elif scope == "per-atom":
        mask = graph_mask[label["batch"]]
    elif scope == "per-edge":
        mask = graph_mask[label["batch"][label["edge_index"][0]]]
    elif scope == "per-pair":
        counts = label["ptr"][1:] - label["ptr"][:-1]
        mask = torch.repeat_interleave(graph_mask, counts.square())
    else:
        raise ValueError(
            f"{property_name} has scope {scope}; only per-system, per-atom, "
            "per-edge and per-pair metrics support weight filtering"
        )

    if property_name == "forces" and "noise_mask" in label:
        mask = mask & (~label["noise_mask"].bool())

    if mask.shape[0] != value.shape[0]:
        raise ValueError(
            f"{property_name} weight mask length {mask.shape[0]} does not match "
            f"value length {value.shape[0]}"
        )
    return mask


def filter_error_by_property_weight(
    error: Tensor,
    label: Dict[str, Tensor],
    property_name: str,
) -> Tensor:
    mask = _property_weight_mask(label, property_name, error)
    return error[mask]


class MaskMAE(Metric):
    def __init__(self, property_name: str, scale: float):
        super().__init__()
        self.property_name = property_name
        self.scale = scale
        self.add_state("sum_abs_error", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor, label: Dict[str, Tensor]):
        error = filter_error_by_property_weight(
            preds - targets,
            label,
            self.property_name,
        )
        self.sum_abs_error += torch.abs(error).sum()
        self.count += error.numel()

    def compute(self):
        value = self.sum_abs_error / self.count.clamp_min(1) * self.scale
        return torch.where(self.count > 0, value, torch.full_like(value, float("nan")))


class MaskRMSE(Metric):
    def __init__(self, property_name: str, scale: float):
        super().__init__()
        self.property_name = property_name
        self.scale = scale
        self.add_state(
            "sum_squared_error", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("count", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: Tensor, targets: Tensor, label: Dict[str, Tensor]):
        error = filter_error_by_property_weight(
            preds - targets,
            label,
            self.property_name,
        )
        self.sum_squared_error += torch.square(error).sum()
        self.count += error.numel()

    def compute(self):
        value = (
            torch.sqrt(self.sum_squared_error / self.count.clamp_min(1)) * self.scale
        )
        return torch.where(self.count > 0, value, torch.full_like(value, float("nan")))


class MaskPerAtomMAE(MaskMAE):
    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        ptr: Tensor,
        label: Dict[str, Tensor],
    ):
        num_nodes = ptr[1:] - ptr[:-1]
        num_nodes = expand_dims_to(num_nodes, preds.ndim, dim=-1)
        super().update(preds / num_nodes, targets / num_nodes, label)


class MaskPerAtomRMSE(MaskRMSE):
    def update(
        self,
        preds: Tensor,
        targets: Tensor,
        ptr: Tensor,
        label: Dict[str, Tensor],
    ):
        num_nodes = ptr[1:] - ptr[:-1]
        num_nodes = expand_dims_to(num_nodes, preds.ndim, dim=-1)
        super().update(preds / num_nodes, targets / num_nodes, label)

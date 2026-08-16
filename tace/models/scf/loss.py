################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict

import torch

from tace.utils.torch_scatter import scatter_sum


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.sum(value * weight) / torch.sum(weight).clamp_min(1.0e-12)


def density_mse(
    reference: Dict[str, torch.Tensor], prediction: Dict[str, torch.Tensor]
) -> torch.Tensor:
    target = reference.get("density_coefficients", reference.get("atomic_multipoles"))
    if target is None:
        raise KeyError(
            "density loss requires density_coefficients or atomic_multipoles"
        )
    weight = reference.get("density_coefficients_weight")
    if weight is None:
        weight = torch.ones_like(target[..., :1])
    error = (prediction["density_coefficients"] - target).square().mean(dim=-1)
    return _weighted_mean(error, weight.reshape(-1))


def total_charge_mse(reference, prediction) -> torch.Tensor:
    error = (prediction["total_charge"] - reference["total_charge"]).square()
    weight = reference.get("total_charge_weight", torch.ones_like(error))
    return _weighted_mean(error, weight)


def fermi_level_mse(reference, prediction) -> torch.Tensor:
    error = (prediction["fermi_level"] - reference["fermi_level"]).square()
    weight = reference.get("fermi_level_weight", torch.ones_like(error))
    return _weighted_mean(error, weight)


def dipole_mse(reference, prediction) -> torch.Tensor:
    error = (prediction["dipole"] - reference["dipole"]).square().mean(dim=-1)
    weight = reference.get("dipole_weight", torch.ones_like(error))
    return _weighted_mean(error, weight)


def fixed_point_stability_loss(reference, prediction) -> torch.Tensor:
    history = prediction["density_history"]
    if history.shape[-1] < 2:
        return history.new_zeros(())
    delta = (history[..., -1] - history[..., -2]).square().mean(dim=-1)
    batch = reference["batch"]
    n_graphs = prediction["scf_residual"].numel()
    per_graph = scatter_sum(delta, batch, dim=0, dim_size=n_graphs)
    counts = scatter_sum(torch.ones_like(delta), batch, dim=0, dim_size=n_graphs)
    return torch.mean(per_graph / counts.clamp_min(1.0))


class SCFLoss(torch.nn.Module):
    """Composable loss for SCF-specific labels."""

    FUNCTIONS = {
        "density": density_mse,
        "total_charge": total_charge_mse,
        "fermi_level": fermi_level_mse,
        "dipole": dipole_mse,
        "stability": fixed_point_stability_loss,
    }

    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        unknown = sorted(set(weights).difference(self.FUNCTIONS))
        if unknown:
            raise ValueError(f"Unknown SCF loss terms: {unknown}")
        self.weights = {key: float(value) for key, value in weights.items()}

    def forward(self, prediction, reference) -> torch.Tensor:
        total = prediction["energy"].new_zeros(())
        for name, weight in self.weights.items():
            total = total + weight * self.FUNCTIONS[name](reference, prediction)
        return total

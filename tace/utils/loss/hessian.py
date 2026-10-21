"""Losses for full, directly predicted Hessians."""

import torch
import torch.distributed as dist

from tace.utils.torch_scatter import scatter_sum

from .mse_fn import register_loss


def direct_hessian_loss(pred, label, loss, reduction="structure", huber_delta=0.01):
    """Reduce Cartesian-entry errors or atom-pair Frobenius norms.

    Parameters
    ----------
    pred, label : dict
        Predictions and labels in packed ``(sum_g N_g**2, 3, 3)`` layout.
    loss : {'mse', 'mae', 'l2mae', 'huber'}
        Elementwise loss, or a Frobenius norm per atom pair for ``l2mae``.
    reduction : {'structure', 'component'}
        Average within each structure, then over structures, or directly over
        valid Cartesian entries. For ``l2mae``, count atom pairs, not entries.
    huber_delta : float
        Huber threshold in the units of the Hessian labels.
    """
    key = "direct_hessian"
    target = label[key]
    if pred[key].shape != target.shape:
        raise ValueError(
            "Direct Hessian predictions and labels must have the same shape."
        )
    counts = label["ptr"][1:] - label["ptr"][:-1]
    graph = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts.square()
    )
    weight = (label["entropy"] * label[f"{key}_weight"]).reshape(-1)
    mask = torch.isfinite(target) & (weight[graph, None, None] > 0)
    if f"{key}_mask" in label:
        mask = mask & label[f"{key}_mask"]
    error = torch.where(mask, pred[key], 0) - torch.where(mask, target, 0)
    if loss == "mse":
        values = error.square()
    elif loss == "mae":
        values = error.abs()
    elif loss == "l2mae":
        values = torch.linalg.vector_norm(error.flatten(1), dim=-1, keepdim=True)
        mask = mask.flatten(1).any(dim=-1, keepdim=True)
    elif loss == "huber":
        values = torch.nn.functional.huber_loss(
            error, torch.zeros_like(error), reduction="none", delta=huber_delta
        )
    else:
        raise ValueError(f"Unknown Hessian loss: {loss}")
    totals = scatter_sum(
        values.flatten(1).sum(-1), graph, dim=0, dim_size=counts.numel()
    )
    valid = scatter_sum(mask.flatten(1).sum(-1), graph, dim=0, dim_size=counts.numel())
    if reduction == "structure":
        numerator = (totals / valid.clamp_min(1) * weight).sum()
        denominator = (valid > 0).sum()
    elif reduction == "component":
        numerator = (totals * weight).sum()
        denominator = valid.sum()
    else:
        raise ValueError("reduction must be 'structure' or 'component'.")
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    return numerator * world_size / denominator.clamp_min(1)


@register_loss
def mse_direct_hessian(pred, label, reduction="structure"):
    return direct_hessian_loss(pred, label, "mse", reduction)


@register_loss
def mae_direct_hessian(pred, label, reduction="structure"):
    return direct_hessian_loss(pred, label, "mae", reduction)


@register_loss
def l2mae_direct_hessian(pred, label, reduction="structure"):
    return direct_hessian_loss(pred, label, "l2mae", reduction)


@register_loss
def huber_direct_hessian(pred, label, huber_delta=0.01, reduction="structure"):
    return direct_hessian_loss(
        pred, label, "huber", reduction=reduction, huber_delta=huber_delta
    )

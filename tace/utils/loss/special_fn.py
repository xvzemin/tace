################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from typing import Dict

import torch

from .mse_fn import register_loss


@register_loss
def conditional_huber_forces(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
    huber_delta: float = 0.01,
) -> torch.Tensor:
    pred_forces = pred["forces"]
    label_forces = label["forces"]
    factors = huber_delta * torch.tensor(
        [1.0, 0.7, 0.4, 0.1], device=label_forces.device, dtype=label_forces.dtype
    )
    norm_forces = torch.norm(label_forces, dim=-1)
    c1 = norm_forces < 100
    c2 = (norm_forces >= 100) & (norm_forces < 200)
    c3 = (norm_forces >= 200) & (norm_forces < 300)
    c4 = ~(c1 | c2 | c3)
    se = torch.zeros_like(pred_forces)
    se[c1] = torch.nn.functional.huber_loss(
        label_forces[c1], pred_forces[c1], reduction="none", delta=factors[0]
    )
    se[c2] = torch.nn.functional.huber_loss(
        label_forces[c2], pred_forces[c2], reduction="none", delta=factors[1]
    )
    se[c3] = torch.nn.functional.huber_loss(
        label_forces[c3], pred_forces[c3], reduction="none", delta=factors[2]
    )
    se[c4] = torch.nn.functional.huber_loss(
        label_forces[c4], pred_forces[c4], reduction="none", delta=factors[3]
    )
    return torch.mean(se)


# TODO
@register_loss
def mse_direct_forces_curl(
    pred: Dict[str, torch.Tensor],
    label: Dict[str, torch.Tensor],
) -> torch.Tensor:

    num_pairs = 1
    forces = pred["direct_forces"]
    positions = label["positions"]

    if not positions.requires_grad:
        raise RuntimeError(
            "positions must have requires_grad=True before model forward."
        )

    F_flat = forces.reshape(-1)
    R_dim = positions.numel()

    losses = []

    for _ in range(num_pairs):
        a = torch.randint(0, F_flat.numel(), (1,), device=F_flat.device).item()
        b = torch.randint(0, R_dim, (1,), device=F_flat.device).item()

        grad_a = torch.autograd.grad(
            outputs=F_flat[a],
            inputs=positions,
            retain_graph=True,
            create_graph=True,
        )[0].reshape(-1)

        grad_b = torch.autograd.grad(
            outputs=F_flat[b],
            inputs=positions,
            retain_graph=True,
            create_graph=True,
        )[0].reshape(-1)

        curl_ab = grad_a[b] - grad_b[a]

        losses.append(curl_ab**2)

    return torch.mean(torch.stack(losses))

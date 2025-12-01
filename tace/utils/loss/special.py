################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

import torch


from .fn import *


# huber loss | energy + conditional_huber_forces + stress
class UniversalLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        huber_delta=0.01,
        **kwargs,
    ) -> None:
        "From MACE https://github.com/ACEsuit/mace, used for UMLIPs training"
        super().__init__()
        self.huber_delta = huber_delta
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label) -> Tensor:
        num_atoms = (label.ptr[1:] - label.ptr[:-1])
        batch = label["batch"]
        energy_weight = label.energy_weight
        forces_weight = label.forces_weight[batch].unsqueeze(-1)
        stress_weight = label.stress_weight.unsqueeze(-1).unsqueeze(-1)

        loss_energy = torch.nn.functional.huber_loss(
            energy_weight * label["energy"] / num_atoms,
            energy_weight * pred["energy"] / num_atoms,
            reduction="mean",
            delta=self.huber_delta,
        )
        loss_forces = conditional_huber_forces(
            forces_weight * pred["forces"],
            forces_weight * label["forces"],
            huber_delta=self.huber_delta,
        )
        loss_stress = torch.nn.functional.huber_loss(
            stress_weight * label["stress"],
            stress_weight * pred["stress"],
            reduction="mean",
            delta=self.huber_delta,
        )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f})"
        )

    
@register_loss
def omat24_salex_mptrj_conditional_huber_forces(
    pred_forces: Tensor, label_forces: Tensor, huber_delta: float
) -> Tensor:
    "Code From MACE, see same logic as NequIP"
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


class OMat24sAlexMPtrjLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=5.0,
        stress_weight=0.01,
        energy_huber_delta=0.01,
        forces_huber_delta=0.01,
        stress_huber_delta=0.1,
        normalize: bool = True,
        **kwargs,
    ) -> None:
        "Modify code From MACE, see same logic as NequIP"
        super().__init__()
        if normalize:
            normalizer = energy_weight + forces_weight + stress_weight
            self.register_buffer(
                "energy_weight",
                torch.tensor(energy_weight / normalizer , dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "forces_weight",
                torch.tensor(forces_weight / normalizer, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "stress_weight",
                torch.tensor(stress_weight / normalizer, dtype=torch.get_default_dtype()),
            )
        else:
            self.register_buffer(
                "energy_weight",
                torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "forces_weight",
                torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
            )
            self.register_buffer(
                "stress_weight",
                torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
            )
        self.energy_huber_delta = energy_huber_delta
        self.forces_huber_delta = forces_huber_delta
        self.stress_huber_delta = stress_huber_delta

    def forward(self, pred, label) -> Tensor:
        num_atoms = (label.ptr[1:] - label.ptr[:-1])
        batch = label["batch"]
        energy_weight = label.energy_weight
        forces_weight = label.forces_weight[batch].unsqueeze(-1)
        stress_weight = label.stress_weight.unsqueeze(-1).unsqueeze(-1)

        loss_energy = torch.nn.functional.huber_loss(
            energy_weight * label["energy"] / num_atoms,
            energy_weight * pred["energy"] / num_atoms,
            reduction="mean",
            delta=self.energy_huber_delta,
        )
        loss_forces = conditional_huber_forces(
            forces_weight * pred["forces"],
            forces_weight * label["forces"],
            huber_delta=self.forces_huber_delta,
        )
        loss_stress = torch.nn.functional.huber_loss(
            stress_weight * label["stress"],
            stress_weight * pred["stress"],
            reduction="mean",
            delta=self.stress_huber_delta,
        )
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, stress_weight={self.stress_weight:.3f}, "
            f"energy_huber_delta={self.energy_huber_delta:.3f}, forces_huber_delta={self.forces_huber_delta:.3f}, "
            f"stress_huber_delta={self.stress_huber_delta:.3f})"
        )
################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################
"""
This file is designed to be compatible with earlier development versions. The 
framework of the early loss function originated from MACE, However, as the number 
of physical quantities predicted by TACE increases, it becomes less convenient.
"""

import torch


from .fn import *


# total energy
class EnergyLoss(torch.nn.Module):
    def __init__(self, energy_weight: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))

    def forward(self, pred, label):
        return self.energy_weight * mse_energy(pred, label)

    def __repr__(self):
        return f"{self.__class__.__name__}"


# per atom energy
class EnergyPerAtomLoss(torch.nn.Module):
    def __init__(self, energy_weight: float = 1.0, **kwargs) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))

    def forward(self, pred, label):
        return self.energy_weight * mse_energy_per_atom(pred, label)

    def __repr__(self):
        return f"{self.__class__.__name__}"


# dipole
class DipolePerAtomLoss(torch.nn.Module):
    def __init__(self, dipole_weight=1.0, **kwargs) -> None:
        super().__init__()
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label) -> Tensor:
        loss = mse_dipole(pred, label) * 100.0
        return self.dipole_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(dipole_weight={self.dipole_weight:.3f})"


# polarizability
class PolarizabilityPerAtomLoss(torch.nn.Module):
    def __init__(self, polarizability_weight=1.0, **kwargs) -> None:
        super().__init__()
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label) -> Tensor:
        loss = mse_polarizability_per_atom(pred, label) * 100.0
        return self.polarizability_weight * loss

    def __repr__(self):
        return f"{self.__class__.__name__}(polarizability_weight={self.polarizability_weight:.3f})"


# total energy + forces
class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=100.0, **kwargs) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("forces_weight", torch.tensor(forces_weight))

    def forward(self, pred, label):
        return self.energy_weight * mse_energy(
            pred, label
        ) + self.forces_weight * mse_forces(pred, label)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item()}, "
            f"forces_weight={self.forces_weight.item()})"
        )


# energy + forces
class WeightedEnergyPerAtomForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=100.0, **kwargs) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("forces_weight", torch.tensor(forces_weight))

    def forward(self, pred, label):
        return self.energy_weight * mse_energy_per_atom(
            pred, label
        ) + self.forces_weight * mse_forces(pred, label)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item()}, "
            f"forces_weight={self.forces_weight.item()})"
        )


# energy + forces + stress
class WeightedEnergyPerAtomForcesStressLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=100.0, stress_weight=1.0, **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("forces_weight", torch.tensor(forces_weight))
        self.register_buffer("stress_weight", torch.tensor(stress_weight))

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.stress_weight * mse_stress(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item()}, "
            f"forces_weight={self.forces_weight.item()}, "
            f"stress_weight={self.stress_weight.item()})"
        )


# huber loss | energy + forces + stress
class WeightedHuberEnergyForcesStressLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        stress_weight=1.0,
        huber_delta=0.01,
        **kwargs,
    ) -> None:
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

    def forward(self, pred, label) -> torch.Tensor:
        num_nodes = label.ptr[1:] - label.ptr[:-1]

        loss_energy = torch.nn.functional.huber_loss(
            pred["energy"] / num_nodes,
            label["energy"] / num_nodes,
            reduction="mean",
            delta=self.huber_delta,
        )
        loss_forces = torch.nn.functional.huber_loss(
            pred["forces"], label["forces"], reduction="mean", delta=self.huber_delta
        )
        loss_stress = torch.nn.functional.huber_loss(
            pred["stress"], label["stress"], reduction="mean", delta=self.huber_delta
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


# energy + forces + virials
class WeightedEnergyPerAtomForcesVirialsPerAtomLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=100.0, virials_weight=1.0, **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("forces_weight", torch.tensor(forces_weight))
        self.register_buffer("virials_weight", torch.tensor(virials_weight))

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.virials_weight * mse_virials_per_atom(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item()}, "
            f"forces_weight={self.forces_weight.item()}, "
            f"virials_weight={self.virials_weight.item()})"
        )


# total energy + forces + hessians
class WeightedEnergyForcesHessiansLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=0.08, hessians_weight=0.02, **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("forces_weight", torch.tensor(forces_weight))
        self.register_buffer("hessians_weight", torch.tensor(hessians_weight))

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.hessians_weight * mse_hessians(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item()}, "
            f"forces_weight={self.forces_weight.item()}, "
            f"hessians_weight={self.hessians_weight.item()})"
        )


# total energy + forces + hessians
class WeightedEnergyPerAtomForcesHessiansLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, hessians_weight=1.0, **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("forces_weight", torch.tensor(forces_weight))
        self.register_buffer("hessians_weight", torch.tensor(hessians_weight))

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.hessians_weight * mse_hessians(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item()}, "
            f"forces_weight={self.forces_weight.item()}, "
            f"hessians_weight={self.hessians_weight.item()})"
        )


# energy + forces + dipole
class WeightedEnergyPerAtomForcesDipoleLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0, **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.dipole_weight * mse_dipole(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f})"
        )


# energy + forces + charges
class WeightedEnergyPerAtomForcesChargesLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=100.0, charges_weight=1.0, **kwargs
    ) -> None:
        super().__init__()
        self.register_buffer("energy_weight", torch.tensor(energy_weight))
        self.register_buffer("forces_weight", torch.tensor(forces_weight))
        self.register_buffer("charges_weight", torch.tensor(charges_weight))

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.charges_weight * mse_charges(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight.item()}, "
            f"forces_weight={self.forces_weight.item()}, "
            f"charges_weight={self.charges_weight.item()})"
        )


# energy(electric enthalpy) + forces + polarization
class WeightedEnergyPerAtomForcesPolarizationPerAtomLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        polarization_weight=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarization_weight",
            torch.tensor(polarization_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.polarization_weight * mse_polarization_per_atom(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, polarization_weight={self.polarization_weight:.3f}"
        )


# energy + forces + dipole + polarizability
class WeightedEnergyPerAtomForcesDipolePolarizabilityPerAtomLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        polarizability_weight=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.dipole_weight * mse_dipole(pred, label)
            + self.polarizability_weight * mse_polarizability_per_atom(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.dipole_weight:.3f}, "
            f"polarizability_weight={self.polarizability_weight:.3f})"
        )


# energy(electric enthalpy) + forces + polarization + polarizability
class WeightedEnergyPerAtomForcesPolarizationPerAtomPolarizabilityPerAtomLoss(
    torch.nn.Module
):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        polarization_weight=1.0,
        polarizability_weight=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarization_weight",
            torch.tensor(polarization_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.polarization_weight * mse_polarization_per_atom(pred, label)
            + self.polarizability_weight * mse_polarizability_per_atom(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.polarization_weight:.3f}, "
            f"polarizability_weight={self.polarizability_weight:.3f})"
        )


# energy(electric enthalpy) + forces + polarization + born_charges
class WeightedEnergyPerAtomForcesPolarizationPerAtomBornEffectiveChargesLoss(
    torch.nn.Module
):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        polarization_weight=1.0,
        born_effective_charges_weight=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarization_weight",
            torch.tensor(polarization_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "born_effective_charges_weight",
            torch.tensor(
                born_effective_charges_weight, dtype=torch.get_default_dtype()
            ),
        )

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.polarization_weight * mse_polarization_per_atom(pred, label)
            + self.born_effective_charges_weight
            * mse_born_effective_charges(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, polarization_weight={self.polarization_weight:.3f}, "
            f"born_effective_charges_weight_weight={self.born_effective_charges_weight:.3f})"
        )


# energy(electric enthalpy) + forces + polarization + polarizability + born_charges
class WeightedEnergyPerAtomForcesPolarizationPerAtomPolarizabilityPerAtomBornEffectiveChargesLoss(
    torch.nn.Module
):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        polarization_weight=1.0,
        polarizability_weight=1.0,
        born_effective_charges_weight=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarization_weight",
            torch.tensor(polarization_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "born_effective_charges_weight",
            torch.tensor(
                born_effective_charges_weight, dtype=torch.get_default_dtype()
            ),
        )

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.polarization_weight * mse_polarization_per_atom(pred, label)
            + self.polarizability_weight * mse_polarizability_per_atom(pred, label)
            + self.born_effective_charges_weight
            * mse_born_effective_charges(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.polarization_weight:.3f}, "
            f"polarizability_weight={self.polarizability_weight:.3f}), "
            f"born_effective_charges_weight_weight={self.born_effective_charges_weight:.3f}), "
        )


class WeightedEnergyPerAtomForcesStressPolarizationPerAtomPolarizabilityPerAtomBornEffectiveChargesLoss(
    torch.nn.Module
):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        polarization_weight=1.0,
        polarizability_weight=1.0,
        born_effective_charges_weight=1.0,
        stress_weight=1.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarization_weight",
            torch.tensor(polarization_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "polarizability_weight",
            torch.tensor(polarizability_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "born_effective_charges_weight",
            torch.tensor(
                born_effective_charges_weight, dtype=torch.get_default_dtype()
            ),
        )
        self.register_buffer(
            "stress_weight",
            torch.tensor(stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label):
        return (
            self.energy_weight * mse_energy_per_atom(pred, label)
            + self.forces_weight * mse_forces(pred, label)
            + self.polarization_weight * mse_polarization_per_atom(pred, label)
            + self.polarizability_weight * mse_polarizability_per_atom(pred, label)
            + self.born_effective_charges_weight
            * mse_born_effective_charges(pred, label)
            + self.stress_weight * mse_stress(pred, label)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, dipole_weight={self.polarization_weight:.3f}, "
            f"polarizability_weight={self.polarizability_weight:.3f}), "
            f"born_effective_charges_weight_weight={self.born_effective_charges_weight:.3f}), "
            f"stress_weight_weight={self.stress_weight:.3f})"
        )


# energy per atom L1 + forces L2
class WeightedEnergyPerAtomForcesL1L2Loss(torch.nn.Module):
    "From MACE https://github.com/ACEsuit/mace."

    def __init__(self, energy_weight=1.0, forces_weight=1.0, **kwargs) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label) -> torch.Tensor:
        loss_energy = mae_energy_per_atom(pred, label)
        loss_forces = mean_normed_forces(pred, label)
        return self.energy_weight * loss_energy + self.forces_weight * loss_forces

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


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

# huber loss | energy + conditional_huber_forces + stress (direct version)
class UniversalDirectLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        direct_forces_weight=1.0,
        direct_stress_weight=1.0,
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
            "direct_forces_weight",
            torch.tensor(direct_forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "direct_stress_weight",
            torch.tensor(direct_stress_weight, dtype=torch.get_default_dtype()),
        )

    def forward(self, pred, label) -> Tensor:
        num_atoms = (label.ptr[1:] - label.ptr[:-1])
        batch = label["batch"]
        energy_weight = label.energy_weight
        forces_weight = label.direct_forces_weight[batch].unsqueeze(-1)
        stress_weight = label.direct_stress_weight.unsqueeze(-1).unsqueeze(-1)

        loss_energy = torch.nn.functional.huber_loss(
            energy_weight * label["energy"] / num_atoms,
            energy_weight * pred["energy"] / num_atoms,
            reduction="mean",
            delta=self.huber_delta,
        )
        loss_forces = conditional_huber_forces(
            forces_weight * pred["direct_forces"],
            forces_weight * label["direct_forces"],
            huber_delta=self.huber_delta,
        )
        loss_stress = torch.nn.functional.huber_loss(
            stress_weight * label["direct_stress"],
            stress_weight * pred["direct_stress"],
            reduction="mean",
            delta=self.huber_delta,
        )
        return (
            self.energy_weight * loss_energy
            + self.direct_forces_weight * loss_forces
            + self.direct_stress_weight * loss_stress
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"direct_forces_weight={self.direct_forces_weight:.3f}, direct_stress_weight={self.direct_stress_weight:.3f})"
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
        **kwargs,
    ) -> None:
        "Modify code From MACE, see same logic as NequIP"
        super().__init__()
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
            f"energy_huber_delta={self.energy_huber_delta:.3f}, forces_huber_delta={self.forces_weight:.3f}, "
            f"stress_huber_delta={self.stress_huber_delta:.3f})"
        )
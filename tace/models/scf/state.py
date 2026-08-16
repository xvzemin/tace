################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


class SCFState(NamedTuple):
    density_coefficients: torch.Tensor
    fermi_level: torch.Tensor
    field_features: Optional[torch.Tensor]
    density_history: torch.Tensor
    num_steps: int
    status: str
    residual: torch.Tensor


@dataclass(frozen=True)
class FixedPointSCFOptions:
    num_scf_steps: int = 100
    scf_tolerance: float = 1.0e-6
    mixing_parameter: float = 0.25
    constant_charge: bool = True
    initial_density: str = "local_guess"
    initial_fermi_level: str = "from_data"
    differentiation: str = "unrolled"
    num_autograd_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_scf_steps < 1:
            raise ValueError("num_scf_steps must be positive")
        if self.scf_tolerance <= 0.0:
            raise ValueError("scf_tolerance must be positive")
        if not 0.0 < self.mixing_parameter <= 1.0:
            raise ValueError("mixing_parameter must be in (0, 1]")
        if self.initial_density not in {"local_guess", "zeros", "from_data"}:
            raise ValueError(
                "initial_density must be 'local_guess', 'zeros', or 'from_data'"
            )
        if self.initial_fermi_level not in {"zeros", "from_data"}:
            raise ValueError("initial_fermi_level must be 'zeros' or 'from_data'")
        if self.differentiation not in {"unrolled", "implicit"}:
            raise ValueError("differentiation must be 'unrolled' or 'implicit'")


@dataclass(frozen=True)
class EnergyFunctionalSCFOptions:
    num_scf_steps: int = 100
    scf_tolerance: float = 1.0e-6
    learning_rate: float = 0.1
    initial_density: str = "local_guess"
    differentiation: str = "unrolled"
    num_autograd_steps: Optional[int] = None

    def __post_init__(self) -> None:
        if self.num_scf_steps < 1:
            raise ValueError("num_scf_steps must be positive")
        if self.scf_tolerance <= 0.0:
            raise ValueError("scf_tolerance must be positive")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.initial_density not in {"local_guess", "zeros", "from_data"}:
            raise ValueError(
                "initial_density must be 'local_guess', 'zeros', or 'from_data'"
            )
        if self.differentiation not in {"unrolled", "implicit"}:
            raise ValueError("differentiation must be 'unrolled' or 'implicit'")

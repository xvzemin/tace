################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from .normal import NormalLoss
from .registry import ensure_loss_functions_registered
from .uncertainty import UncertaintyLoss

ensure_loss_functions_registered()

__all__ = ["NormalLoss", "UncertaintyLoss"]

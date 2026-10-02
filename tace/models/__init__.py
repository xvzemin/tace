import eqx  # Initialize native operators and legacy e3nn constants first.

from ._e3nn import e3nnTACE
from ._e3nn.tece import TECE
from .adapter import TensorModel
from .compile import CompileTensorModel

# Compatibility export for configurations created before cartTACE was removed.
cartTACE = None

try:
    from .scf import SCFTACE
except Exception:
    SCFTACE = None

__all__ = [
    "cartTACE",
    "e3nnTACE",
    "TECE",
    "TensorModel",
    "CompileTensorModel",
    "SCFTACE",
]

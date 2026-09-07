from ._e3nn import e3nnTACE
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
    "TensorModel",
    "CompileTensorModel",
    "SCFTACE",
]

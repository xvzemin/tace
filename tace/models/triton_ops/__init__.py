################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Temporary Triton kernels used by the TACE EQX preview integration.

EQX is intended to become the general operator library for O(3), O(2),
Wigner-6j, and Cartesian equivariant computations.  This package is only a
temporary implementation placeholder: users select it through
``TACE_USE_EQX=1`` and should not treat ``triton_ops`` as a public backend.
"""

from .uu_so2_scatter import UUSO2Scatter

__all__ = ["UUSO2Scatter"]

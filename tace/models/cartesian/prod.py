################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

from ..spherical.prod import NonlinearCgtpACE


class NonlinearIctdACE(NonlinearCgtpACE):
    """Channel-wise ICTD products with invariant path gates.

    Parameters
    ----------
    **kwargs
        Parameters of :class:`~tace.models.spherical.base.Product`.

    Notes
    -----
    Select ``product_basis.type: nonlinear_ictd`` with ``CartesianTACE``.
    Features use flattened ``mul_ir`` storage with ``3**l`` Cartesian entries
    per channel. Paths, parameters and normalization match ``NonlinearCgtpACE``.
    Linear up generates independent factors and invariant scalars. Scalars
    and element embeddings produce one gate per path and channel.
    The shared ``NonlinearCgtpACE.asymmetric`` class setting controls the input
    factors for both representations. Only symmetric-product paths are retained.
    """

    _compiled_cg_rows = False

    def _setup(self):
        from .layers import cartesian_modules

        super()._setup()
        cartesian_modules(self)

    def build_eqx(self):
        """Keep the low-rank eager fusion; native ICTD owns each training stage."""
        from equivariantx import co3

        # Conversion may start from a host with valid *spherical* programs.
        # In particular eqx_products contains raw CG rows, not an ICTD STF
        # projector. A recursive product must project before the next factor.
        self._clear_eqx()
        if not isinstance(self.irreps_in, co3.Irreps):
            return  # Temporary spherical layout during _setup.
        from equivariantx_foundation.models.tace.nonlinear_ace import (
            MAX_INLINE_STF_RANK,
            build,
        )

        if any(
            stage.tp.irreps_out[i].ir.l > MAX_INLINE_STF_RANK
            for stage in self.aces
            if stage.tp.project
            for i in stage.tp.projection_indices
        ):
            # Do not inline a (3**l)**2 STF matrix in the eager expression.
            # The existing ICTD module has its own factored projection.
            return
        self.eqx_metadata, self.eqx_bindings = build(self)

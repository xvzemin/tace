"""Nonlinear products in irreducible Cartesian tensor storage."""

from .._e3nn.prod import NonlinearCgtpACE


class NonlinearIctdACE(NonlinearCgtpACE):
    """Channel-wise ICTD products with invariant path gates.

    Parameters
    ----------
    **kwargs
        Parameters of :class:`~tace.models._e3nn.base.Product`.

    Notes
    -----
    Select ``product_basis.type: nonlinear_ictd`` with ``cartTACE``.
    Features use flattened ``mul_ir`` storage with ``3**l`` Cartesian entries
    per channel. Paths, parameters and normalization match ``NonlinearCgtpACE``.
    Linear up generates independent factors and invariant scalars. Scalars
    and element embeddings produce one gate per path and channel.
    The shared ``NonlinearCgtpACE.asymmetric`` class setting controls the input
    factors for both representations. Only symmetric-product paths are retained.
    """

    def _setup(self):
        from .layers import cartesian_modules

        super()._setup()
        cartesian_modules(self)

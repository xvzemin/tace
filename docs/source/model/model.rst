Model Arch
==========

``tace.models.SphericalTACE`` is the default model. Select
``tace.models.CartesianTACE`` through ``model.config._target_`` for Cartesian
storage. Their components are in ``tace.models.spherical`` and
``tace.models.cartesian``. For ``.pt`` parameter-dictionary models, saved
configuration names using ``e3nnTACE``, ``cartTACE``, ``_e3nn`` or ``_cart``
are resolved during loading. This compatibility does not cover serialized
model objects; new code uses the names above.

.. toctree::
   :maxdepth: 2
   :caption: Contents

   node_embedding.rst
   edge_embedding.rst
   node_update.rst
   edge_update.rst
   inter.rst
   prod.rst
   cartesian.rst

Legacy model building blocks
----------------------------

Cutoff placement
   Always set ``radial_basis.apply_cutoff: false`` for new models. This applies
   the cutoff to the convolution weights or the final edge message, after the
   radial MLP. With ``true``, only the radial basis is multiplied by the
   cutoff, so a biased MLP can still produce nonzero weights at the cutoff.

Density normalization without a cutoff
   ``atomic_basis.scatter_norm: no_cutoff_density`` retains a development bug
   solely for compatibility with existing models. It omits the cutoff from
   each neighbor's contribution to the learned density, so the normalization
   can jump when a neighbor leaves the cutoff. For new models that use learned
   density normalization, set ``atomic_basis.scatter_norm: density``.

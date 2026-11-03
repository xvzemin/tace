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

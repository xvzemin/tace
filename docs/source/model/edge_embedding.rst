Edge Embedding
==============

We recommend using either ``IdentityEdgeEmbedding`` or ``NonLinearEdgeEmbedding`` in most cases.

The identity embedding is the most conservative choice. 
The nonlinear edge embedding is generally recommended when used together with element-dependent edge update.

.. autoclass:: tace.models.spherical.edge.IdentityEdgeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models.spherical.edge.LinearEdgeEmbedding
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models.spherical.edge.NonLinearEdgeEmbedding
   :no-members:
   :show-inheritance:

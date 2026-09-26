Edge Update
===========

The available edge updates are ``IdentityEdgeUpdate`` and ``Element2EdgeUpdate``.

The most conservative choice is ``IdentityEdgeUpdate``, which does not introduce any additional information during edge updates. 
For datasets with relatively rich sampling in configuration space, we recommend using ``Element2EdgeUpdate`` to incorporate element-dependent information. 

``Element2EdgeUpdate`` keeps radial, target-element and source-element features
separate. The first MLP linear map projects the node features before gathering
them onto edges. This is equivalent to projecting their concatenation, while
avoiding repeated element projections on edges.

.. autoclass:: tace.models._e3nn.edge.IdentityEdgeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.edge.Element2EdgeUpdate
   :no-members:
   :show-inheritance:

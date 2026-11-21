Edge Update
===========

Select an edge update with ``edge_update.type``: ``identity`` or ``element2``.

The most conservative choice is ``IdentityEdgeUpdate``, which does not introduce any additional information during edge updates. 
For datasets with relatively rich sampling in configuration space, we recommend using ``Element2EdgeUpdate`` to incorporate element-dependent information. 

``Element2EdgeUpdate`` keeps radial, target-element and source-element features
separate, using independent ``torch.nn.Embedding`` tables for the two endpoints.
The first MLP linear map projects the node features before gathering
them onto edges. This is equivalent to projecting their concatenation, while
avoiding repeated element projections on edges.

.. autoclass:: tace.models.spherical.edge.IdentityEdgeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models.spherical.edge.Element2EdgeUpdate
   :no-members:
   :show-inheritance:

Edge Update
===========

Select an edge update with ``edge_update.type``: ``identity``, ``element2``
or ``element2_v2``.

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

``element2_v2`` uses separate source and target lookup tables. Branch-wise
scaling prevents the added element channels from diluting the radial input.
The element embeddings start with a small standard deviation fixed by the
class constant ``ELEMENT_STD``; it is not a configuration option.

This update requires ``radial_basis.apply_cutoff: false`` so that the envelope
acts after the radial MLP, including the element-dependent contributions.

.. autoclass:: tace.models._e3nn.edge.Element2EdgeUpdateV2
   :no-members:
   :show-inheritance:

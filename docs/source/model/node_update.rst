Node update
==============

``NodeUpdate`` transforms the magnetic radial basis on nodes and returns
separate source and target feature tensors. Edge indexing is deferred to the
magnetic interaction. ``magnetic_type`` accepts ``identity``, ``element``, or
``element2``.


.. autoclass:: tace.models._e3nn.node.IdentityNodeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.ElementNodeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.node.Element2NodeUpdate
   :no-members:
   :show-inheritance:

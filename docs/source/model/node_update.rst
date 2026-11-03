Node update
==============

``NodeUpdate`` transforms the magnetic radial basis on nodes and returns
separate source and target information tensors. Edge indexing is deferred to the
magnetic interaction. ``magnetic_type`` accepts ``identity``, ``element``, or
``element2``.


.. autoclass:: tace.models.spherical.node.IdentityNodeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models.spherical.node.ElementNodeUpdate
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models.spherical.node.Element2NodeUpdate
   :no-members:
   :show-inheritance:

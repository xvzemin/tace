Product
=======

The following modules implement many-body expansions using different strategies.
We recommend using ``CgtpACE`` in practice. For other ACE variants, interested
users are encouraged to refer to the source code for further exploration.

.. autoclass:: tace.models._e3nn.prod.CgtpACE
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.prod.MaceACE
   :no-members:
   :show-inheritance:

Select MACE symmetric contractions with the existing product configuration:

.. code-block:: yaml

   product_basis:
     type: mace
     correlation: 3
     l1l2: "<="

The block applies ``Linear up → symmetric contraction → Linear`` and then the
configured residual connection. ``Linear up`` is omitted when the input and
contraction channel counts match. Select fused contractions with ``TACE_USE_EQX=1``
or ``TACE_USE_CUE=1``. EQX takes priority when both are enabled. With neither
enabled, or on CPU, the contraction uses PyTorch. The weight layout is unchanged.

``l1l2`` restricts each recursive CG coupling: the accumulated intermediate
degree is compared with the next input degree. Options are ``null``, ``<``,
``<=``, ``>``, ``>=``, ``==`` and ``!=``. Degree-one terms are unchanged.
Each retained path has its own element-dependent coefficient. Changing the
restriction changes the path count and parameter shapes.

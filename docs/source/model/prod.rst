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

The block applies ``Linear up → symmetric contraction → Linear`` and then the
configured residual connection. Select fused contractions with ``TACE_USE_EQX=1``
or ``TACE_USE_CUE=1``. EQX takes priority when both are enabled. With neither
enabled, or on CPU, the contraction uses PyTorch. The weight layout is unchanged.

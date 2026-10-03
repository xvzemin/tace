.. _equivariantx-api-co3:

Cartesian O(3)
==============

Features use flattened ``mul_ir`` storage with ``3**l`` entries per channel.
Tensor-product inputs lie in the symmetric traceless subspace.

Representations
---------------

.. autoclass:: eqx.co3.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.co3.Irreps
   :members:

Harmonics
---------

.. autoclass:: eqx.co3.CartesianHarmonics
   :members: forward

Linear maps and activations
---------------------------

.. autoclass:: eqx.co3.Linear
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co3.Activation
   :members: forward

.. autoclass:: eqx.co3.Gate
   :members: forward

Tensor products
---------------

.. autoclass:: eqx.co3.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co3.FullyConnectedTensorProduct

.. autoclass:: eqx.co3.ElementwiseTensorProduct

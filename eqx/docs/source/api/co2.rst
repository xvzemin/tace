.. _equivariantx-api-co2:

Cartesian O(2)
==============

Features use flattened ``mul_ir`` storage with ``2**m`` entries per channel.
Positive-order tensors lie in the symmetric traceless subspace.

Representations
---------------

.. autoclass:: eqx.co2.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.co2.Irreps
   :members:

Harmonics
---------

.. autoclass:: eqx.co2.CartesianHarmonics
   :members: forward

Linear maps
-----------

.. autoclass:: eqx.co2.Linear
   :members: forward, weight_view_for_instruction, weight_views

Activations
-----------

.. autoclass:: eqx.co2.Activation
   :members: forward

.. autoclass:: eqx.co2.Gate
   :members: forward

Tensor products
---------------

.. autoclass:: eqx.co2.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co2.FullyConnectedTensorProduct

.. autoclass:: eqx.co2.ElementwiseTensorProduct

Harmonic O(3) coupling
----------------------

.. autoclass:: eqx.co2.O3TensorProduct
   :members: forward, forward_scatter, from_tensor_product, weight_views

.. autoclass:: eqx.co2.SphericalCoupling
   :members: forward

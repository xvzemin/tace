.. _equivariantx-api-co2:

Cartesian O(2)
==============

Features use flattened ``mul_ir`` storage with ``2**m`` entries per channel.
Positive-order tensors lie in the two-dimensional symmetric traceless
subspace. See :ref:`equivariantx-cartesian-o2` for basis conversion and
restriction of three-dimensional tensors.

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

Linear maps and activations
---------------------------

.. autoclass:: eqx.co2.Linear
   :members: forward, weight_view_for_instruction, weight_views

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

These operators couple O(3) features to harmonics of a nonzero direction.
They do not take an arbitrary second feature tensor.

.. autoclass:: eqx.co2.O3TensorProduct
   :members: forward, forward_scatter, from_tensor_product, weight_views

.. autoclass:: eqx.co2.SphericalCoupling
   :members: forward

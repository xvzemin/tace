.. _equivariantx-api-convolutions:

Convolutions
============

O(3) convolutions
-----------------

.. autoclass:: eqx.conv.O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.GauntTensorProductConv
   :members: forward

.. autoclass:: eqx.conv.MatrixTensorProductConv
   :members: forward

.. autoclass:: eqx.conv.Wigner6jTensorProductConv
   :members: forward, forward_reference

.. autoclass:: eqx.conv.CartesianTensorProductConv
   :members: forward

O(2) convolutions
-----------------

.. autoclass:: eqx.conv.O2O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UuO2TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UvO2TensorProductConv
   :members: forward

Cartesian output maps
---------------------

.. autoclass:: eqx.conv.co3.Linear
   :members: forward

Graph attention
---------------

.. autofunction:: eqx.conv.graph_softmax

.. autoclass:: eqx.conv.StreamingGraphAttention
   :members: forward

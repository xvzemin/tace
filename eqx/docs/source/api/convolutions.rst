.. _equivariantx-api-convolutions:

Convolutions
============

Fused spherical convolutions use ``ir_mul`` layout. Cartesian, Gaunt and
Matrix convolutions use ``mul_ir``, unless compact Cartesian storage is selected.

|eqx-convolutions|

Tensor-product convolutions
---------------------------

.. autoclass:: eqx.conv.O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.GauntTensorProductConv
   :members: forward

.. autoclass:: eqx.conv.MatrixTensorProductConv
   :members: forward

.. autoclass:: eqx.conv.CartesianTensorProductConv
   :members: forward

.. autoclass:: eqx.conv.O2O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UuO2TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UvO2TensorProductConv
   :members: forward

Compact Cartesian output
------------------------

.. autoclass:: eqx.conv.co3.Linear
   :members: forward

Graph attention
---------------

.. autofunction:: eqx.conv.graph_softmax

.. autoclass:: eqx.conv.StreamingGraphAttention
   :members: forward

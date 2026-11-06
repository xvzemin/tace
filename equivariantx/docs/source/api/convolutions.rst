.. _equivariantx-api-convolutions:

Convolutions
============

CUDA contractions support float32 and float64. Floating operands within a
kernel must share a dtype and device. Convolution wrappers can evaluate geometry
in float64 and cast the resulting angular features to the network dtype;
these casts preserve autograd. FP16 and BF16 CUDA contractions are not supported.

O(3) convolutions
-----------------

.. autoclass:: equivariantx.conv.O3TensorProductConv
   :members: forward

.. autoclass:: equivariantx.conv.GauntTensorProductConv
   :members: forward

.. autoclass:: equivariantx.conv.MatrixTensorProductConv
   :members: forward

.. autoclass:: equivariantx.conv.Wigner6jTensorProductConv
   :members: forward, forward_reference

.. autoclass:: equivariantx.conv.CartesianTensorProductConv
   :members: forward

O(2) convolutions
-----------------

.. autoclass:: equivariantx.conv.O2O3TensorProductConv
   :members: forward

.. autoclass:: equivariantx.conv.UuO2TensorProductConv
   :members: forward

.. autoclass:: equivariantx.conv.UvO2TensorProductConv
   :members: forward

Cartesian output maps
---------------------

.. autoclass:: equivariantx.conv.co3.Linear
   :members: forward

Graph attention
---------------

.. autofunction:: equivariantx.conv.graph_softmax

.. autoclass:: equivariantx.conv.StreamingGraphAttention
   :members: forward

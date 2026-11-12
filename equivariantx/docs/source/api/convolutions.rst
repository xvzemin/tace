.. _equivariantx-api-convolutions:

Convolutions
============

Torch is the default backend; CUDA kernels are optional. Operands support
FP32 and FP64, with the caller's TF32 matrix policy. FP16, BF16 and reduced
precision autocast are unsupported. Geometry may remain FP64 with FP32 features.

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

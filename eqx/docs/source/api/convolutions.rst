.. _equivariantx-api-convolutions:

Convolutions
============

O(3) CG, O(2)/O(3), Uu, and Uv convolutions use ``ir_mul`` layout.
Cartesian, Gaunt, matrix, and Wigner-6j convolutions use ``mul_ir`` by default.
Gaunt and matrix convolutions use PyTorch operations rather than fused kernels.

|eqx-convolutions|

Tensor-product convolutions
---------------------------

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

.. _equivariantx-api-convolutions:

Fused convolutions
==================

Spherical features use flattened ``ir_mul`` layout. Cartesian features use
``mul_ir`` or the explicitly selected compact layout. Each convolution
provides a PyTorch reference with automatic differentiation through
``backend="torch"``. See :ref:`equivariantx-convolutions` for CUDA methods
and fusion boundaries.

.. autoclass:: eqx.conv.O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.CartesianTensorProductConv
   :members: forward

.. autoclass:: eqx.conv.O2O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UuO2TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UvO2TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.EceO2TensorProductConv
   :members: forward, set_algorithm

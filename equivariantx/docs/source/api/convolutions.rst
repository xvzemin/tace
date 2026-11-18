.. _equivariantx-api-convolutions:

Convolutions
============

``backend=None`` selects CUDA when available and optimized PyTorch otherwise.
Use ``backend="torch"`` to select PyTorch explicitly. Operands support
FP32 and FP64, with the caller's TF32 matrix policy. FP16, BF16 and reduced
precision autocast are unsupported. Geometry may remain FP64 with FP32 features.

For CGTP and UV-O(2), ``radial_mode="full"`` includes the radial network in
the convolution schedule. Use ``radial_mode="last"`` to keep its hidden
features and fuse only the output projection.

``EQX_USE_CUDA_GRAPH=1`` enables native convolution replay after warmup.
Eligible local O(2) programs can reuse a capture across nearby node and edge
counts. Replay reserves additional memory.

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

.. _equivariantx-api-convolutions:

Fused convolutions
==================

Spherical features use flattened ``ir_mul`` storage. Cartesian features use
``mul_ir`` unless compact storage is selected explicitly.

|eqx-convolutions|

CUDA backends support float32, float64, and higher derivatives, including
force training. Use ``backend="torch"`` for the differentiable PyTorch
reference on either device. CPU inputs use PyTorch. Atomic reductions may
change floating-point summation order.

Warm up the required derivatives and automatic method selection before
``torch.compile`` or timing. Exported models require EQX operator registration
and the CUDA runtime. Node-level channel maps and preceding radial MLP layers
remain outside the fused CGTP operators.

Tensor-product convolutions
---------------------------

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

Compact Cartesian output
------------------------

.. autoclass:: eqx.conv.co3.Linear
   :members: forward

Graph attention
---------------

.. autofunction:: eqx.conv.graph_softmax

.. autoclass:: eqx.conv.StreamingGraphAttention
   :members: forward

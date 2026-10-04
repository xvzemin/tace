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
and the CUDA runtime. Node-level channel maps remain outside the operators.

Pass ``radial_network`` to stream the preceding radial MLP in bounded edge
tiles. Nonlinear activations are recomputed during backward and full-edge
convolution weights are not retained. Linear layers, SiLU,
sigmoid, tanh, LayerNorm, and RMSNorm are supported. For CGTP and UuO2 operators,
``projection`` supplies the final linear weight separately. Omit
``radial_network`` to stream only this final projection.
Recomputation reduces activation storage but may increase runtime on small graphs.

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

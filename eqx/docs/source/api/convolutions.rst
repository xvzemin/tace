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
and the CUDA runtime. Node-level channel maps and edge embeddings remain outside
the fused operators.

Pass ``radial_network`` to evaluate the full radial MLP in bounded CUDA tiles.
Hidden activations and path weights are not retained over all edges; backward
recomputes them within each tile. Dense layers use matrix multiplication,
including wide hidden layers. This is exact evaluation, not interpolation.
The PyTorch backend evaluates the same network with ordinary autograd.

For tensor-product convolutions, ``radial_network`` supplies the layers before
``projection``. For ``UvO2TensorProductConv`` it includes the final layer that
produces the convolution coefficients. Linear, SiLU, sigmoid, tanh, LayerNorm
and RMSNorm layers are supported.

Radial inputs may be partitioned as ``((edge, "edge"), (node, "source"),
(node, "target"))``. Their channels are concatenated in the declared order.
Node contributions to the first linear layer are projected before gathering.
Edge embeddings themselves are unchanged.

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

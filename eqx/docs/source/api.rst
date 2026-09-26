.. _equivariantx-api:

API Reference
=============

See :ref:`equivariantx-tutorials` for representation conventions and
:ref:`equivariantx-convolutions` for execution backends and fusion boundaries.

Native O(2) operations
----------------------

Features use a flattened ``ir_mul`` axis with trailing size ``irreps.dim``.
Iterating over :class:`eqx.o2.Irreps` yields ``(ir, mul)`` entries. These
operators use PyTorch on CPU and GPU without a compiled extension.

.. autoclass:: eqx.o2.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.o2.Irreps
   :members:

.. autoclass:: eqx.o2.Linear
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.o2.UuLinear
   :members: forward

.. autoclass:: eqx.o2.Activation
   :members: forward

.. autoclass:: eqx.o2.Gate
   :members: forward

.. autoclass:: eqx.o2.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.o2.AsymmetricContraction
   :members: forward

.. autofunction:: eqx.o2.circular_harmonics

.. autoclass:: eqx.o2.CircularHarmonics
   :members: forward

O(3)/O(2) frame conversion
--------------------------

Global representation metadata follows ``e3nn.o3.Irreps``. Global and local
feature tensors both use EQX's flattened ``ir_mul`` layout. Wigner construction
has a PyTorch method and an optional CUDA method; ``LocalFrame`` itself applies
the supplied matrices with PyTorch.

.. autofunction:: eqx.o2.rotation_matrix_to_x_axis

.. autofunction:: eqx.o2.rotation_matrix_to_y_axis

.. autofunction:: eqx.o2.rotation_matrix_to_z_axis

.. autoclass:: eqx.o2.WignerD
   :members: forward, forward_packed, matrix_blocks

.. autoclass:: eqx.o2.LocalFrame
   :members: restrict, forward, to_local, to_global

.. autoclass:: eqx.o2.O3TensorProduct
   :members: forward, forward_local, forward_scatter

Fused O(3)/O(2) convolutions
----------------------------

These interfaces retain their supplied paths, weights, and normalization.
All feature operands use flattened ``ir_mul`` layout. CUDA kernels support
float32 and float64, force training, and recursive higher derivatives.
CGTP and uu convolutions also provide a PyTorch reference backend. The uv
interface is CUDA-only; its reference is the composition of native operators.

.. autoclass:: eqx.conv.O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.O2O3TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UuO2TensorProductConv
   :members: forward

.. autoclass:: eqx.conv.UvO2TensorProductConv
   :members: forward

.. autofunction:: eqx.conv.graph_softmax

.. autoclass:: eqx.conv.StreamingGraphAttention
   :members: forward

Supporting operators
--------------------

``eqx.o3`` supplies indexed element-dependent linear maps, not a second O(3)
representation algebra. These maps and ``eqx.ace.TACE`` use flattened ``mul_ir``
features. Weights are external and biases are applied by the caller.

.. autoclass:: eqx.o3.ElementLinear
   :members: forward

.. autoclass:: eqx.o3.MoEElementLinear
   :members: forward

.. autoclass:: eqx.ace.TACE
   :members: forward

.. autofunction:: eqx.kernels.wigner_D

Model-specific fusion
---------------------

Specialized kernels and adapters are organized by application under
``eqx.conv.models``. TACE maintains its PyTorch model definition and checkpoint
conversion separately; the MACE interface converts an existing model.

.. autoclass:: eqx.conv.models.tace.tece_oam_rra.BilinearACE
   :members: forward

.. autofunction:: eqx.conv.models.mace.convert_mace_to_eqx

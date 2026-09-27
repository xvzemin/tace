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

Cartesian O(3) operations
--------------------------

Features use flattened ``mul_ir`` storage with ``3**l`` Cartesian entries
per channel. Inputs to tensor products are symmetric traceless tensors.
These operators use native PyTorch on CPU and GPU.

.. autoclass:: eqx.co3.Irrep
   :members:

.. autoclass:: eqx.co3.Irreps
   :members:

.. autoclass:: eqx.co3.ChangeOfBasis
   :members: forward

.. autoclass:: eqx.co3.Projector
   :members: forward

.. autoclass:: eqx.co3.CartesianHarmonics
   :members: forward

.. autoclass:: eqx.co3.Linear
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co3.Activation
   :members: forward

.. autoclass:: eqx.co3.Gate
   :members: forward

.. autoclass:: eqx.co3.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co3.FullyConnectedTensorProduct

.. autoclass:: eqx.co3.ElementwiseTensorProduct

.. autofunction:: eqx.co3.path_matrix

.. autofunction:: eqx.co3.path_normalization

Supporting operators
--------------------

``eqx.o3`` supplies indexed element-dependent linear maps and gated activations.
These operators and ``eqx.ace.TACE`` use flattened ``mul_ir`` features. Linear
weights are external and biases are applied by the caller.

.. autoclass:: eqx.o3.Gate
   :members: forward

.. autoclass:: eqx.o3.ElementLinear
   :members: forward

.. autoclass:: eqx.o3.MoEElementLinear
   :members: forward

.. autoclass:: eqx.ace.TACE
   :members: forward

.. autofunction:: eqx.kernels.wigner_D

Model-specific operators
------------------------

Specialized operators and adapters are organized by application under
``eqx.models``. TACE maintains its PyTorch model definition and checkpoint
conversion separately. Other adapters convert instantiated models.

.. autoclass:: eqx.models.tace.tece_oam_rra.BilinearACE
   :members: forward

.. autofunction:: eqx.models.mace.convert_mace_to_eqx

.. autofunction:: eqx.models.nequip.convert_nequip_to_eqx

.. autofunction:: eqx.models.sevennet.convert_sevennet_to_eqx

.. autofunction:: eqx.models.prophet.convert_prophet_to_eqx

.. autofunction:: eqx.models.equflash.convert_equflash_to_eqx

Utilities
---------

.. autofunction:: eqx.utils.parse_metadata

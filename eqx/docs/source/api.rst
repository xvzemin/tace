.. _equivariantx-api:

API Reference
=============

See :ref:`equivariantx-tutorials` for representation conventions and examples.

Representations
---------------

Representation metadata defines the flattened ``ir_mul`` feature axis used by
all :mod:`eqx.o2` layers. Iterating over :class:`eqx.o2.Irreps` yields
``(ir, mul)`` entries.

.. autoclass:: eqx.o2.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.o2.Irreps
   :members:

Layers
------

These layers accept real tensors with trailing shape ``(irreps.dim,)``.
External weights may carry leading dimensions that broadcast with the inputs.

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

Harmonics and rotations
-----------------------

Circular harmonics operate directly in two dimensions. ``WignerD`` and
``LocalFrame`` convert global three-dimensional features to edge-aligned local
features and back while retaining flattened ``ir_mul`` storage.

.. autofunction:: eqx.o2.circular_harmonics

.. autoclass:: eqx.o2.CircularHarmonics
   :members: forward

.. autofunction:: eqx.o2.rotation_matrix_to_x_axis

.. autofunction:: eqx.o2.rotation_matrix_to_y_axis

.. autofunction:: eqx.o2.rotation_matrix_to_z_axis

.. autoclass:: eqx.o2.WignerD
   :members: forward, forward_packed

.. autoclass:: eqx.o2.LocalFrame
   :members: restrict, forward, to_local, to_global

.. autoclass:: eqx.o2.O3TensorProduct
   :members: forward, forward_local, forward_scatter

Element-dependent spatial linear maps
-------------------------------------

``eqx.o3`` uses flattened ``mul_ir`` features. CUDA kernels read weights
by element index instead of gathering a weight matrix for each node.
Expert channels remain independent within each irrep. Weights are external,
and biases are applied by the caller. The torch backend provides a reference
implementation; CUDA supports float32, float64 and higher derivatives.

.. autoclass:: eqx.o3.ElementLinear
   :members: forward

.. autoclass:: eqx.o3.MoEElementLinear
   :members: forward

Fused convolutions
------------------

``eqx.conv`` contains kernels for specific convolution architectures.
``O3TensorProductConv`` evaluates sparse global CG contractions with indexed
gather and reduction. Its optional ``vectors`` argument replaces general edge
attributes with spherical harmonics. The CUDA kernel evaluates fixed Cartesian
polynomials and contracts their derivatives directly into vector gradients,
without storing harmonic cotangents. ``amplitudes`` supplies independent radial
or cutoff factors. ``normalize=False`` selects regular solid harmonics.
The same polynomial derivative rule supports force training and higher orders.

.. autoclass:: eqx.conv.O3TensorProductConv
   :members: forward

``O2O3TensorProductConv`` evaluates an O(3) tensor product in aligned O(2)
frames, including indexed gather and reduction. Its default
``backend="cuda"`` generates fused CUDA kernels
on GPU and uses PyTorch on CPU. GPU execution requires the ``cuda`` extra
and a CUDA toolkit. CUDA supports ``uvu`` instructions. ``backend="torch"``
selects the reference implementation, including ``uvw`` instructions.
Both backends support recursive higher derivatives.

Passing ``vectors=edge_vector`` to ``O2O3TensorProductConv.forward`` enables direct
direction derivatives. Supply their packed alignment matrices, for example
with ``eqx.kernels.wigner_D(frame, edge_vector.detach())``. The matrices are then cached
values, and sparse rotation-generator contractions provide the geometry
derivatives, including higher orders. Degree-zero harmonic paths bypass the
rotations. Omitting ``vectors`` preserves differentiation with respect to the
matrix entries themselves.

Direction-derivative kernels stage only the Wigner rows required by each
sparse angular contraction. The packed matrices, rotation values and analytic
derivative rules are unchanged.

.. autoclass:: eqx.conv.O2O3TensorProductConv
   :members: forward

Atomic cluster expansions
-------------------------

``TACE`` evaluates unweighted channel-wise products with external element
coefficients. It preserves every supplied coupling path. CUDA execution
avoids expanded CG products and per-node coefficient matrices, and fuses
the final correlation order with its coefficient contraction. Intermediate
orders are retained for reuse. Features use flattened ``mul_ir`` storage.
The torch backend provides a reference implementation. Forward and recursively
transposed CUDA contractions support float32, float64 and higher derivatives.

.. autoclass:: eqx.conv.TACE
   :members: forward

``BilinearTACE`` fuses a gated ``uuu`` product with an element-dependent or
MoE coefficient map. It preserves expert-channel grouping after adjacent
irreps are simplified. Small node tiles reuse shared coefficients, and an
optional shared map reuses the same block-local angular tile as the experts.
Input gradients combine both maps before differentiating the
product, while coefficient gradients reuse the angular tile across output
channels. No full product or product-gradient array is stored. Biases,
order-one terms and the final expert normalization remain separate.

.. autoclass:: eqx.conv.BilinearTACE
   :members: forward

Shared geometry kernels
-----------------------

``eqx.kernels`` provides geometry kernels shared by convolution architectures.
``wigner_D`` uses direct quaternion polynomials by default on CUDA. Set
``method="recursive"`` to use sparse CG degree contractions instead, or
``backend="torch"`` for the recursive PyTorch reference. Both CUDA methods
support float32, float64, and higher derivatives. Coefficients and compiled
kernels are cached independently of the number of edges.

.. autofunction:: eqx.kernels.wigner_D

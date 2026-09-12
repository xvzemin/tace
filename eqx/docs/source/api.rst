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

.. autoclass:: eqx.o2.Activation
   :members: forward

.. autoclass:: eqx.o2.Gate
   :members: forward

.. autoclass:: eqx.o2.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.o2.O3TensorProduct
   :members: forward, forward_local

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
   :members: forward

.. autoclass:: eqx.o2.LocalFrame
   :members: restrict, forward, to_local, to_global

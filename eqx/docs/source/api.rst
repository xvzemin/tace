.. _equivariantx-api:

API Reference
=============

This page documents the public EquivariantX API directly from the installed
implementation. The tensor layouts and representation conventions are
described in :ref:`equivariantx-tutorials`.

O(2) representations
---------------------

Representation metadata defines the flattened ``ir_mul`` feature axis used by
all :mod:`eqx.o2` layers. Iterating over :class:`eqx.o2.Irreps` yields
``(irrep, multiplicity)`` entries.

.. autoclass:: eqx.o2.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.o2.Irreps
   :members:

O(2) layers
------------

These layers accept real tensors with trailing shape ``(irreps.dim,)``.
External weights may carry leading batch dimensions when the corresponding
module is configured without shared internal weights.

.. autoclass:: eqx.o2.Linear
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.o2.Activation
   :members: forward

.. autoclass:: eqx.o2.Gate
   :members: forward

.. autoclass:: eqx.o2.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.o2.AsymmetricContraction
   :members: forward

O(2) angular and local-frame tools
----------------------------------

Circular harmonics operate directly in two dimensions. ``WignerD`` and
``LocalFrame`` convert global three-dimensional features to edge-aligned local
features and back while retaining flattened ``ir_mul`` storage.

.. autofunction:: eqx.o2.circular_harmonics

.. autoclass:: eqx.o2.CircularHarmonics
   :members: forward

.. autofunction:: eqx.o2.init_edge_rot_mat_quaternion

.. autoclass:: eqx.o2.WignerD
   :members: get_wigner

.. autoclass:: eqx.o2.LocalFrame
   :members: restrict, to_local, to_global

Cartesian O(3) representations
-------------------------------

Cartesian representations store each rank-``l`` tensor in an ambient axis of
size ``3**l`` while tracking its symmetric traceless degrees of freedom.

.. autoclass:: eqx.co3.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.co3.Irreps
   :members:

Cartesian O(3) layers
----------------------

Cartesian layers use the same flattened ``(..., irreps.dim)`` convention as
the O(2) layers. Each entry uses ``ir_mul`` order and is viewed internally as
``(..., ir.dim, mul)``.

.. autoclass:: eqx.co3.Linear
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co3.Activation
   :members: forward

.. autoclass:: eqx.co3.Gate
   :members: forward

.. autoclass:: eqx.co3.TensorProduct
   :members: forward, project_output, weight_view_for_instruction, weight_views

Cartesian O(3) harmonics and tensors
------------------------------------

These utilities construct and project ambient Cartesian tensors while keeping
the public representation axis flattened.

.. autoclass:: eqx.co3.CartesianHarmonics
   :members: forward

.. autofunction:: eqx.co3.project

.. autofunction:: eqx.co3.delta

.. autofunction:: eqx.co3.levi_civita

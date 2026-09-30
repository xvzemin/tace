.. _equivariantx-api-tools:

Tools
=====

Rotations and local frames
--------------------------

Global metadata uses ``e3nn.o3.Irreps``. Both the global and local tensors
passed to ``LocalFrame`` use flattened ``ir_mul`` layout.

.. autofunction:: eqx.o2.rotation_matrix_to_x_axis

.. autofunction:: eqx.o2.rotation_matrix_to_y_axis

.. autofunction:: eqx.o2.rotation_matrix_to_z_axis

.. autoclass:: eqx.o2.WignerD
   :members: forward, forward_packed, matrix_blocks

.. autoclass:: eqx.o2.LocalFrame
   :members: restrict, forward, to_local, to_global

.. autofunction:: eqx.kernels.wigner_D

Cartesian O(3) basis conversion
-------------------------------

.. autoclass:: eqx.co3.ChangeOfBasis
   :members: forward

.. autoclass:: eqx.co3.Projector
   :members: forward

.. autofunction:: eqx.co3.path_matrix

.. autofunction:: eqx.co3.path_normalization

Cartesian O(2) basis conversion
-------------------------------

.. autoclass:: eqx.co2.ChangeOfBasis
   :members: forward

.. autoclass:: eqx.co2.Projector
   :members: forward

.. autofunction:: eqx.co2.path_matrix

Plane projection and restriction
--------------------------------

These operators retain three-dimensional Cartesian indices, with trailing
size ``3**m`` for a rank-``m`` tensor in the transverse plane.

.. autofunction:: eqx.co2.plane_projector

.. autoclass:: eqx.co2.PlaneProjector
   :members: forward

.. autoclass:: eqx.co2.PlanarDetracer
   :members: forward, matrix

.. autoclass:: eqx.co2.TransverseProjector
   :members: forward

.. autoclass:: eqx.co2.Restriction
   :members: forward, inverse

.. autofunction:: eqx.co2.restriction_scale

.. autofunction:: eqx.co2.restriction_matrix

.. autofunction:: eqx.co2.coupling_coefficients

Cartesian tensor decomposition
------------------------------

ICTD retains all irreducible paths, including repeated irreps.
See :ref:`equivariantx-ictd` for layouts and examples.

.. autoclass:: eqx.o3.ICTD
   :members: forward, inverse, path_matrix, project

.. autofunction:: eqx.o3.path_matrices

.. autoclass:: eqx.o2.ICTD
   :members: forward, inverse, path_matrix, project

.. autofunction:: eqx.o2.path_matrices

Graph attention
---------------

.. autofunction:: eqx.conv.graph_softmax

.. autoclass:: eqx.conv.StreamingGraphAttention
   :members: forward

Operator metadata
-----------------

.. autofunction:: eqx.utils.parse_metadata

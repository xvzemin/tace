.. _equivariantx-api-tools:

Tools
=====

Rotations
---------

.. autofunction:: equivariantx.o3.so3_generators

.. autofunction:: equivariantx.o2.rotation_matrix_to_x_axis

.. autofunction:: equivariantx.o2.rotation_matrix_to_y_axis

.. autofunction:: equivariantx.o2.rotation_matrix_to_z_axis

.. autoclass:: equivariantx.o2.WignerD
   :members: forward, forward_packed, matrix_blocks

Local frames
------------

.. autoclass:: equivariantx.o2.LocalFrame
   :members: restrict, forward, to_local, to_global, gather, scatter

For Cartesian frames, see :class:`equivariantx.co2.LocalFrame`.

Cartesian basis conversion
--------------------------

.. autoclass:: equivariantx.co3.ChangeOfBasis
   :members: forward

.. autoclass:: equivariantx.co3.Projector
   :members: forward

.. autofunction:: equivariantx.co3.path_matrix

.. autofunction:: equivariantx.co3.path_normalization

.. autoclass:: equivariantx.co2.ChangeOfBasis
   :members: forward

.. autofunction:: equivariantx.co2.path_matrix

Plane projection and restriction
--------------------------------

See :class:`equivariantx.co2.PlaneProjector`, :class:`equivariantx.co2.PlanarDetracer`, and
:class:`equivariantx.co2.Restriction` for the corresponding modules.

.. autofunction:: equivariantx.co2.plane_projector

.. autofunction:: equivariantx.co2.restriction_scale

.. autofunction:: equivariantx.co2.restriction_matrix

.. autofunction:: equivariantx.co2.coupling_coefficients

Tensor decomposition
--------------------

.. autoclass:: equivariantx.o3.ICTD
   :members: forward, inverse, path_matrix, project

.. autofunction:: equivariantx.o3.path_matrices

.. autoclass:: equivariantx.o2.ICTD
   :members: forward, inverse, path_matrix, project

.. autofunction:: equivariantx.o2.path_matrices

Module utilities
----------------

.. autofunction:: equivariantx.utils.default_dtype

.. autofunction:: equivariantx.utils.copy_model

.. autofunction:: equivariantx.utils.convert_modules

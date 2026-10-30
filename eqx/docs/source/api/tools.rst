.. _equivariantx-api-tools:

Tools
=====

Rotations
---------

.. autofunction:: eqx.o3.so3_generators

.. autofunction:: eqx.o2.rotation_matrix_to_x_axis

.. autofunction:: eqx.o2.rotation_matrix_to_y_axis

.. autofunction:: eqx.o2.rotation_matrix_to_z_axis

.. autoclass:: eqx.o2.WignerD
   :members: forward, forward_packed, matrix_blocks

Local frames
------------

.. autoclass:: eqx.o2.LocalFrame
   :members: restrict, forward, to_local, to_global, gather, scatter

For Cartesian frames, see :class:`eqx.co2.LocalFrame`.

Cartesian basis conversion
--------------------------

.. autoclass:: eqx.co3.ChangeOfBasis
   :members: forward

.. autoclass:: eqx.co3.Projector
   :members: forward

.. autofunction:: eqx.co3.path_matrix

.. autofunction:: eqx.co3.path_normalization

.. autoclass:: eqx.co2.ChangeOfBasis
   :members: forward

.. autofunction:: eqx.co2.path_matrix

Plane projection and restriction
--------------------------------

See :class:`eqx.co2.PlaneProjector`, :class:`eqx.co2.PlanarDetracer`, and
:class:`eqx.co2.Restriction` for the corresponding modules.

.. autofunction:: eqx.co2.plane_projector

.. autofunction:: eqx.co2.restriction_scale

.. autofunction:: eqx.co2.restriction_matrix

.. autofunction:: eqx.co2.coupling_coefficients

Tensor decomposition
--------------------

.. autoclass:: eqx.o3.ICTD
   :members: forward, inverse, path_matrix, project

.. autofunction:: eqx.o3.path_matrices

.. autoclass:: eqx.o2.ICTD
   :members: forward, inverse, path_matrix, project

.. autofunction:: eqx.o2.path_matrices

Module utilities
----------------

.. autofunction:: eqx.utils.default_dtype

.. autofunction:: eqx.utils.copy_model

.. autofunction:: eqx.utils.convert_modules

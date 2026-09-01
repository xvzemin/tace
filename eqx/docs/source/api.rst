.. _equivariantx-api:

API Reference
=============

This page documents the public EquivariantX API directly from the installed
implementation. The tensor layouts and representation conventions are
described in :ref:`equivariantx-tutorials`.

O(2) representations
---------------------

.. autoclass:: eqx.o2.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.o2.Irreps
   :members:

O(2) layers
------------

.. autoclass:: eqx.o2.Linear
   :members: forward

.. autoclass:: eqx.o2.Gate
   :members: forward

.. autoclass:: eqx.o2.TensorProduct
   :members: forward

.. autoclass:: eqx.o2.AsymmetricContraction
   :members: forward

O(2) angular and local-frame tools
----------------------------------

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

.. autoclass:: eqx.co3.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.co3.Irreps
   :members:

Cartesian O(3) layers
----------------------

.. autoclass:: eqx.co3.Linear
   :members: forward

.. autoclass:: eqx.co3.Gate
   :members: forward

.. autoclass:: eqx.co3.TensorProduct
   :members: forward, project_output

.. autoclass:: eqx.co3.Layout
   :members: to_grouped, from_grouped

Cartesian O(3) harmonics and tensors
------------------------------------

.. autoclass:: eqx.co3.CartesianHarmonics
   :members: forward

.. autofunction:: eqx.co3.project

.. autofunction:: eqx.co3.delta

.. autofunction:: eqx.co3.levi_civita

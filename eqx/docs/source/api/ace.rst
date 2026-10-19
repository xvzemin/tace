.. _equivariantx-api-ace:

Atomic cluster expansion
========================

Features use flattened ``mul_ir`` layout.
Spherical and Cartesian products retain the same coupling paths and coefficients.
Cartesian products use delta/epsilon contractions and STF projection.

.. autoclass:: eqx.ace.TACE
   :members: forward

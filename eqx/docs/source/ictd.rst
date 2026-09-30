.. _equivariantx-ictd:

Cartesian tensor decomposition
==============================

``eqx.o3.ICTD(rank)`` decomposes a three-dimensional Cartesian tensor;
``eqx.o2.ICTD(rank)`` provides the two-dimensional decomposition.
Both retain all irreducible paths,
including repeated irreps. This differs from ``co2.path_matrix(m)`` and
``co3.path_matrix(l)``, which select only the highest-order STF subspace.

Successive vector couplings construct the normalized path matrices
:math:`\widehat P^{(\mathrm{path})}`. Their concatenation is the orthogonal
``change_of_basis`` matrix :math:`P`. For row-vector inputs,
:math:`h=xP`, :math:`x=hP^{\mathsf T}`, and projection onto one path is
:math:`x\widehat P^{(\mathrm{path})}\widehat P^{(\mathrm{path})\mathsf T}`.
Construction follows the path expansion in
`ICTD, Algorithm 1 <https://www.jmlr.org/papers/v26/25-0134.html>`_
and its `reference implementation <https://github.com/ShihaoShao-GH/ICT-decomposition-and-equivariant-bases>`_.
The two-dimensional version uses real O(2) coupling coefficients.

.. code-block:: python

   import torch
   from eqx import o2, o3

   decomposition = o3.ICTD(rank=2)
   x = torch.randn(8, 3, 3)
   h = decomposition(x.flatten(-2))
   reconstructed = decomposition.inverse(h).reshape_as(x)
   torch.testing.assert_close(reconstructed, x)

   planar = o2.ICTD(rank=2)
   y = torch.randn(8, 4)
   torch.testing.assert_close(planar.inverse(planar(y)), y)

   # Each path is identified by its sequence of intermediate irreps.
   for index, path in enumerate(decomposition.paths):
       matrix = decomposition.path_matrix(index)
       projected = decomposition.project(x.flatten(-2), index).reshape_as(x)

Flatten Cartesian indices into the last axis. Batch and channel dimensions
precede this axis. ``irreps_out`` contains one multiplicity-one entry per
path, without regrouping. At rank two,

.. math::

   d=3:\quad 1o\otimes1o=2e\oplus1e\oplus0e,
   \qquad
   d=2:\quad 1m\otimes1m=2m\oplus0e\oplus0o.

These are the symmetric traceless, antisymmetric, and trace subspaces;
the output order follows the displayed decompositions. Two-dimensional
paths retain the reflection sign of every zero-order intermediate irrep.

The basis is constructed on CPU in float64 and stored as a module buffer.
``dtype`` and ``device`` select its runtime precision and location.
Forward, inverse, and projection use PyTorch matrix multiplication and
support autograd, higher derivatives, and compilation. Casting to a higher
precision regenerates the constants instead of promoting rounded values.

The full change-of-basis matrix stores :math:`d^{2\,\mathrm{rank}}` entries.
Individual square projectors are not stored. For higher ranks,
``eqx.o3.path_matrices(rank)`` and ``eqx.o2.path_matrices(rank)`` yield
rectangular matrices one path at a time without assembling the full matrix.
Use the existing STF-only
``path_matrix`` functions when other irreducible subspaces are not needed.

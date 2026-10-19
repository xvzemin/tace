.. _equivariantx-api-tools:

Tools
=====

Rotations and local frames
--------------------------

|eqx-frames|

Both sides of ``LocalFrame`` use flattened ``ir_mul`` storage.
Convert each irrep entry from ``mul_ir`` before passing features in that layout.
Keeping all orders gives an invertible transformation; truncating ``mmax``
does not.

.. code-block:: python

   import torch
   from eqx import o2
   from e3nn import o3

   irreps = o3.Irreps("4x0e + 4x1o + 4x1e + 4x2e")
   frame = o2.LocalFrame(irreps)
   rotation = o2.WignerD(irreps.lmax, irreps.lmax, method="recursive")
   features = torch.randn(8, irreps.dim)  # Flattened ir_mul.
   wigner, inverse = rotation(torch.randn(8, 3))
   local = frame.to_local(features, wigner)
   output = frame.to_global(local, inverse)
   torch.testing.assert_close(output, features, atol=1e-5, rtol=1e-5)

.. autofunction:: eqx.o3.so3_generators

.. autofunction:: eqx.o2.rotation_matrix_to_x_axis

.. autofunction:: eqx.o2.rotation_matrix_to_y_axis

.. autofunction:: eqx.o2.rotation_matrix_to_z_axis

.. autoclass:: eqx.o2.WignerD
   :members: forward, forward_packed, matrix_blocks

.. autoclass:: eqx.o2.LocalFrame
   :members: restrict, forward, to_local, to_global, gather, scatter

Indexed frame operations
~~~~~~~~~~~~~~~~~~~~~~~~

``LocalFrame.gather`` rotates indexed node features directly into edge frames.
``LocalFrame.scatter`` inverse-rotates and sums at receiving nodes. Both accept
``backend="torch"`` (default) or ``backend="cuda"``. The CUDA implementation
avoids storing gathered global features and inverse-rotated edge messages.
Its multilinear adjoints support repeated differentiation, including derivatives
of the supplied Wigner matrices.

Together with ``Gate(backend="cuda")``, these operations fuse data movement and
pointwise operations around the unchanged ``Linear`` matrix products, following
the execution strategy of `TrimSO(2) <https://arxiv.org/abs/2610.20029>`_.
They preserve O(2) reflection and time-reversal labels; no additional complex
weights are introduced.

.. code-block:: python

   device, backend = "cpu", "torch"  # Use "cuda", "cuda" for fused execution.
   frame = o2.LocalFrame("4x0e + 4x1o + 4x1e").to(device)
   gate = o2.Gate(
       "4x0e + 4x0o", [torch.nn.SiLU(), torch.tanh],
       "8x0e", [torch.sigmoid], "8x1m", backend=backend,
   ).to(device)
   up = o2.Linear(frame.irreps_out, gate.irreps_in).to(device)
   down = o2.Linear(gate.irreps_out, frame.irreps_out).to(device)
   rotation = o2.WignerD(1, 1, method="recursive").to(device)
   nodes = torch.randn(8, frame.input_dim, device=device)
   source, target = torch.randint(8, (2, 24), device=device)
   wigner = rotation.forward_packed(torch.randn(24, 3, device=device))
   local = frame.gather(nodes, wigner, source, backend=backend)
   output = frame.scatter(
       down(gate(up(local))), wigner, target, nodes.size(0), backend=backend,
   )

CUDA execution requires float32 or float64 inputs and the CUDA toolkit.
Local edge features and matrix-product outputs are still materialized.

.. autofunction:: eqx.kernels.wigner_D

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

Tensor decomposition
--------------------

ICTD retains every irreducible path of an arbitrary Cartesian tensor.
The ``co2`` and ``co3`` ``path_matrix`` functions instead select the
highest-order symmetric traceless subspace.

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

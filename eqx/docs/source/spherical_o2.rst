.. _equivariantx-spherical-o2:

Spherical O(2)
==============

``eqx.o2`` provides real O(2) irreps, harmonics, linear maps, gates, and
tensor products. Features use flattened ``ir_mul`` storage:
``(..., ir.dim, mul)`` within each entry. Iteration over ``Irreps`` yields
``(ir, mul)``.

Representations
---------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Irrep
     - Dimension
     - Transformation
   * - ``0ee``, ``0eo``
     - 1
     - Reflection-even scalar; the final letter gives time parity.
   * - ``0oe``, ``0oo``
     - 1
     - Reflection-odd scalar; the final letter gives time parity.
   * - ``1me``, ``1mo``, ...
     - 2
     - Positive-order irrep; the final letter gives time parity.

``0e``, ``0o``, and ``1m`` abbreviate time-even irreps.
``CircularHarmonics`` evaluates harmonics of a two-dimensional vector;
``normalize=False`` gives homogeneous polynomials that are smooth at zero.

Linear maps and gates
---------------------

``Linear`` mixes channels with identical irrep labels. ``UuLinear`` applies
channelwise weights. ``Gate`` combines normalized scalar activations with
scalar-tensor products, including the basis change for odd scalar gates.

.. code-block:: python

   import torch
   from eqx import o2

   irreps = o2.Irreps("4x0e + 2x0o + 4x1m + 4x2m")
   gate = o2.Gate(
       "4x0e + 2x0o", [torch.nn.functional.silu, torch.tanh],
       "8x0e", [torch.sigmoid], "4x1m + 4x2m",
   )
   linear_up = o2.Linear(irreps, gate.irreps_in)
   linear_down = o2.Linear(gate.irreps_out, irreps)
   features = irreps.randn(8, -1)
   output = linear_down(gate(linear_up(features)))

Tensor products
---------------

``TensorProduct`` couples input irreps through explicit instructions.
Each weighted path has independent weights. For example,
``1m`` times ``1m`` decomposes into ``0e``, ``0o``, and ``2m``:

.. code-block:: python

   tp = o2.TensorProduct(
       "4x1m", "1x1m", "4x0e + 4x0o + 4x2m",
       [(0, 0, i, "u1u", True) for i in range(3)],
       internal_weights=False, shared_weights=False,
   )
   x = tp.irreps_in1.randn(8, -1)
   y = tp.irreps_in2.randn(8, -1)
   weights = torch.randn(8, tp.weight_numel)
   output = tp(x, y, weights)

O(3)/O(2) frame conversion
----------------------------

|eqx-frames|

``WignerD`` builds rotations that align a nonzero direction with the y axis.
``LocalFrame`` rotates features, groups local orders, and applies the fixed
reflection-basis change. Restriction preserves time parity:

.. math::

   (\ell,p,t)\downarrow
   =\left(0,p(-1)^\ell,t\right)
   \oplus\bigoplus_{m=1}^{\ell}(m,0,t).

Both sides of ``LocalFrame`` use flattened ``ir_mul`` layout. Convert each
entry from ``mul_ir`` before passing features stored in that layout.

.. code-block:: python

   from e3nn import o3

   irreps = o3.Irreps("4x0e + 4x1o + 4x1e + 4x2e")
   frame = o2.LocalFrame(irreps)
   rotations = o2.WignerD(irreps.lmax, irreps.lmax, method="recursive")
   features = torch.randn(8, irreps.dim)  # Flattened ir_mul.
   vectors = torch.randn(8, 3)
   wigner, wigner_inv = rotations(vectors)
   local = frame.to_local(features, wigner)
   reconstructed = frame.to_global(local, wigner_inv)
   torch.testing.assert_close(reconstructed, features, atol=1e-5, rtol=1e-5)

With all orders retained, the transformation is invertible. Setting
``mmax`` below the largest degree truncates local orders and no longer
gives an exact round trip.

See :ref:`equivariantx-api-o2` for operators and
:ref:`equivariantx-api-tools` for frame and rotation parameters.

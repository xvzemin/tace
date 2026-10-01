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

O(2) grid
----------

``O2Grid`` evaluates flattened ``ir_mul`` features on uniform circular grids
and projects values back by quadrature. Inputs require ``C x 0e + C x 0o``
and ``2C`` copies of every positive order through ``mmax``, with ``C > 0``.
Each grid channel contains two coefficient sets. A positive order can be
declared as ``2C x m`` or split into ``C x m + C x m``; combined features
follow the declared irrep order. Time-odd irreps are not supported.

.. code-block:: python

   import torch
   from eqx import o2
   from eqx.nn import PolynomialActivation

   irreps = o2.Irreps("64x0e + 64x0o + 128x1m + 128x2m + 128x3m")
   act = PolynomialActivation("silu", degree=8, bound=3.0)
   grid = o2.O2Grid(irreps, resolution=(act.degree + 1) * irreps.mmax + 1)
   features = irreps.randn(8, -1)
   values = grid(features)  # (8, 64, 2, 28)
   output = grid.from_grid(act(values))
   assert output.shape == features.shape

The two sets may also be passed separately. Each uses ``C`` channels at
every order, with ``0e`` in the first and ``0o`` in the second:

.. code-block:: python

   first = grid.irreps_in1.randn(8, -1)   # 64x0e + 64x1m + 64x2m + 64x3m
   second = grid.irreps_in2.randn(8, -1)  # 64x0o + 64x1m + 64x2m + 64x3m
   values = grid(first, second)          # (8, 64, 2, 28)
   first_out, second_out = grid.from_grid(act(values), split=True)
   assert first_out.shape == first.shape
   assert second_out.shape == second.shape

For a combined positive-order entry, concatenate the two ``(..., 2, C)``
tensors along their channel axis before flattening. To concatenate two
complete feature vectors instead, declare ``irreps_in1 + irreps_in2``:

.. code-block:: python

   separate = o2.O2Grid(
       grid.irreps_in1 + grid.irreps_in2, resolution=grid.resolution
   )
   joined = torch.cat((first, second), dim=-1)
   torch.testing.assert_close(separate(joined), grid(first, second))

In this construction, ``0e`` and ``0o`` occupy the scalar and
pseudoscalar fields, respectively. Positive-order copies fill the scalar
set first, then the pseudoscalar set; the latter uses the fixed inverse
basis change :math:`J^{-1}(h_{+m},h_{-m})=(h_{-m},-h_{+m})`.
The two sampled fields are combined as
:math:`(f_{\mathrm e}+f_{\mathrm o})/\sqrt{2}` and
:math:`(f_{\mathrm e}-f_{\mathrm o})/\sqrt{2}`. Reflection exchanges these
sheets while reversing the angle, so apply the same activation to both.
Reconstruction reverses the sum-and-difference transform and projects onto
the requested irreps.

Pointwise squaring includes antisymmetric couplings between the two sets:
:math:`h^A_{+m}h^B_{-m}-h^A_{-m}h^B_{+m}` contributes to ``0o``. Distinct
coupling paths are summed in the projected output, not retained separately.

The default ``resolution=4*mmax+1`` integrates a cubic pointwise polynomial
followed by projection without aliasing. In general, a degree-:math:`d`
polynomial requires :math:`N>(d+1)m_{\max}` samples. ``PolynomialActivation``
uses a fixed Chebyshev approximation and supports ``silu``, ``gelu``, ``relu``,
``tanh``, ``sigmoid``, and ``softplus``. It approximates the activation on
``[-bound, bound]`` without clipping inputs. The approximation error does not
break equivariance when the grid meets the sampling condition. Outside the
interval, the polynomial may grow rapidly.

The eighth-degree default activation needs a finer grid than the default
cubic sampling rule; set ``resolution`` as in the examples. Applying the
original SiLU directly instead requires a resolution-convergence check.
Finite sampling does not guarantee exact equivariance for non-polynomial
activations. See
`Nonlinearities in Steerable SO(2)-Equivariant CNNs
<https://arxiv.org/abs/2109.06861>`_ and :ref:`equivariantx-api-nn`.

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

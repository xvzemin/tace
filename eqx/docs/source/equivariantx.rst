.. _equivariantx-tutorials:

Tutorials
=========

EquivariantX (``eqx``) provides real :math:`O(2)` operators and conversion
between global :math:`O(3)` features and local :math:`O(2)` frames.

.. note::

   **EquivariantX is under active development.**

   APIs, module names, and behaviors may change without notice, and backward
   compatibility is not guaranteed at this stage.

   A stable release will be published as a separate package.

``eqx.o2``
   General real :math:`O(2)\times\mathbb{Z}_2^T` irreps and operations, plus the specialized
   conversion between global spherical :math:`O(3)` features and local
   :math:`O(2)` frames.

Quick start
-----------

.. code-block:: python

   import torch

   from eqx import o2

   irreps_o2 = o2.Irreps("8x0ee + 4x0oe + 6x1me + 3x2me")

Common tensor conventions
-------------------------

The :mod:`eqx.o2` operators use one flattened feature axis:

.. math::

   (\ldots,\ D_{\mathrm{irreps}}).

Inside every ``(ir, mul)`` entry, values are ordered as ``ir_mul``. The entry
can therefore be viewed directly as

.. math::

   (\ldots,\ d_{\mathrm{irrep}},\ \mathrm{mul}).

Linear, activation, gate, tensor-product, and local-frame modules use this
flattened ``ir_mul`` representation. Circular harmonics produce the same
layout. Asymmetric contraction accepts a sequence of independent flattened
inputs, one for each correlation order. Inputs, internal parameters, and
external weights use real floating-point dtypes unless an API states
otherwise.

Real O(2) with time reversal
-----------------------------

Representations
~~~~~~~~~~~~~~~

The real irreps of :math:`O(2)\times\mathbb{Z}_2^T` carry a spatial
reflection parity and an independent time-reversal parity:

``0ee``, ``0eo``
   One-dimensional scalars even under reflection. The final letter is the
   time parity.

``0oe``, ``0oo``
   One-dimensional pseudoscalars odd under reflection.

``1me``, ``1mo``, ``2me``, ``2mo``, ...
   Two-dimensional real irreps. The components are stored as the cosine-like
   and sine-like pair for positive order :math:`m`. ``m`` denotes the spatial
   representation and the final letter denotes time parity.

The legacy names ``0e``, ``0o``, ``1m``, ``2m``, and so on remain accepted
and denote their time-even counterparts.

For a rotation by :math:`\theta`, a positive-order block transforms as

.. math::

   D_m(\theta)=
   \begin{pmatrix}
   \cos(m\theta)&-\sin(m\theta)\\
   \sin(m\theta)& \cos(m\theta)
   \end{pmatrix}.

Reflection distinguishes ``0ee`` and ``0oe`` and acts on the two-dimensional
blocks through the reflection component of :math:`O(2)`. Time reversal
multiplies an irrep by its time parity :math:`t=\pm1` without changing its
spatial components.

The tensor-product rules are

.. list-table:: Real O(2) tensor products
   :header-rows: 1
   :widths: 30 70

   * - Inputs
     - Outputs
   * - ``0ee x a``
     - ``a``
   * - ``0oo x 0oo``
     - ``0ee``
   * - ``0oo x 1mo``
     - ``1me``
   * - ``1mo x 2me``
     - ``1mo + 3mo``
   * - ``2mo x 2mo``
     - ``0ee + 0oe + 4me``

For every product, time parity follows

.. math::

   t_{\mathrm{out}}=t_1t_2.

Linear, Gate, and TensorProduct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.Linear` connects only identical irreps and uses a dense
input-output multiplicity matrix for every instruction. Missing output irreps
are returned as differentiable zeros; only ``0ee`` can receive a bias.

:class:`eqx.o2.Gate` applies an arbitrary scalar activation to ``0ee``. An
activation acting on a scalar that is odd under reflection or time reversal
must itself be even or odd. Gate products multiply both reflection and time
parities.

:class:`eqx.o2.TensorProduct` supports three channel contracts:

.. list-table:: Tensor-product path modes
   :header-rows: 1
   :widths: 18 32 50

   * - Mode
     - Channel constraint
     - Weight layout per path
   * - ``u1u``
     - :math:`C_2=1`, :math:`C_3=C_1`
     - One weight per output channel
   * - ``uuu``
     - :math:`C_1=C_2=C_3`
     - One weight per matched channel
   * - ``uvw``
     - No equality constraint
     - Dense :math:`C_1\times C_2\times C_3` weights

A minimal nonlinear block is

.. code-block:: python

   irreps = o2.Irreps("4x0ee + 2x0oe + 3x1me + 2x2me")
   irreps_scalars = o2.Irreps("4x0ee + 2x0oe")
   irreps_gated = o2.Irreps("3x1me + 2x2me")
   irreps_gates = o2.Irreps("5x0ee")

   nonlinearity = o2.Gate(
       irreps_scalars,
       [torch.nn.SiLU(), torch.nn.Tanh()],
       irreps_gates,
       [torch.nn.Sigmoid()],
       irreps_gated,
   )
   linear_up = o2.Linear(
       irreps,
       nonlinearity.irreps_in,
       biases=True,
   )
   linear_down = o2.Linear(
       nonlinearity.irreps_out,
       irreps,
       biases=False,
   )

   node_feats = torch.randn(32, irreps.dim)
   node_feats = linear_down(nonlinearity(linear_up(node_feats)))

Circular harmonics
~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.CircularHarmonics` constructs native two-dimensional angular
features. With ``normalize=True`` the output depends only on direction. With
``normalize=False``, order :math:`m` is homogeneous of degree :math:`m` in the
input vector. ``time_reversal=True`` declares a time-odd input and assigns
time parity :math:`(-1)^m` to order :math:`m`.

.. code-block:: python

   vectors_2d = torch.randn(128, 2)
   harmonics = o2.CircularHarmonics(mmax=3, normalize=True)
   edge_attrs = harmonics(vectors_2d)

   assert harmonics.irreps_out == o2.Irreps("0ee + 1me + 2me + 3me")
   assert edge_attrs.shape == (128, harmonics.irreps_out.dim)

For a time-odd two-dimensional vector:

.. code-block:: python

   magnetic_harmonics = o2.CircularHarmonics(
       mmax=3,
       normalize=True,
       time_reversal=True,
   )
   assert magnetic_harmonics.irreps_out == o2.Irreps(
       "0ee + 1mo + 2me + 3mo"
   )

Global O(3) to local O(2)
~~~~~~~~~~~~~~~~~~~~~~~~~

A directed three-dimensional vector defines a local axis. Restricting an
:math:`O(3)\times\mathbb{Z}_2^T` irrep ``(l, p, t)`` to the
:math:`O(2)\times\mathbb{Z}_2^T` isotropy subgroup gives

.. math::

   (l,p,t)\downarrow
   =\left(0,p(-1)^l,t\right)
   \oplus\bigoplus_{m=1}^{\min(l,m_{\max})}(m,0,t).

Time parity is retained by every local entry. For example, an axial,
time-odd vector restricts as ``1eo -> 0oo + 1mo``.

:class:`eqx.o2.WignerD` constructs the global-to-local and local-to-global
matrices from three-dimensional vectors. :class:`eqx.o2.LocalFrame` applies
those matrices. Its global input and local output both use flattened
``ir_mul`` layout. ``mmax`` may truncate local positive orders while inverse
rescaling preserves the intended variance.

.. code-block:: python

   channels = 64
   lmax = 3
   mmax = 2
   global_irreps = (
       "64x0e + 64x0o + 64x1o + 64x1e + "
       "64x2e + 64x2o + 64x3o + 64x3e"
   )
   edge_index = torch.randint(0, 16, (2, 48))
   edge_vectors = torch.randn(48, 3)

   wigner = o2.WignerD(lmax=lmax, mmax=mmax)
   D, D_inv = wigner.get_wigner(edge_vectors)
   frame = o2.LocalFrame(
       global_irreps,
       lmax=lmax,
       mmax=mmax,
   )
   node_feats = torch.randn(16, frame.global_irreps.dim)

   local_features = frame.to_local(node_feats[edge_index[0]], D)
   global_messages = frame.to_global(local_features, D_inv)

``node_feats`` must already use flattened ``ir_mul`` order inside every O(3)
entry. ``local_features`` follows ``frame.irreps_out`` in the same flattened
order.

Sparse edge tensor products
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.O3TensorProduct` evaluates a feature--spherical-harmonic
CGTP in an edge-aligned frame. At the positive y-axis,

.. math::

   Y^{(l_f)}_{m_f}(\widehat{\boldsymbol y})
   = \sqrt{2l_f+1}\,\delta_{m_f0}

for component-normalized harmonics. The local coupling therefore reduces to

.. math::

   \widetilde z^{(l_o)}_{m_o}
   = \sum_{l_i,l_f,m_i}
       W_{l_i l_f l_o}
       C^{l_o m_o}_{l_i m_i,l_f0}
       \sqrt{2l_f+1}\,
       \widetilde x^{(l_i)}_{m_i}.

In the real basis, only ``m_i = +/- m_o`` can contribute. The module stores
only the nonzero coefficients and evaluates indexed products followed by PyG
sparse summation. Harmonics are neither evaluated nor rotated at runtime.
The input features are rotated into the frame and the output is rotated back.
All original angular paths, weights, and normalization factors are retained;
this is not an unconstrained local linear layer or a magnetic-order truncation.

The second input must be time-even edge spherical harmonics with multiplicity
one. Channelwise ``uvu`` and channel-mixing ``uvw`` paths are supported.
An optional ``harmonic_scale`` supplies one invariant multiplier per degree,
for example to reproduce unnormalized solid harmonics.

.. code-block:: python

   from e3nn import o3
   from eqx import o2
   import torch

   irreps_in = o3.Irreps("8x1o")
   irreps_sh = o3.Irreps("1x1o")
   irreps_out = o3.Irreps("8x0e + 8x1e + 8x2e")
   tensor_product = o2.O3TensorProduct(
       irreps_in, irreps_sh, irreps_out,
       [(0, 0, i, "uvu", True) for i in range(3)],
       internal_weights=False, shared_weights=False,
   )
   edge_vectors = torch.randn(32, 3)
   features = torch.randn(32, irreps_in.dim)  # flattened ir_mul
   D, D_inv = o2.WignerD(2, 2).get_wigner(edge_vectors)
   weights = torch.randn(32, tensor_product.weight_numel)
   output = tensor_product(features, D, D_inv, weights)

The Wigner matrices can be shared across layers. Adjacent identical output
irreps share a rotation across their channels, while the public output retains
the declared irrep-entry order.

Asymmetric contraction
~~~~~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.AsymmetricContraction` contracts independent input features up
to a requested correlation order. All weights are supplied externally.
``algorithm="edge"`` recursively evaluates paths and minimizes coefficient
storage. ``algorithm="node"`` stores generalized CG tensors and evaluates
larger dense contractions, trading memory for node-level speed. Both
algorithms enumerate the same paths. ``path_mode="sum"`` is the default and
accumulates paths with the same output irrep using variance-preserving
normalization. ``path_mode="expand"`` retains every path as an output
multiplicity so that a following :class:`eqx.o2.Linear` performs the
compression.

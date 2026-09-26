.. _equivariantx-tutorials:

Tutorials
=========

EquivariantX (``eqx``) provides PyTorch-native :math:`O(2)` operations,
e3nn-compatible global :math:`O(3)` to local :math:`O(2)` frame conversions,
and CUDA-fused :math:`O(3)`/:math:`O(2)` graph convolutions. O(2)
representations can also carry time-reversal parity.

Installation
------------

EquivariantX currently supports installation from source only and does not
require TACE. A standalone package release is planned once the library is
fully mature.

.. code-block:: bash

   git clone https://github.com/xvzemin/tace.git
   pip install ./tace/eqx

The native operators use PyTorch >= 2.4 on CPU and CUDA. Installation includes
``e3nn>=0.4.4`` for representation and coupling conventions.
PyG and external CUDA extensions are not required.

With e3nn 0.4.x and recent PyTorch, import ``eqx`` before ``e3nn.o3``. The
older release loads packaged CG constants containing Python ``slice`` objects;
EQX allows these only within the import context, leaving ``torch.load`` and
its defaults unchanged. The available angular degrees depend on the installed
e3nn coefficient tables. Global time-odd irreps require the time-reversal
extension; ordinary spatial operations work with upstream e3nn.

Install the optional generated CUDA convolution backend with:

.. code-block:: bash

   pip install './tace/eqx[cuda]'

A CUDA toolkit is required for GPU execution; set ``CUDA_HOME`` if needed.
This requirement applies to fused kernels, not native PyTorch execution on
GPU. The C++ launcher and NVRTC kernels compile lazily and are cached.
Installing or importing EQX does not compile an extension. See
:ref:`equivariantx-convolutions` for the backends supported by each convolution.

Package organization
--------------------

``eqx.o2`` contains the native operators and frame conversions. ``eqx.conv``
contains the four general convolution interfaces: ``O3TensorProductConv``,
``O2O3TensorProductConv``, ``UuO2TensorProductConv``, and
``UvO2TensorProductConv``. Model-specific operators and adapters live in
``eqx.models.tace`` and ``eqx.models.mace``, respectively for TACE
and MACE. The MACE interface converts existing models for ASE and training.
``eqx.kernels`` supplies shared geometry, CUDA compilation, and execution.
Supporting operators remain separate: ``eqx.o3`` provides element-dependent
linear maps and ``eqx.ace`` provides atomic cluster expansions.

Quick start
-----------

.. code-block:: python

   import torch

   from eqx import o2

   irreps = o2.Irreps("8x0ee + 4x0oe + 6x1me + 3x2me")
   linear = o2.Linear(irreps, "4x0ee + 2x1me")
   features = irreps.randn(32, -1)
   output = linear(features)
   assert output.shape == (32, linear.irreps_out.dim)

Tensor layout
-------------

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

Native O(2) operations
----------------------

Operator arguments use ``irreps_in``, ``irreps_out``, ``instructions``,
``internal_weights``, and ``shared_weights``. Tensor products distinguish
``irreps_in1`` and ``irreps_in2``. ``weight_numel`` gives the size of the flat
weight axis; ``weight_views()`` exposes individual instruction weights.
Feature tensors retain EQX's ``ir_mul`` layout throughout.

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
   Two-dimensional real irreps. The components are stored as the cosine
   and sine pair for positive order :math:`m`. ``m`` denotes the spatial
   representation and the final letter denotes time parity.

The names ``0e``, ``0o``, ``1m``, and ``2m`` omit the even time parity.

``Irreps.sort()`` returns ``(irreps, p, inv)``: the sorted representation,
the original-to-sorted entry permutation, and its inverse. Entries are sorted
by order, reflection parity, and time parity, with even parity first.
``simplify()`` merges adjacent identical entries; ``regroup()`` sorts and
then simplifies. These operations change representation metadata, not tensors.

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
must itself be even or odd. Gated outputs follow the tensor-product rules
for the activated gate and the gated irrep.

:class:`eqx.o2.TensorProduct` supports three connection modes:

.. list-table:: Tensor-product connection modes
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

Activations are rescaled so that :math:`\mathbb{E}[\phi(z)^2]=1` for
:math:`z\sim\mathcal{N}(0,1)`. Linear and tensor-product path normalization
uses the declared input variances and unit-variance weight initialization.
``path_normalization="element"`` normalizes by the total number of contributing
input elements; ``"path"`` assigns equal variance to each contributing path.

A gated update can be constructed as follows:

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

A tensor product is configured independently of a convolution. The following
``uuu`` instructions retain all three couplings of two order-one inputs:

.. code-block:: python

   product = o2.TensorProduct(
       "4x1m", "4x1m", "4x0e+4x0o+4x2m",
       [(0, 0, i, "uuu", True) for i in range(3)],
       internal_weights=False,
       shared_weights=False,
   )
   features1 = product.irreps_in1.randn(16, -1)
   features2 = product.irreps_in2.randn(16, -1)
   weights = torch.randn(16, product.weight_numel)
   coupled = product(features1, features2, weights)
   assert coupled.shape == (16, product.irreps_out.dim)

Circular harmonics
~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.CircularHarmonics` constructs two-dimensional angular
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

Asymmetric contraction
~~~~~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.AsymmetricContraction` contracts independent input features up
to a requested correlation order. All weights are supplied externally.
``algorithm="recursive"`` evaluates successive channel-wise tensor products.
``algorithm="dense"`` contracts precomputed generalized Clebsch--Gordan
tensors, using more coefficient storage. Both enumerate the same paths and
accept inputs at any batch size. ``path_mode="sum"`` sums paths to each output
irrep and scales by the inverse square root of the path count.
``path_mode="expand"`` retains paths in the output multiplicity, allowing a
following :class:`eqx.o2.Linear` to mix them.

Global O(3) to local O(2)
--------------------------

A directed three-dimensional vector defines a local axis. Restricting an
:math:`O(3)\times\mathbb{Z}_2^T` irrep ``(\ell, p, t)`` to the
:math:`O(2)\times\mathbb{Z}_2^T` isotropy subgroup gives

.. math::

   (\ell,p,t)\downarrow
   =\left(0,p(-1)^\ell,t\right)
   \oplus\bigoplus_{m=1}^{\ell}(m,0,t).

Time parity is retained by every local entry. For example, an axial,
time-odd vector restricts as ``1eo -> 0oo + 1mo``. Ordinary e3nn irreps are
treated as time-even. Global time-odd irreps require an e3nn version that
supports time-reversal labels.

:class:`eqx.o2.WignerD` constructs the global-to-local and local-to-global
matrices from three-dimensional vectors. :class:`eqx.o2.LocalFrame` applies
those matrices. Its global input and local output both use flattened
``ir_mul`` layout. ``mmax`` may truncate local positive orders while inverse
rescaling preserves the intended variance. Truncation is a projection, not
an invertible change of representation. ``LocalFrame`` derives the required
degree from its irreps and the Wigner layout from matrix shapes. Shared
matrices may cover additional degrees or orders.

``WignerD(method="auto")`` uses direct quaternion polynomials on CUDA for
float32 and float64 inputs, and recursive PyTorch contractions otherwise.
The CUDA method requires the ``cuda`` extra and a CUDA toolkit. Fixed
polynomial coefficients are generated analytically, with no numerical fit.
All degrees are evaluated without degree-to-degree recursion, and higher
derivatives are supported. ``method="recursive"`` explicitly selects the
PyTorch construction; ``method="quaternion"`` requires supported CUDA inputs.
``forward_packed`` returns only the degree blocks, without zero padding or
a separate inverse copy.

By default, ``basis_change=True`` gives positive-order features a uniform
reflection convention. Setting it to ``False`` retains the spherical harmonic
basis without changing the regrouped layout. Channels with different reflection
matrices must then be handled explicitly rather than mixed by standard O(2)
operators. :class:`eqx.o2.O3TensorProduct` uses this setting to contract the
original CG coefficients directly.

.. code-block:: python

   from e3nn import o3

   num_channels = 64
   lmax = 3
   mmax = lmax
   global_irreps = o3.Irreps(" + ".join(
       f"{num_channels}x{l}{p}"
       for l in range(lmax + 1)
       for p in ("e", "o")
   ))
   edge_index = torch.randint(0, 16, (2, 48))
   edge_vectors = torch.randn(48, 3)

   wigner = o2.WignerD(lmax=lmax, mmax=mmax, method="recursive")
   D, D_inv = wigner(edge_vectors)
   frame = o2.LocalFrame(
       global_irreps,
       mmax=mmax,
   )
   # Transpose within each O(3) entry, once at node level.
   e3nn_features = global_irreps.randn(16, -1)
   node_feats = torch.cat([
       e3nn_features[:, s].reshape(16, mul, ir.dim).transpose(-1, -2).flatten(1)
       for (mul, ir), s in zip(global_irreps, global_irreps.slices())
   ], dim=-1)

   local_features = frame.to_local(node_feats[edge_index[0]], D)
   global_messages = frame.to_global(local_features, D_inv)
   torch.testing.assert_close(
       global_messages, node_feats[edge_index[0]], atol=1e-5, rtol=1e-5
   )

``node_feats`` must already use flattened ``ir_mul`` order inside every O(3)
entry. ``local_features`` follows ``frame.irreps_out`` in the same flattened
order. For returning tensors to an e3nn layer, transpose each output entry
from ``(..., ir.dim, mul)`` to ``(..., mul, ir.dim)`` before flattening. Irrep
metadata compatibility does not imply identical feature storage.

The aligned CGTP operator :class:`eqx.o2.O3TensorProduct` preserves the declared
coupling instructions for harmonic edge inputs. See
:ref:`equivariantx-convolutions` for a runnable comparison of its direct and
aligned convolution forms and the corresponding CUDA interfaces.

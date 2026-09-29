.. _equivariantx-cartesian:

Cartesian O(3)
==============

``eqx.co3`` implements Cartesian irreps, harmonics, linear maps, gates, and
tensor products using PyTorch. It does not require a CUDA extension or
TorchScript.

Representations and basis
-------------------------

``Irrep(l, p)`` labels angular degree and spatial inversion parity.
Both natural and unnatural parities are supported. ``Irreps`` iterates over
``(mul, ir)`` entries. Each entry uses ``(..., mul, 3**l)`` storage, flattened
into the final feature axis. Thus ``ir.dim`` is the Cartesian storage size,
while ``ir.spherical_dim`` is the dimension of the symmetric traceless (STF)
subspace. Redundant Cartesian entries are not independent channels.

Let :math:`C_\ell` denote the orthonormal path matrix, with shape
:math:`3^\ell\times(2\ell+1)`. For column vectors,
:math:`X=C_\ell h`, :math:`h=C_\ell^T X`, and the STF projector is
:math:`C_\ell C_\ell^T`. ``ChangeOfBasis`` and ``Projector`` apply these
rectangular matrices without materializing the square projector.

For arbitrary tensors, :ref:`eqx.ICTD <equivariantx-ictd>` retains all
irreducible paths rather than selecting only the STF subspace.

.. code-block:: python

   import torch
   from eqx import co3

   irreps = co3.Irreps("2x0e + 3x1o + 2x2e + 1x2o")
   to_cartesian = co3.ChangeOfBasis(irreps)
   to_spherical = co3.ChangeOfBasis(irreps, inverse=True)
   h = torch.randn(8, to_cartesian.irreps_in.dim)
   x = to_cartesian(h)
   torch.testing.assert_close(to_spherical(x), h)

``CartesianHarmonics`` projects tensor powers of a vector onto the STF
subspace. ``normalize=False`` evaluates homogeneous solid harmonics, including
at the zero vector. ``normalize=True`` first normalizes the input vector.
The ``integral``, ``component``, and ``norm`` conventions refer to orthonormal
irreducible coordinates, not to the redundant Cartesian entries.

.. code-block:: python

   harmonics = co3.CartesianHarmonics([0, 1, 2], normalize=False,
                                      normalization="component")
   y = harmonics(torch.randn(8, 3))

Linear maps and gates
---------------------

``Linear`` mixes channels with identical representation labels and retains
the input/output entry partitions. ``Gate`` uses normalized scalar
activations and scalar-tensor products. Odd scalar activations must have
definite parity. Activations and gates act on projected features.

.. code-block:: python

   gate = co3.Gate("2x0e", [torch.nn.functional.silu],
                   "3x0e", [torch.sigmoid], "2x1o + 1x2e")
   linear_up = co3.Linear(irreps, gate.irreps_in)
   linear_down = co3.Linear(gate.irreps_out, irreps)
   output = linear_down(gate(linear_up(x)))

Tensor products and normalization
---------------------------------

``TensorProduct`` retains the supplied instructions and independent path
weights, including repeated output irreps. It supports ``uvu``, ``uvv``,
``uuu``, ``uuw``, ``uvw``, and ``uvuv`` channel connections. Cartesian
contractions use Kronecker deltas for even :math:`\ell_1+\ell_2-\ell_3`
and one Levi-Civita tensor for odd :math:`\ell_1+\ell_2-\ell_3`.
The odd branch is necessary to retain all allowed couplings.

.. code-block:: python

   tp = co3.TensorProduct(
       "2x1o", "1x1o", "2x0e + 2x1e + 2x2e",
       [(0, 0, 0, "uvu", True),
        (0, 0, 1, "uvu", True),
        (0, 0, 2, "uvu", True)],
       shared_weights=False, internal_weights=False, project=True,
   )
   x = tp.irreps_in1.randn(8, -1)
   y = tp.irreps_in2.randn(8, -1)
   weights = torch.randn(8, tp.weight_numel)
   output = tp(x, y, weights)

Set :math:`S=\ell_1+\ell_2+\ell_3` and
:math:`k=\lfloor(\ell_1+\ell_2-\ell_3)/2\rfloor`. The raw contractions
include :math:`3^{-k/2}` for the delta branch and
:math:`(2\,3^k)^{-1/2}` for the epsilon branch. Their projected coefficients
have magnitude :math:`s_{\ell_1\ell_2\ell_3}` times the unit-norm CG coefficients, where

.. math::

   s_{\ell_1\ell_2\ell_3}^2 =
   \frac{(S+1)!\prod_{a=1}^{3}(S-2\ell_a)!}
        {3^k\prod_{a=1}^{3}(2\ell_a)!}.

Each instruction applies its normalized path weight divided by this scale.
The path sign is fixed during construction from the installed CG phase
convention; no CG contraction is evaluated during the forward pass.
The spherical and Cartesian implementations therefore use identical
trainable weights. Equality is algebraic, not bitwise: contraction order
and floating-point precision affect the last digits.

Deferred projection
-------------------

With ``project=False``, a tensor product returns raw Cartesian tensors.
Projection commutes with neighbor summation and channel-linear maps, so it
can follow both operations. ``Linear(..., output_basis="spherical")`` first
compresses Cartesian channels, then applies :math:`C_\ell^T` to return
spherical features. Use ``Linear(..., project=True)`` when retaining Cartesian
storage instead. Do not pass unprojected features into another tensor product
or nonlinearity.

The native operators expose full Cartesian storage, which grows as
:math:`3^\ell`. The fused convolution below can retain compact storage.
All operators support ordinary autograd, including force training and
higher derivatives.

Fused convolution
------------------

``eqx.conv.CartesianTensorProductConv`` accepts a Cartesian ``TensorProduct``
with ``uvu`` instructions. Its arguments match ``O3TensorProductConv``, but
features use Cartesian dimensions and ``mul_ir`` layout. Supply
``vectors=...`` to evaluate harmonics and their derivatives inside CUDA,
without materializing edge harmonics or tensor-product messages.
Final radial projections use bounded temporary workspaces and are recomputed
in backward. Output paths remain separate. With ``project=False``, aggregate
raw tensors, apply the node-level channel Linear, and then project as above.
With ``symmetric_inputs=True``, a rank-:math:`\ell` input uses
:math:`\binom{\ell+2}{2}` exponent triples. Contracting :math:`k` indices sums
over :math:`\binom{k+2}{2}` triples with integer permutation multiplicities.
Raw outputs retain separate symmetric free-index groups; no symmetrization
is performed between groups. Instructions sharing an output retain only
their common permutation symmetries. Independent path weights are unchanged.
Cartesian inputs are averaged over index permutations, including the
corresponding transpose in backward. ``input_basis="spherical"`` instead
applies the path matrix directly into compact storage on nodes.

``compact_output=True`` retains packed raw outputs after aggregation and
requires ``project=False``. ``eqx.conv.co3.Linear`` consumes these outputs,
mixes channels with the same storage layout, and then applies the transposed
path matrix. The full path-channel tensor is never expanded. The expansion
indices passed to Linear describe storage only and do not alter its weights.

.. code-block:: python

   from eqx.conv.co3 import CartesianTensorProductConv, Linear

   tp = co3.TensorProduct(
       "2x1o", "1x1o", "2x0e + 2x1e + 2x2e",
       [(0, 0, i, "uvu", True) for i in range(3)],
       shared_weights=False, internal_weights=False, project=False,
   )
   conv = CartesianTensorProductConv(
       tp, symmetric_inputs=True, input_basis="spherical", compact_output=True,
       backend="torch",
   )
   linear = Linear(tp.irreps_out, "4x0e + 4x1e", conv.harmonic_output_index)
   h = torch.randn(8, 6)
   edge_index = torch.randint(8, (2, 24))
   rij = torch.randn(24, 3)
   radial = torch.randn(24, 4)
   projection = torch.randn(4, tp.weight_numel)
   message = conv(h, None, radial, projection, edge_index, vectors=rij)
   output = linear(message)

Without compact output, the public result retains flattened ``mul_ir`` order.
Delta/epsilon coefficients and permutation multiplicities remain integers;
their common normalization is applied once through the path factor.
Harmonic indices use the same exponent-triple layout, so CUDA tiles cover
unique entries rather than redundant Cartesian permutations. Only requested
harmonic polynomials are generated. Higher derivatives use
the same transposed contraction rules, including derivatives at zero vectors
when ``normalize=False``.
High-rank contractions are tiled over Cartesian indices to limit register
usage. Tiles sharing a path use the same weight and sum their contributions;
this does not truncate the tensor or remove coupling paths.

In TACE, select ``atomic_basis.type: co3`` and ``TACE_USE_EQX=1``. Existing
spherical models can be converted with ``convert_cgtp(model, "co3")``;
the product basis remains spherical.

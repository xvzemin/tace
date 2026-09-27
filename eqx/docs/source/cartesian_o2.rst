.. _equivariantx-cartesian-o2:

Cartesian O(2)
==============

``eqx.co2`` provides two-dimensional Cartesian tensor operations and a
coordinate-free restriction of three-dimensional tensors. All operations use
PyTorch, including on GPU. No compiled extension is required.

Spherical storage without alignment
-----------------------------------

``SphericalCoupling(l1, l2, l3)`` evaluates the same transverse coupling on
spherical coefficients, without constructing Cartesian tensors or selecting
transverse axes. Inputs and outputs have last dimensions :math:`2\ell_1+1`
and :math:`2\ell_3+1`, respectively.

For a unit direction :math:`\mathbf n`, let :math:`G_{\mathbf n}` be the
rotation generator in the smaller input or output representation. The minimum-degree harmonic
coupling, of degree :math:`\delta=|\ell_1-\ell_3|`, connects matching transverse
orders. A polynomial of :math:`-G_{\mathbf n}^2` supplies their required
weights; odd couplings additionally apply :math:`G_{\mathbf n}`. Generator
actions are placed on the smaller representation using equivariance. Its
coefficients are obtained from the reference-axis coupling matrices in
float64 and stored in a Chebyshev basis. No directional sampling is used.

.. code-block:: python

   import torch
   from eqx import co2

   coupling = co2.SphericalCoupling(3, 2, 3)
   h = torch.randn(16, 7)
   rij = torch.randn(16, 3, requires_grad=True)
   output = coupling(h, rij)
   gradient = torch.autograd.grad(output.square().sum(), rij, create_graph=True)

``eqx.conv.O2O3TensorProductConv`` uses this construction when edge vectors
are supplied. Its CUDA backend retains spherical features, independent path
weights and outputs, and supports force training and higher derivatives.
The generated CUDA code selects between the factored transverse expression
and its direct sparse harmonic contraction by static operation count. Both
expressions avoid alignment. Direction derivatives use the exact harmonic
polynomial extension, with normalization differentiated through PyTorch.
Directions are smooth on the nonzero-vector domain; no alignment chart is used.

Representations and conversion
------------------------------

``Irrep(m, p, t)`` uses the same reflection and time-reversal labels as
``eqx.o2``. Positive orders are two-dimensional irreducible representations.
Their Cartesian storage contains :math:`2^m` entries per channel, restricted
to the symmetric traceless (STF) subspace. Order zero includes both scalars
and pseudoscalars. ``ir.dim`` is the storage size; ``ir.circular_dim`` is
the number of independent coordinates.

``Irreps`` iterates over ``(mul, ir)``. Features use flattened ``mul_ir``
storage, with shape ``(..., mul, 2**m)`` within each entry.
``ChangeOfBasis`` also performs the layout conversion to or from the compact
``eqx.o2`` layout, which uses ``(..., ir.circular_dim, mul)``.

.. code-block:: python

   import torch
   from eqx import co2

   irreps = co2.Irreps("2x0e + 3x1m + 2x2mo")
   embed = co2.ChangeOfBasis(irreps)
   extract = co2.ChangeOfBasis(irreps, inverse=True)
   h = irreps.circular().randn(8, -1)
   x = embed(h)
   torch.testing.assert_close(extract(x), h)

The columns of ``path_matrix(m)`` are orthonormal Cartesian harmonic
tensors. Embedding, extraction, and projection use :math:`C_m`,
:math:`C_m^T`, and :math:`C_m C_m^T`, respectively. ``Projector`` applies
the two rectangular matrices without constructing a square projector.

Operators
---------

``CartesianHarmonics`` evaluates selected orders of a two-dimensional
vector. ``normalize=False`` gives homogeneous polynomials, smooth at zero.
``component``, ``norm``, and ``integral`` normalization refer to independent
circular coordinates, not redundant Cartesian entries. Integral normalization
uses the measure :math:`d\phi` on the unit circle.

``Linear`` mixes channels of identical irreps. ``Activation`` and ``Gate``
reuse the normalized scalar activations and parity rules of ``eqx.o2``.
Odd scalar gates include the fixed quarter-turn map on positive-order
tensors. Gate computations use compact coordinates internally.

``TensorProduct`` uses Cartesian products, contractions, and antisymmetric
contractions. It retains instruction order and independent weights, with
``u1u``, ``uuu``, and ``uvw`` channel connections. Normalization and weight
layout match ``eqx.o2.TensorProduct``.

.. code-block:: python

   tp = co2.TensorProduct(
       "2x1m", "1x1m", "2x0e + 2x0o + 2x2m",
       [(0, 0, i, "u1u", True) for i in range(3)],
       internal_weights=False, shared_weights=False,
   )
   x = tp.irreps_in1.randn(8, -1)
   y = tp.irreps_in2.randn(8, -1)
   weights = torch.randn(8, tp.weight_numel)
   output = tp(x, y, weights)

``project=False`` returns unprojected Cartesian products. Projection may
follow sums and channel-linear maps, but must precede another tensor product
or nonlinearity. ``Linear(output_basis="circular")`` mixes channels before
extracting compact coordinates.

Coordinate-free restriction
---------------------------

``Restriction(l)`` accepts three-dimensional STF tensors with trailing size
:math:`3^\ell` and unit directions :math:`\bm n`. It returns one transverse
tensor for each order :math:`m=0,\ldots,\ell`. These tensors retain global
Cartesian indices, with trailing sizes :math:`3^m`, rather than the
:math:`2^m` storage of ``co2.Irreps``. No transverse axes or Wigner rotation
matrices are constructed.

With :math:`P=I-\bm n\bm n^T`, the normalized restriction is

.. math::

   B_{\ell m}T = a_{\ell m}\operatorname{STF}_{\perp}
   \left[P^{\otimes m}
   \left(T\mathbin{\lrcorner}\bm n^{\otimes(\ell-m)}\right)\right],
   \qquad
   a_{\ell m}^2=2^{m-\ell}\binom{2\ell}{\ell-m}.

It satisfies :math:`\sum_m B_{\ell m}^{\dagger}B_{\ell m}=I` on STF
inputs. ``inverse`` reconstructs the tensor. The transverse projection
uses finite trace-removal formulas; its constants are prepared at
construction time.

Intermediate symmetric tensors use :math:`\binom{m+2}{2}` normalized
coordinates instead of :math:`3^m` repeated entries. Longitudinal contractions
are shared across orders. Packing uses successive normalized symmetrizations
rather than atomic sums over repeated entries. For an STF input, the unprojected transverse tensors
satisfy :math:`\operatorname{tr}^k U_{\ell m}=(-1)^k U_{\ell,m-2k}`;
restriction reuses these lower orders rather than computing their traces again.
Plane transformations contract symmetric index groups, and trace corrections
use nested symmetric products. Ranks one and two use direct formulas.
External tensor shapes and normalization are unchanged.

Exact harmonic tensor products
-------------------------------

``O3TensorProduct`` couples features to unit-direction harmonics of a nonzero edge vector.
Its representation labels are O(3) labels. Input and output storage may be
spherical or Cartesian, both in flattened ``mul_ir`` layout.
The second input is implicit and must contain natural-parity, time-even
harmonics with one channel per entry. It is not an arbitrary second feature
tensor.

For each retained path :math:`\pi=(\ell_1,\ell_2,\ell_3)`,

.. math::

   K_\pi(\bm n)=
   \sum_{m=0}^{\min(\ell_1,\ell_3)}
   c_{\pi m} B_{\ell_3m}^{\dagger}(\bm n)
   J_m^{s_\pi}(\bm n) B_{\ell_1m}(\bm n),
   \qquad s_\pi=(\ell_1+\ell_2+\ell_3)\bmod 2.

Here :math:`J_m` is the normalized transverse rotation generator. The odd
branch vanishes at :math:`m=0`. Coefficients are fixed from the reference-axis
CG tensor, including the harmonic normalization. No coupling paths are
discarded or assigned shared trainable weights.

.. code-block:: python

   from e3nn import o3
   from eqx import co2

   reference = o3.TensorProduct(
       "2x1o", "1x1o", "2x0e + 2x1e + 2x2e",
       [(0, 0, i, "uvu", True) for i in range(3)],
   )
   operator = co2.O3TensorProduct.from_tensor_product(
       reference, input_basis="spherical", output_basis="spherical",
       normalization="component",
   )
   h = reference.irreps_in1.randn(8, -1)
   vectors = torch.randn(8, 3)
   harmonics = o3.spherical_harmonics(
       reference.irreps_in2, vectors, normalize=True,
       normalization="component",
   )
   torch.testing.assert_close(operator(h, vectors), reference(h, harmonics))

The converter also accepts ``co3.TensorProduct``. It copies normalized
instruction weights directly, including variance and path normalization,
and does not renormalize them a second time. The Cartesian and spherical
operators use identical trainable weights. Equality is algebraic; rounding
depends on precision and contraction order.

``forward_scatter`` embeds nodes before gathering, accumulates unprojected
Cartesian messages, then projects on nodes. With Cartesian output and
``project=False``, a subsequent ``co3.Linear`` can compress channels before
projection. Edge-dependent transverse projections cannot be moved after
neighbor summation.
Reconstruction uses nested products with the direction, avoiding a full-rank
Cartesian temporary for every local order. Fixed STF projection remains a
pair of rectangular path-matrix contractions.

Scope and numerical behavior
-----------------------------

All formulas support arbitrary integer degrees, but explicit Cartesian
storage grows exponentially with rank. This implementation establishes
equivalence and conversion; it does not claim a speed advantage over fused
CGTP kernels. Large-degree trace removal can suffer cancellation and should
be validated at the intended dtype. Use float64 for strict comparisons.

For nonzero edge vectors, the construction supports ordinary autograd,
including mixed parameter/position derivatives and higher derivatives.
It uses no angular sampling or FFT approximation. Direction normalization
is undefined at zero; ``O3TensorProduct`` requires nonzero edge vectors,
and ``Restriction`` requires unit directions supplied by the caller.

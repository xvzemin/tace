.. _equivariantx-tutorials:

Tutorials
=========

EquivariantX (``eqx``) provides native :math:`O(2)` operators, e3nn-compatible
:math:`O(3)`/:math:`O(2)` frame conversion, and fused CUDA convolutions.
Time-reversal parity is optional.

Installation
------------

Install from source without installing TACE:

.. code-block:: bash

   git clone https://github.com/xvzemin/tace.git
   pip install ./tace/eqx

Native operations require PyTorch >= 2.4 and e3nn >= 0.4.4. They run on CPU
and GPU without PyG or custom CUDA extensions. For the fused backend, install
the build dependencies and provide a CUDA toolkit:

.. code-block:: bash

   pip install './tace/eqx[cuda]'

Set ``CUDA_HOME`` if the toolkit is not detected. Fused kernels compile on first
use and are cached; installation does not compile them. A standalone package
release is planned.

With e3nn 0.4.x, import ``eqx`` before ``e3nn.o3``. Global time-odd irreps
require the time-reversal e3nn extension.

Representations and layouts
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Irrep
     - Dimension
     - Transformation
   * - ``0ee``, ``0eo``
     - 1
     - Reflection-even scalar; final letter gives time parity.
   * - ``0oe``, ``0oo``
     - 1
     - Reflection-odd scalar; final letter gives time parity.
   * - ``1me``, ``1mo``, ...
     - 2
     - Positive-order real irrep; final letter gives time parity.

``0e``, ``0o``, and ``1m`` abbreviate time-even irreps. Iteration over
``o2.Irreps`` yields ``(ir, mul)``. ``sort()``, ``simplify()``, and ``regroup()``
change metadata, not feature tensors.

Every feature tensor has shape ``(..., irreps.dim)``. Within each entry,
EQX stores ``(..., ir.dim, mul)`` before flattening: the ``ir_mul`` layout.
e3nn stores ``(..., mul, ir.dim)`` before flattening. Transpose within each
entry when crossing this boundary; a reshape alone does not convert layouts.

Native operators
----------------

``Linear`` mixes channels of identical irreps. Only the invariant scalar
``0ee`` can receive a bias.

.. code-block:: python

   import torch
   from eqx import o2

   irreps = o2.Irreps("4x0e + 2x0o + 3x1m")
   linear = o2.Linear(irreps, "2x0e + 2x1m", biases=True)
   features = irreps.randn(16, -1)
   output = linear(features)
   assert output.shape == (16, linear.irreps_out.dim)

``Gate`` combines scalar activations with equivariant tensor gating. Scalars
odd under reflection or time reversal require an even or odd activation;
an even activation changes their output parity. Gated output irreps follow
the tensor-product rules.

.. code-block:: python

   gate = o2.Gate(
       "4x0e + 2x0o", [torch.nn.SiLU(), torch.nn.Tanh()],
       "3x0e", [torch.nn.Sigmoid()],
       "3x1m",
   )
   linear_up = o2.Linear(irreps, gate.irreps_in)
   linear_down = o2.Linear(gate.irreps_out, irreps)
   output = linear_down(gate(linear_up(features)))
   assert output.shape == features.shape

``TensorProduct`` couples two inputs using explicit instructions. This example
retains the three couplings ``1m x 1m -> 0e + 0o + 2m``:

.. code-block:: python

   product = o2.TensorProduct(
       "4x1m", "4x1m", "4x0e + 4x0o + 4x2m",
       [(0, 0, i, "uuu", True) for i in range(3)],
       internal_weights=False,
       shared_weights=False,
   )
   x = product.irreps_in1.randn(16, -1)
   y = product.irreps_in2.randn(16, -1)
   weights = torch.randn(16, product.weight_numel)
   output = product(x, y, weights)
   assert output.shape == (16, product.irreps_out.dim)

.. list-table:: Channel connections
   :header-rows: 1
   :widths: 20 40 40

   * - Mode
     - Constraint
     - Weights per path
   * - ``u1u``
     - :math:`C_2=1,\ C_3=C_1`
     - :math:`C_1`
   * - ``uuu``
     - :math:`C_1=C_2=C_3`
     - :math:`C_1`
   * - ``uvw``
     - Independent channel widths
     - :math:`C_1 C_2 C_3`

Time parity multiplies along a tensor-product path.
Activations are normalized to unit second moment under a standard Gaussian.
Linear and tensor-product normalization uses the declared input variances:
``path_normalization="element"`` counts contributing input elements, whereas
``"path"`` balances contributing paths.

Additional operators are described in :ref:`equivariantx-api`:

* ``CircularHarmonics`` constructs 2D angular features. ``normalize=False``
  gives homogeneous polynomials, and ``time_reversal=True`` assigns time parity
  :math:`(-1)^m`.
* ``AsymmetricContraction`` contracts independent inputs across correlation
  orders. ``path_mode="sum"`` sums paths; ``"expand"`` retains them as channels.

O(3)/O(2) frame conversion
----------------------------

|eqx-frames|

An edge direction defines the alignment axis. Restriction preserves time
parity and separates local orders:

.. math::

   (\ell,p,t)\downarrow
   =\left(0,p(-1)^\ell,t\right)
   \oplus\bigoplus_{m=1}^{\ell}(m,0,t).

``WignerD`` builds rotation matrices; ``LocalFrame`` applies them and performs
the reflection-basis change. Both global and local features use flattened
``ir_mul`` storage. Convert layouts at node level before gathering.

.. code-block:: python

   from e3nn import o3

   torch.set_default_dtype(torch.float64)
   irreps = o3.Irreps("4x0e + 4x1o + 4x1e + 4x2e")
   features = irreps.randn(8, -1)
   features = torch.cat([
       features[:, s].reshape(8, mul, ir.dim).transpose(-1, -2).flatten(1)
       for (mul, ir), s in zip(irreps, irreps.slices())
   ], dim=-1)

   edge_index = torch.randint(0, 8, (2, 24))
   vectors = torch.randn(24, 3)
   frame = o2.LocalFrame(irreps)
   wigner = o2.WignerD(mmax=2, lmax=2, method="recursive")
   packed = wigner.forward_packed(vectors)

   local = frame.to_local(features[edge_index[0]], packed)
   restored = frame.to_global(local, packed)
   torch.testing.assert_close(restored, features[edge_index[0]])

   linear = o2.Linear(frame.irreps_out, frame.irreps_out)
   messages = frame.to_global(linear(local), packed)
   output = features.new_zeros(8, irreps.dim).index_add(
       0, edge_index[1], messages
   )

Packed matrices are shared by both rotation directions. Transpose each output
entry back to ``mul_ir`` before passing it to an e3nn layer.

.. list-table:: Frame options
   :header-rows: 1
   :widths: 35 65

   * - Option
     - Behavior
   * - ``WignerD(method="auto")``
     - Quaternion CUDA construction for float32/float64 CUDA inputs; recursive
       PyTorch construction otherwise.
   * - ``method="recursive"``
     - PyTorch construction, without the optional CUDA backend.
   * - ``method="quaternion"``
     - CUDA-only quaternion construction.
   * - ``LocalFrame(mmax=...)``
     - Truncate local orders. Truncation is a projection, not an invertible map.
   * - ``basis_change=True`` (default)
     - Put positive-order irreps in the same reflection convention.

``basis_change=False`` retains the spherical-harmonic basis. It is used by
``o2.O3TensorProduct`` for the original CG coefficients; do not mix these
unadjusted features using ordinary O(2) operators.

See :ref:`equivariantx-convolutions` for equivalent CGTPs and fused execution.

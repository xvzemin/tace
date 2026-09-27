.. _equivariantx-convolutions:

Fused O(3)/O(2) Convolutions
============================

EQX provides spherical, aligned-frame, and Cartesian convolutions. The CGTP
forms retain the same instructions, independent path weights, and
normalization. Uu and Uv instead parameterize native O(2) operations.

|eqx-convolutions|

Choosing an operator
--------------------

.. list-table::
   :header-rows: 1
   :widths: 34 38 28

   * - Interface
     - Input / operation
     - Backend
   * - ``O3TensorProductConv``
     - O(3) features and arbitrary edge irreps; ``uvu`` CGTP
     - PyTorch / CUDA
   * - ``O2O3TensorProductConv``
     - O(3) features and harmonic edges; transverse or aligned CGTP
     - PyTorch / CUDA
   * - ``CartesianTensorProductConv``
     - Cartesian features and harmonics; ``uvu`` delta/epsilon contractions
     - PyTorch / CUDA
   * - ``UuO2TensorProductConv``
     - Source features and one externally weighted ``UuLinear``
     - PyTorch / CUDA
   * - ``UvO2TensorProductConv``
     - Linear--Gate--Linear; optional edge features and attention
     - PyTorch / CUDA

Cartesian features use flattened ``mul_ir`` storage; the other convolutions
use flattened ``ir_mul`` storage. The interfaces default to CUDA on GPU and
use PyTorch on CPU. Set ``backend="torch"`` to use tensor operations on either
device without loading CUDA extensions.

PyTorch references
------------------

Each algorithm has a native PyTorch forward, selected by ``backend="torch"``.
The reference retains the chosen mathematical formulation, not just an
equivalent output computed by another convolution. Its gradients, including
higher derivatives, are obtained by autograd without custom backward functions.
CUDA uses its own fused derivatives, which can be checked against these
references.

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Algorithm
     - PyTorch reference
   * - O3 CGTP
     - CG contraction with supplied edge features or spherical harmonics
   * - O2 CGTP, ``generator``
     - Minimum-degree CG coupling and a Chebyshev polynomial of the rotation generator
   * - O2 CGTP, ``cg``
     - Full CG contraction with directional spherical harmonics
   * - O2 CGTP, ``wigner``
     - Wigner-D rotation, order-zero CG contraction and inverse rotation
   * - Uu O2
     - Wigner rotations and UuLinear, or transformed order weights and directional couplings
   * - Uv O2
     - Linear--Gate--Linear in Wigner or transverse representations, including attention
   * - Cartesian CGTP
     - Cartesian harmonic construction, delta/epsilon contractions and optional STF projection

The references retain path outputs, normalization and external weight layouts.
They may materialize edge intermediates. When supplying Wigner matrices from
outside a convolution, construct them with ``o2.WignerD(..., method="recursive")``
to keep the entire reference graph in PyTorch. O2 CGTP's ``method="wigner"``
does this internally for vector inputs.

``graph_softmax(..., fused=False)`` provides native segmented normalization.
``StreamingGraphAttention(..., backend="torch")`` uses the same tiled online
softmax as its replay implementation, but retains the ordinary autograd graph.

Aligned and transverse evaluations
----------------------------------

The three O(2)-based convolutions expose two equivalent evaluations, independent
of the backend:

.. list-table::
   :header-rows: 1
   :widths: 25 35 40

   * - Interface
     - Wigner evaluation
     - Transverse evaluation
   * - ``O2O3TensorProductConv``
     - Rotate, contract local CG coefficients, rotate back
     - Directional couplings with the original CGTP path weights
   * - ``UuO2TensorProductConv``
     - Rotate, apply UuLinear, rotate back
     - Transform order weights and contract directional couplings
   * - ``UvO2TensorProductConv``
     - Rotate, Linear--Gate--Linear, rotate back
     - Restrict to spherical order subspaces, Linear--Gate--Linear, lift

Supply Wigner matrices for the aligned evaluation. Uu and Uv select transverse
evaluation with ``vectors`` and ``wigner=None``; Uv additionally takes
``wigner_inv=None``. For O2 CGTP, ``method`` selects the evaluation of vector
inputs as described below. Transverse evaluation constructs neither alignment
rotations nor transverse axes. Both forms preserve parameters and normalization.

O2 CGTP evaluation methods
~~~~~~~~~~~~~~~~~~~~~~~~~

``O2O3TensorProductConv(..., method="auto")`` selects among complete evaluations
when edge vectors are supplied. A call may override ``method`` without changing
the module or its weights.

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Method
     - Angular evaluation
     - Direction derivatives
   * - ``baseline``
     - Original static choice of generator or sparse CG expressions per path
     - Harmonic polynomial derivatives
   * - ``generator``
     - Minimum-degree coupling and generator polynomials
     - Product-rule derivatives of the same generator expression
   * - ``cg``
     - Sparse CG contraction of harmonic polynomials
     - Derivatives of those polynomials
   * - ``wigner``
     - Wigner-D construction, alignment, order-zero CG contraction, inverse rotation
     - Derivatives through the Wigner matrices

All paths, output multiplicities, radial weights and normalization are retained.
The first three methods do not construct frames. The generator method still
uses fixed CG coefficients for its minimum-degree coupling; it does not replace
the full coupling with a direct CG expression. ``baseline`` is a scheduling
policy over these expressions, not an additional coupling formula. Its PyTorch
reference uses the generator expression; ``cg`` independently checks the direct
contraction. Every explicit method has its own differentiable PyTorch forward.

On CUDA, ``auto`` measures geometry construction, radial projection, contraction
and reduction together. Gradient-enabled inputs include a reverse pass;
training with differentiable vectors also measures the reverse pass of a force
loss. Results are checked against the baseline before selection. Measurements
exclude compilation and are cached by device, precision, channel shapes,
node/edge size ranges and derivative requirements. ``selected_method`` reports
the last autotuned method and ``tuning_results`` contains the measured
milliseconds. CPU and the PyTorch backend use the baseline without timing.

Warm up in eager mode with the intended gradient requirements before
``torch.compile`` or CUDA Graph capture. Compiled and captured calls use the
last measured method, or the baseline if none has been measured. They never
benchmark during tracing or capture. Use an explicit method for controlled
comparisons. Without vectors, supplied Wigner matrices retain their existing
evaluation and are not included in method selection.

.. code-block:: python

   convolution = conv.O2O3TensorProductConv(tp, method="auto")
   # Keep the previous implementation for a controlled comparison.
   convolution.method = "baseline"
   # Other fixed choices: "generator", "cg", "wigner".

TACE's ``o2_cgtp`` CUDA interaction uses the same default selection. To fix its
method, set ``interaction.rejector.eqx_tp.method`` on the corresponding fused
convolution, or select all ``O2O3TensorProductConv`` modules through
``model.modules()``.

CUDA CGTPs support ``uvu`` instructions and float32/float64.
The aligned PyTorch CGTP also supports ``uvw``. Its harmonic input must have
natural spatial parity, even time parity, and one channel per degree.
This equivalence does not extend to arbitrary second-input tensors.

Equivalent CGTPs
----------------

For source node :math:`j` and destination node :math:`i`, the two CGTP forms
evaluate the same sum:

.. math::

   h'_i=\sum_j \operatorname{TP}(h_j,a_{ij};W_{ij}),
   \qquad W_{ij}=z_{ij}W.

Here :math:`z_{ij}` is the radial input and :math:`W` is its final projection.
Paths remain independent unless the supplied instructions explicitly sum them.

The following example runs without a CUDA toolkit. Construct coefficients
in the desired dtype; promoting float32 coefficients later does not recover
float64 accuracy.

.. code-block:: python

   import torch
   from eqx import conv, o2
   from e3nn import o3

   torch.set_default_dtype(torch.float64)
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   irreps_in = o3.Irreps("4x1o")
   irreps_sh = o3.Irreps("1o")
   irreps_out = o3.Irreps("4x0e+4x1e+4x2e")
   instructions = [(0, 0, i, "uvu", True) for i in range(3)]
   options = dict(internal_weights=False, shared_weights=False)

   tp = o3.TensorProduct(irreps_in, irreps_sh, irreps_out, instructions, **options)
   aligned_tp = o2.O3TensorProduct(
       irreps_in, irreps_sh, irreps_out, instructions, **options
   )
   direct = conv.O3TensorProductConv(tp, backend="torch").to(device)
   aligned = conv.O2O3TensorProductConv(aligned_tp, backend="torch").to(device)

   features = torch.randn(8, irreps_in.dim, device=device, requires_grad=True)
   vectors = torch.randn(24, 3, device=device, requires_grad=True)
   edge_index = torch.randint(0, 8, (2, 24), device=device)
   radial = torch.randn(24, 6, device=device, requires_grad=True)
   projection = torch.randn(6, tp.weight_numel, device=device, requires_grad=True)
   harmonics = o3.spherical_harmonics(irreps_sh, vectors, True, "component")
   amplitudes = torch.ones(24, len(irreps_sh), device=device)

   output_o3 = direct(features, harmonics, radial, projection, edge_index)
   output_o2 = aligned(
       features, radial, projection, None, amplitudes, edge_index, features.size(0),
       vectors=vectors,
   )
   torch.testing.assert_close(output_o2, output_o3, atol=1e-10, rtol=1e-10)
   output_o2.square().sum().backward()

To compare the O2 CGTP algorithms without changing the backend or parameters:

.. code-block:: python

   for method in ("generator", "cg", "wigner"):
       output = aligned(
           features, radial, projection, None, amplitudes, edge_index,
           features.size(0), vectors=vectors, method=method,
       )
       torch.testing.assert_close(output, output_o3, atol=1e-10, rtol=1e-10)
       forces = -torch.autograd.grad(output.square().sum(), vectors, create_graph=True)[0]
       weight_gradient = torch.autograd.grad(forces.square().sum(), projection)[0]

Use ``backend="cuda"`` with CUDA operands for fusion. If radial inputs already
contain path weights, supply an empty projection of shape
``(0, weight_numel)``. Harmonic amplitudes remain separate differentiable
inputs, for example for a distance cutoff.

Equivalence uses the installed e3nn coefficient conventions. Different releases
can use different CG signs; do not assume checkpoints share those conventions.

Fusion boundaries
-----------------

.. list-table::
   :header-rows: 1
   :widths: 23 42 35

   * - Operation
     - Fused work
     - Intermediate storage
   * - O3 / O2--O3 CGTP
     - Gather, angular contraction, radial weighting, scatter;
       aligned CGTP also fuses feature rotations
     - No full edge messages; bounded radial-projection workspaces
   * - Uu O(2)
     - Gather, aligned or transverse contractions, channelwise weighting, scatter
     - No full edge messages; bounded radial-projection workspaces
   * - Cartesian O(3)
     - Gather, harmonic polynomials, delta/epsilon contractions, scatter
     - No edge harmonics or messages with vector inputs; bounded radial workspaces
   * - Uv O(2)
     - Restriction, radial multiplication, gate, optional attention, lift/scatter
     - Radial weights and channel-GEMM operands remain explicit
   * - TECE-OAM-RRA
     - Tiled matrix products and CUDA expressions; two-pass attention
     - Attention scores span edges; local features are recomputed per tile

Final radial projections are recomputed during backward where needed.
Preceding radial-MLP layers and node-level ``linear_down`` remain outside
the CGTP kernels. Shared subexpressions do not merge learned path weights.

Derivatives and compilation
---------------------------

.. code-block:: text

   energy forward
        |
        +--> geometry derivatives --> forces / stress
        |                               |
        |                               +--> force-loss parameter gradients
        |
        +--> parameter gradients --> energy-loss training

CUDA contraction programs support recursive derivatives, including the
mixed second derivatives required by force training. Registered operators
provide fake implementations and autograd rules for ``torch.compile``.
Atomic reductions can change floating-point summation order.
For force training, compile the graph including force evaluation. This avoids
requesting double backward through an AOT-compiled energy-only wrapper.

CUDA vector interfaces avoid storing angular intermediates for differentiation:

* ``O3TensorProductConv.forward(..., vectors=...)`` differentiates fixed harmonic
  polynomials. ``normalization`` selects ``component``, ``integral``, or
  ``norm``; ``normalize=False`` selects regular solid harmonics.
* ``CartesianTensorProductConv.forward(..., vectors=...)`` similarly evaluates
  STF harmonic polynomials and their derivatives inside the contraction.
  ``TensorProduct(project=False)`` defers projection until after node aggregation
  and channel compression. See :doc:`cartesian`.
* ``O2O3TensorProductConv.forward(..., vectors=...)`` selects a complete evaluation
  through ``method``. ``generator`` differentiates the generator polynomial;
  ``cg`` uses harmonic polynomials; ``baseline`` uses harmonic derivatives with
  its selected forward expression. These methods require no alignment matrices.
  ``wigner`` constructs and differentiates the Wigner-D matrices. If vectors are
  omitted, supplied matrices are used and differentiated instead.
* ``UuO2TensorProductConv.forward(..., vectors=...)`` preserves every local
  order weight while evaluating in spherical storage without alignment.
  A fixed, degree-wise change of coefficients is applied to the radial
  projection before processing edges. ``forward_wigner`` retains the aligned
  evaluation. With vector inputs, CUDA uses generator-based direction derivatives;
  PyTorch constructs Wigner-D matrices and differentiates the full forward.

``UvO2TensorProductConv`` accepts packed degree matrices from
``WignerD.forward_packed``. Pass ``wigner_inv=None`` to reuse these matrices
for the inverse rotation. This avoids zero-filled dense storage and a separate
inverse matrix; it does not remove frame alignment.

For transverse Uv evaluation, order :math:`m>0` is embedded isometrically in a
degree-:math:`m` spherical tensor of width :math:`2m+1`. The tensor stays in a
two-dimensional subspace, with no explicit Cartesian tensor or selected axes.
Channel maps, even scalar gates, and attention inner products preserve this
subspace. An odd scalar gate additionally applies the rotation generator about
the unit edge direction, divided by :math:`m`, replacing the fixed local
quarter-turn. Order-zero activations are unchanged. CUDA kernels evaluate
restriction/lifting, scalar expressions, and segmented attention; channel
mixing uses PyTorch GEMMs. These larger intermediates can cost more than the
compact Wigner implementation, so frame-free evaluation is not necessarily
faster or smaller. The Wigner path remains available as a reference.

CUDA kernels compile lazily and are cached. Warm up the forward and derivatives
used by the workload before measuring speed. ``EQX_USE_CUDA_GRAPH=1`` enables
optional replay for CGTP and ACE; it is off by default and can increase memory.
Custom-operator registrations and the CUDA runtime remain required after
model export.

Supporting operators
--------------------

.. list-table::
   :header-rows: 1
   :widths: 42 58

   * - Module
     - Role
   * - ``eqx.ace.TACE``
     - Atomic cluster expansion, separate from convolution
   * - ``eqx.o3``
     - Element-dependent Linear, MoE Linear, and Gate in ``mul_ir`` layout
   * - ``eqx.conv.graph_softmax``
     - Destination-wise normalization of ``(edges, heads)`` scores
   * - ``eqx.models.tace.tece_oam_rra``
     - TECE-OAM-RRA interaction and bilinear ACE fusion
   * - ``eqx.models.mace``
     - Conversion of existing MACE models
   * - ``eqx.models.nequip``
     - NequIP interaction conversion
   * - ``eqx.models.sevennet``
     - SevenNet convolution conversion
   * - ``eqx.models.prophet``
     - Prophet spatial convolution conversion

TACE owns its model definition and checkpoint migration. TECE-OAM-RRA fusion
stores attention scores, recomputes bounded edge tiles, and reduces shared
parameter gradients without per-edge weight matrices.
``StreamingGraphAttention`` is also available for tiled callbacks; those live
Python callbacks are not standalone AOTI artifacts.

See :doc:`models` for model conversion, training and ASE usage.

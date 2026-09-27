.. _equivariantx-convolutions:

Fused O(3)/O(2) Convolutions
============================

EQX provides four convolution interfaces. The two CGTP forms retain the same
instructions, independent path weights, and normalization. Uu and Uv instead
parameterize native O(2) operations.

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
     - O(3) features and spherical-harmonic edges; aligned CGTP
     - PyTorch / CUDA
   * - ``UuO2TensorProductConv``
     - Source features and one externally weighted ``UuLinear``
     - PyTorch / CUDA
   * - ``UvO2TensorProductConv``
     - Linear--Gate--Linear; optional edge features and attention
     - CUDA

Convolution features use flattened ``ir_mul`` storage. The first three
interfaces default to CUDA on GPU and use PyTorch on CPU. Set
``backend="torch"`` to select their reference implementation on either device.
For a native Uv reference, compose ``LocalFrame``, ``Linear``, and ``Gate``.

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
   irreps_in = o3.Irreps("4x1o")
   irreps_sh = o3.Irreps("1o")
   irreps_out = o3.Irreps("4x0e+4x1e+4x2e")
   instructions = [(0, 0, i, "uvu", True) for i in range(3)]
   options = dict(internal_weights=False, shared_weights=False)

   tp = o3.TensorProduct(irreps_in, irreps_sh, irreps_out, instructions, **options)
   aligned_tp = o2.O3TensorProduct(
       irreps_in, irreps_sh, irreps_out, instructions, **options
   )
   direct = conv.O3TensorProductConv(tp, backend="torch")
   aligned = conv.O2O3TensorProductConv(aligned_tp, backend="torch")

   features = torch.randn(8, irreps_in.dim, requires_grad=True)
   vectors = torch.randn(24, 3, requires_grad=True)
   edge_index = torch.randint(0, 8, (2, 24))
   radial = torch.randn(24, 6, requires_grad=True)
   projection = torch.randn(6, tp.weight_numel, requires_grad=True)
   harmonics = o3.spherical_harmonics(irreps_sh, vectors, True, "component")
   frame = o2.WignerD(mmax=2, lmax=2, method="recursive")
   packed = frame.forward_packed(vectors)
   amplitudes = torch.ones(24, len(irreps_sh))

   output_o3 = direct(features, harmonics, radial, projection, edge_index)
   output_o2 = aligned(
       features, radial, projection, packed, amplitudes, edge_index, features.size(0)
   )
   torch.testing.assert_close(output_o2, output_o3, atol=1e-10, rtol=1e-10)
   output_o2.square().sum().backward()

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
     - Gather, rotations, channelwise weighting, scatter
     - No full edge messages; bounded radial-projection workspaces
   * - Uv O(2)
     - Rotations, radial multiplication, gate, optional attention, scatter
     - Radial weights and local GEMM operands remain explicit
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

Transposed contraction programs support recursive derivatives, including the
mixed second derivatives required by force training. Registered operators
provide fake implementations and autograd rules for ``torch.compile``.
Atomic reductions can change floating-point summation order.

Two geometry interfaces avoid differentiating stored angular intermediates:

* ``O3TensorProductConv.forward(..., vectors=...)`` differentiates fixed harmonic
  polynomials. ``normalization`` selects ``component``, ``integral``, or
  ``norm``; ``normalize=False`` selects regular solid harmonics.
* ``O2O3TensorProductConv.forward(..., vectors=...)`` uses analytic angular derivatives
  with matching cached Wigner matrices. Construct them with
  ``eqx.kernels.wigner_D(frame, vectors.detach())``. If vectors are omitted,
  gradients are taken with respect to the matrix operands instead.

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

TACE owns its model definition and checkpoint migration. TECE-OAM-RRA fusion
stores attention scores, recomputes bounded edge tiles, and reduces shared
parameter gradients without per-edge weight matrices.
``StreamingGraphAttention`` is also available for tiled callbacks; those live
Python callbacks are not standalone AOTI artifacts.

MACE models
-----------

Install ``'./tace/eqx[mace,cuda]'`` from the parent of the cloned repository.
The adapter supports spatial RealAgnostic MACE interactions, not magnetic
models or all-even SO(3)-only harmonic inputs.

Given an existing model:

.. code-block:: python

   from eqx.models.mace import convert_mace_to_eqx
   from mace.calculators import MACECalculator

   model = convert_mace_to_eqx(model)
   calculator = MACECalculator(
       models=model, device="cuda", default_dtype="float32"
   )

Conversion replaces interaction tensor products and their final radial
projections, retaining the other model operations and all coupling paths.
``enable_cueq=True`` converts the remaining supported operations through MACE;
do not request a second conversion in the calculator.

For training, convert before constructing the optimizer and call the model
with ``training=True`` for force losses. Use a separate model instance for ASE,
whose calculator disables parameter gradients. Load original checkpoints before
conversion; converted state dictionaries have different parameter names.

See :ref:`equivariantx-api` for signatures and supported interaction classes.

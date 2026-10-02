.. _equivariantx-convolutions:

Fused Convolutions
==================

CGTP convolutions preserve the supplied instructions, independent path weights
and normalization. Uu and Uv convolutions parameterize spherical O(2) maps.

|eqx-convolutions|

Interfaces
----------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Interface
     - Operation
   * - ``O3TensorProductConv``
     - O(3) features coupled to edge irreps through ``uvu`` CGTPs
   * - ``O2O3TensorProductConv``
     - Equivalent harmonic CGTPs through aligned or transverse evaluation
   * - ``CartesianTensorProductConv``
     - Cartesian delta/epsilon contractions with optional deferred projection
   * - ``UuO2TensorProductConv``
     - Source features, externally weighted UuLinear, and aggregation
   * - ``UvO2TensorProductConv``
     - Linear--Gate--Linear with optional edge representations and attention
   * - ``EceO2TensorProductConv``
     - Radial UuLinear--Linear--edge expansion--Linear and aggregation

Each interface defaults to CUDA on GPU and PyTorch on CPU.
``backend="torch"`` selects PyTorch operations on either device,
without custom backward functions or CUDA extensions. The CUDA backend
supports float32, float64 and higher derivatives, including force training.

Radial projections use tiled matrix products in the CGTP and Uu convolutions.
Fused edge programs also use this schedule for radial input widths above 128;
64- and 128-channel projections retain their cooperative CUDA evaluation.
In the matrix-product schedule, parameter gradients are reduced without
per-edge outer products. Temporary tiles are released after their last use and are
recomputed for higher derivatives instead of being retained for every edge.

Spherical features use flattened ``ir_mul`` storage. Cartesian features use
flattened ``mul_ir`` storage, except for explicitly requested compact output.
See :doc:`cartesian` for Cartesian storage and projection.

Equivalent CGTPs
----------------

For source node :math:`j` and destination node :math:`i`, both spherical
CGTP interfaces evaluate

.. math::

   h'_i=\sum_j \operatorname{TP}(h_j,a_{ij};W_{ij}),
   \qquad W_{ij}=z_{ij}W.

Here :math:`z_{ij}` is the radial input and :math:`W` its final projection.
Paths are summed only where the instructions specify the same output.

``O3TensorProductConv`` supports arbitrary edge irreps and ``uvu``
connections. ``O2O3TensorProductConv`` requires natural-parity, time-even
harmonics with one channel per irrep entry. It supports ``uvu`` in CUDA and
additionally ``uvw`` in PyTorch; it does not replace a tensor product with
arbitrary learned second inputs.

This example uses only PyTorch. Construct coefficients in the intended dtype;
casting float32 coefficients to float64 does not restore their precision.

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

If radial inputs already contain path weights, supply a projection of shape
``(0, weight_numel)``. Harmonic amplitudes are separate differentiable
inputs, for example for a distance cutoff. Weight equivalence follows the
installed e3nn CG convention; different coefficient conventions may require
checkpoint conversion.

O2 CGTP methods
---------------

The ``method`` parameter selects how vector inputs are evaluated. Each
explicit method has its own PyTorch reference and CUDA implementation.

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Method
     - Evaluation
   * - ``generator``
     - Minimum-degree CG coupling and a Chebyshev polynomial of the rotation generator
   * - ``recurrence``
     - Minimum-degree coupling and a fixed-parity CG recurrence
   * - ``cg``
     - Sparse CG contraction with directional harmonic polynomials
   * - ``wigner``
     - Rotation, order-zero CG contraction and inverse rotation
   * - ``baseline``
     - Static per-path generator/CG selection in CUDA; generator evaluation in PyTorch
   * - ``auto`` (default)
     - Measured CUDA selection; baseline in PyTorch

All methods preserve paths, normalization and weights. Only ``wigner``
constructs alignment matrices. Generator and recurrence methods still use CG
coefficients for the minimum-degree coupling.

.. code-block:: python

   for method in ("generator", "recurrence", "cg", "wigner"):
       output = aligned(
           features, radial, projection, None, amplitudes, edge_index,
           features.size(0), vectors=vectors, method=method,
       )
       torch.testing.assert_close(output, output_o3, atol=1e-10, rtol=1e-10)
       forces = -torch.autograd.grad(output.square().sum(), vectors, create_graph=True)[0]
       weight_gradient = torch.autograd.grad(forces.square().sum(), projection)[0]

CUDA autotuning includes geometry, radial projection, contraction and reduction.
It checks results against the baseline, excludes compilation from timing, and
includes derivatives when requested. Training with differentiable vectors also
measures a force-loss backward pass. ``selected_method`` reports the result;
``tuning_results`` stores timings by device, precision, shape range and
derivative requirements.
All candidates compile before timing, which can be expensive at high degrees.
Choose an explicit method to avoid compiling unused candidates.

Warm up in eager mode before ``torch.compile`` or CUDA Graph capture.
Tracing and capture use the last selected method, or the baseline if none has
been selected. They do not run autotuning. An explicit ``method`` disables
selection. When vectors are omitted, supplied Wigner matrices are used
directly, without autotuning.

Spherical O(2) convolutions
----------------------------

Uu applies a single externally weighted channelwise Linear to source features.
Uv mixes source, target and optional edge features using Linear--Gate--Linear.

Both support aligned and transverse evaluation. Supply Wigner matrices for
the aligned form. For the transverse form, supply ``vectors`` and
``wigner=None``; Uv also takes ``wigner_inv=None``.
No alignment matrices or transverse axes are constructed in the latter form.

Uv transverse features of order :math:`m>0` occupy a two-dimensional
subspace of a degree-:math:`m` spherical representation. Linear maps and
scalar gates preserve this subspace. Odd scalar gates use the rotation
generator about the edge direction divided by :math:`m`.
These redundant intermediates can cost more than compact aligned features.

For aligned evaluation, packed matrices from ``WignerD.forward_packed``
avoid zero-filled dense storage. In Uv, ``wigner_inv=None`` reuses the
packed matrices in the inverse direction. For an entirely PyTorch reference,
construct external matrices with ``WignerD(method="recursive")``.

Fusion and derivatives
----------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Operator
     - Fusion boundary
   * - O3 / O2--O3 CGTP
     - Gather, angular coupling, radial weighting and scatter; no full edge messages
   * - Cartesian CGTP
     - Delta/epsilon contractions and scatter; vector inputs also fuse harmonic construction
   * - Uu
     - Gather, frame or transverse contractions, channelwise weighting and scatter
   * - Uv
     - Frame operations, radial scaling, gates and aggregation; channel GEMMs remain explicit
   * - TECE-OAM-RRA
     - Tiled interaction and bilinear ACE fusion; attention scores remain explicit

CGTP radial projections use bounded temporary workspaces, recomputed during
backward. Preceding MLP layers and node-level Linear remain outside these
kernels. See :doc:`models` for model integration.

CUDA operators register fake implementations and autograd rules for
``torch.compile``. Force training requires mixed parameter/position
derivatives: compile the graph that includes force evaluation, rather than
requesting double backward through an AOT-compiled energy-only wrapper.
Atomic reductions may change floating-point summation order.

Kernels compile on first use and are cached. Warm up the derivatives used by
the workload before timing. ``EQX_USE_CUDA_GRAPH=1`` enables optional replay
for CGTP and ACE; it is disabled by default and can increase memory.
Exported models still require EQX operator registration and the CUDA runtime.

``eqx.ace.TACE`` provides atomic cluster expansion independently of
convolution. ``graph_softmax(..., fused=False)`` and
``StreamingGraphAttention(..., backend="torch")`` provide differentiable
PyTorch attention references. Streaming attention's Python callbacks are not
standalone AOTI artifacts.

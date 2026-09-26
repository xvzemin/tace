.. _equivariantx-convolutions:

Fused O(3)/O(2) Convolutions
============================

EQX separates the mathematical operator from its execution backend. CGTP
convolutions preserve the supplied instructions, path normalization, weight
ordering, and output multiplicities. Repeated output irreps are not merged
unless the supplied instructions already sum into the same output entry.
Native uu/uv convolutions instead use independently parameterized local O(2)
maps; they are not constrained to the CGTP parameterization.

CGTP equivalence refers to the conventions of the installed e3nn version.
Releases can differ in the overall signs of CG tensors. EQX reads the installed
coefficients and fixes rotation-generator signs independently; API compatibility
does not imply that checkpoints are interchangeable across coefficient conventions.

Choosing an operator
--------------------

.. list-table:: Convolution interfaces
   :header-rows: 1
   :widths: 30 40 30

   * - Interface
     - Operation
     - Backend
   * - ``O3TensorProductConv``
     - Sparse O(3) CGTP with ``uvu`` paths
     - PyTorch or CUDA
   * - ``O2O3TensorProductConv``
     - Harmonic-edge O(3) CGTP in aligned frames
     - PyTorch (``uvu``, ``uvw``) or CUDA (``uvu``)
   * - ``UuO2TensorProductConv``
     - Externally weighted channelwise O(2) linear map
     - PyTorch or CUDA
   * - ``UvO2TensorProductConv``
     - O(2) Linear--Gate--Linear, optional edge features and attention
     - CUDA

The first three default to ``backend="cuda"`` and use their PyTorch reference
on CPU. Explicit ``backend="torch"`` also permits the reference on GPU.
``UvO2TensorProductConv`` is a CUDA interface; use the constituent
``LocalFrame``, ``Linear``, and ``Gate`` modules for native PyTorch execution.

All convolution features use flattened ``ir_mul`` layout. The second input
to ``O2O3TensorProductConv`` is restricted to natural-parity, time-even
spherical harmonics with one channel per entry. Their order-zero values in
the aligned frame are included in the coupling coefficients. General edge
representations should instead use ``O3TensorProductConv`` or an appropriate
native O(2) convolution.

An equivalent CGTP example
--------------------------

This example evaluates the same weighted paths with direct and aligned-frame
contractions. It uses the reference backend so it runs without a CUDA toolkit.
Construct coefficients in the intended dtype, rather than constructing in
float32 and only later promoting the module.

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

For fused execution, place operands and modules on CUDA and use
``backend="cuda"``. Supply a projection of shape ``(0, weight_numel)`` if
``radial`` already contains the path weights. Harmonic amplitudes remain
separate differentiable operands, for example for a distance cutoff.

Fusion and memory
-----------------

The O3 and O2-O3 kernels fuse gathering, angular coupling, and target
aggregation. Aligned contractions additionally fuse feature rotations and
reuse rotations across compatible paths. Uu convolutions use the same aligned
contraction engine with the coefficients of ``UuLinear``. These kernels do not
materialize full edge messages. Final radial projections use bounded matrix
workspaces; projected weights are recomputed instead of saved for backward.

Uv convolutions preserve the native Linear and Gate instruction layouts,
including biases, normalized activations, and time-odd gates. CUDA fuses
rotations, basis changes, radial multiplication, gate operations, attention
scores, and inverse rotation with aggregation. Dense channel maps use batched
GEMM. Radial weights and local GEMM operands remain explicit; global edge
messages are not allocated.

Derivatives and geometry
------------------------

Transposed contraction programs implement input and parameter gradients and
can be differentiated recursively. This supports force training and higher
derivatives. Operators provide fake implementations and registered autograd
rules for ``torch.compile``. Floating-point reductions may use atomic additions
and are not generally bitwise deterministic.

``O3TensorProductConv`` can take ``vectors`` instead of explicit harmonic edge
attributes. Fixed Cartesian polynomials then provide direct vector gradients
without storing harmonic cotangents. ``normalization`` selects ``component``,
``integral``, or ``norm`` harmonics; ``normalize=False`` selects regular solid
harmonics.

``O2O3TensorProductConv`` accepts ``vectors`` together with matching packed
alignment matrices. In this mode, matrices are treated as cached values and
direction derivatives use rotation generators. Construct them with
``eqx.kernels.wigner_D(frame, vectors.detach())``; their degrees must match the
frame and their directions must match ``vectors``. Omitting ``vectors``
instead differentiates the matrix operands themselves. CUDA Wigner
construction uses direct quaternion polynomials by default, with a recursive
method available as an alternative.

Fixed angular contractions
--------------------------

CUDA generates contractions for the requested degrees and preserves every
instruction and weight. Compatible paths share scalar products and factored
contractions. Cartesian harmonic derivatives collect equal multi-indices,
rather than enumerating every ordering of derivative directions.
Scalar expressions are shared across reordered monomials and overall sign
changes. Only exact coefficient cancellations are removed; independent paths
and their weights are retained.

With direction inputs, harmonic degrees zero, one and two use direct sparse
contractions, including paths between different feature degrees. Same-degree
dipole and quadrupole maps additionally use finite generator polynomials.
The generators are divided by ``sqrt(l * (l + 1))`` before multiplication;
no dense generator powers are stored. Higher harmonic degrees retain the
aligned contraction. The specialization does not limit the feature degree.

Angular derivatives apply ordered sparse generators in the harmonic space.
Vector adjoints reuse the forward suffix and transposed prefix of this product
instead of evaluating three separate chains. Harmonic cotangents are shared
with amplitude and weight adjoints. Neither Jacobians nor Hessians are stored
on edges. Directions are read from the degree-one Wigner matrix without a
separate copy.

Angular derivative coefficients are constructed in float64 by differentiating
the harmonic factor before coupling it to features. The derivative vector
indices are differentiated as well. This avoids subtracting input- and
output-generator terms whose magnitudes grow with the feature degree.

Shared and specialized kernels
------------------------------

``eqx.conv.graph_softmax`` normalizes scores over incoming edges. Scores have
shape ``(edges, heads)``; optional nonnegative weights have shape
``(edges, 1)`` or ``(edges, heads)``. PyTorch handles CPU inputs and CUDA uses
segmented reductions, including derivatives of the weights.

``StreamingGraphAttention`` additionally evaluates score/value callbacks in
tiles. Callback plans belong to the live Python process and are not portable
standalone AOTI artifacts. Model-specific fusion in
``eqx.models.tace.tece_oam_rra`` instead uses serialized expressions and
explicit parameter operands, without a live model callback. It supplies
TECE-OAM-RRA interaction kernels and ``BilinearACE``; TACE owns the reference
model. MACE adapters are separate in ``eqx.models.mace``.

``eqx.ace.TACE`` remains an independent atomic cluster expansion operator.
``eqx.o3.ElementLinear`` and ``MoEElementLinear`` read external element weights
directly instead of allocating one weight matrix per node. Unlike the
convolution interfaces, these supporting operators use ``mul_ir`` features.

MACE models
-----------

``eqx.models.mace.convert_mace_to_eqx`` converts an existing MACE model
without changing MACE source code. MACE is an optional dependency, imported
only when conversion is requested. Install it separately or use
``pip install './tace/eqx[mace,cuda]'`` from the parent of the cloned repository.
The adapter targets MACE >= 0.3.17 and supports the six spatial RealAgnostic
interaction variants listed in the API reference. Magnetic interactions and
all-even, SO(3)-only edge harmonics are not supported.

Conversion retains the original interaction methods, node linear maps,
product bases and readouts. The last radial projection is moved into
``O2O3TensorProductConv``; its bias, when present, is represented by a constant
radial channel. The unchanged interaction applies the cutoff before the
fused projection. All tensor-product paths and learned parameters are retained.
Feature layouts are converted between the active MACE backend and EQX.

ASE inference
~~~~~~~~~~~~~

.. code-block:: python

   import torch
   from eqx.models.mace import convert_mace_to_eqx
   from mace.calculators import MACECalculator

   device = "cuda" if torch.cuda.is_available() else "cpu"
   model = torch.load("mace.model", map_location=device, weights_only=False)
   model = convert_mace_to_eqx(model.float())
   calculator = MACECalculator(
       models=model, device=device, default_dtype="float32"
   )
   atoms.calc = calculator
   energy = atoms.get_potential_energy()
   forces = atoms.get_forces()
   stress = atoms.get_stress()

Only load checkpoints from trusted sources. Conversion returns a copy by
default; ``inplace=True`` reuses the input model where possible. Existing
cuEquivariance operators are retained. ``enable_cueq=True`` first converts the
remaining operators with MACE's own converter and requires its optional
dependencies. Do not request a second backend conversion in the calculator.

Training and checkpoints
~~~~~~~~~~~~~~~~~~~~~~~~

The returned model keeps the original MACE forward interface. Convert after
loading foundation weights and before creating an optimizer or distributed
wrapper. A force loss requires ``training=True`` to retain the derivative
graph:

.. code-block:: python

   model = convert_mace_to_eqx(model)
   model.train()
   model.requires_grad_(True)  # Omit for intentionally frozen parameters.
   optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
   optimizer.zero_grad(set_to_none=True)
   prediction = model(batch.to_dict(), training=True, compute_stress=True)
   loss = loss_fn(pred=prediction, ref=batch)
   loss.backward()
   optimizer.step()

Use a separate model copy for ASE: its calculator disables parameter gradients
for inference. The converter is a Python interface, not an additional MACE
training CLI option. Select ``backend="torch"`` to use reference contractions
on either device; the default uses fused CUDA kernels on CUDA and PyTorch on CPU.

Moved radial parameters have new state-dict names. Load original MACE weights
before conversion. A converted checkpoint can be saved as a module with
``torch.save(model, path)`` and loaded in an environment containing MACE and
EQX. Alternatively, load its state dict into the same architecture converted
with this interface. Reapplying the converter does not wrap operators again.

Compilation
-----------

Install the optional CUDA dependencies and a CUDA toolkit as described in
:ref:`equivariantx-tutorials`. The model-independent C++ launcher is built
lazily. NVRTC compiles generated CUDA programs on first use and caches binaries
by their static specification. Compilation is not performed at installation
or ordinary import. Forward and derivative programs can require separate
warmup calls.

``EQX_USE_CUDA_GRAPH=1`` enables optional CUDA Graph replay for CGTP and ACE
contractions. It is disabled by default. Capture needs warmup and additional
static buffers, so measure latency and memory for the actual workload. An
outer CUDA Graph capture bypasses this internal replay mechanism.

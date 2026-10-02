TECE
====

TECE stores node features and computes edge cluster expansions within each
interaction. It reuses TACE's energy readouts, scales and shifts, and derivatives
for forces, stress and virials. With ``num_layers=N``, it applies ``N-1`` edge
cluster expansions and one final node ACE, giving ``N`` many-body expansions.

Only tensor node embeddings are accepted: ``spherical_tensor``,
``spherical_tensor_element2``, ``wigner_tensor``, and ``wigner_tensor_element2``.
The Wigner variants lift local ``0e`` features into component-normalized
spherical harmonics using the inverse rotation, without evaluating harmonics
separately. Unnatural-parity initialization is zero when ``parity: true``.

The computation is:

.. code-block:: text

   tensor node embedding -> neighbor sum -> node Gate

   repeat N-1 times:
       source / target node features
       -> local frame and channel concatenation
       -> Linear up (equal channel counts)
       -> symmetric edge cluster expansion
       -> radial UuLinear (retain all radial paths)
       -> Linear down -> global frame -> neighbor sum
       -> neighbor normalization -> Gate -> node Linear
       -> single residual addition

   final node ACE -> projected residual addition -> readout

``num_layers=1`` uses tensor embedding and node ACE without an ECE block.
``atomic_basis.correlation`` controls ECE; ``product_basis.correlation``
controls the final ACE. Intermediate ECE outputs may also contribute to
readout through ``readout_emlp.use_alllayer``.

Two element tables of shape ``(num_elements, weight_numel)`` supply the
expansion coefficients: ``source_weight[Z_source] * target_weight[Z_target]``.
These coefficients do not depend on distance and do not pass through an MLP.
Only the UuLinear weights are produced by a radial MLP. By default they
depend on distance alone. ``atomic_basis.element_dependent: true`` adds
two element tables that multiplicatively modulate these radial weights.
The cutoff multiplies the complete edge message after the expansion.

Gate acts after every neighbor sum, including the initial embedding.
It follows TACE's ``atomic_basis.nonlinear`` (a single value or one per layer),
``gate_m0``, ``scalar_act`` and ``tensor_act`` settings. ``nonlinear: null``
disables it. Each ECE block adds its input once, after the gated node update.
The final ACE uses one projected skip connection. There is no skip inside
the tensor embedding or the edge expansion.

``atomic_basis.algorithm`` chooses the evaluation order without changing the
basis, parameters or output:

* ``recursive`` reuses coupling-path prefixes and uses less working memory.
* ``dense`` contracts external weights with generalized coupling tensors,
  then contracts each feature factor in turn. Its temporary tensor size grows
  exponentially with correlation; CUDA skips zero coupling coefficients.

Call ``model.representation.set_algorithm("dense")`` to switch a constructed
model. Symmetric contractions are the default in the TECE configuration.
``use_asymmetric_contraction: true`` constructs a different model with
independently projected factors; it is not an execution switch.

``TACE_USE_EQX=1`` enables the CUDA edge program. Rotations, the final radial
projection, channel maps, expansion and graph reduction are fused. The radial
MLP's hidden features and Wigner matrices remain materialized. Internal edge
working storage is bounded by the launch size rather than the edge count.
Backward and higher derivatives use generated CUDA adjoint programs, not
activation checkpointing. The PyTorch path remains available without this flag.

From ``example/train/benchmark_configs``:

.. code-block:: bash

   TACE_USE_EQX=1 tace-train -cn 3bpa_tece.yaml

The current model accepts positions and element types. Distributed LAMMPS and
long-range interactions are not implemented.

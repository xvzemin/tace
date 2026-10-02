TECE
====

TECE stores node features and computes edge cluster expansions within each
interaction. It reuses TACE's energy readouts, scales and shifts, and derivatives
for forces, stress and virials. There is no node product module.

Each interaction applies:

.. code-block:: text

   source / target node features
       -> local frame and channel concatenation
       -> radial UuLinear (retain all paths)
       -> Linear up (equal channel counts)
       -> symmetric edge cluster expansion
       -> Linear down -> global frame -> neighbor sum
       -> residual addition

Two element tables of shape ``(num_elements, weight_numel)`` supply the
expansion coefficients: ``source_weight[Z_source] * target_weight[Z_target]``.
Only the radial weights are produced by an MLP. The cutoff multiplies the
complete edge message after the expansion.

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

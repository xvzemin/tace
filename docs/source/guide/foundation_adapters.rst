EquivariantX Foundation adapters
================================

Move the loaded model to an NVIDIA CUDA device before conversion.
Use the TACE planner to inspect a whole-model schedule or the
choices for individual interaction and product layers. Plan and convert when
setting up the calculator or trainer, not inside a molecular dynamics step
or training batch.

Adding a TACE architecture
--------------------------

Start with ``plan(model, task="inference", regional_only=True)`` and repeat
with ``task="training"`` on the training model. A new checkpoint, channel
count or ordering of existing blocks does not in itself require a new
architecture package. Inspect the selected provider for each region.

* Implement model-independent tensor operations in ``equivariantx``.
* Implement reusable TACE interaction, product and readout schedules in
  ``equivariantx_foundation.models.tace``. An unmatched region retains its
  host implementation. Register node interaction and product matchers in
  ``models/tace/regional.py``. Enable ``rank_local`` for a matcher only after
  validating owned outputs and reverse halo exchange against native ML-IAP.
* When a measured bottleneck spans several blocks and depends on the whole
  architecture, add ``models/tace/<architecture>/`` to Foundation. Declare
  its ``TASKS`` and implement ``matches(model, *, task)`` and ``optimize``.
  Match the loaded architecture, not the checkpoint name.
* Keep ASE, AOTI and ML-IAP integration in the shared ``targets`` package.
  Do not copy a model implementation for each deployment target.

A new TACE architecture does not need a new adapter family. Register a new
family in ``equivariantx_foundation.registry.ADAPTERS`` only for a different
host library. If a schedule exports a new custom operator namespace, add its
registration module to ``REGISTRATION_MODULES``. The loader must be able to
register those operators without importing the training model's library.

Share execution policies separately from the operators they call. For example,
receiver tiling, halo discovery and reverse force accumulation live in
``equivariantx_foundation.execution``. The TACE adapter supplies the per-layer
state and energy readout. A whole-model schedule supplies only the layer
calculation that depends on its architecture. LAMMPS owns MPI communication;
a single-device tiling policy does not provide an MPI implementation.

For example, a magnetic TACE model can reuse the regional Wigner-6j
interaction and the joint AOTI training target. It does not need a new
whole-model directory just to train on magnetic forces. The property and its
loss remain defined in TACE. A dedicated schedule is warranted if profiling
shows a bottleneck that cannot be handled by those shared blocks.

For another example, a fully periodic TACE-LES charge head can use the same
reciprocal-grid reduction after either a spherical or a Cartesian node stack.
The selection checks the head, solver state and execution mode, not the number
of layers or channels of the network that produced the charges. This optimizes
the eager property head; it does not by itself make a new deployment target
supported.

Validate force inference, force-loss training and distributed execution
separately. Compare predictions, parameter gradients and an optimizer update
to native TACE. Measure complete steps against native TACE and generic EQX
with the same weights, precision and graphs. A planner match identifies a
code path, not a measured speedup.

Track coverage by architecture and task. For each result, record the selected
regions, native fallbacks, predicted properties, precision and deployment
target. Test an exported artifact on its target, not just the eager model
before export. For energy-domain tiling, the energy must be additive over
centers and the dependency halo must include every message-passing layer.
Do not apply that policy to a global interaction without an explicit
distributed reduction or other exact decomposition.

Use a separate qualification row for each execution contract:

.. list-table::
   :header-rows: 1

   * - Workload
     - What to compare with native TACE
   * - ASE or eager inference
     - Requested properties, including forces and stress when enabled.
   * - Frozen AOTI
     - The reloaded artifact on its supported devices and graph sizes.
   * - Ordinary or joint training
     - Loss, live parameter gradients, optimizer groups, update and resume.
       For a numerically sensitive optimizer, repeat the native step and
       compare updates from identical gradients separately.
   * - Distributed ML-IAP
     - Owned atomic energies and forces after reverse ghost exchange.
       Single-device capacity is not a multi-rank result.

For a new configuration, qualify the smallest change first:

#. Run the native model, generic EQX and the regional plan on the same
   structures and weights. New weights with the same structure normally reuse
   the same selection rules; rebuild frozen packs from the new weights.
#. Profile the entire requested operation: an energy/force/stress call for
   inference, or loss, backward and optimizer step for training. Record
   startup separately from steady-state time and peak memory. A contraction
   benchmark cannot establish a model-level speedup.
#. Add a regional or whole-model schedule only where it improves the target
   workload. Keep the original parameters live during training and verify an
   optimizer step and checkpoint reload. Frozen inference may prepack weights;
   ordinary training must not reuse a frozen copy after an update.

In CI, exercise each distinct execution contract on a small input. Benchmark
representative small and large graphs on the target hardware before enabling
a fast path by default. Record unmatched regions explicitly. A configuration
can run correctly through native fallback without qualifying as accelerated.

.. _foundation-lightning-training:

Joint training with Lightning
-----------------------------

For energy and force training, with or without stress, the 7M, OAM-L and
TECE-OAM-RRA plans can replay a microbatch with a fixed capacity. Replay
also supports single-fidelity regional plans with standard scalar energy
readouts on one GPU or with Lightning NCCL DDP:

* Spherical: streamed CGTP or local UU/UV O(2) interactions with CGTP,
  nonlinear CGTP, MACE symmetric contraction, or a supported second-order
  bilinear-gate product.
* Cartesian: ICTD O(3), or direct planar UU/UV O(2) with 64 channels and
  maximum degree and order two. The planar stack can start with a scalar
  input layer. Use ICTD products or nonlinear ICTD products with
  CUDA support for every factor contraction and projection.

In memory-first IEEE FP32, the direct UV receiver is selected automatically
for two measured layouts: the scalar input stage with 128 radial weights
and 192 scalar hidden features, and the tensor stage with 1,408 radial
weights. Both use the native path below 8,192 directed edge rows or for
distinct input and output rotations. During replay the edge count includes
padding. Automatic selection preserves the host UV implementation under
TF32. To request CUDA explicitly, including for bilinear-gate plans:

.. code-block:: yaml

   callbacks:
     foundation:
       _target_: tace.utils.callbacks.FoundationTrainingCallback
       backend: cuda
       replay_bucket: [128, 8192]

Use this with ``tace-fast-train -cn YOUR_CONFIG``. The first two values
bound the real atoms and directed edges across the **entire microbatch**.
The captured model processes ``max_atoms + 1`` atoms and
``max_edges + max_atoms`` edges, even for a smaller batch. Grouping
similar-size structures and choosing tight bounds therefore matters. An
oversized batch raises an error. With a two-entry capacity, the standard TACE
graph loader uses its per-rank batch size as the structure bound when it
retains the final partial batch and the size is between 2 and the atom
capacity. Other loaders capture again when the structure count changes.
To specify the maximum real structures per microbatch explicitly, use
``replay_bucket: [128, 8192, 8]``. Empty structure slots add no atoms or
edges and do not contribute to the loss.

For mixed sizes, specify fixed capacities using the same key, for example
``replay_bucket: [[32, 1024, 4], [128, 8192, 8]]``. Each rank selects the
fitting capacity with the fewest padded edges (then atoms, then structures).
Selection uses batch shapes and does not reorder the samples. Each capacity
is captured on first use and retains a private CUDA graph until validation
or teardown. Extra capacities reduce padding but cost capture time and GPU
memory. Use a single capacity for uniform workloads.

Replay requires a
supported ``NormalLoss``, FP32 model weights without autocast and one optimizer.
The model and loss must request the same physical properties. With
``loss_property: [energy, forces]``, the replay does not construct a strain
tensor, read any cached stress labels, or reduce stress counts.
Lightning still handles gradient accumulation, optimizer updates and EMA.
Capture is repeated after validation or checkpoint resume. CUDA graphs are
not stored in checkpoints. Do not combine this option with ``training_package``
or the specialized OAM-v2 schedule.

Supply the original directed edges, not a reciprocal-pair map. A paired
Cartesian ICTD E/F plan shares radial values between reciprocal edges in
eager mode. Its replay uses the same trainable parameters but evaluates one
radial row per directed edge to keep the graph shape fixed. The extra work
and capture memory can outweigh the launch savings for loosely filled buckets.

For DDP, use the default NCCL group (``trainer.strategy: ddp``).
Every trainable parameter must participate in the loss. Set the
bucket capacity for **one GPU**, not the sum across GPUs. All ranks must
take the same number of microbatches.
Their graph counts, sizes and stress masks may differ. Without a structure
capacity, changing a rank's graph count recaptures only that rank. DDP
handles parameter synchronization and gradient accumulation. Custom DDP
communication hooks and ``SimpleDDPStrategy`` are not supported for this replay mode.

For the 3BPA ``o3.yaml`` recipe (5 Å cutoff, batch size five), the 500
structures in ``train_300K.xyz`` each have 27 atoms and at most 514 directed
edges. Use ``replay_bucket: [135, 2570, 5]`` to cover any five structures.
By default, ``tace-fast-train`` lets Foundation select the operators for its
training plan. Ordinary ``tace-train`` uses the generic EQX backend.

To keep synchronized parameter gradients in DDP's communication buffers, set
``trainer.strategy`` to a ``lightning.pytorch.strategies.DDPStrategy`` with
``gradient_as_bucket_view: true``. This does not guarantee the same saving
at peak: accumulation microbatches can still allocate separate gradients.

For epoch-based training and resume, use ``max_epochs`` with ``max_steps: -1``.
With Lightning 2.6.6, setting both limits can make an epoch-end checkpoint
skip the next epoch on resume, including without Foundation.

To use a precompiled regional training step with ``tace-fast-train``, export the
same regional model, objective and precision as the training configuration.
For example, for a CUDA model and a labelled local graph batch:

.. code-block:: python

   from equivariantx_foundation import optimize
   from equivariantx_foundation.targets.aoti import export_training

   model = optimize(model.train(), family="tace", task="training",
                    backend="cuda", regional_only=True)
   export_training(model, objective.train(), batch, "train.pt2", dynamic_shapes=True)

.. code-block:: yaml

   callbacks:
     foundation:
       _target_: tace.utils.callbacks.FoundationTrainingCallback
       backend: cuda
       training_package: /absolute/path/to/train.pt2

The first training batch verifies the model and objective against the package.
Subsequent training batches run the packaged loss and gradients. Validation
uses the eager regional model. Lightning retains gradient accumulation, optimizer,
scheduler, EMA and checkpoint handling. To resume, specify the same callback
and the usual Lightning checkpoint. A training package is not a checkpoint.
Keep the module structure, trainable parameter set and precision fixed.
Dynamic packages accept different local atom and edge counts but keep the
number of structures fixed. Choose batches within the exported shape range.
Re-export after changing the model computation or the installed build of its
custom operators. In ordinary DDP, pass local batches to Lightning as usual;
do not apply the standalone gradient reduction in addition to DDP.

For standalone data-parallel joint packages, call ``distributed_training_inputs``
and ``mean_training_result`` on every rank before a replicated optimizer step.
The first helper supports standard energy/force/stress, magnetic-force and
direct-Hessian MSEs, and the HIP recipe's ``mae_energy``, ``mae_forces`` and
``mae_direct_hessian`` with the component mean. Hessian labels are packed by
complete structure. Ranks without a Hessian label supply shape-correct blocks
and zero label weights; their energy and force terms still contribute.
Huber is not supported.

API
---

.. autofunction:: equivariantx_foundation.optimize

.. autofunction:: equivariantx_foundation.models.tace.plan

.. autofunction:: equivariantx_foundation.models.tace.training.supports_training_buckets

.. autoclass:: equivariantx_foundation.models.tace.training.BucketedEFSTrainingStep

.. autofunction:: equivariantx_foundation.targets.ase.replay_calculator

.. autofunction:: equivariantx_foundation.targets.aoti.load

.. autofunction:: equivariantx_foundation.targets.aoti.export_training

.. autofunction:: equivariantx_foundation.targets.aoti.load_training

.. autofunction:: equivariantx_foundation.targets.aoti.training_inputs

.. autofunction:: equivariantx_foundation.targets.aoti.training_batch

.. autofunction:: equivariantx_foundation.targets.aoti.validate_training_model

.. autofunction:: equivariantx_foundation.targets.aoti.distributed_training_inputs

.. autofunction:: equivariantx_foundation.targets.aoti.mean_training_result

.. autofunction:: equivariantx_foundation.targets.lammps.export_distributed

.. autofunction:: equivariantx_foundation.adapters.mace.convert_mace_to_eqx

.. autofunction:: equivariantx_foundation.adapters.nequip.convert_nequip_to_eqx

.. autofunction:: equivariantx_foundation.adapters.sevennet.convert_sevennet_to_eqx

.. autofunction:: equivariantx_foundation.adapters.prophet.convert_prophet_to_eqx

.. autofunction:: equivariantx_foundation.adapters.equflash.convert_equflash_to_eqx

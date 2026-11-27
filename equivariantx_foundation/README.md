# EquivariantX Foundation

Model-specific execution for EquivariantX. Install from the TACE repository
root with `pip install .`. EQX and Foundation are included. Importing Foundation
and inspecting plans do not require CUDA. Run model-specific plans on an NVIDIA
CUDA device. Other adapters require their corresponding host library.

```python
from equivariantx_foundation import optimize
from equivariantx_foundation.models.tace import plan

model.cuda().eval()
report = plan(model, task="inference")
print(report.schedule)
for region in report.regions:
    print(region.module_path, region.provenance, region.provider, region.reason)
model = optimize(model, family="tace", task="inference")
```

Released L, 7M and TECE-RRA architectures have whole-model schedules. Other
TACE models can reuse matching spherical CGTP, local Uu/Uv O(2), spin-orbit
Uv O(2), Cartesian ICTD and planar Cartesian Uu/Uv implementations. Selection is
per layer and depends on its layout, paths, precision and backend. Unmatched
layers keep their original forward, including
any provider already selected in TACE. Requesting optimization when no region
or whole-model schedule applies raises an error. You can re-plan a regional model for
another backend where its installed regions can be converted without losing
state. For a different backend on a whole-model schedule, pass the original
TACE model. Selecting a backend does not override providers owned by the host.
For TACE's MACE-style product, the planner selects the existing EQX symmetric
contraction per instance and leaves an existing CuEq contraction in place.
The report distinguishes a new regional schedule from a retained host
provider. It describes dispatch, not measured speed.
For aligned O(2) CGTP, the two-layer C64 energy/force configuration with
FP32 features, FP64 geometry and the validated 4/17-path layouts uses CUDA
automatically in memory-first IEEE execution. Other layouts keep the host
convolution unless a backend is selected.
For reciprocal radial streaming on this architecture, set `TACE_USE_EQX=1`
and `TACE_MEMORY_FIRST=1` **before building the TACE model**, then call
`optimize`. The plan retains the host's streamed convolutions; applying
Foundation to a native model instead uses the directed, precomputed weights.
An explicit backend selects the precomputed and streamed reductions and
their angular operators together. A mixed host path has no single provider.
Use `region.module_path` for the path from the TACE property adapter.
The report distinguishes node representation, pair interactions and active
host readouts. A host-selected operator is not a new Foundation lowering;
optimizing the interaction stack does not fuse the physical readout.

For HIP-TACE, explicit regional CUDA inference uses a row-wise contraction
for the matched 64-channel spherical pair interaction. Ordinary training
keeps the host pair contraction. Both paths produce the same complete
ordered-pair Hessian; the regional node stack is selected independently.

For periodic charge-only TACE-LES, the memory-first CUDA plan can retain just
one reciprocal hemisphere. Automatic selection supports the standard LES 0.2.0
E/F/stress solver (`N_max=10`, `dl=2`, `sigma=1`, FP32 model and FP64 grid
factors); it does not depend on the node channel width or layer count.
Other eligible grids require an explicit backend. LES still applies the
cell-dependent cutoff on each call. This is a PyTorch grid reduction, not a
custom CUDA kernel. Mixed-boundary, multipole and Born-effective-charge
readouts keep their original grid.

TACE can export field-free, periodic charge-only LES 0.2.0 models with
`tace-export-eval --backend aoti` for frozen graph inference in both spherical
and Cartesian representations. The separate single-rank CUDA joint
training package supports the spherical representation only. Neither path is
an ML-IAP export. See the [LES guide](../docs/source/guide/les.rst).

| TACE architecture | Whole-model schedule | Other configurations |
| --- | --- | --- |
| OAM-L, OAM-7M, TECE-OAM-RRA | Inference, differentiable training and single-rank FP32 CUDA ML-IAP for matching models | Request per-layer matching with `regional_only=True`. |
| OAM-XL-v2 | Specialized joint E/F/S training | Ordinary inference and training use per-layer matching. |
| Other architectures | No dedicated schedule required | Match individual blocks and keep their host implementation where no rule applies. |

ASE, AOTI and ML-IAP are deployment targets, not separate copies of any
architecture. An eager training schedule does not by itself imply support for
CUDA Graph replay, AOTI training or multi-rank ML-IAP.

For spherical CGTP with degree-wise `ejn` or `spb` features, explicit
memory-first CUDA can evaluate the basis and grouped radial MLP on reciprocal
pairs. The convolution still accumulates directed messages. This option
reduces activation memory and is not necessarily faster than materializing
directed weights. It applies to matched inference and training layers,
not to directed or customized radial branches.

For a two-layer C64 spherical CGTP stack with node degree two and edge degree
three, memory-first regional CUDA can evaluate the shared angular basis in a
single row program. Automatic selection requires IEEE matmul; TF32 permits
explicit CUDA selection. FP64 directions retain FP64 evaluation before
conversion to the model dtype. Read the `o3_angular_basis` region in the plan; CPU evaluation,
custom harmonic functions and host-selected convolution providers retain
their own implementation.

For extension and benchmarking criteria, see [Adding a TACE execution schedule](#adding-a-tace-execution-schedule).

The Foundation TACE adapter fuses compatible spherical and ICTD second-order
bilinear products with their coefficient maps on CUDA. Ordinary TACE evaluates
the product and coefficient maps separately. Expert, shared and gate weights
remain trainable. Small training batches gather only the species rows used by
the bilinear contraction. Explicit CUDA can also fuse eligible asymmetric
path-gated products and use CUDA for eligible FP32 multi-block spherical
interaction RMS norms. In memory-first mode (the default), regional CUDA
streams compatible TECE rotary attention on the host's existing frames.
In mixed CGTP/rotary stacks, eligible directed O(3) layers also stream their
radial MLP and reuse the host's spherical harmonics. This does not change
the frame format or the graph's edge order. In AOTI exports, indexed radial
tables and their row maps are inputs to the O(3) and rotary programs; the maps
need not alias the source or target indices. The
speed-first policy (`TACE_MEMORY_FIRST=0` before planning the original
model) retains the host rotary interaction. Compare complete training steps
at the intended batch size.

For explicit memory-first CUDA training, a compatible pure O(2) stack
mixing Uu and Uv layers reuses the packed frame already needed by Uv.
This schedule covers 32 channels, degree and order two, FP32 features
and FP64 geometry. It does not change inference or the automatic policy.

For spherical magnetic Wigner-6j interactions, the regional plan selects the
backend of the host's existing EQX convolution. The coupling paths, radial
weights and magnetic inputs are unchanged. EQX owns the choice between
node recoupling and the standard coupling tree. If the host already uses the
requested provider, the plan reports `host_selected` and leaves the layer
unchanged. For compatible full-order magnetic Uv O(2) convolutions, degrees
one and three use regional CUDA in memory-first mode; speed-first keeps
the host implementation. Explicitly time-odd gates use regional CUDA only
for the supported degree-two FP32 program in memory-first mode.

For non-SOC Uv layers, the FP32 48-channel memory-first plan uses a CUDA
convolution on the host's projected spin scalars and precomputed radial
weights. All three spin paths are retained. A matching three-layer eager
stack shares one packed degree-two spatial frame. Export uses dense
frames. An already enabled host EQX convolution is not replaced. For
other channel widths or FP64, this reduction requires an explicit CUDA backend.

The single-device FP32 memory-first plan also selects eligible projected
Cartesian ICTD factors in the two-layer 64-channel nonlinear 3BPA layout
with correlation three and IEEE matrix multiplication. Factors exceeding
the CUDA program budget keep their native implementation. The gates,
coefficient maps and projections between successive products are unchanged.

For the two-layer C64 Cartesian O(3) E/F layout with a wide scalar radial
MLP (ordinary correlation-three ICTD or the measured nonlinear ICTD layout),
explicit `backend="cuda"` in memory-first mode can compute radial
features on reciprocal bonds and stream the live MLP into the convolution.
The original directed Cartesian angular attributes and output packing are
unchanged. This is an eager memory tradeoff, not a throughput default:
automatic selection, speed-first execution, per-edge force requests and
customized radial calls keep the directed path. Nonlinear gates and the
per-factor selection of CUDA or Torch are independent of this choice.

For ordinary training, use `model.train()` and `task="training"`. Regional
conversion shares the original parameters and preserves the checkpoint keys.
The explicit `joint_training` task is the specialized XL-v2 training entry,
not a prerequisite for training other architectures.
Use `regional_only=True` to request per-layer matching on an eager model
even when its architecture also supports a whole-model schedule.
`tace-fast-train` selects the captured schedule for the matching from-scratch
XL-v2 E/F/S recipe and the usual Foundation planner otherwise.
`+fast_schedule=foundation` selects the latter explicitly.
`tace-train` keeps the host model and its generic EQX/AOTInductor path.
The default `replay_bucket: auto` learns recurring E/F(/S) batch sizes for
supported release models on one GPU or with standard NCCL DDP. Each rank can
replay or run an ordinary step for its own batch. Regional spherical CGTP
also supports automatic recording with two layers, 64 channels, hidden
degree 2 and IEEE FP32 matmul. E/F uses one GPU, edge degree 3 and correlation
3. E/F/S uses one GPU or two NCCL ranks, edge degree 2 and correlation 2 with the
[single-fidelity parity-complete configuration](../docs/source/guide/foundation_adapters.rst).
Its recordings can persist across measured, read-only validation with the
standard graph loader, including the standard device-local EMA callback
after the planner when memory covers its temporary backup. Custom validation
hooks and unsupported EMA configurations release the recordings.
The planner checks memory headroom before
recording. Set `replay_bucket: null`
to retain ordinary Foundation training without graph recording.
For explicit fixed-capacity E/F or E/F/S training, set `replay_bucket` in the
Foundation callback. Supported spherical and Cartesian plans can replay on
one GPU or with NCCL DDP.
On the fast entry, the optional `training_package` on
`tace.utils.callbacks.FoundationTrainingCallback` selects the regional plan
and runs a packaged joint AOTI loss. Export with the same model, loss, regional backend and
precision. The first training microbatch checks the compiled computation
before attaching its gradients. Lightning still handles accumulation, clipping,
EMA, the optimizer and DDP. Do not run a separate gradient all-reduce.
Validation uses the eager model, not the packaged training loss. See
[the training configuration](../docs/source/guide/foundation_adapters.rst).

The other adapter families are `mace`, `nequip`, `sevennet`, `equflash`, and
`prophet`. Ordinary tensor products, convolutions and many-body operators live in
`equivariantx`. Model schedules live in `models/`, host conversions in `adapters/`,
and deployment code in `targets/`. The TACE CLI owns checkpoint loading, ASE graph
construction and ML-IAP entry points. The specialized ML-IAP path uses the local
CUDA rank and FP32.

For a loaded single-model ASE calculator, use
`targets.ase.optimize_calculator(calc, family="tace")`. For ML-IAP, select
`tace-export-lammps --backend eqx-foundation` (capacity) or
`--backend eqx-foundation-throughput` (speed) on one rank. During tiled force
evaluation the energy readout uses only the layers it consumes. For a final-layer
readout, an energy domain reads its final core rows directly, without gathering
earlier descriptors. All message-passing layers still contribute to the forces.
For eager multi-rank ML-IAP, use `--backend eqx-regional-mpi --device cpu`.
Export only serializes the model. Each rank requires a CUDA device at runtime. Eligible spherical CGTP, spherical
Uu/Uv O(2), TECE rotary-attention and Cartesian ICTD convolutions and owned-node
products can use regional schedules. TECE retains all incoming edges, including
ghost senders, but allocates its softmax statistics and output only for owned
receivers. On this target, the memory-first spherical CGTP schedule streams
compatible directed radial MLPs. With `TACE_MEMORY_FIRST=0`, large wide
FP32 projections fuse only the last linear map; small ones precompute edge
weights. TACE retains its ordinary forward and reverse halo exchanges.

To package an exported tensor graph directly, use
`equivariantx_foundation.targets.aoti.export(program, "model.pt2")`. Load it
with `equivariantx_foundation.targets.aoti.load("model.pt2")`; this path
does not require TACE. This flat tensor runner leaves the physical input
contract to the caller. For a frozen TACE graph, `node_type` must contain
valid zero-based indices in the package's element order, not atomic numbers.
A frozen periodic charge-only LES graph requires every PBC flag to be true
and every cell to be finite with a positive determinant. Use TACE's graph
loader if you need its input conversion and validation.

For repeated CUDA inference with the same input sizes, use
`step = load("model.pt2", replay_inputs=tuple(inputs))`, then
`energy, forces, stress = step(*new_inputs)` for a model with these outputs.
Positions, element types and neighbors may change. The captured package uses one
thread and CUDA stream, and returns fresh tensors. Use ordinary `load` if atom
or edge counts change. Replay is opt-in and reserves a CUDA Graph memory pool.
For a TACE graph exported for ASE:

```python
from equivariantx_foundation.targets.ase import replay_calculator

atoms.calc = replay_calculator("model.pt2", device="cuda")
```

For the default dynamic TACE package, the calculator keeps one model and
one capture. If a frame changes shape it uses the same model without replay,
and reuses the capture when the original shape returns. A TACE graph exported
with `--fixed-shapes --sample` accepts only the exact sample input shapes,
with or without replay.

`replay_calculator(..., neighbor_skin=0.5)` also reuses candidate image
pairs between ASE frames. Physical edges are filtered at every step. This
host-side option is independent of CUDA Graph capture and also works in
the ordinary TACE ASE calculator.

For single-device E/F/stress training with an external optimizer:

```python
from equivariantx_foundation.targets.aoti import export_training, load_training, training_inputs

model = optimize(model.train(), family="tace", task="training")
export_training(model, objective, batch, "train.pt2")
step = load_training("train.pt2")
data, parameters, buffers = training_inputs(model, objective, batch)
optimizer.zero_grad(set_to_none=True)
predictions, loss, gradients = step(data, parameters, buffers)
for name, gradient in gradients.items():
    parameters[name].grad = gradient
optimizer.step()
```

The training package reads current parameters on every step. Shapes are fixed by
default. Pass `dynamic_shapes=True` to `export_training` for batches with the
same number of structures but varying atom, edge and pair counts. A count of
zero or one at export stays fixed; a dynamic count must be at least two.
If the data loader bounds the total number of directed edges per local batch,
pass that bound as
`max_edges` (for example, `dynamic_shapes=True, max_edges=2048`). The package
executes the actual graph without padding and rejects batches above the bound.
After the first `training_inputs` call, ordinary in-place optimizer steps
can reuse the parameter and buffer mappings. Use
`training_batch(model, objective, next_batch)` for new labels and graph
inputs; refresh both mappings if parameter or buffer objects are replaced.
The optimizer and its state are not serialized inside the package. Buffers unused by the traced
predictions, loss and gradients are omitted from the compiled argument list.
Ordinary steps accept the full buffer mapping or only the buffers read by
the package. CUDA Graph replay binds the original full state. Export and batch
preparation use TACE. With prepared tensors and live state mappings, the
compiled step can run in a separate CUDA process without importing TACE.
The step returns local gradients and
does not perform a distributed all-reduce. For a multi-head model,
`training_inputs` supplies one head index per structure, even when the
export sample used the model's default head. A new default or mixed explicit
indices do not require re-export.

For a deterministic fixed-shape CUDA training package, pass
`replay_inputs=(data, parameters, buffers)` to `load_training`. The returned
step has the same signature. It stages each batch and reads the bound
parameter and buffer storage directly, including in-place optimizer updates.
Use one thread and CUDA stream. Recreate the runner after replacing storage
or changing precision. Capture has a startup cost and reserves GPU memory.

For data-parallel training, initialize the default PyTorch process group, set
each rank's CUDA device, and start every rank from the same model, loss and
optimizer state. Each rank needs a package compatible with its local batch
shapes. Each rank owns complete structures; different ranks may have different
atom, edge and structure counts. For supported `NormalLoss` terms, including
the HIP recipe's E/F/direct-Hessian MAE:

```python
from equivariantx_foundation.targets import aoti

step = aoti.load_training(package_for_this_shape, device_index=local_rank)
data, params, buffers = aoti.distributed_training_inputs(model, objective, local_batch)
predictions, local_loss, local_grads = step(data, params, buffers)
loss, grads = aoti.mean_training_result(local_loss, local_grads, params)
optimizer.zero_grad(set_to_none=True)
for name, grad in grads.items():
    params[name].grad = grad
optimizer.step()
```

Call both helpers on every rank, once per step. They account for energy per
structure, forces (including noncollinear magnetic forces) per atom, and
stress per valid label. Supported terms are the existing E/F/S, magnetic-force
and direct-Hessian MSEs, plus `mae_energy`, `mae_forces` and
`mae_direct_hessian`.
For direct-Hessian MSE or MAE, the denominator follows
the selected structure or component reduction over valid structures and their
ordered atom pairs. A rank without Hessian labels still supplies zero weights
and correctly shaped placeholder blocks. An atom pair never crosses a
structure or rank boundary.
Averaging unadjusted local means is generally **not** the global-batch objective. Nonempty shards
and floating FP32/FP64 energy/force label weights are required. Globally unused
parameters keep `grad=None`; tied parameters are updated once. Do not add a
second DDP reduction. Huber, arbitrary custom or uncertainty-weighted objectives,
graph partitioning and automatic synchronization of replicas are not covered.

For single-GPU eager training with fully labelled, unweighted E/F or E/F/stress
batches, a CUDA Graph bucket can reuse one capture across different atom and
edge counts:

```python
from equivariantx_foundation.models.tace.training import (
    BucketedEFSTrainingStep, supports_training_buckets,
)

model = optimize(model.train(), family="tace", task="training")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
if supports_training_buckets(model):
    step = BucketedEFSTrainingStep(model, optimizer, example_batch,
                                   max_atoms=256, max_edges=4608)
    metrics = step(batch)  # includes optimizer.step()
```

The bucket uses energy-per-atom MSE plus force-component MSE and optional
stress-component MSE. The number of structures stays fixed. Atom and directed
edge counts may vary within the chosen capacities. For FP32 regional CUDA
models the predicate checks the installed graph-local operations and J0/C2
radial geometry. Spherical CGTP, Uu/Uv O(2), and compatible stacks mixing
both groups can share a bucket. Cartesian O(3) supports per-edge ICTD and
node-local ICTD products. Cartesian local O(2) supports the FP32, 64-channel,
degree/order-two direct Uu and Uv receivers. Uv uses both endpoints, the
original radial weights and the live Linear–Gate–Linear maps. Below 8192 edges
or with distinct analysis and synthesis rotations the receiver uses the native
path; the padded edge count determines dispatch during replay.
Automatic Uv selection in memory-first IEEE FP32 covers the scalar input
stage (128 radial weights, 192 scalar hidden features) and the tensor stage
(1,408 radial weights). Explicit CUDA also supports TF32, with the normal
floating-point differences between contraction orders.
An all-native Cartesian stack does not select this bucket.

A single-head rank-two spin-orbit magnetic Uv stack with a supported CGTP
product can also replay E/F/stress/noncollinear magnetic-force SGD or AdamW
steps. Supply complete unit-weight spins and labels on every microbatch.
The magnetic-force term averages over the three components of real atoms.
This bucket does not cover explicitly time-odd gates.

Directed stacks combining spherical CGTP with legacy TECE rotary attention
can also use a regional bucket with `backend="cuda"`, `regional_only=True`
and memory-first execution. Supported blocks include directed element updates,
the host's density normalization, and the scheduled quadratic bilinear MoE.
Provide the original directed neighbor list, not an undirected-pair map.

Explicit regional CUDA can schedule nonlinear Cartesian ICTD factors with
their native STF projections. The gates and coefficient maps are unchanged.
Bucket replay accepts this path when every factor fits the CUDA program and
all gates act independently on each node.

Spherical products may also use the standard EQX MACE-style symmetric
contraction or the native nonlinear CGTP gate with scheduled CUDA channel
products. This does not enable generalized-CG, CuEq or alternate fused
nonlinear product implementations for bucket replay.

An element table may be shared between endpoints (`element`) or directed
(`element2`). Identity, mean-neighbor, square-root-mean-neighbor and cutoff
scatter normalization are supported. Padding images are placed beyond the
model cutoff. Standard ScaleShift and C2-cutoff ZBL can be enabled with live
trainable parameters. For regional multi-head energy readouts, pass one
`fidelity_idx` per real structure or use the optimized model's current
default head. Changing heads does not require a new capture.
The bucket keeps isolated atoms distinct from one-atom
periodic cells with neighbors. For gradient accumulation set `microbatches=K` and call
`step([batch1, ..., batchK])`; one optimizer update uses all real atoms for
the force denominator and all real structures for energy and stress. Each
microbatch must contain the same number of structures. Arbitrary losses, AMP
and distributed training are not supported by this bucket.

Capture trades startup time and GPU memory for lower per-step latency. For
one C32 two-layer Cartesian ICTD example on an RTX PRO 5000, alternating
216 atoms / 3888 edges and 125 atoms / 2250 edges, ten steady E/F/stress SGD
steps had median times of 66.1 ms (regional eager) and 7.9 ms (bucket) in
isolated processes. Peak reserved memory rose from 144 to 172 MiB; replay
setup took 4.44 s for warmup plus 0.11 s for capture. These measurements are
specific to that workload. Use ordinary differentiable execution for other
configurations.

For the eight-layer TECE-RRA release, alternating two OMat batches
(18 atoms / 578 edges and 21 atoms / 650 edges) on an RTX PRO 5000 gave
522.00 ms (regional eager) and 123.12 ms (regional bucket) per complete
FP32 E/F/stress SGD step. These are steady-state medians over eight steps.
Peak allocated memory fell from 1873 to 1713 MiB, while peak *reserved*
memory rose from 2426 to 3838 MiB. The capture itself took 0.78 s after
17.66 s of setup and warmup.

### Adding a TACE execution schedule

Start with `plan(model, regional_only=True)`. A new checkpoint or a different
ordering of supported blocks does not require a new package. Runtime adapters
belong in `targets/ase.py`, `targets/aoti.py` or `targets/lammps/`.

| What changes? | Where it belongs |
| --- | --- |
| Weights, training data or a combination of already supported blocks | Reuse the existing plan. No new architecture directory. |
| A convolution or contraction usable outside TACE | `equivariantx`; Foundation selects it for compatible blocks. |
| A TACE interaction, product, readout, or a value reused across layers | A regional rule in `models/tace/`. |
| The complete sequence of layers must be known to fuse or tile it | A task-specific schedule in `models/tace/<architecture>/`. |
| Neighbor exchange, batch staging, capture or artifact loading | A shared execution or target adapter, not a copy of the model. |

1. Keep the model definition, weights and checkpoint format in TACE. Put a
   model-independent tensor operation in EQX, not in the Foundation adapter.
2. For a reusable TACE block, add a structural matcher in `models/tace/`
   and register its host types in [regional.py](models/tace/regional.py).
   Match the representation, layout, paths, normalization and derivative
   contract. Leave unsupported blocks and previously enabled providers intact.
   Ordinary execution and multi-rank ML-IAP use the same table. A rule is
   single-device by default. Set `rank_local=True` only for native block
   types verified with ghost senders, owned outputs and reverse force exchange.
   The multi-rank target retains the host's layer loop and geometry.
3. Apply optimizations shared across compatible layers (e.g. constructing their
   common Wigner frames once) in the regional planner. Do not tie them to a
   release name.
4. When an optimizer needs the complete architecture (e.g. a fused, fixed-layout
   receiver program), list its package in `models/tace/planner.py:WHOLE_MODEL_SCHEDULES`.
   The package defines `TASKS`, `matches(model, *, task)`, and `optimize(model, **options)`.
   Match the architecture, not the checkpoint filename, before importing its kernels. Only
   declare `lammps` if the package also implements the local-owner exporter.
   Training, joint E/F/S training and bucketed replay are distinct contracts.
5. Validate each advertised task. Force inference needs position derivatives;
   force or stress training also needs mixed parameter derivatives. Report
   execution time and peak memory against both native TACE and generic EQX.
   Single-rank ML-IAP and distributed force exchange are separate contracts.
   For a new architecture, first run `plan` on its normal TACE configuration
   and benchmark the native model and the regional plan on the same weights,
   batches and precision. Add a whole-model package only for a measured
   bottleneck that crosses the existing block boundaries.

For example, changing channel width and depth creates a new configuration,
not a new schedule package. Re-plan its interactions and products. If it
retains the same eligible periodic LES solver, the reciprocal-grid optimization
can still apply even when the node representation changes from spherical to
Cartesian. Frozen Cartesian LES inference and spherical LES joint training
are separate capabilities: inheriting the solver does not certify either
export target.

Track coverage by **block and task**, not by checkpoint count. Distinguish a
structural match, an observed kernel call, numerical agreement and a measured
speed or memory improvement. For training, compare a full optimizer step and
its live parameter gradients. For deployment, test the exported artifact on
the intended target. Record the batch shapes, device, precision and first-call
cost separately from steady-state time and peak allocated/reserved memory.

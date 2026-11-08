dataset
=======

Configure structure sources, property keys, graph storage, and data loaders here.

Reading and training
--------------------

.. code-block:: text

   ASE / ASE database / ASE-LMDB
                |
        indexed FP64 records
                |
      batched neighbor construction
                |
       immutable graph shards ------> full training-set statistics
                |              \----> complete validation / test loaders
       optional train filter
                |
      batched loader + augmentation
                |
        model precision policy

``memory`` constructs graphs once in RAM and suits small datasets.
``clmdb`` constructs losslessly compressed graphs once on disk and is
recommended for large, repeated training. ``lmdb`` writes the uncompressed
format. Both read existing compressed and uncompressed shards automatically;
the mode controls new writes and does not convert completed caches.
Source decoding and neighbor search are not repeated each epoch.
``on_the_fly`` reads records and constructs neighbors for each requested batch;
it saves graph storage, but may reduce training throughput. It is opt-in.

ASE-LMDB and SQLite are opened read-only and read by index. XYZ and trajectory
files also support random access. Other ASE formats are converted once to a
losslessly compressed record cache under ``shard_dirs[0]/records``. This cache is
independent of graph cutoff and targets. ``record_cache_dir`` overrides its location; ``null``
disables it. Original source files are never rewritten.

.. code-block:: yaml

   dataset:
     reader: aselmdb
     train_file: /data/omat24/train
     valid_file: /data/omat24/val
     storage_mode: clmdb
     shard_dirs: [graphCache]
     graph_batch_size: 256
     graph_workers: 0
     neighborlist_backend: matscipy
     cache_size: 0
     train_dataloader:
       _target_: torch_geometric.loader.DataLoader
       batch_size: 32
       shuffle: true
       num_workers: 4
       persistent_workers: true
       pin_memory: true

``matscipy`` remains the default CPU neighbor backend. ``graph_workers``
controls CPU preprocessing threads, not training-loader workers.
``alchemiops`` performs FP64 neighbor search for packed graph batches.
For offline ``lmdb``, ``clmdb`` or ``memory`` construction,
``graph_device: null`` uses all visible CUDA devices, or CPU if none are available:

.. code-block:: yaml

   dataset:
     neighborlist_backend: alchemiops
     storage_mode: clmdb
     graph_device: null  # all visible GPUs; CPU fallback
     graph_batch_size: 256

Use ``graph_device: [cuda:0, cuda:1]`` to select devices, ``cuda:0`` for one
GPU, or ``cpu`` for CPU. Explicit ``all`` requires at least one visible GPU.
Device indices follow ``CUDA_VISIBLE_DEVICES``; graph devices are independent
of training devices and do not split individual structures across GPUs.

With multiple GPUs and either LMDB storage mode, each GPU process independently
reads records, constructs graphs, and writes its own LMDB shards. The parent
assigns work and merges metadata without transferring or rewriting graphs.
Sample order is preserved. Work partitions are balanced across GPUs and
distributed across ``shard_dirs``. Graphs are constructed and encoded in bounded
batches; shard and write-buffer limits are internal constants.
``graph_batch_size`` is per GPU; ``graph_workers`` does not set GPU worker
count. Completed caches load without starting GPU workers.

Offline construction shows one progress bar with exact completed/total graph
counts, throughput, and estimated remaining time. LMDB counts committed records;
multiple GPUs share one parent-process bar. Cache hits show no construction bar,
and failed builds close it without advancing the count.

More GPUs can increase preprocessing throughput, but source decoding, shared
disk bandwidth, and process startup can limit scaling. Multi-rank DDP rejects
multi-GPU ``memory`` preprocessing; use either LMDB mode for rank-zero shared-cache
construction, or select a single rank-local graph device.

``on_the_fly`` remains single-device: ``null`` selects the current CUDA device
or CPU, while ``all`` and device lists are not accepted. CUDA online
construction requires ``num_workers: 0`` for training, validation, and
statistics loaders. Benchmark online throughput before long training runs.

Graph precision and cache
-------------------------

Positions, cells, floating-point properties, and property weights are stored in
float64. Atomic numbers, edge indices, periodic image indices, and fidelity
indices are integers. Element features are constructed from indices in the
model; graphs do not store one-hot element arrays.

Graphs are stored directly in ``shard_dirs``. Each ``train_shard*.lmdb``,
``valid_shard*.lmdb``, or ``test0_shard*.lmdb`` directory contains its own
metadata and graph counts. There is no configuration-hash directory.

The reader accepts ordinary ``typed-array-v1`` records and losslessly compressed
``typed-array-zlib-v1`` records, including a mixture of the two shard formats.
Compression preserves the encoded bytes, float64 values, integer indices and
graph counts. Compressed copies must retain their shard names and ``counts.bin``;
changing storage location or codec then leaves the statistics key unchanged.
New compressed stores use per-record zlib compression with an internal fixed
level. There is no separate compression switch or level in the training config.
Loader workers decode records before collation; model inputs and precision are
identical for both formats. The standalone graph and record writers default to
compression, while ``storage_mode: lmdb`` explicitly selects ordinary graph writes.

To combine datasets, move their completed ``*.lmdb`` directories into the same
directory, preserving their generated names:

.. code-block:: text

   graphCache/
     train_shard<unique-id>_000000.lmdb/
     train_shard<another-id>_000000.lmdb/
     valid_shard<unique-id>_000000.lmdb/

.. code-block:: yaml

   dataset:
     storage_mode: lmdb
     shard_dirs: [graphCache]

Restarting reuses the saved graphs before opening source files, including when
statistics were not finished. Sources can be moved or removed after graph
construction. Train, validation, and test splits retain their saved assignments;
changing source paths or split settings does not rebuild existing graphs.
Use another directory to construct a new dataset.

Only matching cutoffs, neighbor limits, and the required label fields are needed.
Unused labels are omitted during decoding. Fidelity indices must have the same
meaning in all combined datasets. Statistics are computed from the graphs and
refreshed when shards are added or removed.

Training uses batched LMDB reads without repeating neighbor search. Statistics
read only the required fields and accumulate in FP64. LMDB capacity grows
automatically; write buffers and open-shard limits are set in Python. Each
worker retains up to 4,096 read-only shard handles, reduced on POSIX systems
to one eighth of the file-descriptor limit. This avoids reopening shards on
every random batch; mapped files do not load the entire dataset into RAM.
``cache_size`` controls optional decoded samples per worker and defaults to zero;
the operating-system page cache remains active.

Statistics have an independent loader:

.. code-block:: yaml

   dataset:
     statistics_dataloader:
       batch_size: 16384
       num_workers: 0

For disk-backed LMDB graphs, ``misc.device: cuda`` uses all visible GPUs for
statistics. Select visible GPUs with ``CUDA_VISIBLE_DEVICES`` or use an explicit
device such as ``misc.device: cuda:0`` to use one GPU. CPU and other storage modes
keep the configured single device. An externally launched distributed rank uses
its own GPU instead of expanding to all visible devices again.
Each statistics process reads its own graph shards; only small FP64 moment
states are transferred to the parent and merged. Every training graph is
included once, regardless of ``epoch_size``. The parent writes the statistics
files after all processes finish.
Statistics display their own progress bar showing completed structures,
throughput, and estimated time remaining. Multi-GPU statistics readers update
one statistics progress bar in the parent process.

The statistics batch size is per device. Larger batches also require more
host memory for decoding and collating graphs; benchmark throughput instead of
selecting a batch solely from free GPU memory. Statistics loaders default to
``num_workers: 0`` independently of training. Explicit loader workers use one
prefetched batch by default, and their shared-memory demand is multiplied by
the number of statistics devices. Adjust these values for the dataset and
available memory.

Incomplete builds are preserved and not reused as complete datasets.
Augmentation creates sample-local tensors without modifying cached graphs.

Filtering cached structures
---------------------------

``dataset.filter`` selects training structures from completed graphs without
rewriting LMDB files or rebuilding neighbors. It is disabled by default.
When enabled, the default edge limit is 24,000 per structure; all other limits default to
``null``. Set any limit to ``null`` to disable it.

.. code-block:: yaml

   dataset:
     storage_mode: lmdb
     filter:
       enabled: true
       max_edges: 24000
       max_atoms: null
       max_forces: null
       max_energy_per_atom: null
       max_stress: null

A structure is retained only if it satisfies every enabled upper bound;
equality is retained. ``max_forces`` uses the largest Euclidean norm of any
atom's stored ``forces`` vector (eV/Angstrom). ``max_energy_per_atom`` compares
the signed stored total ``energy`` divided by the atom count (eV/atom), without
subtracting reference atomic energies. ``max_stress`` uses the Frobenius norm
of the full stored 3-by-3 ``stress`` tensor (eV/Angstrom^3), including both
off-diagonal entries. Atom and edge bounds count the full cached structure;
edges are counted exactly as stored in ``edge_index``, including both directed
edges when present. No atoms or edges are truncated inside a retained graph.

Filtering applies only to the training loader, before training augmentation
and epoch sampling. Validation and test loaders traverse their complete splits.
Enabled physical bounds require the corresponding cached label
and weight fields, even if the current model does not train on that label.
Structures with nonfinite or zero-weight labels fail that physical bound;
zero-weight placeholders are not treated as measured zeros. Missing fields
or a training selection with no remaining structures produce a clear error.

The first training loader scans the count index for atom/edge bounds without
decoding graphs.
Physical bounds decode only the needed labels for structures that pass the
count bounds. A single training-filter progress bar and retained/removed counts
are shown. The resulting fixed index is held in memory; training keeps the
existing batched LMDB reader and does not repeat filtering each epoch.
When all structures pass, the original dataset is reused without a selection
index allocation.
The same selection also works for ``storage_mode: memory``;
``on_the_fly`` requires preparing a completed graph cache first.

``EpochSampler`` uses the retained dataset: a fractional ``epoch_size`` is
relative to its size, and an integer must not exceed it. Distributed ranks
sample the same filtered order, and full-checkpoint recovery checks the
selection identity, including changes that leave the sample count unchanged.
Changing the selected structures requires starting a new sampling stream.
Statistics and fitted atomic reference energies always use the complete training
split, before filtering or epoch sampling. Validation and test labels do not
enter these statistics. Changing or disabling filters does not invalidate the
statistics cache. The original shards, labels, and statistics options still
determine its identity. Finetuning and checkpoint resume retain their existing
model statistics.

Training epoch size
-------------------

``dataset.epoch_sampler`` groups ``epoch_size`` and ``balance_batches``.
Omitting the group or setting ``epoch_size: null`` preserves ordinary
full-dataset training. Set an integer
to choose the requested global number of training samples per epoch across all
GPUs,
or a float in ``(0, 1]`` to choose a fraction of the training dataset, rounded
down with a minimum of one sample. The requested count must not exceed the
dataset size; ``1`` requests one sample, while ``1.0`` requests a full pass,
before batch alignment.

.. code-block:: yaml

   misc:
     global_seed: 42
   dataset:
     epoch_sampler:
       epoch_size: 100000  # global samples, not samples per GPU
       balance_batches: null  # null, atoms, edges, or [atom_weight, edge_weight]
     train_dataloader:
       batch_size: 32  # per GPU
       shuffle: true
       drop_last: false

With ``epoch_size`` enabled, epochs take consecutive windows from one repeating
dataset order. ``shuffle: true`` selects a fixed permutation using
``misc.global_seed``; ``false`` uses source order. Every full traversal visits
each sample once before repeating the same order, even when an epoch crosses
the traversal boundary. Statistics still use the full training dataset;
validation and test loaders are unchanged.

Set ``dataset.epoch_sampler.balance_batches: edges`` or ``atoms`` to redistribute each
global batch across GPUs using cached edge or atom counts. The selected
structures, global batch size and equal per-GPU structure counts are preserved;
only their rank assignment changes. A largest-first partition is used unless
the original partition has a smaller maximum load. This requires
``epoch_size`` and cached graph sizes. The default ``null`` keeps strided
assignment. Validation, test and statistics are unaffected.

Alternatively, supply ``balance_batches: [20, 1]`` to use a weighted sum of
atom and edge counts, here ``20 * atoms + edges``. Both weights must be
nonnegative integers and at least one must be positive. Compare complete
training steps on the intended model and hardware before choosing a cost;
equal atom or edge counts alone do not guarantee equal execution time.

Balancing reads the size sidecars once and retains one int64 cost per selected
training structure in each rank's CPU memory. Graph records remain lazy.

With ``drop_last: false``, the requested count is rounded up to a multiple of
``world_size``, giving every rank the same sample count and retaining its final
partial batch. With ``drop_last: true``, it is rounded up to a multiple of
``world_size * batch_size``, so every batch is full. The actual count is logged
when alignment changes it.

Extra samples come from the continuing stream, not repeated padding from the
current epoch. At the end of the complete dataset, sampling continues from the
start of the next pass, without dropping the remaining structures. For example,
with two GPUs, per-GPU batch size four and ``epoch_size: 11``, an epoch contains
12 samples with ``drop_last: false`` or 16 with ``drop_last: true``. Repeats are
possible within an epoch only if alignment makes it longer than the dataset.

Resume through the usual top-level ``resume_from_model: /path/to/model.ckpt``
using a full training checkpoint (``save_weights_only: false``). The saved
position tracks completed training batches, not worker-prefetched samples.
The checkpoint must have been saved with this sampler enabled.
Keep the same dataset and ordering, requested epoch count, seed, shuffle setting,
world size, per-rank batch size, ``drop_last`` and ``balance_batches`` when resuming. This restores
sample order, not exact random augmentation or dropout state. Optional ``trainer.max_steps``
and ``ModelCheckpoint.every_n_train_steps`` control step-based stopping and
checkpoint intervals independently of the shortened epochs.
Leave ``trainer.limit_train_batches`` at ``1.0`` and disable
``trainer.overfit_batches``; use ``epoch_size`` to control the training subset.

Data augmentation
-----------------

``augmentation`` is one symmetry-preserving transformation or a list of
transformations applied only when training samples are read. ``null`` and an
empty list disable augmentation. The available entries are:

``spin_rotation``
  Apply one uniformly sampled global spin rotation to
  ``initial_noncollinear_magmoms`` and ``noncollinear_magnetic_forces``.
  Positions are unchanged. This teaches the independent spin-rotation
  symmetry of data without spin--orbit coupling.

``time_reversal``
  Randomly reverse ``initial_noncollinear_magmoms`` and
  ``noncollinear_magnetic_forces`` together. This enforces global time
  reversal while preserving the consistency of the input and its derivative.

The SOC architecture permits the broadest function space because it enforces
only coupled space--spin rotations. When it is fitted to non-SOC data,
``spin_rotation`` is required to sample the independent
:math:`SO(3)_{\mathrm{spin}}` symmetry. If the installed e3nn does not track
time-reversal parity, ``time_reversal`` is required as well. These
augmentations sample
:math:`O(3)_{\mathrm{space}}\times SO(3)_{\mathrm{spin}}
\times\mathbb Z_2^{\mathcal T}` in the training data; they do not make it an
exact architectural symmetry.

The explicit non-SOC architecture already enforces the full group
:math:`O(3)_{\mathrm{space}}\times SO(3)_{\mathrm{spin}}
\times\mathbb Z_2^{\mathcal T}` and therefore requires neither augmentation.
For zero-field SOC data, the complete time-reversal model requires no
augmentation, whereas the standard-e3nn SOC model uses ``time_reversal``.

No transformation acts on validation, test, or statistics loaders. Spin
rotations and reversals are global per structure; independent per-atom
transformations would change the physical exchange interactions.


.. note::
  
   - The priority order is:  ``no_valid_set`` > ``valid_file`` > ``valid_from_index`` > ``valid_ratio``.


Property scopes
---------------

``per-system`` stores one value per structure; ``per-atom`` stores one per atom.
``per-edge`` follows the entries of ``edge_index`` and ``edge_shifts``, for example
``edge_forces`` with shape ``(E, 3)``. Edges depend on the neighbor list and need
not include all atom pairs. Periodic images can give multiple edges for the same
atom indices.

``per-pair`` contains all ordered atom pairs within each structure, including
``i == j``, in ``i * N + j`` order. Hessians use ``(N**2, 3, 3)`` regardless of
the cutoff. A batch concatenates ``sum(E)`` edge entries or ``sum(N**2)`` pair
entries, without pairs between structures. Edge and pair labels belong in
``atoms.info``, not the per-atom ``atoms.arrays``.

Periodic and non-periodic structures
------------------------------------

Molecules and crystals can share a dataset and a batch. The physical cell and
``pbc`` are preserved, including zero cells for isolated atoms and molecules.
Neighbor-list construction does not assign a physical volume to those systems.

Missing properties (absent keys or ``None``) receive zero weight. Provided
labels must contain only finite values; NaN and infinite values are rejected.

Stress requires at least one periodic direction and a non-singular cell.
Non-periodic structures contribute to energy and force training, but not
stress training. Omit their stress labels, or set ``stress_weight: 0`` in
the structure metadata.

Internally, undefined stresses are zero placeholders, not physical predictions.
The ASE calculator raises ``PropertyNotImplementedError`` when stress is
requested for a non-periodic or zero-volume structure.

TorchSim 0.6.2 requires identical boundary conditions within a ``SimState``;
run periodic and non-periodic inference batches separately through that interface.
This restriction does not apply to TACE training batches.

Fidelity per data source
-----------------------

``train_file``, ``valid_file``, and each entry in ``test_files`` accept a
path or a mapping with ``path`` and ``fidelity_idx``. Paths may refer to
files or directories. Fidelity indices follow the order of
``model.config.fidelity``.

.. code-block:: yaml

   dataset:
     train_file:
       - {path: mad-1.6-r2scan-train.xyz, fidelity_idx: 0}
       - {path: mad-1.6-pbe.xyz, fidelity_idx: 1}
     valid_file:
       - {path: r2scan-valid.xyz, fidelity_idx: 0}
       - {path: pbe-valid.xyz, fidelity_idx: 1}
     test_files:
       - {path: r2scan-test.xyz, fidelity_idx: 0}
       - {path: pbe-test.xyz, fidelity_idx: 1}

When a source specifies a fidelity, missing structure metadata is filled with
that value. Existing metadata must match, otherwise reading fails with the
file path, zero-based structure index, and conflicting values. The metadata
key is selected by ``dataset.keys.fidelity_idx_key`` (default: ``fidelity_idx``).
Without a source fidelity, existing metadata is preserved; unmarked structures
still use head 0. These assignments do not modify the source files.

Source checks run when raw data is read. Completed graph caches retain their
saved fidelity assignments; use a new graph directory to rebuild them with
different assignments.

``dataset.reader`` selects ``ase`` (default), ``ase_db``, or ``aselmdb``.
A string applies uniformly to every training, validation, and test source.
A list assigns one reader per configured address in ``train_file`` order,
then ``valid_file`` order, then ``test_files`` order. Its length must exactly
match the total address count, including when completed graph caches are reused.
Directories count as one address; all their discovered files use that address's
reader. Seven addresses therefore require seven values even when an address
contains many files. File discovery and sorting preserve these assignments.
When validation is split from training with ``valid_file: null``, it uses the
training readers and requires no additional list entries.

ASE formats are always detected from file names and contents. Directory
discovery includes formats recognized by ASE's filename rules, such as XYZ,
trajectory and CIF files. There are no ``dataset.type``, ``format``, or
``options`` settings.

.. code-block:: yaml

   dataset:
     reader: [aselmdb, ase, ase_db, ase]
     train_file: [crystal_databases, molecule_files]  # first two readers
     valid_file: validation.db                     # third reader
     test_files: [test.xyz]                        # fourth reader

Explicit property keys take precedence across ASE arrays, metadata, database
data, and calculator results. Canonical names and ASE aliases are fallbacks;
no calculator is executed. Malformed shapes, non-finite labels, negative
weights, non-integer fidelities, and corrupt source files raise errors instead
of silently dropping structures.

Example
-------

.. yaml-config:: ../../../../example/train/TACE/tace.yaml
   :path: dataset

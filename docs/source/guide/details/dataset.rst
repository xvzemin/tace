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
       immutable graph shards ------> projected statistics
                |
      batched loader + augmentation
                |
        model precision policy

``memory`` constructs graphs once in RAM and suits small datasets.
``lmdb`` constructs graphs once on disk and is recommended for large, repeated
training. Source decoding and neighbor search are not repeated each epoch.
``on_the_fly`` reads records and constructs neighbors for each requested batch;
it saves graph storage, but may reduce training throughput. It is opt-in.

ASE-LMDB and SQLite are opened read-only and read by index. XYZ and trajectory
files also support random access. Other ASE formats are converted once to a
typed record cache under ``shard_dirs[0]/records``. This cache is independent of
graph cutoff and targets. ``record_cache_dir`` overrides its location; ``null``
disables it. Original source files are never rewritten.

.. code-block:: yaml

   dataset:
     type: aselmdb
     train_file: /data/omat24/train
     valid_file: /data/omat24/val
     storage_mode: lmdb
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
For offline ``lmdb`` or ``memory`` construction, ``graph_device: null`` uses
all visible CUDA devices, or CPU if none are available:

.. code-block:: yaml

   dataset:
     neighborlist_backend: alchemiops
     storage_mode: lmdb
     graph_device: null  # all visible GPUs; CPU fallback
     graph_batch_size: 256

Use ``graph_device: [cuda:0, cuda:1]`` to select devices, ``cuda:0`` for one
GPU, or ``cpu`` for CPU. Explicit ``all`` requires at least one visible GPU.
Device indices follow ``CUDA_VISIBLE_DEVICES``; graph devices are independent
of training devices and do not split individual structures across GPUs.

With multiple GPUs and ``storage_mode: lmdb``, each GPU process independently
reads records, constructs graphs, and writes its own LMDB shards. The parent
assigns work and merges metadata without transferring or rewriting graphs.
Sample order is preserved. Work partitions are balanced across GPUs and
distributed across ``shard_dirs``. Each shard obeys ``shard_size`` and
``shard_bytes``; graphs are constructed and encoded in bounded batches.
``graph_batch_size`` is per GPU; ``graph_workers`` does not set GPU worker
count. Completed caches load without starting GPU workers.

Offline construction logs completed graphs, throughput, and estimated remaining
time every ten seconds. LMDB progress counts committed records; multiple GPUs
share one aggregate report. Completion is logged after publishing the cache.

More GPUs can increase preprocessing throughput, but source decoding, shared
disk bandwidth, and process startup can limit scaling. Multi-rank DDP rejects
multi-GPU ``memory`` preprocessing; use ``lmdb`` for rank-zero shared-cache
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

LMDB caches use versioned subdirectories under ``shard_dirs``. Source paths,
file metadata, property keys, dataset splits, and neighbor-list settings select
the cache. Network dtype and element ordering do not. Completed shard metadata
allows loading and statistics without rereading source structures. Train,
validation, and test caches are independent. Changing only the test set does
not rebuild training graphs. Test graphs are prepared when testing begins.

Shards contain typed arrays, not serialized Python/PyG objects. Loaders read
several indices in one shard transaction; statistics decode only their input
fields and use stored neighbor counts. Elemental energy fitting and moments
accumulate in FP64 with bounded memory.

``shard_size`` limits records per shard; ``shard_bytes`` additionally limits
encoded bytes (default 2 GiB). ``transaction_bytes`` bounds each write batch
(16 MiB), and ``max_buffer_bytes`` bounds buffered encoded records (64 MiB).
An individual record exceeding that limit raises an error; increase the limit
for very large pair properties. LMDB map capacity grows automatically.
``cache_size`` controls optional decoded samples per worker (default zero);
the operating-system page cache remains active.
``max_open_shards`` limits open shards per dataset and process (default 32).
Increase it for caches with many small shards to avoid reopening them during
random training reads. Larger shards do not increase the write-buffer budget.

Old caches are left untouched and rebuilt from source in the new namespace.
Casting an old FP32 cache to FP64 cannot recover discarded precision. Incomplete
caches are rejected; move them aside or select a new cache directory to rebuild.

Augmentation creates sample-local tensors and does not overwrite cached data.

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

Source checks run when raw data is read. Changing configured source fidelities
selects a new graph-cache namespace automatically.

Sources also accept ``reader``, ``format``, and ``options`` for mixed formats:

.. code-block:: yaml

   train_file:
     - {path: crystals.aselmdb, reader: aselmdb, fidelity_idx: 0}
     - {path: molecules.xyz, reader: ase, format: extxyz, fidelity_idx: 1}

Explicit property keys take precedence across ASE arrays, metadata, database
data, and calculator results. Canonical names and ASE aliases are fallbacks;
no calculator is executed. Malformed shapes, non-finite labels, negative
weights, non-integer fidelities, and corrupt source files raise errors instead
of silently dropping structures.

Example
-------

.. yaml-config:: ../../../../example/train/TACE/tace.yaml
   :path: dataset

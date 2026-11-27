dataset
=======

Configure structure sources, property keys, graph storage and data loaders.

Sources
-------

``reader`` selects ``ase`` (default), ``ase_db`` or ``aselmdb``.
A string applies to every source. A list assigns one reader to each configured
address, in ``train_file``, ``valid_file``, then ``test_files`` order. The list
length must match the total address count, including when reusing a graph cache.
A directory counts as one address; all its files use the same reader.
ASE detects file formats automatically.

.. code-block:: yaml

   dataset:
     reader: [aselmdb, ase, ase_db, ase]
     train_file: [crystal_databases, molecule_files]
     valid_file: validation.db
     test_files: [test.xyz]

Each source may instead be a mapping with ``path`` and ``fidelity_idx``.
Fidelity indices follow ``model.config.fidelity``. A single fidelity mapping
defines one head, just like a list containing that mapping. Missing structure metadata
uses the source fidelity; conflicting metadata raises an error. Without a source
fidelity, unmarked structures use head 0.

.. code-block:: yaml

   dataset:
     reader: ase
     train_file:
       - {path: r2scan-train.xyz, fidelity_idx: 0}
       - {path: pbe-train.xyz, fidelity_idx: 1}

Validation selection follows this priority:
``no_valid_set`` > ``valid_file`` > ``valid_from_index`` > ``valid_ratio``.
Validation split from training uses the training readers without additional
reader-list entries.

Source files are read-only. ASE-LMDB, SQLite, XYZ and trajectory readers support
indexed access. Other ASE formats are converted once to a compressed record cache
at ``shard_dirs[0]/records``. ``record_cache_dir`` overrides this location;
``null`` disables the record cache.

Graph storage
-------------

``storage_mode`` selects:

* ``memory``: construct graphs once in RAM.
* ``clmdb``: construct losslessly compressed graph shards on disk.
* ``lmdb``: construct uncompressed graph shards on disk.
* ``on_the_fly``: construct neighbors for each requested batch.

Both disk modes automatically read compressed and uncompressed shards, including
mixed caches. The mode controls new writes; it does not convert existing shards.
Compression preserves all values and has no separate configuration switch.

.. code-block:: yaml

   dataset:
     reader: aselmdb
     train_file: /data/omat24/train
     valid_file: /data/omat24/val
     storage_mode: clmdb
     shard_dirs: [graphCache]
     cache_size: 0
     neighborlist_backend: matscipy
     graph_batch_size: 256
     graph_workers: 0
     train_dataloader:
       _target_: torch_geometric.loader.DataLoader
       batch_size: 32
       shuffle: true
       num_workers: 4
       persistent_workers: true
       pin_memory: true

Positions, cells, floating labels and weights are stored in float64. Atomic
numbers, edge indices, periodic image indices and fidelity indices are integers.
The model constructs element features from atomic numbers.

Shards are stored directly in ``shard_dirs``:

.. code-block:: text

   graphCache/
     train_shard<unique-id>_000000.lmdb/
     train_shard<another-id>_000000.lmdb/
     valid_shard<unique-id>_000000.lmdb/

Completed caches load before source files are opened. Sources may therefore be
moved after graph construction. Saved splits and fidelity assignments are reused;
use another graph directory to change them. Incomplete builds are not loaded.

Combine compatible datasets by placing their completed shard directories together,
retaining their names and contents. Cutoffs, neighbor limits and required label
fields must match, and fidelity indices must have the same meaning. Adding or
removing shards invalidates statistics; changing only their location or codec does
not. Unused labels are omitted during decoding.

Training reads batches directly from LMDB without rebuilding neighbors.
``cache_size`` sets decoded samples retained per worker; zero disables this cache
while retaining the operating-system page cache.

Graph construction
------------------

``matscipy`` is the default CPU neighbor backend. ``graph_workers`` controls CPU
preprocessing threads independently of data-loader workers.
``alchemiops`` constructs neighbors for packed batches in float64:

.. code-block:: yaml

   dataset:
     neighborlist_backend: alchemiops
     graph_device: null
     graph_batch_size: 256

For offline construction, ``graph_device: null`` uses all visible GPUs or CPU
when no GPU is available. An explicit device, such as ``cpu`` or ``cuda:0``,
selects one device; ``[cuda:0, cuda:1]`` selects several. ``all`` requires a GPU.
Indices follow ``CUDA_VISIBLE_DEVICES``. ``graph_batch_size`` is per device;
``graph_workers`` does not select the number of GPUs.

Disk construction uses independent workers and one progress bar, preserving
sample order. Multi-rank DDP requires disk storage for multi-GPU preprocessing,
or a single rank-local graph device for ``memory`` storage.

``on_the_fly`` supports one device: ``null`` selects the current GPU or CPU.
CUDA online construction requires ``num_workers: 0`` in every loader.

Statistics
----------

Statistics and fitted atomic reference energies use the complete training split,
before filtering, epoch sampling or augmentation. Validation and test labels do
not enter these statistics. Finetuning and resume reuse model statistics.

.. code-block:: yaml

   dataset:
     statistics_dataloader:
       batch_size: 16384
       num_workers: 0

Disk LMDB statistics use all visible GPUs when ``misc.device: cuda``. An explicit
device, such as ``cuda:0``, selects one GPU. Externally launched distributed ranks
use their own device. CPU and other storage modes use the configured single
device. Reductions use float64 and display one statistics progress bar.

Batch size is per device. Decoding also consumes host memory. Statistics workers
default to zero independently of training; explicit workers prefetch one batch
by default, with memory use multiplied by the number of devices.

Training filters
----------------

``dataset.filter`` selects complete training graphs without rewriting the cache.
It is disabled by default. When enabled, connectivity, translational and rotational
force consistency, and force–stress consistency are checked. Other numeric bounds default to ``null``
(no limit). An isolated reference atom (one site with no periodic image inside
the cutoff) always passes, regardless of its labels or any enabled limit.
One-site crystals with an active periodic image, and all multi-atom structures,
are subject to the enabled rules. For example:

.. code-block:: yaml

   dataset:
     filter:
       enabled: true
       max_edges: 24000
       max_atoms: null
       min_covalent_ratio: null
       drop_isolated: false
       drop_disconnected: true
       max_translation_forces_rmse: 0.01
       max_rotation_forces_rmse: 1.0e-4
       max_forces_stress_residual: 1.0e-6
       max_forces: null
       max_energy_per_atom: null
       max_stress: null

* ``max_atoms`` and ``max_edges`` count the full structure, including directed
  edges as stored. They do not truncate atoms or edges.
* ``min_covalent_ratio`` bounds :math:`r_{ij}/(R_i+R_j)` from below for every
  pair, using the ASE covalent radii of the two elements. For example, ``0.5``
  rejects a structure if any distance is less than 50% of its pair's radius sum.
  Nonzero-shift periodic self images count. Graphs without edges pass this bound.
  The graph cutoff must cover the largest pair threshold for the dataset's elements.
* ``drop_isolated: true`` rejects a multi-atom structure if any atom has no
  neighbor within the graph cutoff. Single-atom structures pass this filter.
  Nonzero-shift periodic images count as neighbors.
* ``drop_disconnected`` rejects multi-atom structures with more than one
  connected component in the cutoff graph, even if every atom has neighbors.
  Periodic bonds connect their two sites in the input cell.
* ``max_translation_forces_rmse`` bounds the minimum force correction required
  for zero net force in each connected component, including periodic and
  single-atom components. For :math:`N` atoms the metric is
  :math:`[\sum_C |\sum_{i\in C}\mathbf F_i|^2/(3N |C|)]^{1/2}` in
  eV/Angstrom. For a connected graph it is the norm of the mean force divided
  by :math:`\sqrt{3}`. Values equal to the bound pass. Missing or zero-weight
  force labels skip this rule; ``null`` disables it. Labels are not modified.
* ``max_rotation_forces_rmse`` bounds the force component incompatible with an
  independently rotatable, finite fragment, in eV/Angstrom. Finite fragments
  are unwrapped across cell boundaries. Components with periodic winding and
  single atoms contribute zero. The squared Cartesian force corrections are
  summed over fragments and divided by :math:`3N` for the entire structure
  before taking the square root. Net translation is not included. Structures
  without a positive-weight force label skip this rule. Set it to ``null`` to
  disable it independently of ``drop_disconnected``.
* ``max_forces`` bounds the largest atomic force norm in eV/Angstrom.
* ``max_forces_stress_residual`` bounds the Frobenius norm of the stress
  mismatch with the force virial in eV/Angstrom^3, not the stress magnitude.
  It applies to configurational stress (ASE tensile-positive convention) for
  a short-range energy without explicit cell terms. Stress components
  undetermined by the periodic cutoff graph are projected out; any component
  winding in three independent directions leaves the total stress unconstrained.
  Nonperiodic structures, zero-volume cells and records without positive-weight
  force and stress labels skip this rule. Values equal to the bound pass.
  Set it to ``null`` to disable it independently of the force thresholds.
* ``max_energy_per_atom`` bounds signed total energy divided by the atom count,
  in eV/atom, without subtracting atomic reference energies.
* ``max_stress`` bounds the Frobenius norm of the full stress matrix in
  eV/Angstrom^3.

The maximum energy, force norm and stress bounds require cached labels and weights. Nonfinite or zero-weight
labels fail the bound. Missing fields or an empty selection raise an error.
Edgeless reference atoms pass using the shard count index alone. Other filters
read the required fields with CPU readers. Connected bulk structures with three
independent periodic winding directions need no rotational or stress projection.
In distributed training, ranks scan
disjoint ranges and exchange a bit-packed selection mask. All ranks retain the
same ordered selection in memory and reuse it each epoch.

Filters support disk and ``memory`` caches. ``on_the_fly`` requires preparing a
completed cache first. Only training is filtered; validation and test traverse
their complete splits. Filters do not invalidate full-dataset statistics.

Epoch sampling
--------------

``dataset.epoch_sampler.epoch_size`` is the requested global sample count across
all GPUs. ``null`` uses ordinary full-dataset training; an integer requests a
count, and a float in ``(0, 1]`` requests a fraction of the retained training set,
rounded down to at least one. The request cannot exceed the dataset size.
``1`` means one sample; ``1.0`` means a full pass before batch alignment.

.. code-block:: yaml

   misc:
     global_seed: 42
   dataset:
     epoch_sampler:
       epoch_size: 100000
       balance_batches: null
     train_dataloader:
       batch_size: 32  # per GPU
       shuffle: true
       drop_last: false

Epochs take consecutive windows from a repeating dataset order. ``shuffle: true``
uses a fixed permutation determined by ``misc.global_seed``; otherwise source
order is used. Each traversal visits all retained structures before repeating.

``balance_batches`` redistributes each global batch across GPUs while preserving
its structures and equal per-GPU sample counts. Choose ``atoms``, ``edges`` or
nonnegative integer weights ``[atom_weight, edge_weight]`` with at least one
positive entry. For example, ``[20, 1]`` uses ``20 * atoms + edges``. ``null``
keeps strided assignment. Balancing requires ``epoch_size`` and cached graph
sizes, and retains one int64 cost per selected structure per rank. Choose the
cost by measuring complete training steps on the intended model and hardware.

The requested count rounds up to a multiple of ``world_size`` when
``drop_last: false``, or ``world_size * batch_size`` when ``true``. Additional
samples come from the continuing stream. For two GPUs, batch size four and
``epoch_size: 11``, the actual count is respectively 12 or 16. Alignment may
cause repeats within an epoch if the count exceeds the dataset size.

Resume using ``resume_from_model`` with a full checkpoint
(``save_weights_only: false``). The sampler restores completed batches, excluding
prefetched samples. Keep the same dataset selection and order, epoch size, seed,
shuffle setting, world size, batch size, ``drop_last`` and ``balance_batches``.
This restores sampling order, not random augmentation or dropout state.
Keep ``trainer.limit_train_batches: 1.0`` and ``trainer.overfit_batches: 0``.

Augmentation
------------

``augmentation`` accepts a name or list; ``null`` and ``[]`` disable it.
Transformations act only on training samples and preserve cached graphs.

* ``spin_rotation`` applies a uniform global rotation to
  ``initial_noncollinear_magmoms`` and ``noncollinear_magnetic_forces``.
* ``time_reversal`` randomly reverses both properties together.

Rotations are global per structure, preserving exchange interactions. For a SOC
architecture trained on non-SOC data, spin rotation samples the independent spin
symmetry; standard e3nn also needs time reversal. The explicit non-SOC
architecture already enforces both symmetries. For zero-field SOC data, standard
e3nn uses time reversal; the complete time-reversal architecture needs neither
augmentation.

Properties
----------

Explicit ``dataset.keys`` take precedence over canonical names and ASE aliases.
No calculator is executed while reading labels. Absent or ``None`` labels receive
zero weight; supplied labels must be finite. Invalid shapes, negative weights,
non-integer fidelities and corrupt records raise errors.

``per-system`` stores one value per structure; ``per-atom`` stores one per atom.
``per-edge`` follows ``edge_index`` and periodic images. ``per-pair`` includes all
ordered atom pairs, including self pairs, in ``i * N + j`` order. Hessians use
``(N**2, 3, 3)`` independently of cutoff. Edge and pair labels belong in
``atoms.info``. Batches concatenate entries without adding inter-structure pairs.

Periodic and non-periodic structures may share training batches. Physical cells
and PBC are preserved, including zero cells. Stress requires a periodic direction
and a nonsingular cell; omit stress or set its weight to zero otherwise.
Non-periodic structures still contribute energy and forces. The ASE calculator
raises ``PropertyNotImplementedError`` for undefined stress.

For TorchSim boundary-condition requirements, see :doc:`../torchSim`.

Example
-------

.. yaml-config:: ../../../../example/train/TACE/tace.yaml
   :path: dataset

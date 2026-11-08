.. _neighbor-lists:

Neighbor Lists
==============

Backends
--------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Backend
     - Device
     - Implementation
   * - ``matscipy``
     - CPU
     - Default; included with TACE
   * - ``ase``
     - CPU
     - ASE neighbor list; included with TACE
   * - ``vesin``
     - CPU
     - Vesin; install with ``pip install "tace[nl]"``
   * - ``alchemiops``
     - CUDA / CPU
     - ``nvalchemiops`` batch neighbor lists; FP64 geometry

The AlchemiOps package is ``nvalchemi-toolkit-ops``, imported as
``nvalchemiops``; use ``alchemiops`` as the TACE backend value:

.. code-block:: bash

   pip install nvalchemi-toolkit-ops

The full NValCHEMI simulation interface has additional dependencies; see
:ref:`nvalchemi-installation`.

.. _training-neighbor-lists:

Training
--------

Set ``dataset.neighborlist_backend`` (default: ``matscipy``).
For large-scale training, use AlchemiOps batch neighbor lists and an LMDB
graph cache. Each GPU call searches a packed batch of structures.

.. code-block:: yaml

   dataset:
     neighborlist_backend: alchemiops
     storage_mode: lmdb
     shard_dirs: [graphCache]
     graph_device: null
     graph_batch_size: 256
     graph_workers: 0

Apply these overrides to your training configuration, then launch on eight
visible GPUs:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 tace-train -cn tace.yaml

- ``graph_device: null`` uses all visible GPUs, or CPU if none are available.
  Use ``all`` to require GPUs, ``cuda:0`` for one GPU,
  ``[cuda:0, cuda:1]`` for selected GPUs, or ``cpu``.
- ``graph_batch_size`` is the number of structures per GPU construction
  batch; it is independent of the training-loader batch size.
- ``graph_workers`` controls CPU preprocessing threads, not the GPU count.
  Each selected GPU process reads structures and writes its own graph shards.
- Later training runs reuse completed graphs without repeating neighbor search.
  To rebuild with a different backend, use a fresh ``shard_dirs`` directory.

Reference throughput for this batched path is approximately **100 million
structures in 4 hours on one node with 8 GPUs**. Actual time depends on
GPU hardware, atoms per structure, cutoff, and source/storage throughput.

``memory`` constructs graphs once in RAM. ``on_the_fly`` rebuilds
neighbors for each requested batch and supports one graph device per process;
CUDA construction requires ``num_workers: 0`` for training, validation,
and statistics loaders. See :doc:`details/dataset` for storage and cache details.

Inference
---------

Each interface selects neighbors through its own entry point:

.. list-table::
   :header-rows: 1
   :widths: 25 40 35

   * - Interface
     - Selection
     - Example
   * - ASE / General ASE
     - ``neighborlist_backend``
     - :ref:`ase-neighbor-lists`
   * - TorchSim
     - ``neighbor_list_fn`` callable
     - :ref:`torchsim-neighbor-lists`
   * - ``tace-eval``
     - ``--nl_backend``; cached graphs reuse stored edges
     - :ref:`eval-neighbor-lists`
   * - OpenMM via ASE
     - ``neighborlist_backend`` on the ASE calculator
     - :doc:`openmm`
   * - LAMMPS / USPEX via LAMMPS
     - LAMMPS neighbor settings
     - :doc:`lammps`
   * - NValCHEMI
     - NValCHEMI constructs neighbors for ``TACEWrapper``
     - :doc:`nvalchemi`

With ASE and ``tace-eval``, AlchemiOps uses the current CUDA device when
available, otherwise CPU. The model's ``device`` option does not select
the neighbor-search device; use ``CUDA_VISIBLE_DEVICES`` to select the GPU.

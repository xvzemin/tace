.. _checkpoint-tutorial:

Checkpoint
==========

Every TACE or TECE model path must end with one of the following suffixes:
``.ckpt``, ``.pt``, or ``.pt2``. The suffix determines how TACE
loads the model and which workflows the file supports.

.. list-table:: Model formats
   :header-rows: 1
   :widths: 14 28 31 27

   * - Suffix
     - Contents
     - Primary use
     - Portability
   * - ``.ckpt``
     - Lightning training checkpoint
     - Resume training
     - Coupled to the TACE version
   * - ``.pt``
     - Model configuration and parameter state dictionary
     - Long-term storage and eager inference
     - Portable while the required TACE modules remain supported
   * - ``.pt2``
     - AOTInductor compiled package
     - Hardware-specific accelerated inference
     - Coupled to the deployment environment

Lightning Checkpoints: ``.ckpt``
--------------------------------

A ``.ckpt`` file is produced by Lightning during training. It stores the model
parameters together with the training state, including information required by
the optimizer, learning-rate scheduler, callbacks, epoch, and global step. It
is therefore the correct format for resuming an interrupted training run.

Lightning checkpoints are tied to the TACE version and internal model layout
used during training. Changes to model classes, configuration fields, or module
construction can prevent an older checkpoint from loading in a newer version.
A ``.ckpt`` file should consequently not be the only copy of a model intended
for long-term use.

After training, export the checkpoint to the state-dictionary ``.pt`` format:

.. code-block:: bash

   tace-export-eval \
     -m checkpoints_epoch/last.ckpt \
     -o model.pt \
     --backend state_dict \

Keep the original ``.ckpt`` while training may still need to be resumed, but
use the exported ``.pt`` file for evaluation, deployment, and archival.

Portable Models: ``.pt``
------------------------

The recommended TACE ``.pt`` model is a state-dictionary package containing:

* the model configuration;
* the parameter state dictionary;
* target and embedding properties;
* model statistics.

TACE reconstructs the model from the saved configuration and then loads its
parameters. The package remains usable across TACE versions as long as the
required modules and their configuration remain supported. Any model that must
be retained for long-term use should therefore be exported and stored as a
``.pt`` file.

For eager inference, load the exported package directly:

.. code-block:: python

   from tace.lightning import load_tace

   model = load_tace("model.pt", device="cuda")
   model.eval()

AOTI Packages: ``.pt2``
-----------------------

A ``.pt2`` file is an AOTInductor package compiled specifically for inference.
It does not contain optimizer state and cannot be used to resume training. Its
compiled native code depends on the deployment environment, including the
operating system, PyTorch and CUDA ABI, device type, and GPU architecture. A
CUDA package cannot be used for CPU inference, and a package should generally
be compiled on the deployment machine or a compatible system. TACE AOTI export
requires ``torch>=2.13``.

Keep the portable ``.pt`` model as the source model and generate ``.pt2``
packages only when inference acceleration is required. For ASE, TorchSim, or
other native TACE inference workflows, use:

.. code-block:: bash

   tace-export-eval \
     -m model.pt \
     -o model.pt2 \
     --backend aoti \
     --device cuda

For accelerated molecular dynamics through LAMMPS, use:

.. code-block:: bash

   tace-export-lammps \
     -m model.pt \
     -o model-lammps.pt \
     --aoti-package model-lammps.pt2 \
     --backend aoti \
     --device cuda

Recommended Lifecycle
---------------------

Use the formats in the following order:

.. code-block:: text

   training -> .ckpt -> state-dictionary .pt -> deployment-specific .pt2

* retain ``.ckpt`` files while exact training recovery is required;
* always export a final state-dictionary ``.pt`` model for long-term storage;
* regenerate ``.pt2`` packages from the ``.pt`` model for each deployment
  environment that requires AOTI acceleration.

All three formats---``.ckpt``, ``.pt``, and ``.pt2``---can be used directly
for inference. A ``.pt2`` package is optional and is only needed when AOTI
acceleration is desired. If compiling and managing a hardware-specific
``.pt2`` package is inconvenient, use the portable ``.pt`` model directly.

For additional backend and export options, see the
:ref:`TACE Export Tutorial <tace-export-tutorial>`.

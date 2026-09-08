TACE Scripts Tutorial
=====================

TACE installs command-line scripts for training, inference, data preparation,
model conversion, and deployment. Most scripts use ``argparse`` and support
``-h`` directly:

.. code-block:: bash

   tace-eval -h
   tace-export-eval -h

.. important::

   ``tace-train`` is launched through Hydra. Hydra resolves the configuration
   before running the command, so the corresponding YAML file must be available
   even when only requesting ``-h``. Run the command from the configuration
   directory and select the YAML file with ``-cn`` when needed:

   .. code-block:: bash

      cd example/train
      tace-train -cn tace.yaml -h

Script Overview
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Command
     - Purpose
   * - ``tace-train``
     - Train, validate, or test from a Hydra configuration
   * - ``tace-eval``
     - Run inference on an ASE-readable structure file
   * - ``tace-export-train``
     - Export an editable model for training or transfer
   * - ``tace-export-eval``
     - Export state-dict, full-model, or AOTI inference artifacts
   * - ``tace-export-lammps``
     - Export eager or AOTI-backed LAMMPS ML-IAP artifacts
   * - ``tace-compile``
     - Alias for ``tace-export-eval``
   * - ``tace-split``
     - Split an ASE-readable dataset into train/validation/test files
   * - ``tace-finetune``
     - Generate a fine-tuning and LoRA configuration
   * - ``tace-convert``
     - Convert model parameters, currently including LoRA merging
   * - ``tace-update``
     - Update model statistics such as atomic energies, scale, and shift
   * - ``tace-average``
     - Average parameters from models with identical architectures
   * - ``tace-plot-diatom``
     - Plot energy and force curves for every homonuclear pair in a model
   * - ``tace-clean``
     - Remove standard training outputs from the current directory

Training
--------

``tace-train`` uses Hydra. The ``-cn`` argument selects a YAML configuration
without its extension:

.. code-block:: bash

   tace-train -cn tace

If the configuration is named ``tace.yaml`` in the configured search path, the
default command is sufficient:

.. code-block:: bash

   tace-train

Hydra overrides can be appended on the command line. This is useful for short
experiments without editing the source YAML:

.. code-block:: bash

   tace-train -cn tace trainer.max_epochs=10 \
     dataset.train_dataloader.batch_size=8

Training checkpoints contain optimizer, scheduler, callback, and model state,
so keep ``.ckpt`` files when a run may need to be resumed. For resume settings,
see :doc:`details/resume`.

Inference
---------

``tace-eval`` reads structures through ``ase.io.read`` and writes predictions
through ``ase.io.write``. At minimum, provide the input and model paths:

.. code-block:: bash

   tace-eval \
     -i structures.xyz \
     -m model.ckpt \
     -o predict.xyz \
     --device cuda \
     -b 16

Supported model inputs include checkpoints, state-dict packages, serialized
full models, and compatible ``.pt2`` AOTI packages. Use ``-t 1`` to report test
metrics when references are present and ``-e 0`` to disable EMA checkpoint
parameters.

Property names in a dataset can be remapped with arguments such as
``--energy_key``, ``--forces_key``, and the other key options shown by
``tace-eval -h``.

Model Export
------------

Export commands, formats, AOTI requirements, output names, and deployment
examples are documented in the dedicated
:ref:`TACE Export Tutorial <tace-export-tutorial>`.

The three primary entry points are:

.. code-block:: bash

   tace-export-train -m model.ckpt
   tace-export-eval -m model.ckpt --backend state_dict
   tace-export-lammps -m model.pt --backend mliap

Use ``tace-export-eval --backend aoti`` for native ASE/TorchSim deployment and
``tace-export-lammps --backend aoti`` for compiled LAMMPS inference. AOTI
requires ``torch>=2.13``. ASE/TorchSim AOTI export automatically creates
dynamic synthetic graph inputs, so ``--sample`` is not required. Use ``-f`` or
``--fidelity_idx`` to select a fidelity during export.

Dataset Preparation
-------------------

``tace-split``
~~~~~~~~~~~~~~

Split a structure file into train, validation, and test sets by specifying the
number of configurations in each output:

.. code-block:: bash

   tace-split -i dataset.xyz -n 800 100 100 -s 42

For ``dataset.xyz``, the command writes ``dataset_train.xyz``,
``dataset_valid.xyz``, and ``dataset_test.xyz``. Each selected structure keeps
its original position in ``atoms.info["tace_index"]``. The requested counts
must not exceed the number of input structures.

Fine-Tuning and Model Conversion
--------------------------------

``tace-finetune``
~~~~~~~~~~~~~~~~~

Generate a fine-tuning template from an existing model:

.. code-block:: bash

   tace-finetune -m foundation.pt

This writes ``finetune_config.yaml`` in the current directory. By default the
base parameters are frozen and LoRA adapters are enabled. Review the generated
file before launching ``tace-train``.

``tace-convert``
~~~~~~~~~~~~~~~~

Merge trained LoRA parameters into the base model for inference or export:

.. code-block:: bash

   tace-convert -m lora-model.pt -t merge_lora

The output is ``lora-model.pt-merged_lora.pt``. The merged model no longer
requires separate LoRA adapter parameters.

Model Statistics and Averaging
------------------------------

``tace-update``
~~~~~~~~~~~~~~~

Update stored statistics using ``statistics_<fidelity_idx>.yaml`` files in the
current directory:

.. code-block:: bash

   tace-update \
     -m model.pt \
     -u atomic_energy scale shift

Available update fields are ``atomic_energy``, ``scale``, and ``shift``. The
result is written to ``new_statistics.pt``. Verify that every statistics file
uses the same element order and fidelity indexing as the model.

``tace-average``
~~~~~~~~~~~~~~~~

Average parameters from checkpoints or exported models with identical
architectures:

.. code-block:: bash

   tace-average -m epoch-90.ckpt epoch-95.ckpt epoch-100.ckpt -e 0

The command writes ``average_model-state.pt``. This is a manual stochastic
weight averaging workflow. EMA is disabled by default and should generally
remain disabled when averaging multiple time-adjacent checkpoints.

Diatomic Curves
---------------

``tace-plot-diatom`` reads the supported elements and cutoff directly from a
model and evaluates each supported homonuclear pair from 0 to cutoff:

.. code-block:: bash

   tace-plot-diatom -m model.ckpt

Pass ``--heteronuclear`` to include all supported unlike-element pairs as
well. This option applies both to the automatic scan and to structures selected
from a DFT input file.

The command selects CUDA when available, otherwise CPU, and writes
``diatom.png`` in the current directory.
An optional DFT dataset may be supplied as follows:

.. code-block:: bash

   tace-plot-diatom -m model.ckpt -i diatoms.xyz \
     --energy_key energy --forces_key forces

Inspection and Maintenance
--------------------------

``tace-clean``
~~~~~~~~~~~~~~

Remove standard run outputs from the current working directory:

.. code-block:: bash

   tace-clean

Targets include Hydra outputs, Lightning/W&B logs, checkpoint directories,
index files, and generated configuration files. The command intentionally
refuses cleanup when a target exceeds its size limit. Run it only from the
training directory whose generated outputs should be removed.

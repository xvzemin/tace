Training
========

.. note::

  Example config yaml files are provided in the 
  `GitHub repository example <https://github.com/xvzemin/tace/tree/main/example/train>`_. 

  Inherit the supplied base configuration and override only the settings needed
  for your experiment. See :doc:`details/defaults` for examples of configuration
  inheritance and merging.

  - Be aware that Python's ``None`` must be written as ``null`` in YAML format.
  - Bool values are recommended to be written as ``true`` or ``false`` in YAML format.

Once you have prepared the input files, you can start training using the command:

.. code-block:: bash

    tace-train -cn tace.yaml

For inference, and usage of other scripts, please refer to the scripts documentation: :doc:`Scripts <scripts>`

Input Files
-----------

At least **two types of input files** are required for training:

1. **tace.yaml** specify the model architecture, optimizer, scheduler, and other training parameters.  
2. **train/valid/test sets** provide the atomic structures and corresponding reference data for training.

- The **training set** is mandatory.  
- The **validation set** and **test set** are optional.

YAML Configuration
~~~~~~~~~~~~~~~~~~

The configuration file of **TACE** is organized into **different fields**.  
All other parameters must be placed under these fields. The examples in the
detailed pages below are extracted directly from the canonical
``example/train/TACE/tace.yaml`` file, so their values and comments remain
synchronized with the actual training configuration.

  defaults:
  resume_from_model: 
  finetune_from_model: 
  dtype:
  misc: 
  logger:
  trainer:
  callbacks:
  dataset:
  optimizer:
  scheduler:
  synth_metric: 
  loss:
  model:


Field Descriptions
^^^^^^^^^^^^^^^^^^

- **defaults** (Hydra feature)
  Selects inherited YAML files and their merge order. Keep ``_self_`` last so
  the current file overrides its bases; see :doc:`details/defaults`.

- **resume_from_model** 
  Path to a previously saved checkpoint. The model specified here must end with .ckpt 
  and must be a model saved using ``lightning.pytorch.callbacks.ModelCheckpoint``. 
  If this field is provided, training will **automatically resume** 
  from the given checkpoint instead of starting from scratch.  

- **finetune_from_model** 
  Path to a pretrained model. You can use either your own trained model or TACE's pretrained model we provide for 
  fine-tuning. The filename must end with .pt, .pth, or .ckpt.

- **misc** 
  Miscellaneous options that control settings of the experiment.  

- **dtype**
  Selects network precision, optional autocast, and matrix multiplication mode.
  ``follow_model`` also applies the model dtype to geometry, physical reductions,
  losses, and metrics. Graph data remain in float64.

- **logger** 
  Specifies the logger to use, such as ``wandb`` or ``tensorboard``.  
  If you are already using the WandB logger added in the ``default field`` it has already been specified in the logger 
  file, then you do not need to set this field again.

- **trainer**
  This section directly corresponds to an instance of  
  `PyTorch Lightning's Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.  
  Precision is controlled by ``dtype``; other ``Trainer`` arguments can be
  specified here.

- **callbacks**
  Functions executed during training, e.g., checkpointing, early stopping, learning rate monitoring, EMA, SWA, etc.

- **dataset**
  Defines paths and formats of training/validation/test data and dataloader.  

- **optimizer**
  Optimizer configuration (e.g., Adam, AdamW, Muon, SOAP), with parameters like learning rate and weight decay.  

- **scheduler**
  Learning rate scheduler (e.g., ReduceLROnPlateau, CosineAnnealingLR, WSD). 
  Defines how the learning rate changes during training.  

- **synth_metric**
  You can define a custom metric by assigning your own weights to different validation-set errors. 
  This metric can then be used for early stopping, learning-rate scheduling, and similar training controls.

- **loss**
  Specifies the loss function. (Mainly uses ``tace.utils.loss.NormalLoss``; for other advanced loss functions, refer to 
  the specific documentation.)

- **model**
  Specifies which model and model architecture parameters to use.

For detailed parameters in every field, see below:

.. toctree::
   :maxdepth: 1
   :caption: Detailed hyper-parameter description for each field.

   details/defaults.rst
   details/resume.rst
   details/finetune.rst
   details/misc.rst
   details/logger.rst
   details/trainer.rst
   details/callbacks.rst
   details/dataset.rst
   details/optimizer.rst
   details/scheduler.rst
   details/metric.rst
   details/loss.rst
   details/model.rst

Dataset Format
~~~~~~~~~~~~~~

Now, we support:

- **Any file readable by** `ase.io.read <https://docs.ase-lib.org/ase/io/io.html#module-ase.io>`_.

- **ASE database files** (``.db``) created with `ase.db <https://docs.ase-lib.org/ase/db/db.html>`_.

- **Fair aselmdb** (``.aselmdb``) created with `ase.db <https://docs.ase-lib.org/ase/db/db.html>`_.  

For large datasets, set ``dataset.storage_mode`` to ``lmdb``. ``tace-train``
constructs and caches the graphs automatically, and later runs reuse compatible
graphs from the configured shard directories.

Precision
~~~~~~~~~

.. code-block:: yaml

   dtype:
     model: float32
     follow_model: false
     autocast: null
     matmul: null

``model`` selects ``float32`` or ``float64`` for learned network parameters and
features. With ``follow_model: true``, geometry, physical constants, output
reductions, losses, and metrics also use this dtype. With the default ``false``,
these operations use float64. ``autocast`` selects ``null``, ``bfloat16``, or
``float16`` for supported network operations; mixed precision requires
``model: float32``. Unsupported fused low-precision
backends are rejected with a clear error; set ``autocast: null`` to use them.

``matmul`` accepts ``ieee`` or ``tf32``. With ``null``, it resolves from
``TACE_MATMUL_PRECISION`` or ``TACE_USE_TF32``; without either variable, training
enables TF32 for FP32 operations and inference disables it. Explicit conflicting
settings raise an error. Autocast and TF32 cannot be enabled together.
Configure precision here rather than through ``trainer.precision``.

Data loading, graph caches, and statistics use float64 regardless of these
settings. Changing network precision therefore reuses the same graph cache.
Using ``follow_model: true`` with float32 also rounds atomic reference energies
and large energy sums; check energy errors as well as training speed.
Lower-precision network arithmetic can still affect energies and derivatives;
float64 geometry and output accumulation do not make the network itself FP64.

Output Files
------------

During training, **TACE** automatically generates several directories and files to organize results and logs:

- **checkpoints_epoch/**  
  Stores trained model checkpoints, these files can be used to resume training, inference or test.

- **outputs/**  
  Contains standard output files, such as training logs and evaluation summaries.

- **lightning_logs or wandb_logs/**
  Logs experiment metrics and training progress if using the `Weights & Biases <https://wandb.ai/site>`_ logger. Other 
  logging backends can also be used.

- **statistics_0.yaml, statistics_1.yaml ...**  
  Stores statistical information about the training set, such as RMS information of forces.
  If you are using multi-head or multi-fidelity training, multiple statistics files will be generated for each computation fidelity_idx.

  Compatible files are reused. The data schema, source fingerprint, elements,
  and reference energies are checked before reuse; outdated statistics are
  recomputed from the graph cache when available.
  
- **_tace.yaml**  
  A full copy of the configuration file used for this run, which ensures reproducibility of the experiment.

Training Tutorial
=================

.. note::

  Example config yaml files are provided in the `GitHub repository example <https://github.com/xvzemin/tace/tree/main/example/train>`_. 

  During the process of using TACE, what you need to do is to modify the default yaml files we provide instead of writing a new one yourself.

  - Be aware that Python's ``None`` must be written as ``null`` in YAML format.
  - Bool values are recommended to be written as ``true`` or ``false`` in YAML format.
  - Not all fields in the ``YAML`` file are allowed to be omitted. We recommend using the official input file as much as possible and making only minimal modifications.
  

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
``example/train/tace.yaml`` file, so their values and comments remain
synchronized with the actual training configuration.

  defaults:
  resume_from_model: 
  finetune_from_model: 
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
  If you are not familiar with Hydra, you can safely ignore this field and keep the default configuration. 
  We will not cover it here.

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

- **logger** 
  Specifies the logger to use, such as ``wandb`` or ``tensorboard``.  
  If you are already using the WandB logger added in the ``default field`` it has already been specified in the logger 
  file, then you do not need to set this field again.

- **trainer**
  This section directly corresponds to an instance of  
  `PyTorch Lightning's Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.  
  All arguments accepted by ``Trainer`` can be specified here.

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

  These statistics are computed before each training starting. However, if the file already exists in the current directory, 
  the computation will be skipped and the existing file will be read instead. 
  This means you can modify the statistical information manually if needed.
  
- **_tace.yaml**  
  A full copy of the configuration file used for this run, which ensures reproducibility of the experiment.

.. note::
  Automatically reading the statistics is equivalent to running the dataloader once in advance, which may sometimes 
  cause subtle effects, though in most cases these can be ignored.






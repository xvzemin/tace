Training Tutorial
=================

Once you have prepared the input files, you can start training using the command:

.. code-block:: bash

    tace-train -cn *.yaml # Replace the wildcard with the actual name of your YAML file.

For inference, and usage of other scripts, please refer to the scripts documentation: :doc:`Scripts <scripts>`

Input Files
-----------

At least **two types of input files** are required for training:

1. **Configuration file(s)** in ``YAML`` format, which specify the model architecture, optimizer, scheduler, and other training parameters.  
2. **Dataset**, which provides the atomic structures and corresponding reference data for training.

- The **training set** is mandatory.  
- The **validation set** and **test set** are optional.

YAML Configuration
~~~~~~~~~~~~~~~~~~

All training configurations can either be defined in a **single YAML file** or split into **multiple YAML files** and merged together through a main configuration file. This flexible design is powered by `Hydra <https://hydra.cc/>`_, making it easy to reuse and modify components.

The configuration file of **TACE** is organized into **different fields**.  
All other parameters must be placed under these fields.  
Among them, ``defaults`` is a special keyword provided by `Hydra <https://hydra.cc/>`_ for loading additional configuration files.  
It can be safely ignored if everything is written in a single YAML file.

.. note::

  Example yaml files are provided in the `GitHub repository example <https://github.com/xuzemin/tace/tree/main/example>`_. 

  The configurations we provide are:

  - A directory named ``config`` containing some yaml files.
  - A file named ``tace.yaml``.

  You can copy both the ``config`` directory and the ``tace.yaml`` file into the directory where your task starts.

  During the process of using TACE, what you need to do is to modify the default yaml files we provide instead of writing a new one yourself.

  - Be aware that Python's ``None`` must be written as ``null`` in YAML format.
  - Bool values are recommended to be written as ``true`` or ``false`` in YAML format.
  - Not all fields in the ``YAML`` file are allowed to be omitted. We recommend using the official input file as much as possible and making only minimal modifications.
  
.. code-block:: yaml
  # Main yaml file

  defaults: 
  dataset:
  misc: 
  trainer:
  callbacks:
  optimizer:
  scheduler:
  loss:
  logger:
  model:
  resume_from_model: 
  finetune_from_model:

Field Descriptions
^^^^^^^^^^^^^^^^^^

- **defaults** (Hydra feature)  
  Used to compose configurations from multiple YAML files. Advanced users can modularize configs, but beginners may use the config directory and yaml files we provide.  

- **dataset**
  Defines paths and formats of training/validation/test data and dataloader.  
  
- **misc** 
  Miscellaneous options that control settings of the experiment.  

- **trainer**
  This section directly corresponds to an instance of  
  `PyTorch Lightning's Trainer <https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.  
  All arguments accepted by ``Trainer`` can be specified here.

- **callbacks**
  Functions executed during training, e.g., checkpointing, early stopping, learning rate monitoring, EMA, SWA, etc.

- **optimizer**
  Optimizer configuration (e.g., Adam, AdamW), with parameters like learning rate and weight decay.  

- **scheduler**
  Learning rate scheduler (e.g., ReduceLROnPlateau, CosineAnnealingLR). Defines how the learning rate changes during 
  training.  

- **loss**
  Specifies the loss function. (Mainly uses ``tace.utils.loss.NormalLoss``; for other advanced loss functions, refer to 
  the specific documentation.)

- **logger** 
  Specifies the logger to use, such as ``wandb`` or ``tensorboard``.  
  If you are already using the WandB logger added in the ``default field`` it has already been specified in the logger 
  file, then you do not need to set this field again.

- **model**
  Specifies which model and model architecture parameters to use.
  If you are already using tace added in the ``default field`` it has already been specified, then you do not need to 
  set this field again.

- **resume_from_model** 
  Path to a previously saved checkpoint. The model specified here must end with .ckpt and must be a model saved 
  using ``lightning.pytorch.callbacks.ModelCheckpoint``. If this field is provided, training will **automatically resume** 
  from the given checkpoint instead of starting from scratch.  

- **finetune_from_model** 
  Path to a pretrained model. You can use either your own trained model or TACE's pretrained model we provide for 
  fine-tuning. The filename must end with .pt, .pth, or .ckpt. We also support some special fine-tuning workflows, 
  such as pretraining on direct forces and direct stress after converting to a conservative model for further fine-tuning.

For detailed parameters in every field, see below:

.. toctree::
   :maxdepth: 1
   :caption: Detailed hyper-parameter description for each field.

   details/defaults.rst
   details/dataset.rst
   details/misc.rst
   details/trainer.rst
   details/callbacks.rst
   details/optimizer.rst
   details/scheduler.rst
   details/loss.rst
   details/model.rst
   details/logger.rst
   details/resume.rst
   details/finetune.rst
   details/metric.rst
   
Dataset Format
~~~~~~~~~~~~~~

Now, we support:

- **Any file readable by** `ase.io.read <https://wiki.fysik.dtu.dk/ase/ase/io/io.html>`_  

- **ASE database files** (``.db``) created with `ase.db <https://wiki.fysik.dtu.dk/ase/ase/db/db.html>`_.  

For large datasets, we support preprocessing and graph construction in advance using the ``tace-graph`` command :doc:`Scripts <scripts>`

Output Files
------------

During training, **TACE** automatically generates several directories and files to organize results and logs:

- **checkpoints/**  
  Stores trained model checkpoints, these files can be used to resume training, inference or test.

- **outputs/**  
  Contains standard output files, such as training logs and evaluation summaries.

- **wandb_logs/**
  Logs experiment metrics and training progress if using the `Weights & Biases <https://wandb.ai/site>`_ logger. Other 
  logging backends can also be used.

- **statistics_0.yaml, statistics_1.yaml ...**  
  Stores statistical information about the training or validation set, such as RMS information of forces.
  If you are using multi-head or multi-fidelity training, multiple statistics files will be generated for each computation level.

  These statistics are computed before each training starting. However, if the file already exists in the current directory, 
  the computation will be skipped and the existing file will be read instead. 
  This means you can modify the statistical information manually if needed, for example, if you want to adjust certain 
  statistics inside the model.
  
- **_tace.yaml**  
  A full copy of the configuration file used for this run, which ensures reproducibility of the experiment.

.. note::
  Automatically reading the statistics is equivalent to running the dataloader once in advance, which may sometimes 
  cause subtle effects, though in most cases these can be ignored.








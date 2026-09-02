Finetune Tutorial
=================

This section introduces how to finetune **TACE** models.

We provide a variety of pretrained TACE foundation models. You can finetune these
foundation models directly, or finetune your own models.

Currently, TACE supports three major finetuning strategies:

1. Full parameters 
2. Freeze parameters
3. Low-Rank Adaptation (LoRA)


Motivation for Finetuning
-------------------------

The main goal of finetuning is to **preserve as much knowledge as possible from the
foundation model**, while achieving high accuracy on your target task.

Therefore,  when finetuning dataset is relatively small, full-parameter finetuning is discouraged. 
Full parameters finetuning may lead to:

- Overfitting due to limited training data
- Catastrophic forgetting of knowledge learned during pretraining

To mitigate these issues, parameter-efficient finetuning methods such as LoRA and
freezing pretrained parameters are generally preferred.

Finetuning Strategies
---------------------

Below we describe the supported finetuning strategies and how to use them in practice.

Freezing Pretrained Parameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Freezing pretrained parameters is not discussed in detail here. In the finetuning
configuration files that we automatically generate, **all pretrained parameters are
frozen by default**.

This design choice helps preserve the knowledge learned during pretraining and avoids
overfitting when the finetuning dataset is limited.

Low-Rank Adaptation (LoRA)
^^^^^^^^^^^^^^^^^^^^^^^^^^

In short, LoRA introduces additional trainable parameters on top of an existing model,
according to user-specified configurations. These trainable parameters follow the same
structural pattern as the original model layers.

After training, the LoRA weights are **merged into the base model weights**, so the final
exported model does **not introduce any additional parameters** compared to the original
model.

LoRA is mainly controlled by two key hyperparameters:

- ``rank`` (int): determines the number of additional trainable parameters introduced
  during LoRA finetuning.
- ``alpha`` (float): controls the strength of the LoRA update.

The ``rank`` parameter is typically in the range of **4 to 32**. A larger
rank increases the number of trainable LoRA parameters.

The ``alpha`` parameter is commonly set within the range of **r to 2r**, where ``r`` denotes the
LoRA rank.

Replay Data
^^^^^^^^^^^

During finetuning, it is possible to mix in a portion of the training data used for
the foundation model in order to mitigate catastrophic forgetting.

However, in our current version, we do not include replay data such as
multi-fidelity or multi-head training data during finetuning by default.


Updating Atomic Energies
------------------------

When the finetuning dataset uses a different reference-energy convention from
the pretrained model, update the per-element atomic energies before starting
finetuning. The model uses these values as the composition-dependent energy
baseline, while the learned model predicts the remaining energy contribution.

``tace-update`` reads ``statistics_<fidelity_idx>.yaml`` files from the current
directory. For example, for a single-fidelity model containing H, C, N, and O,
prepare ``statistics_0.yaml`` with the new atomic energies:

.. code-block:: yaml

   atomic_energy:
     1: -13.587222780835477
     6: -1029.4889999855063
     7: -1484.9814568572233
     8: -2041.9816003861047

The keys may be atomic numbers, as above, or chemical symbols. The dictionary
must contain the same elements as the pretrained model. For a multi-fidelity
model, ``statistics_0.yaml``, ``statistics_1.yaml``, and so on update the
corresponding fidelity. A missing file or a file without an
``atomic_energy`` entry leaves that fidelity unchanged.

Run the update from the directory containing the statistics files:

.. code-block:: bash

   tace-update -m TACE-OAM-7M.pt -u atomic_energy

Example
-------

First, download a pretrained model from the
`TACE Foundation Models repository <https://github.com/xvzemin/tace-foundations>`_
(you may also use a model pretrained by yourself).

You can then follow the example below, which consists of three main steps:

1. Before training, use ``tace-finetune`` to automatically generate a finetuning
   configuration file named ``finetune_config.yaml``. You can modify this file
   to adjust the desired finetuning parameters.

2. After preparing your training config, start the finetuning process using
   ``tace-train``. TACE will automatically load and apply the settings from
   ``finetune_config.yaml``.

3. After training is completed, use ``tace-convert`` to convert the generated
   LoRA checkpoint (``*.ckpt`` file) into a standard model by merging the LoRA
   weights into the base model. The resulting model can then be deployed for
   production use.

Example commands are shown below:

.. code-block:: bash

   tace-finetune -m TACE-OMat24-7M.pt

   # Start finetuning (configuration file specified as needed)
   tace-train -cn tace

   # Merge LoRA weights into the base model
   tace-convert -m checkpoints_epoch/last.ckpt --type merge_lora

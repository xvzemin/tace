resume_from_model
=================

Set ``resume_from_model`` to a full training checkpoint, or ``null`` to start
without resuming:

.. code-block:: yaml

   resume_from_model: /path/to/model.ckpt

The checkpoint restores optimizer state, including its learning rate. Trainer
settings such as ``max_epochs`` and gradient clipping use the current
configuration. For sampler recovery requirements, see :doc:`dataset`.

The initial validation sanity check covers only a few batches, so its metrics
can differ from a completed validation epoch.

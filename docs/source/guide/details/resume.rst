resume_from_model
=================

The ``resume_from_checkpoint`` field does not contain any additional parameters.  
You only need to provide the path to a ``*.ckpt`` file in order to resume training.  
If you do not need to resume training, this field can be omitted or set to ``null``.

.. note::

   If your training process depends on a validation set, keep in mind that
   the validation metrics may differ before and after resuming.  
   This is expected because we perform an automatic finite small step **sanity check** before
   training starts, where only a few batches are used for validation.

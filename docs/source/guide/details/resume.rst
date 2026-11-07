resume_from_model
=================

The ``resume_from_model`` field does not contain any additional parameters.
You only need to provide the path to a ``*.ckpt`` file to resume training.  
If resuming training is not required, this field should be set to ``null``.

When resuming training, changes to configuration options such as ``max_epoch`` or 
gradient clipping will take effect, as they are not recorded in the checkpoint. 
However, modifications to parameters like the learning rate will have no effect.


.. note::

   If your training process depends on a validation set, keep in mind that
   the validation metrics may differ before and after resuming.  
   This is expected because we perform an automatic finite small step **sanity check** before
   training starts, where only a few batches are used for validation.

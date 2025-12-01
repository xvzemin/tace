finetune_from_model
===================

The ``finetune_from_model`` field does not contain any additional parameters.  
You only need to provide the path to a ``.ckpt, .pt or .pth`` file in order to finetunning.  
If you do not need to finetunnning, this field can be omitted or set to ``null``.

.. note::
    Support direct (forces, stress, virials) pretrained, then switch to conservative model.
    But, at present, our finetuning functionality is limited, as the author has not yet begun implementing finetuning. 
    It does not currently support fixing partial weights, freezing layers, or extracting specific components of the model. 
    More finetuning features will be added in the future.

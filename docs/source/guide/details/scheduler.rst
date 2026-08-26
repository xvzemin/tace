scheduler
=========

Types of Learning Rate Schedulers
---------------------------------

There are generally two types of learning rate schedulers:

1. **Validation-based schedulers**  
   Adjust the learning rate based on performance on the validation set.  
   Example: ``torch.optim.lr_scheduler.ReduceLROnPlateau``

2. **Fixed-step schedulers**  
   Reduce the learning rate in a predefined manner.  

In this document, we take ``torch.optim.lr_scheduler.ReduceLROnPlateau`` as an example.  
For other schedulers, please check the `official PyTorch documentation <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`_.

Custom Learning Rate Schedulers
-------------------------------

In addition to the official PyTorch learning rate schedulers,  
you can also use custom schedulers implemented in the codebase.  

If you define your own scheduler in ``tace.utils.lr_scheduler``,  
you only need to modify the ``_target_`` field accordingly.

Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: scheduler


logger
======

The recommended loggers are ``lightning``. 
By default, wandb is configured to store logs locally. 
If you prefer to use wandb online, manually modify misc.env.WANDB_MODE in ``tace.yaml``.


.. note::

   We do not guarantee that the parameters recorded by Weights & Biases (wandb)
   are always correct. We recommend directly referring to the errors printed in
   the standard output. At the end of each epoch, all error metrics are
   automatically printed.


Example
-------
.. literalinclude:: ../../../../example/train/tace.yaml
   :language: yaml
   :start-at: # # You can specify the logger here
   :end-before: trainer:

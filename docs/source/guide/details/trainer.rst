trainer
=======

For a complete list of parameters, see the `Lightning Trainer documentation
<https://lightning.ai/docs/pytorch/stable/common/trainer.html>`_.

Set numerical precision through the top-level ``dtype`` section, not
``trainer.precision``. TACE preserves float64 data and geometry while selecting
the network precision independently.

Example
-------

.. yaml-config:: ../../../../example/train/TACE/tace.yaml
   :path: trainer

optimizer
=========

Any optimizer supported by PyTorch or third party can be used.

.. note::

   The safest choice is to use Adam or AdamW. Although newer optimizers such as
   Muon and SOAP may improve convergence speed and in-domain accuracy, they can
   potentially reduce the model's extrapolation ability.

Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: optimizer

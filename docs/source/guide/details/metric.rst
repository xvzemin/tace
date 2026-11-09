synth_metric
============

Metric names combine the stage, property and reduction, such as
``val/energy_mae``, ``val/energy_per_atom_rmse`` or ``val/forces_mae``.
Loss is available as ``train/loss`` and ``val/loss``.

``synth_metric`` configures the mixing weights for the validation-only composite
metric ``val/synth_metric``.

Example
-------

.. yaml-config:: ../../../../example/train/TACE/tace.yaml
   :path: synth_metric

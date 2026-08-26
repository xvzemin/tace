synth_metric
============

Monitor_metric_name can be accessed by combining the stage (train | val) with 
the property name and the metric type (mae or rmse), for example:  
``val/energy_mae``, ``val/energy_per_atom_rmse``, ``val/forces_mae``, 
``val/stress_rmse``, ``val/loss``, ``train/loss``, etc.  

Specifically, we have a composite metric, ``val/synth_metric``, for which a mixing 
ratio can be specified. This is particularly useful when training on large datasets with outliers.  

Note that the ``val/synth_metric`` is only supported for the validation, ``train/synth_metric`` is not supported.  

Example
-------

.. yaml-config:: ../../../../example/train/tace.yaml
   :path: synth_metric

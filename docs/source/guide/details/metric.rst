synth_metric
============

Example
-------

.. code-block:: yaml

    synth_metric: 
    # All metrics can be accessed by combining the stage (train | val) with the property name and the metric type (mae or rmse), for example:
    # val/energy_mae, val/energy_per_atom_rmse, val/forces_mae, val/stress_rmse, val/loss, train/loss ...
    # Specifically, we have a composite metric, val/synth_metric, for which we can specify a mixing ratio. This is particularly useful when training on large datasets with outliers.
    monitor_metric_name: val/synth_metric # use for variable interpolation
    # MAE
    val/energy_mae: 1.0
    val/energy_per_atom_mae: 5.0
    val/forces_mae: 5.0
    val/stress_mae: 2.5
    # RMSE
    val/energy_per_atom_rmse: 1.0
    val/forces_rmse: 1.0
    val/stress_rmse: 1.0

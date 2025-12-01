callbacks
=========

Example
-------

.. code-block:: yaml

    callbacks:
    early_stopping:
        _target_: lightning.pytorch.callbacks.EarlyStopping
        verbose: true
        log_rank_zero_only: true
        monitor: ${synth_metric.monitor_metric_name}
        min_delta: 1e-5 
        patience: 80

    ema: # ema is always recommended
        _target_: tace.utils.callbacks.EMACallback
        decay: 0.995 # typically >= 0.99
        use_num_updates: true

    checkpoint: # at leas one checkpoint and key=checkpoint is required
        _target_: lightning.pytorch.callbacks.ModelCheckpoint
        dirpath: checkpoints
        filename: TACE-OMat24-{epoch}-{step}-{${synth_metric.monitor_metric_name}:.4f}
        # monitor: ${synth_metric.monitor_metric_name}
        save_top_k: -1
        save_last: true
        every_n_epochs: 1
        # every_n_train_steps: 2000
        # mode: min
        save_weights_only: false
        auto_insert_metric_name: false
        verbose: true

    # checkpoint_step:
    #   _target_: lightning.pytorch.callbacks.ModelCheckpoint
    #   dirpath: checkpoints
    #   filename: TACE-OMat24-{epoch}-{step}-{${synth_metric.monitor_metric_name}:.4f}
    #   # monitor: ${synth_metric.monitor_metric_name}
    #   save_top_k: -1
    #   # every_n_epochs: 1
    #   every_n_train_steps: 2000
    #   mode: min
    #   save_weights_only: false
    #   auto_insert_metric_name: false
    
    # swa:
    # _target_: lightning.pytorch.callbacks.StochasticWeightAveraging
    # swa_lrs: 1e-4
    # swa_epoch_start: 1
    # annealing_epochs: 1
    # annealing_strategy: linear
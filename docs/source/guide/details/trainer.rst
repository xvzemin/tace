trainer
=======

Example
-------

For a complete list of all parameters and their detailed descriptions, please refer to the official PyTorch Lightning Trainer documentation: https://lightning.ai/docs/pytorch/stable/common/trainer.html

.. code-block:: yaml

  trainer:
    # As long as your system has NVIDIA GPUs, the following configuration will automatically detect and use all available GPUs for training.
    # If you wish to control which GPUs are visible, you can do so by setting the environment variable like export CUDA_VISIBLE_DEVICES=0,2. 
    _target_: lightning.Trainer
    limit_train_batches: 0.1
    num_nodes: 1   
    accelerator: auto
    devices: auto        # auto or list
    max_time: '90:00:00:00' # 90 days
    max_epochs: 1000
    min_epochs: 1        # null or int > 1
    precision: 64       # 64 > 32 > bf16-mixed > 16-mixed > 16-true
    # strategy: auto
    strategy:
      _target_: tace.utils.strategy.SimpleDDPStrategy  # Recommended for multi-GPU training. 
                                                      # For single-GPU training, simply use `auto`.
    gradient_clip_val: 1.0    
    enable_progress_bar: true # If you don't want to see the progress bar, you can turn it off
    log_every_n_steps: 10000

    # Generally, no modification is required
    enable_model_summary: true
    enable_checkpointing: true 
    check_val_every_n_epoch: 1
    detect_anomaly: false
    inference_mode: false
    deterministic: false  
    # benchmark: true  


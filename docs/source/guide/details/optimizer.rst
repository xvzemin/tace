optimizer
=========

Example
-------

.. code-block:: yaml

  optimizer:
    _target_: torch.optim.Adam
    lr: 1e-3
    weight_decay: 1e-8 # 1e-8, 5e-7
    amsgrad: true 

  # optimizer:
  #  _target_: torch.optim.AdamW
  #  lr: 1e-3
  #  weight_decay: 1e-8 # 1e-3, 1e-8
  #  amsgrad: false
Loss
====

We support two major categories of loss functions, along with several
additional custom losses. Users can also easily define and integrate
their own loss functions when needed.

Example
-------
.. code-block:: yaml

  # Loss Type 1
  # - In most cases, if training is stable (no strong oscillation),
  #   it is recommended to use MSE for all loss terms.
  # - If noticeable oscillations or outliers appear during training,
  #   consider switching the corresponding terms to huber or l2mae loss.
  # - When energy weight is fixed to 1.0:
  #     * forces weight: typically in the range [1, 10]
  #     * stress weight: typically in the range [0.5, 10]
  # - For other physical quantities, loss weights must be tuned by yourself
  loss:
    _target_: tace.utils.loss.NormalLoss
    loss_property: [energy, forces, stress] 
    # loss_property: [energy, forces, stress, polarization, conservative_polarizability, bonn_effective_charges] 
    loss_function_name:  # prefix can be one of ["mse", "mae", "l2mae", "huber"]
      - mse_energy_per_atom
      - mse_forces
      - mse_stress
    loss_property_weights: [1, 8, 8] 
    loss_function_kwargs: 
      - {}
      - {}
      - {}

  # # Loss Type 2
  # # This loss does not require manually specified weights.
  # # All loss weights should be set to 1.0, as they will be
  # # automatically adjusted during training.
  # #
  # # Note:
  # # Although convenient, the final convergence quality is
  # # generally inferior to that achieved with well-tuned
  # # manually assigned weights. This loss is therefore mainly
  # # intended for toy experiments.
  # loss:
  #   _target_: tace.utils.loss.UncertaintyLoss
  #   loss_property: [energy, forces, stress] 
  #   loss_function_name: 
  #     - mse_energy_per_atom
  #     - mse_forces
  #     - mse_stress
  #   loss_property_weights: [1, 1, 1] 
  #   loss_function_kwargs: [{}, {}, {}]
  
Notes
-----

- For properties that are already *per-atom* quantities,  
  the ``per_atom`` suffix is not required and is not supported.
- ``loss_function_kwargs`` supplies extra arameters for each loss function. 
  For example, use ``{huber_delta: 0.01}`` with a Huber loss. 
  Parameters are checked.
- Every loss whose property has per-atom scope accepts ``element_weights``.
  The list must contain one value for every element in ``node_attrs`` and must
  follow increasing atomic-number order. Omitting ``element_weights`` is equivalent to
  assigning weight ``1.0`` to every element. For example:

  .. code-block:: yaml

    loss_function_name:
      - mse_energy_per_atom
      - huber_forces
    loss_function_kwargs:
      - {}
      - {huber_delta: 0.01, element_weights: [1.0, 2.0]}

  Here the two element weights correspond to the two elements represented by
  ``node_attrs``, ordered by increasing atomic number.
- You can easily define your own loss function without any additional code.
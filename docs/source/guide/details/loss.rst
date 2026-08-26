Loss
====

We support two major categories of loss functions, along with several
additional custom losses. Users can also easily define and integrate
their own loss functions when needed.

- For properties that are already *per-atom* quantities,  
  the ``per_atom`` suffix is not required and is not supported.
- ``loss_function_kwargs`` supplies extra arameters for each loss function. 
  For example, use ``{huber_delta: 0.01}`` with a Huber loss. 
  Parameters are checked.
- Every loss whose property has per-atom scope accepts ``element_weights``.
  The list must contain one value for every element in ``node_attrs`` and must
  follow increasing atomic-number order. Omitting ``element_weights`` is equivalent to
  assigning weight ``1.0`` to every element. During force-statistics calculation,
  TACE reports candidate lists under ``recommended_force_element_weights`` for
  :math:`\alpha=0`, :math:`0.25`, :math:`0.5`, :math:`0.75`, and :math:`1`.
  For element :math:`Z`, it estimates
  :math:`A_Z=\pi_Z\langle\|\mathbf F\|^2/3\rangle_Z` and uses
  :math:`w_Z\propto A_Z^{-\alpha}`. The weights are clipped to ``[0.25, 4.0]``
  and normalized to have atom-count-weighted mean one. These values are
  recommendations only; copy the selected list into the loss configuration:

  .. code-block:: yaml

    loss_function_name:
      - mse_energy_per_atom
      - huber_forces
    loss_function_kwargs:
      - {}
      - {huber_delta: 0.01, element_weights: [1.0, 2.0]}

Example
-------
.. yaml-config:: ../../../../example/train/tace.yaml
   :path: loss


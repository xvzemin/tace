TACE-QEq (Under Reconstruction)
===============================

TACE-QEq predicts atomic charges with the Lagrangian charge-equilibration
readout while enforcing the total charge of each structure. Enable it by
adding ``charges`` to the training targets and selecting the ``lagrangian``
charge method.

The dataset must contain a per-atom charge array and a per-structure total
charge. Map their names and extend an existing energy/forces configuration as
follows:

.. code-block:: yaml

   dataset:
     keys:
       charges_key: charges
       total_charge_key: total_charge

   loss:
     loss_property: [energy, forces, charges]
     loss_function_name:
       - mse_energy_per_atom
       - mse_forces
       - mse_charges
     loss_property_weights: [1.0, 5.0, 1.0]

   model:
     config:
       special:
         charges:
           method: lagrangian

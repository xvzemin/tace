TorchSim calculator
===================

.. code-block:: bash

    pip install "tace[torchsim]"

.. code-block:: python

    from ase.build import bulk
    from torch_sim.io import atoms_to_state
    from tace.interface.torchsim import TACETorchSimCalc

    calc = TACETorchSimCalc("TACE-OAM-L")
    state = atoms_to_state([bulk("Si"), bulk("Cu")], device=calc.device, dtype=calc.dtype)
    results = calc(state)
    energy, forces, stress = results["energy"], results["forces"], results["stress"]

For a multi-fidelity model, set ``fidelity_idx`` on the calculator or pass
an integer tensor of shape ``(n_systems,)`` in ``state.system_extras["fidelity_idx"]``.

.. _torchsim-neighbor-lists:

Neighbor lists
--------------

``neighbor_list_fn`` defaults to ``torch_sim.neighbors.torchsim_nl``.
A custom callable takes ``(positions, cell, pbc, cutoff, system_idx)``
and returns ``(edge_index, system_mapping, unit_shifts)``.
See :doc:`neighbor_list` and the
`TorchSim examples <https://github.com/xvzemin/tace/tree/main/example/torchSim>`_.

.. autoclass:: tace.interface.torchsim.TACETorchSimCalc
   :no-members:

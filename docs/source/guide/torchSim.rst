TorchSim Calculator
===================

This tutorial demonstrates how to use a TACE model as a calculator within TorchSim.

TorchSim documentation: `torchsim <https://torchsim.github.io/torch-sim/>`_

Installation
------------

Install TACE with TorchSim support:

.. code-block:: bash

    pip install "tace[torchsim]"

TACE requires ``torch-sim-atomistic>=0.6.1``. The recommended version is
``0.6.2``. To install it explicitly:

.. code-block:: bash

    pip install "torch-sim-atomistic==0.6.2"

For optimization, molecular dynamics, and batched examples, see the
`TACE TorchSim examples <https://github.com/xvzemin/tace/tree/main/example/torchSim>`_.

Calculator
----------


.. code-block:: python

    import torch

    from tace.interface.torchsim import TACETorchSimCalc

    dtype = "float32"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "model.pt"
    fidelity_idx = 0

    calc = TACETorchSimCalc(
        model_path,
        fidelity_idx=fidelity_idx,
        device=device,
        dtype=dtype,
        compute_forces=True,
        compute_stress=True,
    )

Direct calls convert the state's inputs to the calculator's device and dtype
without modifying the state. For a multi-fidelity model, ``fidelity_idx`` sets
the default for all systems. To select a different fidelity for each system,
set ``state.system_extras["fidelity_idx"]`` to an integer tensor of shape
``(n_systems,)``.

.. _torchsim-neighbor-lists:

Neighbor lists
--------------

Pass a callable as ``neighbor_list_fn``. The default, ``torchsim_nl``,
selects an installed backend automatically. TorchSim 0.6.2 provides these
functions in ``torch_sim.neighbors``:

.. list-table::
   :header-rows: 1
   :widths: 55 45

   * - Callable
     - Neighbor search
   * - ``torchsim_nl``
     - Automatic: AlchemiOps, then Vesin, then PyTorch
   * - ``alchemiops_nl_n2``, ``alchemiops_nl_cell_list``
     - Batched AlchemiOps all-pairs or cell-list search
   * - ``vesin_nl``, ``vesin_nl_ts``
     - Vesin or its TorchScript implementation
   * - ``torch_nl_n2``, ``torch_nl_linked_cell``
     - PyTorch all-pairs or linked-cell search

Examples
~~~~~~~~

>>> from torch_sim.neighbors import alchemiops_nl_cell_list
>>> from tace.interface.torchsim import TACETorchSimCalc
>>> calc = TACETorchSimCalc(
...     "model.pt", device="cuda", neighbor_list_fn=alchemiops_nl_cell_list
... )

Custom callables must accept ``(positions, cell, pbc, cutoff, system_idx)``
and return ``(edge_index, system_mapping, unit_shifts)``. See the
`TorchSim neighbor interface <https://github.com/TorchSim/torch-sim/blob/v0.6.2/torch_sim/neighbors/__init__.py>`_
and :doc:`neighbor_list`.

.. autoclass:: tace.interface.torchsim.torchsim.TACETorchSimCalc
   :no-members:
   :show-inheritance:

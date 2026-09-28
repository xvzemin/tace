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

.. autoclass:: tace.interface.torchsim.torchsim.TACETorchSimCalc
   :no-members:
   :show-inheritance:

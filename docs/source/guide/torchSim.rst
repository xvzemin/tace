TorchSim Calculator
===================

This tutorial demonstrates how to use a TACE model as a calculator within TorchSim.

TorchSim documentation: `torchsim <https://torchsim.github.io/torch-sim/>`_

Installation
------------

Install TACE with TorchSim support:

.. code-block:: bash

    pip install "tace[torchsim]"

TACE requires ``torch-sim-atomistic>=0.6.1``. Version ``0.6.1`` is the safest
and currently recommended version. TorchSim is under active development, so
compatibility with versions newer than ``0.6.1`` is not guaranteed. To use the
tested version explicitly:

.. code-block:: bash

    pip install "torch-sim-atomistic==0.6.1"

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

.. autoclass:: tace.interface.torchsim.torchsim.TACETorchSimCalc
   :no-members:
   :show-inheritance:

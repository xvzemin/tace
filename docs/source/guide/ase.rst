ASE Calculator Tutorial
=======================

This tutorial demonstrates how to use a TACE model as a calculator within ASE (Atomic Simulation Environment).

ASE Calculator documentation: `ASE Calculator <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html>`_

.. code-block:: python

    from ase.io import read
    from tace.interface.ase.calculator import TACECalculator

    DEVICE = 'cuda'            # Compute device, e.g., 'cpu' or 'cuda'
    DTYPE = 'float32'          # Tensor data type 'float32' or 'float64' or None
    MODEL_PATH = '.pt'         # Path to the model checkpoint, file ends with .pt, .pth or .ckpt
    atoms = read('*.xyz', 0)   #  Any ase readable files

    # For other parameters and usage, see the API documentation.
    calc = TACECalculator(
        model_path=MODEL_PATH,
        device=DEVICE,
        dtype=DTYPE,
        extra_compute_first_derivative = None,
        extra_compute_second_derivative  = None,
        level = 0, # fidelity level
    )
    atoms.calc = calc

.. autoclass:: tace.interface.ase.calculator.TACECalculator
   :no-members:
   :show-inheritance:
NValCHEMI Wrapper
=================

NValCHEMI provides common data, neighbor-list, model, and execution interfaces
for GPU atomistic simulation workflows. The TACE wrapper adapts an
energy-based TACE model to the NValCHEMI model contract and exposes energy,
conservative forces, and stress.

Install both optional dependencies as described in
:ref:`nvalchemi-installation`, then import the existing wrapper with:

.. code-block:: python

   from tace.interface.nvalchemi import TACEWrapper

A TACE checkpoint can be loaded directly through the wrapper factory:

.. code-block:: python

   model = TACEWrapper.from_checkpoint(
       "model.ckpt",
       device="cuda",
   )

Importing this module requires ``nvalchemi-toolkit`` and
``nvalchemi-toolkit-ops``. These packages remain optional for every other TACE
workflow.

Upstream projects:

- `nvalchemi-toolkit <https://github.com/NVIDIA/nvalchemi-toolkit>`_
- `nvalchemi-toolkit-ops <https://github.com/NVIDIA/nvalchemi-toolkit-ops>`_

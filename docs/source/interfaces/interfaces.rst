Interfaces
====================

.. automodule:: tace
    :noindex:

For development convenience, all physical property predictions are currently centralized in ``tace.models.tace.TACE`` 
class. Therefore, this version does not support jit and any acceleration.

Supported Interfaces List
-------------------------

* **Python**

  - ✅ Supports integration with **LAMMPS** (`LAMMPS-MLIAP`).

  - ✅ Supports integration with **ASE Calculator** (`ase.Calculator`).
  
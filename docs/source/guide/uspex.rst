USPEX
=====

This tutorial demonstrates how to use a TACE model within USPEX.

Since USPEX currently does not provide a direct external MLIP interface, 
we integrate TACE through the following workflow::

    USPEX -> LAMMPS-ML-IAP -> TACE

At present (as of May 8, 2026), we restrict support to USPEX-v10.5, since
USPEX-2025 does not yet provide LAMMPS support.

See the `USPEX website <https://uspex-team.org/en>`_ for installation and
licensing information.

To use this workflow, both USPEX and LAMMPS must be installed. Currently,
only NVIDIA GPUs (not allow cpu) are supported.


.. An example can be found at:

.. https://github.com/xvzemin/tace/tree/main/example/uspex


Step 1: Install USPEX
----------------------

.. code-block:: bash

    # Install TACE with Python 3.9 when using USPEX-v10.5,
    # since Python 3.9 is the highest USPEX-v10.5 supported Python version.

    tar -xvf USPEX-10.5.tar.gz
    cd USPEX_v10.5
    bash install.sh
    # ...

Step 2: Install LAMMPS
----------------------

See LAMMPS-ML-IAP Tutorial.


Step 3: Example
---------------

.. code-block:: bash

    mkdir test
    cd test

    # Generate the USPEX tutorial input files
    USPEX -c 04

    # Modify the USPEX input file and the LAMMPS input file
    # according to your own requirements
    # ...

    # Run USPEX
    USPEX -r





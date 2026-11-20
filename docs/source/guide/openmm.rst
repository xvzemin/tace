OpenMM
======

Use the ASE potential in `OpenMM-ML <https://github.com/openmm/openmm-ml>`_.
Install the optional dependency with ``pip install "tace[openmm]"``.
Pass a PDB topology and a TACE ASE calculator to ``createSystem``:

.. code-block:: python

    from openmm.app import PDBFile
    from openmmml import MLPotential
    from tace.interface.ase import TACEAseCalc

    pdb = PDBFile("structure.pdb")
    calculator = TACEAseCalc("TACE-OAM-L")
    system = MLPotential("ase").createSystem(pdb.topology, calculator=calculator)

See :ref:`ase-neighbor-lists` for ``neighborlist_backend``.

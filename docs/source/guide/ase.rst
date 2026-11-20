ASE calculators
===============

.. code-block:: python

    from ase.build import bulk
    from tace.interface.ase import TACEAseCalc

    atoms = bulk("Cu")
    atoms.calc = TACEAseCalc("TACE-OAM-L", dtype="float32")
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress()

For D3 dispersion, use ``atoms.calc = add_dispersion(atoms.calc)`` with
``add_dispersion`` from ``tace.interface.ase``.

.. _ase-neighbor-lists:

Neighbor lists
--------------

``neighborlist_backend`` selects the backend for either calculator.
The default is ``"matscipy"``. See :doc:`neighbor_list`.

Additional inputs
-----------------

``GeneralTACEAseCalc.keys`` maps each model input to a key in
``Atoms.info`` (per system) or ``Atoms.arrays`` (per atom):

.. code-block:: python

    from ase.io import read
    from tace.interface.ase import GeneralTACEAseCalc

    atoms = read("magnetic_structure.xyz")
    atoms.calc = GeneralTACEAseCalc(
        "mtace.pt", keys={"initial_noncollinear_magmoms": "spin"}
    )

.. autoclass:: tace.interface.ase.TACEAseCalc
   :no-members:

.. autoclass:: tace.interface.ase.GeneralTACEAseCalc
   :no-members:

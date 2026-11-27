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

For MD, set ``neighbor_skin=0.5`` (angstroms), for example
``TACEAseCalc("TACE-OAM-7M", neighbor_skin=0.5)``. The calculator stores
one list of candidate image pairs and filters it at the physical cutoff
on every call. At fixed cell a displacement of at least half the skin
rebuilds the list. For a changing, fully periodic cell, deformation also
counts against the skin. Changes to the boundary conditions or atom count
always rebuild, as do cell changes with partial periodicity. A wrapped
position can trigger a rebuild. The default ``0.0`` disables reuse. Per-edge
properties keep the original neighbor ordering and use a fresh list.

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

Magnetic Relaxation (in dev)
============================

TACE provides a development-stage ASE optimizer for self-consistent magnetic
and structural relaxation.  The workflow is fixed rather than a simultaneous
update of all six per-atom components:

#. Hold the atomic structure and cell fixed and relax the non-collinear
   magnetic moments until ``mag_fmax`` is below its threshold.
#. Hold the converged magnetic moments fixed and take one atomic-coordinate
   step.
#. Repeat the magnetic relaxation at the updated structure.
#. Continue until both the magnetic and atomic force criteria are satisfied.

Optional cell relaxation
------------------------

Cell relaxation is disabled by default.  When ``optimize_cell=True``, the cell
is updated together with the atomic coordinates during the structural step and
``cell_fmax`` is also checked.  The calculator must provide stress for this
mode.  Models without a stress output can use the default fixed-cell workflow.

Example
-------

The magnetic input is mapped to the corresponding ``Atoms`` array through
``GeneralTACEAseCalc.keys``.  See :doc:`ase` for the general additional-input
calculator interface.

.. code-block:: python

   from ase.io import read

   from tace.interface.ase import GeneralTACEAseCalc, MagneticFIRE

   atoms = read("input.xyz")
   atoms.calc = GeneralTACEAseCalc(
       model="model.pt",
       keys={"initial_noncollinear_magmoms": "spin"},
   )

   optimizer = MagneticFIRE(
       atoms,
       magmom_key="spin",
       optimize_cell=False,
       position_fmax=0.05,
       magmom_fmax=0.01,
       position_maxstep=0.1,
       magmom_maxstep=0.02,
       magmom_steps=1000,
   )
   optimizer.run(steps=1000)

The complete executable example keeps all settings as uppercase constants in
``example/ase/magnetic_relax.py``.

.. autoclass:: tace.interface.ase.optimizer.MagneticFIRE
   :no-members:
   :show-inheritance:

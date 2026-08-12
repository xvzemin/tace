ASE Calculator Tutorial
=======================

This tutorial demonstrates how to use a TACE model as a calculator within ASE (Atomic Simulation Environment).

ASE Calculator documentation: `ASE Calculator <https://wiki.fysik.dtu.dk/ase/ase/calculators/calculator.html>`_

For optimization, molecular dynamics, and other examples, see the
`TACE ASE examples <https://github.com/xvzemin/tace/tree/main/example/ase>`_.

.. code-block:: python

    from ase import units
    from ase.io import read
    from tace.interface.ase import TACEAseCalc, add_dispersion

    device = "cuda"           # Use "cpu" when CUDA is unavailable
    dtype = "float32"         # "float32" or "float64"
    model_path = "model.pt"   # .pt, .pth, .ckpt, or compatible .pt2
    fidelity_idx = 0
    atoms = read("structure.xyz", index=0)

    dispersion = False

    calc = TACEAseCalc(
        model_path,
        device=device,
        dtype=dtype,
        fidelity_idx=fidelity_idx,
    )
    if dispersion:  # Requires: pip install torch-dftd
        calc = add_dispersion(
            base_calc=calc,
            damping="bj",  # choices: ["zero", "bj", "zerom", "bjm"]
            dispersion_xc="pbe",
            dispersion_cutoff=40.0 * units.Bohr,
        )
    atoms.calc = calc


Models with additional inputs
-----------------------------

``GeneralTACEAseCalc`` is intended for TACE models that require physical input
quantities beyond atomic numbers, positions, and the cell.  Representative
examples are TACE-QEq, which requires the total charge, and mTACE, which
requires initial non-collinear magnetic moments.

The ``keys`` argument maps each TACE embedding property to the corresponding
key stored in ``Atoms.info`` or ``Atoms.arrays``.  The calculator determines the
storage location from the property scope:

- Per-system inputs, such as ``total_charge``, are read from ``Atoms.info``.
- Per-atom inputs, such as ``initial_noncollinear_magmoms``, are read from
  ``Atoms.arrays``.

For a TACE-QEq model whose total charge is stored as ``charge``:

.. code-block:: python

   from ase.io import read

   from tace.interface.ase import GeneralTACEAseCalc

   atoms = read("charged_structure.xyz")
   atoms.info["charge"] = 1.0
   atoms.calc = GeneralTACEAseCalc(
       model="tace_qeq.pt",
       keys={"total_charge": "charge"},
   )

For an mTACE model whose non-collinear magnetic moments are stored as ``spin``:

.. code-block:: python

   from ase.io import read

   from tace.interface.ase import GeneralTACEAseCalc

   atoms = read("magnetic_structure.xyz")
   atoms.calc = GeneralTACEAseCalc(
       model="mtace.ckpt",
       keys={"initial_noncollinear_magmoms": "spin"},
   )

Multiple additional inputs can be mapped in the same ``keys`` dictionary.  Use
the original ``TACEAseCalc`` for models that do not require such inputs.


.. autoclass:: tace.interface.ase.calculator.TACEAseCalc
   :no-members:
   :show-inheritance:

   
.. autoclass:: tace.interface.ase.general_calculator.GeneralTACEAseCalc
   :no-members:
   :show-inheritance:

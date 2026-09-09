OpenMM
======

This tutorial demonstrates how to use a TACE model as a calculator within OpenMM. 
We integrate OpenMM via the ASE interface provided by openmm-ml. Example usage is shown below, 
and for additional details, please refer to the OpenMM documentation.

OpenMM-ML github: `github <https://github.com/openmm/openmm-ml>`_
OpenMM-ML documentation: `docs <https://openmm.github.io/openmm-ml/dev/index.html>`_

.. code-block:: python

    '''
    Based on https://github.com/openmm/openmm-ml/blob/main/test/TestASEPotential.py
    '''
    import os
    import numpy as np
    import openmm as mm
    import openmm.app as app
    import openmm.unit as unit
    from openmmml import MLPotential
    from ase.io import read

    from tace.foundations import tace_foundations
    from tace.interface.ase import TACEAseCalc, add_dispersion

    model = tace_foundations["TACE-OAM-L"]
    calculator = TACEAseCalc(
        model=model,
        dtype='float32',
        device='cuda',
    )

    test_data_dir = '../data/'

    def kJmol2eV(energy):
        return (energy / unit.AVOGADRO_CONSTANT_NA).value_in_unit(unit.ev)

    def testCalculator(platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "toluene", "toluene.pdb"))
        potential = MLPotential('ase')
        system = potential.createSystem(pdb.topology, calculator=calculator)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy() # kJ/mol
        energyRef = -92.44810485839844
        assert np.isclose(energyRef, kJmol2eV(energyML), rtol=1e-6)
        print('testCalculator pass')

    def testAtoms(platform_int):
        path = os.path.join(test_data_dir, "toluene", "toluene.pdb")
        pdb = app.PDBFile(path)
        atoms = read(path)
        atoms.calc = calculator
        potential = MLPotential('ase')
        system = potential.createSystem(pdb.topology, aseAtoms=atoms)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.001), platform)
        context.setPositions(pdb.getPositions(asNumpy=True))
        energyML = context.getState(energy=True).getPotentialEnergy()
        assert np.isclose(-92.44810485839844, kJmol2eV(energyML), rtol=1e-6)
        print('testAtoms pass')

    def testPeriodicSystem(platform_int):
        pdb = app.PDBFile(os.path.join(test_data_dir, "alanine-dipeptide", "alanine-dipeptide-explicit.pdb"))
        potential = MLPotential('ase')
        system = potential.createSystem(pdb.topology, calculator=calculator)
        platform = mm.Platform.getPlatform(platform_int)
        context = mm.Context(system, mm.VerletIntegrator(0.0001), platform)
        positionsOriginal = pdb.getPositions(asNumpy=True)
        energyRef = -10992.9296875
        for i in range(3):
            positions = positionsOriginal + i * 0.9 * unit.nanometers # translate molecule to test PBC
            context.setPositions(positions)
            energyML = context.getState(getEnergy=True).getPotentialEnergy()
            assert np.isclose(energyRef, kJmol2eV(energyML), rtol=1e-5)
        context.setPositions(positionsOriginal)
        energyML = context.getState(getEnergy=True).getPotentialEnergy()
        print('testPeriodicSystem pass')

    def testCreateMixedSystem(platform_int):
        prmtop = app.AmberPrmtopFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.prm7"))
        inpcrd = app.AmberInpcrdFile(os.path.join(test_data_dir, "toluene", "toluene-explicit.rst7"))
        mlAtoms = list(range(15))
        mmSystem = prmtop.createSystem(nonbondedMethod=app.PME)
        potential = MLPotential('ase')
        mixedSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=False, calculator=calculator)
        interpSystem = potential.createMixedSystem(prmtop.topology, mmSystem, mlAtoms, interpolate=True, calculator=calculator)
        platform = mm.Platform.getPlatform(platform_int)
        mmContext = mm.Context(mmSystem, mm.VerletIntegrator(0.001), platform)
        mixedContext = mm.Context(mixedSystem, mm.VerletIntegrator(0.001), platform)
        interpContext = mm.Context(interpSystem, mm.VerletIntegrator(0.001), platform)
        mmContext.setPositions(inpcrd.positions)
        mixedContext.setPositions(inpcrd.positions)
        interpContext.setPositions(inpcrd.positions)
        mmEnergy = mmContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        mixedEnergy = mixedContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpEnergy1 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        interpContext.setParameter('lambda_interpolate', 0)
        interpEnergy2 = interpContext.getState(getEnergy=True).getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        assert np.isclose(mixedEnergy, interpEnergy1, rtol=1e-5)
        assert np.isclose(mmEnergy, interpEnergy2, rtol=1e-5)
        print('testCreateMixedSystem pass')


    print('Available Platform:')
    for i in range(mm.Platform.getNumPlatforms()):
        p = mm.Platform.getPlatform(i)
        print("  -", i, p.getName())
    print()

    testCalculator(2)
    testAtoms(2)
    testPeriodicSystem(2)
    testCreateMixedSystem(2)



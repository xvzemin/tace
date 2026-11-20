################################################################################
# Authors: Zemin Xu
# License: MIT, see LICENSE.md
################################################################################

"""Evaluate a PDB structure with the OpenMM-ML ASE potential."""

import argparse

import openmm as mm
from openmm import unit
from openmm.app import PDBFile
from openmmml import MLPotential

from tace.interface.ase import TACEAseCalc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdb", help="PDB structure")
    parser.add_argument("--model", default="TACE-OAM-L", help="TACE model name or file")
    args = parser.parse_args()

    pdb = PDBFile(args.pdb)
    calc = TACEAseCalc(args.model)
    system = MLPotential("ase").createSystem(pdb.topology, calculator=calc)
    context = mm.Context(system, mm.VerletIntegrator(1 * unit.femtoseconds))
    context.setPositions(pdb.positions)
    print(context.getState(getEnergy=True).getPotentialEnergy())


if __name__ == "__main__":
    main()

# Dataset from [LinearACE](https://pubs.acs.org/doi/10.1021/acs.jctc.1c00647)

# 3BPA benchmark dataset

units: distance Angstrom, energy eV, forces eV / A

energy key : `energy`

forces key : `forces`

## Files:

`train_300K.xyz` : Training data of 500 configs sampled form 300 K MD

`train_mixedT.xyz` : Training data of 500 configs sampled from 300 K, 600 K, 1200 K MD

`test_300K.xyz` : Test set sampled from 300 K MD

`test_600K.xyz` : Test set sampled from 600 K MD

`test_1200K.xyz` : Test set sampled from 1200 K MD

`test_dih_beta.xyz` : DFT optimized dihedral scan with beta fixed to 120, 150 or 180 degrees. 

`iso_atoms.xyz` : DFT energy of the isolated atoms, can be used as the reference potential

## Tasks:

### Train-test at different temperatures 

Train two models using the two training sets and test them on the three MD test sets and the dihedral test set. 

### Dihedral scan

Optimize the geometry using the foce field for different combinations of dihedral angles
to test the shape and smoothness of the potential energy landscape. 

An example code for the geometry optimization is as follows:

```
import numpy as np

from ase.io import read, write

from ase.constraints import FixAtoms

from ase.optimize import BFGS

from ase.optimize import FIRE

calculator =  # ASE calculator object

init_confs = read("dih_scan_beta180.xyz", index=":")

eners = []

# fix the three dihedrals and the amino group (necessary to avoid artiefects from geometry optimization)

c1 = FixAtoms(indices=[1,5,13,14]) 

c2 = FixAtoms(indices=[13,5,1,4])

c3 = FixAtoms(indices=[5,13,14,18])

c4 = FixAtoms(indices=[1,4,6,8])

for at in init_confs:

    at.set_calculator(calculator)

    at.set_constraint([c1,c2,c3,c4])

    # now optimize the geometry with multiple optimizers to ensure convergence

    try:

        opt = BFGS(at)

        opt.run(fmax=5e-3, steps=250)

        opt2 = FIRE(at)

        opt2.run(fmax=5e-3, steps=100)

        opt3 = BFGS(at)

        opt3.run(fmax=5e-3, steps=100)

        eners.append(at.get_potential_energy())

        write("NEW_DIH180.xyz", at, append=True)  

    except RuntimeError:

        pass

# save energies separately as well for plotting

np.save("energy_MODEL_180.npy", eners)


```

It is also possible to simply re-evaluate the DFT optimized landscape using the parametrized force field model.  

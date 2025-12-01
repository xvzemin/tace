Q & A
=====

Q1: What units are used internally in TACE?

A: 

    In principle, most physical quantities are not constrained to any specific units. However, for consistency, we recommend the following units for training labels: energy in eV, forces in eV/Å, and stress in eV/Å³ (note that VASP outputs stress in kBar with an opposite sign). All reported errors have been scaled by a factor of 1000, so the errors are expressed in meV.

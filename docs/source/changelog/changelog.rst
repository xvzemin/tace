Changelog
=========

A simple description of the update:

- v0.0.1: Supported prediction of energy, forces, stress, dipole moment, and polarizability; provided ASE interface.
- v0.0.2: Added support for universal invariants (e.g., charges, fidelities, spin) embedding; supported external field response and charge equilibration (Qeq and uniform).
- v0.0.3: Supported noncolinear magnetic moments embedding while predicting magnetic forces; merged various equivariants into universal equivariant embedding; added long-range plugin LES.
- v0.0.4: Added LAMMPS-MLIAP interface; included edge_forces, atomic_stresses, and atomic_virials.
- v0.0.5: No longer compatible with v0.0.1-v0.0.4; added support for multihead architecture for any rank.
- v0.0.6: Modify the multi-GPU parallel framework to support two modes for graph data — memory and LMDB (stored on disk) — to facilitate training on large-scale datasets. Add 
experimental matrice tensor product, analytical tensor product, SO(2) tensor product.
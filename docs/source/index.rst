Tensor Atomic Cluster Expansion
===============================
.. = - ~ ^ "

Documentation Structure
-----------------------

.. toctree::
   :maxdepth: 2
   :caption: Contents

   install/install
   guide/guide
   equivariantx
   model/model
   
.. toctree::
   :maxdepth: 1
   :caption: Changelog

   changelog/changelog


Overview
--------

Currently, the officially supported properties include:

- Energy
- Forces (conservative | direct)
- Hessian (conservative, predict only)
- Stress (conservative | direct)
- Virials (conservative | direct)
- Charges (lagrangian or uniform_distribution)
- Dipole moment (conservative | direct)
- Polarization (conservative, multi-value for PBC systems)
- Polarizability (conservative | direct)
- Born effective charges (conservative, under electric field)
- Atomic stresses (conservative, predict only)
- Atomic virials (conservative, predict only)
- Absolute final collinear magmoms
- Noncollinear magnetic forces (SOC, full O(3))

For embedding property, we support:

- fidelity_idx (different computational levels)
- charges
- total charge
- electric field
- initial noncollinear magmoms (SOC, full O(3))


Plugins
-------

TACE currently supports the following plugin:

- :doc:`TACE-LES <guide/les>` (Latent Ewald Summation)


Interfaces
----------

- ✅ Supports integration with **ASE Calculator**.

- ✅ Supports integration with **LAMMPS-ML-IAP**.

- ✅ Supports integration with **TorchSim**.

- ✅ Supports integration with :doc:`NVIDIA NValCHEMI
  <guide/nvalchemi>`.

- ✅ Supports integration with **OpenMM-ML (OpenMM-ML -> ASE -> TACE)**.

- ✅ Supports integration with **USPEX (USPEX -> LAMMPS-ML-IAP -> TACE)** (Python=3.9).


Citing
------

If you use TACE, please cite our papers:

.. code-block:: bibtex

   @misc{xu2026spectralspatialtensoratomiccluster,
      title={Spectral/Spatial Tensor Atomic Cluster Expansion with Universal Embeddings in Cartesian Space}, 
      author={Zemin Xu and Wenbo Xie and P. Hu},
      year={2026},
      eprint={2509.14961},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2509.14961}, 
   }

   @misc{xu2026edgeclusterexpansionradial,
      title={Edge Cluster Expansion with Radial Rotary Attention for Interatomic Potentials}, 
      author={Zemin Xu and Wenbo Xie and P. Hu},
      year={2026},
      eprint={2607.10664},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/2607.10664}, 
   }

If you use cartnn, Cartesian-3j, cMACE, cNequIP, cAllegro, please cite our papers:

.. code-block:: bibtex

   @inproceedings{xu2026a,
      title={A Cartesian-3j Framework for Machine Learning Interatomic Potentials},
      author={Zemin Xu and Chenyu Wu and Wenbo Xie and Peijun Hu},
      booktitle={Forty-third International Conference on Machine Learning},
      year={2026},
      url={https://openreview.net/forum?id=9ZWK6gneWq}
   }

Contact
-------

For bugs or feature requests, please use https://github.com/xvzemin/tace/issues.

License
-------

The TACE code is published and distributed under the MIT License.

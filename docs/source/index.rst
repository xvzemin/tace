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
   interfaces/interfaces
   changelog/changelog
   qa/qa
   
Overview
--------

TACE is a Cartesian-based machine learning model designed to predict both scalar and tensorial properties.

In principle, the framework supports any tensorial properties (either direct or conservative) determined by the underlying atomic structure. 
Currently, the officially supported properties include:

- Energy
- Forces (conservative | direct) *(Direct pretrained follow conservative finetuning not test by us)*
- Hessians (conservative, predict only)
- Stress (conservative | direct)
- Virials (conservative | direct)
- Charges (Qeq or uniform)
- Dipole moment (conservative | direct)
- Polarization (conservative, multi-value for PBC systems)
- Polarizability (conservative | direct)
- Born effective charges (conservative, under electric field or LES)  *(LES not tested by us)*
- Magnetic forces 0 (collinear, rank-0)
- Magnetic forces 1 (non-collinear, rank-1)
- Atomic stresses (conservative, predict only)
- Atomic virials (conservative, predict only)
- Magnetization (conservative) *(not tested by us)*
- Magnetic susceptibility (conservative) *(not tested by us)*
- Elastic constant (coming soon)
- Nuclear chemical shift (coming soon)
- Nuclear shielding (coming soon)


For embedding property, we support:

invariant quantities:

- Charges
- Total charge
- Spin multiplicity
- Level (different computational levels)
- Magmoms 0 (collinear, rank-0)
- Electron temperature *(not tested by us)*

equivariant quantities:

- electric field
- magnetic field *(not tested by us)*
- Magmoms_1 (non-collinear, rank-1)


Methodology
-----------

TACE leverages **Irreducible Cartesian Tensor Decomposition (ICTD)** and the **Atomic Cluster Expansion (ACE)** 
to efficiently and accurately learn semi-local chemical environments.  

Plugins
-------

TACE currently supports the following plugin:

- **LES** (Latent Ewald Summation)


Citing
------

If you use TACE, please cite our papers:

.. code-block:: bibtex

   @misc{TACE,
         title={TACE: A unified Irreducible Cartesian Tensor Framework for Atomistic Machine Learning}, 
         author={Zemin Xu and Wenbo Xie and Daiqian Xie and P. Hu},
         year={2025},
         eprint={2509.14961},
         archivePrefix={arXiv},
         primaryClass={stat.ML},
         url={https://arxiv.org/abs/2509.14961}, 
   }

Contact
-------

If you have any problems, suggestions or cooperations, please contact us through xvzemin@smail.nju.edu.cn

For bugs or feature requests, please use https://github.com/xvzemin/tace/issues.

License
-------

The TACE code is published and distributed under the MIT License.


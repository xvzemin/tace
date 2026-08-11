.. .. image:: fig/logo.svg
..    :width: 100%
..    :align: center

Tensor Atomic/Edge Cluster Expansion (TACE/TECE)
================================================
.. = - ~ ^ "
TACE is designed with physical priors and strong inductive biases to enhance extrapolation capability. 
It performs Atomic Cluster Expansion and Edge Cluster Expansion based on spherical tensors 
or irreducible Cartesian tensors, with an optional attention architecture.

Cartesian Architecture
----------------------

.. image:: fig/cartesian_arch.png
   :width: 100%
   :align: center

Spherical/SO(2) Architecture
----------------------------

The architecture of the spherical model is largely the same as that of the Cartesian space. 
For details on the SO(2) component, please refer to our paper and code.

Wigner6j/O(2) Magnetic Architecture
-----------------------------------

For more details on global O(3), local O(2), and Wigner 6j recoupling, 
please look forward to our upcoming preprint paper.

Docs
----

`TACE documentation <https://tace.readthedocs.io/en/latest/index.html>`_


SOTA Foundation Model
---------------------

`TACE Foundation Models <https://github.com/xvzemin/tace-foundations>`_


Default Ranking on Matbench as of July 8, 2026
----------------------------------------------

.. figure:: fig/matbench_tece_rra.png
   :width: 100%
   :align: center


Install, Train and Tutorial
---------------------------

The docs contain a complete tutorial. 

We also provide complete input files and example scripts for training, ASE,
TorchSim, LAMMPS, and other workflows in the
`TACE examples <https://github.com/xvzemin/tace/tree/main/example>`_.


.. code-block:: bash

   # Minimal install and training example
   git clone https://github.com/xvzemin/tace
   cd tace
   pip install . # or pip install tace
   cd example/train
   tace-train -cn tace.yaml

Fine-tuning
-----------

- ✅ Full-parameter.

- ✅ Freeze-parameter.

- ✅ LoRA.


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

- **TACE-LES** (Latent Ewald Summation)
- **TACE-QEq** (Lagrangian)
- **mTACE**    (Magnetic, Spin-Orbit Coupling)

Interfaces
----------

- ✅ Supports integration with **ASE Calculator**.

- ✅ Supports integration with **LAMMPS-ML-IAP**.

- ✅ Supports integration with **TorchSim**.

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

For bugs or feature requests, please use the
`TACE issue <https://github.com/xvzemin/tace/issues>`_.

Development
-----------

Install the development tools and run the same formatting and lint checks used by
pre-commit:

.. code-block:: bash

   pip install -e ".[dev]"
   ruff check --fix tace
   ruff format tace
   ruff check tace
   ruff format --check tace

License
-------

The TACE code is published and distributed under the MIT License.

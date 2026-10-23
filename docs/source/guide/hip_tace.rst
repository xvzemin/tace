HIP-TACE
========

HIP-TACE predicts ``direct_hessian``, without differentiating energy or forces. 

Training Example
----------------

With TACE installed, run from the repository root:

.. code-block:: bash

   pip install 'kagglehub>=1.0.2'
   python example/data/download_HORM.py
   cd example/train
   tace-train -cn hip-tace.yaml

``example/train/hip-tace.yaml`` trains energy, forces and direct Hessians.

The main model options are:

.. code-block:: yaml

   model:
     config:
       parity: true
       Lmax: 2
       lmax: 3
       special:
         direct_hessian:
           cutoff: 16.0
           interaction_type: cgtp  # cgtp or o2

Available Hessian losses are ``mae_direct_hessian``, ``mse_direct_hessian``,
``huber_direct_hessian`` and ``l2mae_direct_hessian``. Set ``reduction: component`` 
to average over entries (pairs for L2MAE), or ``structure`` to give each structure 
equal weight.


Data and output
---------------

Provide Hessian labels under ``atoms.info["direct_hessian"]``
or the ASE database's ``data["direct_hessian"]``.  Supported label shapes are 
``(3N, 3N)``, its row-major flattened form, ``(N, N, 3, 3)`` and ``(N², 3, 3)``. 
Shapes are detected automatically; four-index labels use ``(i, j, a, b)`` order. 
Predictions always use ``(N², 3, 3)`` blocks, ordered by ``i * N + j``.

To recover one structure's full matrix:

.. code-block:: python

   matrix = blocks.reshape(N, N, 3, 3).permute(0, 2, 1, 3).reshape(3 * N, 3 * N)

Citing
------

If you use HIP-TACE, please cite the HIP paper:
`HIP: Hessian Interatomic Potentials without derivatives
<https://arxiv.org/abs/2509.21624>`_.

.. code-block:: bibtex

   @misc{burger2025hiphessian,
     title = {{HIP}: Hessian Interatomic Potentials without derivatives},
     author = {Andreas Burger and Luca Thiede and Nikolaj R{\o}nne and
               Varinia Bernales and Nandita Vijaykumar and Tejs Vegge and
               Arghya Bhowmik and Alan Aspuru-Guzik},
     year = {2025},
     eprint = {2509.21624},
     archivePrefix = {arXiv},
     primaryClass = {cs.LG},
     url = {https://arxiv.org/abs/2509.21624}
   }

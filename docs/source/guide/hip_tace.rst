HIP-TACE
========

HIP-TACE predicts ``direct_hessian`` without differentiating energy or forces.

Training Example
----------------

With TACE installed, run from the repository root:

.. code-block:: bash

   pip install 'kagglehub>=1.0.2'
   python example/data/download_HORM.py
   cd example/train/HIP-TACE
   tace-train -cn hip-tace.yaml

The example trains energy, forces and direct Hessians. Adjust the dataset paths
and keep the ``TACE`` and ``HIP-TACE`` subdirectories together; see
:doc:`details/defaults`.

.. literalinclude:: ../../../example/train/HIP-TACE/hip-tace.yaml
   :language: yaml


Data and output
---------------

Provide Hessian labels under ``atoms.info["direct_hessian"]``
or the ASE database's ``data["direct_hessian"]``. Supported label shapes are
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

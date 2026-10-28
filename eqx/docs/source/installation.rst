.. _equivariantx-installation:

Installation
============

Install EQX from source without installing TACE:

.. code-block:: bash

   git clone https://github.com/xvzemin/tace.git
   pip install ./tace/eqx

Requires PyTorch >= 2.4, e3nn >= 0.4.4, and SciPy >= 1.15.
For fused CUDA operators:

.. code-block:: bash

   pip install './tace/eqx[cuda]'

A CUDA toolkit is required; set ``CUDA_HOME`` if it is not detected.
Kernels compile on first use and are cached. PyTorch operators do not require
the CUDA build dependencies.

With e3nn 0.4.x, import ``eqx`` before ``e3nn.o3``. Available O(3) degrees
are limited by that version's CG table. Global time-odd irreps require:

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"

See :ref:`equivariantx-api` for parameters and usage examples.

Implementation layout
---------------------

``o3``, ``co3``, ``co2``, and ``o2`` define the mathematical operations.
``conv`` and ``ace`` compose convolutions and cluster expansions; ``models``
adapts model parameters and layouts. Internal ``ops`` modules register operator
schemas, shape rules, and autograd. ``backends`` supplies execution providers
without changing the mathematical API.

``backend="torch"`` uses PyTorch on the input device, including GPUs. CUDA
implementations are loaded on demand. Importing EQX does not initialize a GPU,
compile kernels, or require any external model package.

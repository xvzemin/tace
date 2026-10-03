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

Example
-------

.. code-block:: python

   from eqx import o2

   linear = o2.Linear("8x0e + 4x1m", "4x0e + 2x1m")
   features = linear.irreps_in.randn(32, -1)
   output = linear(features)

See :ref:`equivariantx-api` for layouts, parameters, and supported operations.

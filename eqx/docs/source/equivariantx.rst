.. _equivariantx-tutorials:

Overview and installation
=========================

EquivariantX (``eqx``) provides spherical and Cartesian O(3)/O(2) operators,
basis and frame conversions, and fused CUDA convolutions.

.. list-table::
   :header-rows: 1
   :widths: 25 15 30 30

   * - Representation
     - Module
     - Layout within each entry
     - API
   * - Spherical O(3)
     - ``eqx.o3``
     - ``(..., mul, 2 * l + 1)``
     - :ref:`equivariantx-api-o3`
   * - Cartesian O(3)
     - ``eqx.co3``
     - ``(..., mul, 3**l)``
     - :ref:`equivariantx-api-co3`
   * - Cartesian O(2)
     - ``eqx.co2``
     - ``(..., mul, 2**m)``
     - :ref:`equivariantx-api-co2`
   * - Spherical O(2)
     - ``eqx.o2``
     - ``(..., ir.dim, mul)``
     - :ref:`equivariantx-api-o2`

Each entry is flattened into the final feature axis. Basis conversions,
rotations, and tensor decompositions are listed in :ref:`equivariantx-api-tools`.
Spherical fused convolutions use ``ir_mul``; see
:ref:`equivariantx-convolutions` for their input layouts and backends.

Installation
------------

Install from source without installing TACE:

.. code-block:: bash

   git clone https://github.com/xvzemin/tace.git
   pip install ./tace/eqx

Dependencies include PyTorch >= 2.4, e3nn >= 0.4.4, and SciPy >= 1.15.
For CUDA kernels, install the build dependencies and provide a CUDA toolkit:

.. code-block:: bash

   pip install './tace/eqx[cuda]'

Set ``CUDA_HOME`` if the toolkit is not detected. Kernels compile on first
use and are cached; installation does not compile them.
A standalone package release is planned.

With e3nn 0.4.x, import ``eqx`` before ``e3nn.o3``. Available O(3) degrees
are limited by that version's packaged CG table. Global time-odd irreps
require the time-reversal extension:

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"

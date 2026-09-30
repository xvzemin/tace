.. _equivariantx-tutorials:

Tutorials
=========

EquivariantX (``eqx``) provides spherical :math:`O(2)` and Cartesian :math:`O(2)`/:math:`O(3)` operators, e3nn-compatible
:math:`O(3)`/:math:`O(2)` frame conversion, and fused CUDA convolutions.

Installation
------------

Install from source:

.. code-block:: bash

   git clone https://github.com/xvzemin/tace.git
   pip install ./tace/eqx

Pytorch operations require PyTorch >= 2.4 and e3nn >= 0.4.4. For the fused 
backend, install the build dependencies and provide a CUDA toolkit:

.. code-block:: bash

   pip install './tace/eqx[cuda]'

Set ``CUDA_HOME`` if the toolkit is not detected. Fused kernels compile on first
use and are cached; installation does not compile them. A standalone package
release is planned.

With e3nn 0.4.x, import ``eqx`` before ``e3nn.o3``. 

Global time-odd irreps require the time-reversal e3nn extension.

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"

Representations and layouts
---------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Irrep
     - Dimension
     - Transformation
   * - ``0ee``, ``0eo``
     - 1
     - Reflection-even scalar; final letter gives time parity.
   * - ``0oe``, ``0oo``
     - 1
     - Reflection-odd scalar; final letter gives time parity.
   * - ``1me``, ``1mo``, ...
     - 2
     - Positive-order irrep; final letter gives time parity.

``0e``, ``0o``, and ``1m`` abbreviate time-even irreps. Iteration over
``o2.Irreps`` yields ``(ir, mul)``. Spherical O(2) features use the ``ir_mul`` 
layout, while Cartesian O(2) and O(3) features use the ``mul_ir`` layout.
see :ref:`equivariantx-cartesian-o2` and :ref:`equivariantx-cartesian`.

O(3)/O(2) frame conversion
--------------------------

|eqx-frames|

An edge direction defines the alignment axis. Restriction preserves time
parity and separates local orders:

.. math::

   (\ell,p,t)\downarrow
   =\left(0,p(-1)^\ell,t\right)
   \oplus\bigoplus_{m=1}^{\ell}(m,0,t).

Fused Convolution
-----------------

See :ref:`equivariantx-convolutions` for more details.
.. _installation:

Installation
============

Requirements
------------

TACE requires Python 3.9 or newer and PyTorch 2.4 or newer. AOTInductor
export additionally requires ``torch>=2.13``. We recommend installing
TACE in a clean environment:

.. code-block:: bash

   micromamba create -n tace python=3.13 -y
   micromamba activate tace

Install TACE
------------

Install the latest release from PyPI:

.. code-block:: bash

   pip install tace

To install the current source tree instead:

.. code-block:: bash

   pip install git+https://github.com/xvzemin/tace.git@main

The core installation uses the standard e3nn implementation. Acceleration
libraries and simulation interfaces are optional and can be installed
independently as described below. When working from a source checkout, replace
``tace[extra]`` with ``.[extra]`` in the commands.

PyTorch Geometric
-----------------

TACE requires the core ``torch_geometric`` package, but not its optional
binary extensions at import time
(``torch-scatter``, ``torch-sparse``, ``torch-cluster``, ``torch-spline-conv``,
or ``pyg-lib``). Standard sum and mean reductions use native PyTorch operations.
TACE's ``scatter_min``, ``scatter_max``, and ``scatter_mul`` require registered
``torch-scatter`` operators only when called. EquivariantX does not depend on
PyG or its extensions.

OpenEquivariance (OEQ)
----------------------

OEQ provides optimized CUDA or HIP equivariant kernels:

.. code-block:: bash

   pip install "tace[oeq]"

.. important::

   When OEQ is used together with AOTInductor export or deployment, TACE
   requires ``openequivariance>=0.6.4``. Upgrade OEQ before exporting the AOTI
   package:

   .. code-block:: bash

      # OEQ used together with AOTI
      pip install "openequivariance>=0.6.4"

Enable it before constructing or loading a configurable model:

.. code-block:: bash

   export TACE_USE_OEQ=1

cuEquivariance (CUEQ)
---------------------

Install the package matching the CUDA major version used by PyTorch. CUDA 12
and CUDA 13 use different kernel packages:

.. code-block:: bash

   # CUDA 12
   pip install "tace[cueq12]"

   # CUDA 13
   pip install "tace[cueq13]"

Check ``torch.version.cuda`` if the correct CUDA variant is unclear, then
enable the backend with:

.. code-block:: bash

   python -c "import torch; print(torch.version.cuda)"
   export TACE_USE_CUE=1

EquiTorch (EQT)
---------------

The EQT implementation used by TACE is bundled with TACE, so ordinary EQT
usage does not require installing a separate EquiTorch package:

.. code-block:: bash

   export TACE_USE_EQT=1

The sparse higher-order product path uses ``torch-scatter`` when available
and otherwise falls back to native PyTorch reductions. To install the optional
extension, select a wheel matching the PyTorch and CUDA versions in the
environment. For example, for PyTorch 2.11 and CUDA 13.0:

.. code-block:: bash

   pip install torch-scatter \
     -f https://data.pyg.org/whl/torch-2.11.0+cu130.html

Use the `PyTorch Geometric installation guide
<https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html>`_
to select a different PyTorch or CUDA wheel.


Latent Ewald Summation (LES)
----------------------------

LES is an optional external dependency used by TACE-LES for long-range
interactions.  Install TACE first, then install the upstream LES (v0.2.0).

.. code-block:: bash

   pip install git+https://github.com/ChengUCB/les.git@v0.2.0

Verify that TACE can import the backend:

.. code-block:: bash

   python -c "from les import Les; print('LES is available')"

See the :doc:`../guide/les`
tutorial for model configuration, supported latent sources, outputs, and
current compatibility limitations.


TorchSim
--------

Install the optional TorchSim interface with:

.. code-block:: bash

   pip install "tace[torchsim]"

.. important::

   TACE requires ``torch-sim-atomistic>=0.6.1``. Version ``0.6.1`` is the
   safest and currently recommended version:

   .. code-block:: bash

      pip install "torch-sim-atomistic==0.6.1"

   TorchSim is under active development, so compatibility with versions newer
   than ``0.6.1`` is not guaranteed.

See the :doc:`../guide/torchSim` tutorial for calculator usage.

.. _nvalchemi-installation:

NValCHEMI
---------

The NValCHEMI interface is optional and requires ``Python >= 3.11``.
Install TACE together with the interface dependencies using:

.. code-block:: bash

   pip install "tace[nvalchemi]"

For a source checkout, use:

.. code-block:: bash

   pip install ".[nvalchemi]"

The extra installs ``nvalchemi-toolkit`` and its required
``nvalchemi-toolkit-ops`` dependency.

The two upstream NVIDIA repositories are provided for reference:

- `nvalchemi-toolkit <https://github.com/NVIDIA/nvalchemi-toolkit>`_
- `nvalchemi-toolkit-ops <https://github.com/NVIDIA/nvalchemi-toolkit-ops>`_


Time-reversal e3nn
------------------

To enforce time-reversal equivariance in mTACE, install the ``time-reversal``
branch of `e3nn
<https://github.com/xvzemin/e3nn/tree/time-reversal>`_ to ensure time-reversal
equivariance:

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"

The package name and Python import remain ``e3nn``.

TACE detects this capability at runtime; no model option is required. Pure
O(3) models then attach time-reversal parity to magnetic moments, magnetic
fields, and their tensor-product paths automatically.

Time-reversal models use the time-reversal e3nn implementation for global
representation metadata and coupling rules. Native EQX O(2) operators and
compatible fused kernels preserve these labels. EQT, CUEQ, and OEQ operators
that do not support time-odd irreps reject them when selected; support is
checked per operator rather than by removing time-odd coupling paths.

EquivariantX
------------

EquivariantX is currently bundled with TACE and does not require a separate
installation for TACE users. Independent installation currently supports
source builds only and does not require installing TACE. A standalone package
release is planned once the library is fully mature. To install from source:

.. code-block:: bash

   git clone https://github.com/xvzemin/tace.git
   pip install ./tace/eqx

The library imports as ``eqx`` and depends on PyTorch >= 2.4, e3nn >= 0.4.4, and
``opt_einsum_fx``, but not on TACE or PyG. Its scope covers native O(2)
operators, e3nn-compatible O(3)/O(2) frame conversion, and fused CUDA
convolutions. Install ``'./tace/eqx[cuda]'`` and provide a CUDA toolkit to
use the fused backend. See :ref:`equivariantx-tutorials` and
:ref:`equivariantx-convolutions` for conventions, examples, and backend support.
When using e3nn 0.4.x with recent PyTorch, import ``eqx`` before ``e3nn.o3``
so its packaged constants are loaded in the scoped compatibility context.


Acceleration Selection
----------------------

Enabled backends are selected per supported operator in the order
EQX, OEQ, EQT, CUEQ. Multiple backends may be enabled together, for example
OEQ convolutions with EQT product-basis operations.
AOTI is a separate compilation and deployment layer. See the
:ref:`acceleration-tutorial` for backend selection, Python interfaces,
compilation, and AOTI export.

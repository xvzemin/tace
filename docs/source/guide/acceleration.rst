.. _acceleration-tutorial:

Acceleration
============

Combine fused operators with model compilation. Select acceleration settings
before constructing, loading, or exporting the model.

Recommended workflow
--------------------

To maximize model throughput, start with **EQX + compilation for training**
and **EQX + AOTI export for inference**.

.. figure:: ../_static/acceleration.svg
   :alt: Enable EQX, then compile training or export inference with AOTI.
   :width: 100%

   Operator fusion and model compilation are complementary.

Install EQX build dependencies from the TACE source directory and provide a
CUDA toolkit:

.. code-block:: bash

   pip install '.[eqx]'

**Training:** enable both environment variables.

.. code-block:: bash

   TACE_USE_EQX=1 TACE_USE_COMPILE=1 tace-train -cn tace.yaml

**Inference:** keep EQX enabled and export a compiled model. The
``tace-export-*`` entry points are ``tace-export-eval`` for ASE/TorchSim and
``tace-export-lammps`` for LAMMPS.

.. code-block:: bash

   TACE_USE_EQX=1 tace-export-eval \
     -m model.ckpt --backend aoti --device cuda

   TACE_USE_EQX=1 tace-export-lammps \
     -m model.ckpt --backend aoti --device cuda

The AOTI export commands enable the compilation path automatically.
Exporting with ``state_dict`` or ``full_model`` does **not** compile the model.
Measure speed after warming up the actual workload, including force-loss
derivatives during training; gains depend on model, batch size, and hardware.

Kernel selection
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 30 50

   * - Priority
     - Environment variable
     - Scope
   * - 1. EquivariantX
     - ``TACE_USE_EQX=1``
     - Convolutions, ACE, element-dependent Linear, and supported gates
   * - 2. OpenEquivariance
     - ``TACE_USE_OEQ=1``
     - O(3) tensor-product convolutions
   * - 3. EquiTorch
     - ``TACE_USE_EQT=1``
     - Product-basis tensor products
   * - 4. cuEquivariance
     - ``TACE_USE_CUE=1``
     - Supported O(3) tensor-product operations

Each operator selects the highest-priority enabled backend it supports:
**EQX > OEQ > EQT > CUEQ**. Unselected packages need not be installed.
A missing dependency for a selected backend raises an error.
``TACE_USE_COMPILE`` is independent of this selection.
See :ref:`installation` for optional packages.

The equivalent Python setup is:

.. code-block:: python

   from tace.utils.env import enable_acceleration

   enable_acceleration(enable_eqx=True, enable_compile=True)

This preserves existing settings. ``force=True`` also disables unselected
backends. ASE and TorchSim calculators expose backend options, but an exported
AOTI package already contains its chosen operator graph.

.. _eqx-streaming:

EQX in TACE
-----------

|eqx-convolutions|

.. list-table:: Model-to-operator mapping
   :header-rows: 1
   :widths: 34 66

   * - TACE operation
     - EQX implementation
   * - ``cgtp``
     - ``eqx.conv.O3TensorProductConv``
   * - ``o2_cgtp``
     - ``eqx.conv.O2O3TensorProductConv``
   * - ``uu_o2``
     - ``eqx.conv.UuO2TensorProductConv``
   * - ``o2`` / ``o2_mag``
     - ``eqx.conv.UvO2TensorProductConv``
   * - TECE-OAM-RRA ``so2``
     - ``eqx.models.tace.tece_oam_rra``
   * - Standard / gated bilinear ACE
     - ``eqx.ace.TACE`` / ``eqx.models.tace.tece_oam_rra.BilinearACE``

Supported CUDA operations use fused forward and derivative kernels. CPU and
unsupported cases retain their reference implementations. Products with
active coefficient LoRA adapters retain their original tensor products.
Feature layouts, paths, normalization, and checkpoint weights are preserved;
floating-point summation order can differ.

For fusion boundaries and standalone examples, see
:ref:`equivariantx-convolutions`. To switch between equivalent CGTP forms
without retraining:

.. code-block:: python

   from tace.utils.env import enable_acceleration
   from tace.lightning import convert_cgtp, load_tace

   enable_acceleration(enable_eqx=True)
   model = load_tace("model.pt", device="cuda")
   model = convert_cgtp(model, implementation="o2")  # "o3" for direct CGTP

The default ``implementation="auto"`` switches each CGTP interaction to the
other form. Conversion returns a copy; recreate its optimizer before training.

Precision and warmup
--------------------

.. list-table:: TF32 for float32 operations
   :header-rows: 1
   :widths: 40 30 30

   * - ``TACE_USE_TF32``
     - Training
     - Inference
   * - Unset
     - Enabled
     - Disabled
   * - ``1``
     - Enabled
     - Enabled
   * - ``0``
     - Disabled
     - Disabled

TF32 changes eligible PyTorch matrix multiplication and cuDNN arithmetic, not
float32 storage, float64 operations, or custom CUDA contraction arithmetic.
Choose precision before compilation or export; changing the environment does
not recompile an AOTI package.

EQX generates and caches CUDA kernels on first use, including derivative
kernels. ``EQX_USE_CUDA_GRAPH=1`` is an optional replay path, disabled by default.
It can increase memory and is not required for the recommended workflow.

.. _tace-export-tutorial:

Export and deployment
---------------------

.. list-table::
   :header-rows: 1
   :widths: 30 25 45

   * - Command
     - Backend
     - Purpose
   * - ``tace-export-train``
     - State dictionary
     - Continued training or fine-tuning
   * - ``tace-export-eval``
     - ``state_dict`` / ``full_model``
     - Native PyTorch inference
   * - ``tace-export-eval``
     - ``aoti``
     - Compiled graph for ASE and TorchSim
   * - ``tace-export-lammps``
     - ``mliap`` / ``aoti``
     - Eager or compiled LAMMPS ML-IAP model

Commands accept checkpoints, state-dict packages, and serialized models.
Use ``-o`` for the output path, ``-f`` for fidelity, and ``--dtype`` for
precision. ``tace-compile`` aliases ``tace-export-eval``.

.. important::

   AOTI export requires PyTorch >= 2.13. If using OEQ, use OEQ >= 0.6.4.
   Supported outputs include energy, forces, stress, virials, charges, and
   noncollinear magnetic forces. LES is not supported by this compilation path.

ASE and TorchSim
~~~~~~~~~~~~~~~~

The recommended export command above writes ``model.pt2``. TACE creates
synthetic inputs with dynamic node, edge, and graph dimensions; ``--sample``
is an optional override, not a requirement.

.. code-block:: python

   from tace.interface.ase import TACEAseCalc

   calc = TACEAseCalc(model="model.pt2", device="cuda")

``load_tace("model.pt2", device="cuda")`` also loads the compiled graph directly.
The package does not call ``torch.compile`` again, but external operator
registrations and their runtime dependencies remain necessary. EQX-generated
kernels may still require first-use compilation if their cache is absent.
For TECE-OAM-RRA packages, import
``eqx.models.tace.tece_oam_rra.interaction`` before loading to register its
model-specific operators.
Validate the exported model in the deployment environment.

For non-compiled inference:

.. code-block:: bash

   tace-export-eval -m model.ckpt --backend state_dict --device cpu

This writes ``model.ckpt-state_dict.pt``. The ``full_model`` backend instead
writes ``model.ckpt-full_model.pt`` and is more dependent on Python package
versions.

LAMMPS
~~~~~~

The AOTI export writes a package and an ML-IAP loader:

.. code-block:: text

   model.ckpt-lammps_aoti.pt2   compiled package
   model.ckpt-lammps_aoti.pt    ML-IAP loader containing the package

Point LAMMPS to the loader, not the raw package:

.. code-block:: text

   pair_style mliap unified model.ckpt-lammps_aoti.pt 0
   pair_coeff * * H C N

Use ``--aoti-package`` to choose the package path. ML-IAP requires CUDA Kokkos;
multi-rank runs also require CUDA-aware MPI. See :doc:`lammps` for setup.

Compile for a compatible deployment OS, PyTorch/CUDA ABI, and GPU architecture.
A CUDA package cannot be loaded on CPU. Set backend flags before exporting a
full model or AOTI graph; they do not replace operators in an already compiled
artifact.

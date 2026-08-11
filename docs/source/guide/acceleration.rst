.. _acceleration-tutorial:

Acceleration Tutorial
=====================

TACE provides several composable acceleration layers:

* OpenEquivariance (OEQ) and cuEquivariance (CUEQ) provide alternative
  implementations of the same edge-level equivariant operations and are
  mutually exclusive;
* EquiTorch (EQT) independently accelerates supported product-basis tensor
  products and can be combined with either OEQ or CUEQ;
* the EQX preview backend currently accelerates only the scatter calculation
  in ``uuSO2Interaction``;
* PyTorch compilation accelerates a larger part of the model and can either
  run inside the current Python process or produce an AOTInductor package for
  later deployment. AOTI is independent of the kernel-backend selection.

The acceleration backend must be selected before the model is constructed.
The same settings can be used during training, validation, testing, and model
export, subject to the backend limitations described below.
Installation commands for each optional backend are listed separately in
:ref:`installation`.

Kernel Backends
---------------

The following kernel backends are available:

.. list-table::
   :header-rows: 1
   :widths: 28 42 30

   * - Backend
     - Scope
     - Environment variable
   * - OpenEquivariance
     - Alternative edge-level backend; choose OEQ or CUEQ
     - ``TACE_USE_OEQ=1``
   * - cuEquivariance
     - Alternative edge-level backend; choose OEQ or CUEQ
     - ``TACE_USE_CUE=1``
   * - EquiTorch
     - Independent product-basis acceleration
     - ``TACE_USE_EQT=1``
   * - EQX preview
     - Independent ``uuSO2Interaction`` scatter acceleration
     - ``TACE_USE_EQX=1``

For example:

.. code-block:: bash

   export TACE_USE_OEQ=1

Do not enable OEQ and CUEQ at the same time. EQT is independent and may be
enabled together with either one when the model contains a supported product
basis. EQX is also independent. It is planned as a general operator library
for ``O(3)``, ``O(2)``, Wigner-6j, and Cartesian equivariant computations, but
the current TACE integration only accelerates the scatter calculation in
``uuSO2Interaction``. OpenEquivariance is the recommended edge-level backend
for supported NVIDIA GPUs.

The current EQX path calls kernels from ``tace.models.triton_ops``. That module
is a temporary implementation placeholder while the corresponding operators
are developed in EQX; it is not a separate user-facing backend.

The acceleration environment can also be configured through one Python
interface before constructing or loading the model:

.. code-block:: python

   from tace.utils.env import enable_acceleration

   enable_acceleration(enable_oeq=True, enable_eqx=True)

By default, this interface only enables the requested backends and preserves
existing environment settings. Pass ``force=True`` to explicitly write every
backend setting and disable unselected backends.

The ASE and TorchSim calculators expose the same backends as constructor
options. For example:

.. code-block:: python

   from tace.interface.ase import TACEAseCalc

   calc = TACEAseCalc(
       model="model.pt",
       device="cuda",
       enable_oeq=True,
       enable_eqx=True,
   )

.. note::

   Environment variables can replace compatible modules while a checkpoint or
   state-dict package is being loaded. Once the complete Python model has been
   serialized, its modules are already fixed. Set the required acceleration
   variables before exporting a full model or a LAMMPS model.

PyTorch Compilation
-------------------

TACE provides two different compilation workflows.

.. list-table::
   :header-rows: 1
   :widths: 28 34 38

   * - Workflow
     - How to enable it
     - Intended use
   * - In-process compilation
     - ``TACE_USE_COMPILE=1``
     - Training, validation, and inference in the current Python process
   * - AOTInductor (AOTI)
     - ``tace-export-eval --backend aoti`` or
       ``tace-export-lammps --backend aoti``
     - Ahead-of-time deployment without compiling again at startup

In-process compilation caches compiled graphs in memory and does not create a
deployment artifact:

.. code-block:: bash

   export TACE_USE_COMPILE=1
   tace-train -cn tace.yaml

AOTI produces a ``.pt2`` package containing compiled native code. Loading the
package does not call ``torch.compile`` again.

AOTI is an independent compilation and deployment layer. It does not replace
OEQ, CUEQ, EQT, or EQX. Configure the desired compatible acceleration
backends before export; the resulting package captures the model constructed
with those selections.

.. important::

   TACE AOTInductor compilation and export require ``torch>=2.13``.
   Earlier PyTorch versions are not supported for AOTI export.
   When OpenEquivariance is enabled for AOTI export, OEQ 0.6.4 or newer is
   required.

Compilation currently supports energy and its force, stress, and virial
derivatives, together with their direct prediction variants. Models using
unsupported output properties or LES cannot be exported through the current
AOTI path.

AOTI packages contain machine-specific native code. Compile on the deployment
machine, or on a machine with a compatible operating system, PyTorch/CUDA ABI,
and GPU architecture. A package compiled for CUDA cannot be loaded on CPU.

.. _tace-export-tutorial:

TACE Export Tutorial
--------------------

Use the export command that matches the target workflow:

* ``tace-export-train`` creates an editable model package for continued
  training, fine-tuning, or transfer learning;
* ``tace-export-eval`` creates a native PyTorch inference model or an AOTI
  graph package;
* ``tace-export-lammps`` creates a LAMMPS ML-IAP model, optionally backed by
  AOTI.

The commands accept ``.ckpt``, state-dict ``.pt``/``.pth`` packages, and
serialized full models as input. Use ``-f`` to select a fidelity and ``--dtype``
to change model precision during export.

Export for Training
~~~~~~~~~~~~~~~~~~~

Use this form when the result must remain editable by TACE:

.. code-block:: bash

   tace-export-train -m model.ckpt

The default output is ``model.ckpt-state.pt``. It stores the state dictionary
and the model configuration required by ``load_tace`` and training utilities.
The command loads EMA parameters when they are available.

An explicit output, fidelity, and precision can also be selected:

.. code-block:: bash

   tace-export-train \
     -m model.ckpt \
     -o model-fidelity-1.pt \
     -f 1 \
     --dtype float32

Export for Native PyTorch Inference
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default ``state_dict`` backend is portable and reconstructs the model from
its saved configuration:

.. code-block:: bash

   tace-export-eval -m model.ckpt --backend state_dict --device cpu

The default output is ``model.ckpt-state_dict.pt``. This is the recommended
non-compiled format for normal evaluation and Python deployment.

The ``full_model`` backend serializes the complete Python module with
``torch.save``:

.. code-block:: bash

   tace-export-eval -m model.ckpt --backend full_model --device cpu

The default output is ``model.ckpt-full_model.pt``. It is convenient, but more
tightly coupled to the TACE and PyTorch versions used during export.

Both formats are loaded through the same API:

.. code-block:: python

   from tace.lightning import load_tace

   model = load_tace("model.ckpt-state_dict.pt", device="cuda")
   model.eval()

Export Eager or AOTI for ASE and TorchSim
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Set external-kernel variables before export, then select the ``aoti`` backend:

.. code-block:: bash

   export TACE_USE_OEQ=1

   tace-export-eval \
     -m model.ckpt \
     --backend aoti \
     --device cuda

No sample structure is required. TACE automatically builds a synthetic
two-graph input and exports dynamic node, edge, and graph dimensions, so the
resulting package can be used with different structures and batch sizes.
``--sample`` remains available as an optional advanced override, but is not
needed for normal ASE or TorchSim deployment.

The default output is ``model.pt2``. ``tace-compile`` is an alias for
``tace-export-eval`` and accepts the same options. The equivalent short command
is:

.. code-block:: bash

   tace-compile -m model.ckpt --backend aoti --device cuda

The graph ``.pt2`` package can be loaded with ``load_tace`` and shared by native
PyTorch consumers, including the ASE and TorchSim integrations:

.. code-block:: python

   from tace.lightning import load_tace

   model = load_tace("model.pt2", device="cuda")
   outputs = model(batch)

Export Eager or AOTI for LAMMPS
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The regular ML-IAP backend serializes the eager model:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps -m model.pt --backend mliap --device cuda

This creates ``model.pt-lammps_mliap.pt`` by default.

To compile the LAMMPS tensor graph ahead of time:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps \
     -m model.pt \
     --backend aoti \
     --device cuda

The AOTI backend creates two files:

* ``model.pt-lammps_aoti.pt2`` is the compiled AOTInductor package;
* ``model.pt-lammps_aoti.pt`` is the ``MLIAPUnified`` loader used by LAMMPS.

LAMMPS ML-IAP loads a pickled Python interface, so ``pair_style`` must point to
the ``.pt`` loader rather than directly to the ``.pt2`` package. The loader
contains the package bytes and loads the compiled model without recompilation:

.. code-block:: text

   pair_style mliap unified model.pt-lammps_aoti.pt 0
   pair_coeff * * H C N

Use ``--aoti-package`` to choose the package path and ``-o`` to choose the
ML-IAP loader path. TACE ML-IAP currently requires the CUDA Kokkos backend,
and multi-rank runs require CUDA-aware MPI. Native CPU inference remains
available through the ASE and TorchSim interfaces. Detailed LAMMPS setup is
covered in :doc:`lammps`.

.. _acceleration-tutorial:

Acceleration
============

TACE provides several composable acceleration layers:

* EquivariantX(EQX), OpenEquivariance (OEQ) and cuEquivariance (CUEQ) provide 
  alternative implementations of the same edge-level equivariant operations and 
  are mutually exclusive;
* EquiTorch (EQT) accelerates product-basis tensor
  products and can be combined with EQX, OEQ or CUEQ;
* PyTorch compilation accelerates a larger part of the model and can either
  run inside the current Python process or produce an AOTInductor package for
  later deployment. AOTI is independent of the kernel-backend selection.

Unless noted otherwise, select the backend before constructing the model.
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
     - Edge in atomic basis
     - ``TACE_USE_OEQ=1``
   * - cuEquivariance
     - Edge in atomic basis
     - ``TACE_USE_CUE=1``
   * - EquivariantX
     - Edge in atomic basis
     - ``TACE_USE_EQX=1``
   * - EquiTorch
     - Node in product basis
     - ``TACE_USE_EQT=1``

EquivariantX achieves more than 3x lower peak GPU memory usage than
OpenEquivariance and CuEquivariance at larger batch sizes. 
However, EquivariantX is still under active development, and bugs or other 
unexpected behaviors may still exist.

For example:

.. code-block:: bash

   export TACE_USE_OEQ=1

The acceleration environment can also be configured through Python
interface before constructing or loading the model:

.. code-block:: python

   from tace.utils.env import enable_acceleration

   enable_acceleration(enable_oeq=True)

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
   )

.. note::

   Environment variables can replace compatible modules while a checkpoint or
   state-dict package is being loaded. Once the complete Python model has been
   serialized, its modules are already fixed. Set the required acceleration
   variables before exporting a full model or a LAMMPS model.

.. _eqx-streaming:

Streamed CGTP with EquivariantX
-------------------------------

Install the optional backend from the TACE source directory, then select EQX
for ``O2CgtpInteraction``:

.. code-block:: bash

   pip install '.[eqx]'
   TACE_USE_EQX=1 tace-train -cn 3bpa_o2_cgtp.yaml

.. code-block:: python

   from tace.utils.env import enable_acceleration
   from tace.lightning import convert_cgtp, load_tace

   model = load_tace("TACE-OAM-7M.pt", device="cuda")
   model = convert_cgtp(model)  # cgtp -> o2_cgtp, detected per interaction
   enable_acceleration(enable_eqx=True)

The flag switches existing ``O2CgtpInteraction`` modules between PyTorch
and fused execution at runtime. Model loading is unchanged, and ordinary
``cgtp`` interactions retain their e3nn/OEQ/CUE backend. Use ``convert_cgtp``
to convert an existing model explicitly, without retraining or changing its
parameters. By default, conversion switches each CGTP interaction to the
other implementation.

The default ``eqx.o2`` operators use PyTorch without external kernels.
``TACE_USE_EQX=1`` selects generated CUDA convolutions on GPU and PyTorch on
CPU. Install the ``eqx`` extra and provide a CUDA toolkit; set ``CUDA_HOME``
if it is not discovered automatically.

The contraction fuses source gather, both feature rotations, sparse order-zero
CG coupling and target reduction, without retaining edge messages or their
adjoints. Radial projections and their transposes use bounded GEMM workspaces.
The same contraction evaluates recursive adjoints, including the mixed second
derivatives required by force-loss training. Channelwise ``uvu`` paths,
shared weights, and float32/float64 are supported. Other connection modes
are not implemented by the CUDA backend.

Paths sharing input features reuse their local rotation. Derivative programs
are partitioned by shared dependencies and register requirements, without a
fixed angular-degree threshold. Compiled register counts and local-memory
usage refine the partition, and static schedules are cached. Wide channel tiles
share Wigner matrices across warps. Cached launch configurations use compiled
occupancy and, outside CUDA Graph capture, measured latency on private outputs.
Tuning uses a bounded scratch allocation and never modifies model results.
Cached kernel phases are launched together.
Node-owned reductions accumulate before
writing results; split rows and shared gradients use atomic additions.
Floating-point summation order can therefore differ from the reference.

TACE supplies edge vectors together with packed Wigner matrices. Geometry
derivatives contract sparse rotation generators directly, avoiding dense
Wigner-matrix adjoints and backpropagation through frame construction.
The same contraction rule applies recursively to higher derivatives, including
the second derivatives used in force training. Harmonic amplitudes retain
their separate radial derivatives. Degree-zero harmonic paths cancel both
rotations exactly, while keeping each path and its weight independent.

Quaternion alignment and direct quaternion polynomials build the cached
degree matrices without degree-to-degree recursion. Analytically generated
coefficients require no numerical fitting. The matrices retain the packed
layout, alignment convention, and float32/float64 support. The recursive CG
construction remains available through the EQX API with ``method="recursive"``.
The EQX API also retains differentiation of arbitrary matrix entries when edge
vectors are omitted. The radial MLP's preceding layers and node-level
``linear_down`` remain separate.

A small C++ launcher is built once. Independent NVRTC kernels are compiled
concurrently and cached by code, compiler version and GPU architecture.
Edge counts are runtime arguments. Launch tuning is also performed on first use.
Warm up forward and required derivatives
before CUDA Graph capture. Compilation is excluded from warmed throughput
measurements.

This backend targets eager training and inference. AOTInductor deployment
with EQX is not currently validated; use the documented OEQ export path
when an AOTI package is required.

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
OEQ, CUEQ, or EQT. Configure the desired compatible acceleration
backends before export; the resulting package captures the model constructed
with those selections.

.. important::

   TACE AOTInductor compilation and export require ``torch>=2.13``.
   Earlier PyTorch versions are not supported for AOTI export.
   When OpenEquivariance is enabled for AOTI export, OEQ 0.6.4 or newer is
   required.

Compilation currently supports energy force, stress, virial, charge
and noncollinear-magnetic-force. Models using
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

.. Export for Training
.. ~~~~~~~~~~~~~~~~~~~

.. Use this form when the result must remain editable by TACE:

.. .. code-block:: bash

..    tace-export-train -m model.ckpt

.. The default output is ``model.ckpt-state.pt``. It stores the state dictionary
.. and the model configuration required by ``load_tace`` and training utilities.
.. The command loads EMA parameters when they are available.

.. An explicit output, fidelity, and precision can also be selected:

.. .. code-block:: bash

..    tace-export-train \
..      -m model.ckpt \
..      -o model-fidelity-1.pt \
..      -f 1 \
..      --dtype float32

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

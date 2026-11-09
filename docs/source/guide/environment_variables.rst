.. _environment-variables:

Environment variables
=====================

Set environment variables before starting Python or ``tace-train``. This also
ensures that settings read during module import take effect. The tables below
describe the behavior when a variable is unset; API arguments and export
commands may explicitly enable an option.

For TACE switches, use the strings ``1`` and ``0`` to enable and disable them.
Use the exact names ``TACE_USE_CUE`` and ``TACE_USE_COMPILE``;
``TACE_USE_CUEQ`` and ``TACE_USECOMPILE`` are not recognized.

Setting variables
-----------------

For one command in a Linux or macOS shell:

.. code-block:: bash

   TACE_USE_EQX=1 TACE_USE_COMPILE=1 tace-train -cn tace.yaml

In PowerShell, set variables before launching the command:

.. code-block:: powershell

   $env:TACE_USE_EQX = "1"
   $env:TACE_USE_COMPILE = "1"
   tace-train -cn tace.yaml

Choose backends supported by the current platform; see :ref:`installation`.
For Python scripts, set import-time options before importing TACE:

.. code-block:: python

   import os

   os.environ["TACE_USE_EQX"] = "1"
   os.environ["TACE_MEMORY_FIRST"] = "0"

   from tace.lightning import load_tace

   model = load_tace("model.ckpt", device="cuda")

For training, ``misc.env`` sets variables during initialization and overwrites
inherited values with the same names. All values must be quoted strings:

.. code-block:: yaml

   misc:
     env:
       TACE_USE_EQX: "1"
       TACE_USE_COMPILE: "1"
       TACE_MEMORY_FIRST: "0"
       WANDB_MODE: "offline"

Use the shell for ``TACE_USE_MATRIX_WEIGHT``, CUDA device visibility, and
toolkit/build settings: modules may already have been imported before
``misc.env`` is applied.

Operator backends and compilation
---------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 18 54

   * - Variable
     - Unset
     - Effect of ``1``
   * - ``TACE_USE_EQX``
     - Not enabled
     - Use EquivariantX for supported convolutions, ACE contractions,
       element-dependent Linear maps, and Cartesian operators.
   * - ``TACE_USE_LUNA``
     - Not enabled
     - Use LunaEquivariance for supported spherical CGTP convolutions.
       Requires Linux, an NVIDIA GPU, and its CUDA build dependencies.
   * - ``TACE_USE_OEQ``
     - Not enabled
     - Use OpenEquivariance for supported tensor-product convolutions.
   * - ``TACE_USE_EQT``
     - Selected ACE paths enable it automatically
     - Enable EquiTorch for supported ACE tensor products. Without an explicit
       setting, ordinary CGTP ACE with correlation greater than two uses it
       when EQX and time-reversal representations are not selected.
       Set ``0`` to disable that automatic choice.
   * - ``TACE_USE_CUE``
     - Not enabled
     - Use cuEquivariance for supported convolutions and MACE symmetric
       contractions.
   * - ``TACE_USE_COMPILE``
     - Eager model path
     - Jointly compile training predictions, loss, and parameter gradients;
       compile predictions for evaluation. Independent of operator backends.
       See :doc:`acceleration` for supported training settings.

When multiple backends are enabled, each operator selects the first supported
one in **EQX > Luna > OEQ > EQT > CUEQ** order. Install the dependencies for the
chosen backend. Compilation support depends on the backend and requested
outputs; LES is not supported by the current compiled model path.

``enable_acceleration(enable_eqx=True, enable_compile=True)`` and corresponding
calculator arguments set the relevant flags to ``1``. False arguments leave
existing flags unchanged unless ``force=True`` is used. The helper controls
only these six acceleration flags.

For AOTI deployment, use ``tace-export-eval --backend aoti`` or
``tace-export-lammps --backend aoti``; these commands enable compilation
automatically. Set the backend, precision, and fusion options before export.
Changing variables when loading a ``.pt2`` package does not rebuild its graph.
The nvalchemi toolkit wrapper disables in-process compilation and loads AOTI
packages for compiled deployment; see :doc:`nvalchemi`.

.. _tace-memory-first:

Radial MLP fusion
-----------------

``TACE_MEMORY_FIRST`` selects the radial MLP fusion boundary for EQX-backed
CGTP, Cartesian CGTP, O2-CGTP, UU-O2, UV-O2 (including magnetic O2), and TECE
convolutions.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Behavior
   * - Unset or ``0``
     - Evaluate hidden layers with PyTorch and fuse only the final projection
       into the convolution. This is the default; ``torch.compile`` can
       optimize hidden activations and their force-training derivatives.
   * - ``1``
     - Fuse the complete radial MLP into bounded edge tiles and recompute its
       hidden activations during backward, reducing retained radial activations.

Set this option before model construction or compilation. It applies to
training and evaluation without changing model parameters or checkpoint
layout. Measure peak memory and complete training steps for the chosen model:
full fusion can trade throughput for lower activation memory, while the total
memory cost also depends on the convolution and derivative workload.

.. code-block:: bash

   # Default: fuse only the final radial projection.
   TACE_USE_EQX=1 TACE_USE_COMPILE=1 tace-train -cn tace.yaml

   # Fuse the complete radial MLP.
   TACE_USE_EQX=1 TACE_USE_COMPILE=1 TACE_MEMORY_FIRST=1 tace-train -cn tace.yaml

FP32 matrix multiplication
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 22 48

   * - Variable
     - Values
     - Purpose
   * - ``TACE_MATMUL_PRECISION``
     - ``ieee``, ``tf32``
     - Select the arithmetic used for supported FP32 matrix multiplications.
   * - ``TACE_USE_TF32``
     - ``0``, ``1``
     - Shorthand for ``ieee`` and ``tf32``, respectively.

When both variables are unset and ``dtype.matmul`` is ``null``, training
without autocast selects TF32; inference and autocast select IEEE. TF32 is
applied only when the model dtype is float32. An explicit ``dtype.matmul``
must agree with the environment, and the two environment variables must agree
with each other. Conflicts raise an error. Autocast cannot be combined with
an explicit TF32 selection.

These variables do not set parameter storage dtype or geometry precision.
Use ``dtype.model`` and ``dtype.autocast`` for network precision; graph data,
geometry, and physical output reductions remain float64. See :doc:`training`.

.. code-block:: bash

   TACE_MATMUL_PRECISION=ieee tace-train -cn tace.yaml

Parameter layout and training features
--------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 28 12 60

   * - Variable
     - Default
     - Effect of ``1``
   * - ``TACE_USE_MATRIX_WEIGHT``
     - ``0``
     - Store supported TACE equivariant Linear weights as separate path
       matrices instead of one flat parameter. This exposes matrix parameters
       to optimizers such as HybridMuon. Set it before importing TACE.
   * - ``TACE_USE_U_SHIFT``
     - ``0``
     - Correct raw selective-PBE+U energy labels online. Structures containing
       O or F receive negative per-metal offsets for V, Cr, Mn, Fe, Co, Ni, Mo,
       and W. Source files and graph caches retain their original labels.
   * - ``TACE_USE_DENS``
     - ``0``
     - Enable DeNS position-noise augmentation, force conditioning, and noise
       readout during training in spherical models. This changes the training
       objective and adds model parameters.

Matrix-weight defaults are read when the Linear classes are imported;
``misc.env`` is too late for this option. Switching the environment alone does
not convert an existing model's parameter layout. Packed EQX/e3nn parameters
are not automatically split into Muon matrices.

U-shift follows Eq. (4) and Table S1 of
`arXiv:2601.21056 <https://arxiv.org/abs/2601.21056>`_. The correction is
``E_shifted = E_raw + sum(n_M * delta_E_M)`` for structures containing O or F,
including metal atoms outside their neighborhoods. Structures without O/F,
including isolated metal atoms, receive zero correction. Forces, stress,
and label weights remain unchanged because the offset depends only on composition.

Set ``TACE_USE_U_SHIFT=1`` for training, validation, test, and
``tace-eval --test 1`` against raw labels calculated with the Materials Project
selective +U scheme. Statistics, fitted atomic references, and energy filtering
use raw labels. The flag does not change graph or statistics cache keys.
Predictions from the trained model already use the corrected energy convention.
Do not enable this correction for PBE-only or already-corrected datasets.

DeNS augmentation, force conditioning, and noise prediction run only during
training. Validation and test use clean structures and the corresponding
ordinary force/stress losses. DeNS requires force labels (``forces`` or
``direct_forces``), compatible
losses such as ``mse_dens_forces``, and
``model.config.readout_emlp.use_alllayer: false``. Set ``TACE_USE_DENS`` before
constructing or loading the matching model architecture. See
:doc:`details/optimizer` and :doc:`details/loss` for optimizer and loss settings.

EQX CUDA Graph replay
---------------------

``EQX_USE_CUDA_GRAPH`` belongs to EquivariantX and also affects EQX operators
used by TACE. It is independent of ``TACE_USE_COMPILE``.

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Value
     - Behavior
   * - Unset
     - Replay eligible nonlinear node-product programs and their derivatives.
       Other native contraction families do not opt into replay by default.
   * - ``0``
     - Disable EQX's native CUDA Graph replay.
   * - ``1``
     - Enable replay for all supported native contraction families, including
       convolution and ACE paths. This can add capture and cache costs when
       graph sizes change frequently.

Unsupported or small workloads use ordinary execution. This switch controls
EQX's own captures, not CUDA Graphs created by other libraries. See
:ref:`eqx-streaming` for the execution details.

Runtime, build, and logging variables
-------------------------------------

These variables are owned by CUDA, PyTorch, EQX's build runtime, or the logger;
they are separate from the ``TACE_*`` model settings.

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Variable
     - Purpose and default
   * - ``CUDA_VISIBLE_DEVICES``
     - Select and reorder the GPUs visible to the process before CUDA
       initialization. Device indices in TACE refer to this visible order.
       Multi-GPU graph building and ``trainer.devices`` are configured
       separately; see :doc:`neighbor_list`.
   * - ``CUDA_HOME``
     - CUDA toolkit root used to build CUDA extensions. Set it before imports
       when automatic toolkit detection does not find the intended installation.
   * - ``PYTORCH_NVCC``
     - Override the NVCC executable used by PyTorch builds and TACE's Luna
       toolkit check. Otherwise the Luna check uses ``CUDA_HOME/bin/nvcc``.
   * - ``MAX_JOBS``
     - Limit build parallelism. EQX's kernel-compilation pool defaults to
       ``min(16, CPU count)`` workers and clamps the supplied integer between
       one and the CPU count. Set it before the CUDA runtime is imported.
   * - ``OMP_NUM_THREADS``
     - Set the OpenMP CPU thread count per process before Python starts.
       Account for all training ranks when choosing it; this does not set
       DataLoader worker count or GPU count.
   * - ``TORCH_EXTENSIONS_DIR``
     - Override PyTorch's extension build-cache root, including EQX's launcher.
       When unset, PyTorch chooses its standard user cache directory.
   * - ``TORCHINDUCTOR_CACHE_DIR``
     - Override PyTorch's compiler cache directory. Compiled TACE training also
       stores reusable joint programs in its ``tace/joint/v1`` subdirectory;
       no additional TACE switch is required.
   * - ``XDG_CACHE_HOME``
     - EQX stores generated CUDA binaries in ``$XDG_CACHE_HOME/eqx/cuda``;
       its fallback is ``~/.cache/eqx/cuda``. TACE's own cache remains
       ``~/.cache/tace``.
   * - ``WANDB_MODE``
     - Choose Weights & Biases logging mode. The supplied training example sets
       ``offline`` through ``misc.env``; use ``online`` for online logging.
       See :doc:`details/logger`.

Distributed launchers supply variables such as ``RANK``, ``LOCAL_RANK``, and
``WORLD_SIZE``. Configure the launcher and trainer rather than setting these
manually to select a GPU count.

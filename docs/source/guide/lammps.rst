LAMMPS ML-IAP
=============

TACE runs in LAMMPS through the unified Python interface of
``pair_style mliap`` and its Kokkos accelerator variant ``mliap/kk``. The
same exported TACE model can be used by one or more MPI ranks. Each rank owns
one spatial subdomain, loads one model instance, and exchanges ghost features
between interaction layers.

The example input files are available in
`example/lammps <https://github.com/xvzemin/tace/tree/main/example/lammps>`__.
The relevant upstream references are the
`LAMMPS ML-IAP documentation <https://docs.lammps.org/pair_mliap.html>`__,
`Kokkos package guide <https://docs.lammps.org/Speed_kokkos.html>`__, and
`LAMMPS command-line options <https://docs.lammps.org/Run_options.html>`__.

Requirements
------------

* Build LAMMPS with ``KOKKOS``, ``ML-IAP``, ``ML-SNAP``, and ``PYTHON``.
  ``ML-SNAP`` is a build dependency of ``ML-IAP`` even though TACE does not
  use a SNAP descriptor.
* LAMMPS and TACE must use the same Python environment. That environment must
  contain TACE, PyTorch, Cython, and the CuPy package matching the CUDA major
  version.
* TACE ML-IAP currently requires the CUDA Kokkos backend.
* One MPI rank on one GPU does not require CUDA-aware MPI. Any run with more
  than one rank requires a CUDA-aware MPI implementation.
* AOTI export requires ``torch>=2.13``. AOTI packages are specific to
  their deployment software stack and accelerator target, so export on a
  machine compatible with the deployment nodes.

Build LAMMPS
------------

The following example builds LAMMPS for an NVIDIA RTX 4090. Replace
``Kokkos_ARCH_ADA89`` with the architecture of the deployment GPU, such as
``Kokkos_ARCH_AMPERE80`` for A100/A800 or ``Kokkos_ARCH_HOPPER90`` for
H100/H200/H20. The complete architecture list is maintained in the
`LAMMPS Kokkos build documentation
<https://docs.lammps.org/Build_extras.html#kokkos>`__.

.. code-block:: bash

   micromamba activate tace

   pip install cython cupy-cuda12x # cuda12

   git clone https://github.com/lammps/lammps.git
   cd lammps

   cmake -S cmake -B build-tace \
     -D CMAKE_BUILD_TYPE=Release \
     -D CMAKE_INSTALL_PREFIX="$PWD/install-tace" \
     -D CMAKE_C_COMPILER=gcc \
     -D CMAKE_CXX_COMPILER=g++ \
     -D CMAKE_CUDA_HOST_COMPILER=g++ \
     -D Python_EXECUTABLE="$(which python)" \
     -D Python3_EXECUTABLE="$(which python)" \
     -D BUILD_MPI=ON \
     -D BUILD_SHARED_LIBS=ON \
     -D PKG_KOKKOS=ON \
     -D PKG_ML-IAP=ON \
     -D PKG_ML-SNAP=ON \
     -D PKG_PYTHON=ON \
     -D MLIAP_ENABLE_PYTHON=ON \
     -D Kokkos_ENABLE_CUDA=ON \
     -D Kokkos_ENABLE_SERIAL=ON \
     -D Kokkos_ARCH_ADA89=ON

   cmake --build build-tace --parallel 8
   cmake --install build-tace

   export PATH="$PWD/install-tace/bin:$PATH"
   export LD_LIBRARY_PATH="$PWD/install-tace/lib:$LD_LIBRARY_PATH"

For multi-rank GPU runs, configure with the compiler wrappers from the
CUDA-aware MPI installation:

.. code-block:: bash

   -D MPI_C_COMPILER=/path/to/cuda-aware-mpi/bin/mpicc
   -D MPI_CXX_COMPILER=/path/to/cuda-aware-mpi/bin/mpicxx


Export a TACE model
-------------------

The eager ML-IAP export is portable between compatible CUDA devices and does
not compile the model.
Acceleration backends such as OEQ must be selected before export:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps -m model.pt --backend mliap --device cuda

For AOTI deployment:

.. code-block:: bash

   export TACE_USE_OEQ=1
   tace-export-lammps -m model.pt --backend aoti --device cuda


LAMMPS input
------------

.. code-block:: text

   units           metal
   atom_style      atomic
   boundary        p p p
   newton          on

   read_data       structure.lammps-data

   mass 1 1.00794
   mass 2 12.0107
   mass 3 14.0067

   pair_style      mliap unified TACE-OAM-7M.pt-lammps_mliap.pt 0
   pair_coeff      * * H C N

   neighbor        2.0 bin
   thermo_style    custom step pe ke etotal temp press vol fmax fnorm
   thermo          10

   velocity        all create 300 5463576
   fix             1 all nvt temp 300 300 0.1
   timestep        0.001
   run             10000

The final ``0`` in ``pair_style`` disables neighbor lists centered on ghost
atoms. TACE instead exchanges ghost features after each interaction layer.
Keep ``newton on``; it is required by ML-IAP. The element order in
``pair_coeff`` must match the LAMMPS atom types.

Single GPU
----------

Use one MPI rank and one visible GPU:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   lmp -k on g 1 -sf kk \
     -pk kokkos newton on neigh half \
     -in in.lmp

Single node, multiple GPUs
--------------------------

The recommended mapping is one MPI rank per physical GPU. For two GPUs:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0,1 \
   mpirun --bind-to none -np 2 \
     lmp -k on g 2 -sf kk \
       -pk kokkos newton on neigh half \
       -in in.lmp

OpenMPI defines ``OMPI_COMM_WORLD_LOCAL_RANK`` for each process, which LAMMPS
uses to assign distinct GPUs. Other supported launchers provide equivalent
local-rank variables. ``-np`` is the total number of ranks, while ``g 2`` is
the number of GPUs available on each node.

One GPU, multiple workers
-------------------------

Multiple MPI ranks can share one GPU:

.. code-block:: bash

   CUDA_VISIBLE_DEVICES=0 \
   mpirun --bind-to none -np 2 \
     lmp -k on g 1 -sf kk \
       -pk kokkos newton on neigh half \
       -in in.lmp

Multiple nodes
--------------

For Slurm, two nodes with two GPUs per node allocation is typically:

.. code-block:: bash

   srun --nodes=2 --ntasks-per-node=2 \
     --gpus-per-task=1 --gpu-bind=single:1 \
     lmp -k on g 2 -sf kk \
       -pk kokkos newton on neigh half \
       -in in.lmp

Direct Hessian prediction
=========================

``direct_hessian`` predicts the complete Cartesian energy Hessian in
eV/angstrom² without differentiating the energy or forces. The existing
``hessian`` property remains the energy-derived Hessian.

Readout
-------

After message passing, a node Linear predicts the diagonal blocks. A separate
edge interaction predicts off-diagonal blocks:

.. code-block:: text

   cat(h_i, h_j) -> Linear -> radial-weighted TP with spherical harmonics
                 -> Linear -> Gate -> Linear -> Cartesian block

The edge interaction does not aggregate to nodes or apply a product block.
Both readouts produce ``0e + 1e + 2e``, converted to 3-by-3 Cartesian blocks.
Symmetrization enforces ``H_ij = H_ji.T``; an off-diagonal block need not be
symmetric by itself. Diagonal blocks are learned, not obtained by a sum rule.

``model.config.direct_hessian.cutoff`` sets a separate Hessian interaction
radius, defaulting to 16 angstroms. All non-self pairs inside that radius are
evaluated, independently of the backbone neighbor list. Blocks outside it are
zero. Self pairs use only the node readout and need no spherical harmonics.

The edge interaction uses ``num_channel`` and the existing ``radial_basis``
configuration, including its MLP widths, type and normalization. Radial MLP
inputs concatenate the radial basis and both endpoints' element attributes.
The cutoff always multiplies the MLP output; the subsequent edge Linear layers
have no bias. ``atomic_basis.gate_m0``, ``scalar_act`` and ``tensor_act`` control
the Gate. ``readout_emlp.bias`` controls the diagonal scalar bias.

Only the final descriptor is used, irrespective of ``readout_emlp.use_alllayer``.
Fidelity heads are supported. The readout is independent of energy scale/shift
and atomic reference energies.

This implementation supports nonperiodic structures and costs quadratically
in atom count. It does not impose positive definiteness, the acoustic sum
rule, or equality to the derivatives of a separately predicted energy.
Periodic force constants, ghost-atom execution and AOTI export are not supported.

Data layout
-----------

The output always has shape ``(sum_g N_g**2, 3, 3)``. Structures are concatenated;
within a structure, block ``i * N + j`` is ``H[i, :, j, :]``. There are no
cross-structure blocks. To recover a single structure's matrix:

.. code-block:: python

   matrix = blocks.reshape(N, N, 3, 3).permute(0, 2, 1, 3).reshape(3 * N, 3 * N)

Store labels in ``atoms.info["direct_hessian"]`` or in an ASE database's
``data`` dictionary. Map a different key with ``dataset.keys.direct_hessian_key``.
Supported layouts are:

.. list-table::
   :header-rows: 1

   * - Layout
     - Shape and index order
   * - ``matrix``
     - ``(3N, 3N)``: rows ``3i + a``, columns ``3j + b``
   * - ``flat``
     - ``(9N²,)``: flattened ``matrix`` in row-major order
   * - ``ijab``
     - ``(N, N, 3, 3)``
   * - ``blocks``
     - ``(N², 3, 3)``

These four shapes are detected automatically and converted to contiguous
``(N², 3, 3)`` blocks for training. No layout flag is required. For three atoms,
``(3, 3, 3, 3)`` uses ``(i, j, a, b)`` order and ``(9, 9)`` is a matrix.
No mass weighting or unit conversion is applied.
Missing labels have zero weight; NaN entries are masked independently.

Download and training
---------------------

``example/data/download_HORM.py`` downloads the author's latest
`HORM release <https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm>`_
without arguments. Raw files are stored in ``~/dataset/HORM/raw/v<version>``;
the official splits are converted to ``~/dataset/HORM/train.db``, ``valid.db``
and ``test.db``. The current release contains approximately 24 GB of raw
data; conversion requires additional disk space.

From the directory containing the TACE checkout:

.. code-block:: bash

   pip install -e ./tace --no-deps
   pip install 'kagglehub>=1.0.2'
   python tace/example/data/download_HORM.py
   cd hessain
   tace-train --config-path ../tace/example/train --config-name hessian

The converter preserves the original energy, force and Hessian units. It
loads the author's Python pickles. Completed databases from the same release
are reused; existing databases from a different release are not overwritten.

For large training subsets, use ``dataset.storage_mode=lmdb`` to cache graphs
on disk. Dataset conversion is streamed, but the current TACE ASE database
reader still loads structures before graph preprocessing; plan host memory
accordingly when using the full dataset.

Loss
----

.. code-block:: yaml

   loss:
     _target_: tace.utils.loss.NormalLoss
     loss_property: [direct_hessian]
     loss_function_name: [huber_direct_hessian]
     loss_property_weights: [1.0]
     loss_function_kwargs: [{reduction: structure, huber_delta: 0.01}]

``mse_direct_hessian``, ``mae_direct_hessian``, ``l2mae_direct_hessian`` and
``huber_direct_hessian`` are available. Huber is the training example's default;
``huber_delta`` defaults to 0.01 eV/angstrom².

MSE, MAE and Huber operate on Cartesian entries. L2MAE takes the Frobenius norm
of each atom-pair error block, without dividing by its nine entries. Missing
entries are excluded from the norm; fully missing blocks are excluded from
the average. The default ``structure`` reduction first averages within each
structure, then over structures. ``component`` reduction averages all valid
entries directly, or all valid atom pairs for L2MAE. Structure weights multiply
the losses without changing the Huber threshold or the averaging denominator.
MAE/RMSE metrics average valid entries and are reported in meV/angstrom².
Energy and force losses can be added normally; Hessian prediction itself
does not require position derivatives.

Evaluation
----------

.. code-block:: bash

   tace-eval -m checkpoints_epoch/last.ckpt -i ~/dataset/HORM/test.db \
       -o predictions.xyz --test 1 --dtype float32

``TACE_direct_hessian`` in the output contains packed ``(N², 3, 3)`` blocks.
When reusing predictions as labels,
select ``dataset.keys.direct_hessian_key: TACE_direct_hessian``.

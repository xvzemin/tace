HIP-TACE
========

HIP-TACE **directly predicts the Hessian** using ``direct_hessian``, without
differentiating energy or forces. It can be trained together with energy and
forces. The separate ``hessian`` property computes the energy-derived Hessian
for evaluation; it is not the HIP-TACE training target.

Download and train
------------------

With TACE installed, run from the repository root:

.. code-block:: bash

   pip install 'kagglehub>=1.0.2'
   python example/data/download_HORM.py
   cd example/train
   tace-train -cn hip-tace.yaml model.config.parity=true

The download script converts the author's latest
`HORM release <https://www.kaggle.com/datasets/yunhonghan/hessian-dataset-for-optimizing-reactive-mliphorm>`_
to ``~/dataset/HORM/train.db``, ``valid.db`` and ``test.db``. Completed databases
from the same release are reused.

``example/train/hip-tace.yaml`` trains energy, forces and direct Hessians with
MAE loss weights ``1:10:25``. Adjust the dataset paths and batch sizes there.
Keep ``direct_hessian`` in ``loss.loss_property`` to enable the prediction head.

The main model options are:

.. code-block:: yaml

   model:
     config:
       parity: true
       Lmax: 2
       lmax: 3
       special:
         direct_hessian:
           cutoff: 16.0
           interaction_type: cgtp  # cgtp or o2

Use ``parity: true`` to allow the antisymmetric parts of off-diagonal Hessian
blocks. ``Lmax`` limits the edge feature degrees and ``lmax`` sets the spherical
harmonic cutoff. ``special.direct_hessian.cutoff`` is a separate interaction
radius in angstroms; off-diagonal contributions outside it are zero.

Available Hessian losses are ``mae_direct_hessian``, ``mse_direct_hessian``,
``huber_direct_hessian`` and ``l2mae_direct_hessian``. Huber uses
``huber_delta`` (default 0.01 eV/angstrom²); L2MAE measures the Frobenius norm
of each atom-pair block. Set ``reduction: component`` to average over entries
(pairs for L2MAE), or ``structure`` to give each structure equal weight.

For compiled training, prefix the training command with ``TACE_USE_COMPILE=1``.
The example uses LMDB graph caching; existing caches are reused for training
and dataset statistics.

Data and output
---------------

Provide Hessian labels in eV/angstrom² under ``atoms.info["direct_hessian"]``
or the ASE database's ``data["direct_hessian"]``. For a different key, set
``dataset.keys.direct_hessian_key``. No unit conversion or mass weighting is
applied.

Supported label shapes are ``(3N, 3N)``, its row-major flattened form,
``(N, N, 3, 3)`` and ``(N², 3, 3)``. Shapes are detected automatically;
four-index labels use ``(i, j, a, b)`` order. Provided labels must be finite.

Predictions always use ``(N², 3, 3)`` blocks, ordered by ``i * N + j``.
Batches concatenate the blocks of each structure without cross-structure
pairs. To recover one structure's full matrix:

.. code-block:: python

   matrix = blocks.reshape(N, N, 3, 3).permute(0, 2, 1, 3).reshape(3 * N, 3 * N)

Molecular and periodic structures are supported; periodic outputs are
fixed-cell atom-pair blocks, not image-resolved force constants. Output storage
grows as ``N²``. Direct predictions are not constrained to equal the second
derivatives of the predicted energy.

Evaluate and export
-------------------

Replace the checkpoint path with your trained model:

.. code-block:: bash

   tace-eval -m checkpoints_epoch/last.ckpt -i ~/dataset/HORM/test.db \
       -o predictions.xyz --test 1 --dtype float32

The output key is ``TACE_direct_hessian``. MAE and RMSE are reported in
meV/angstrom².

Export an AOTInductor inference package with:

.. code-block:: bash

   tace-export-eval -m checkpoints_epoch/last.ckpt -o model.pt2 \
       --backend aoti --device cuda --dtype float32

Use ``model.pt2`` with ``tace-eval`` or ``load_tace``. FP32 and FP64 are
supported; precision is fixed at export. AOTI packages are for inference,
not training. The compiled input batch must contain at least two atoms;
use the PyTorch model for a single isolated atom.

Citing
------

If you use HIP-TACE, please cite the HIP paper:
`HIP: Hessian Interatomic Potentials without derivatives
<https://arxiv.org/abs/2509.21624>`_.

.. code-block:: bibtex

   @misc{burger2025hiphessian,
     title = {{HIP}: Hessian Interatomic Potentials without derivatives},
     author = {Andreas Burger and Luca Thiede and Nikolaj R{\o}nne and
               Varinia Bernales and Nandita Vijaykumar and Tejs Vegge and
               Arghya Bhowmik and Alan Aspuru-Guzik},
     year = {2025},
     eprint = {2509.21624},
     archivePrefix = {arXiv},
     primaryClass = {cs.LG},
     url = {https://arxiv.org/abs/2509.21624}
   }

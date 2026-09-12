Magnetic TACE
=============

mTACE learns magnetic potential energy surfaces with or without spin--orbit
coupling (SOC).

Installation
------------

If explicit time-reversal symmetry is required, the time-reversal version of 
e3nn should be installed.
If time-reversal symmetry is instead incorporated through data augmentation, 
or if a non-SOC model is used, the standard e3nn package is sufficient.

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"
   python -c "from e3nn import o3; print(o3.Irrep('1eo'))"


Time-reversal e3nn supplies :math:`O(3)\times\mathbb Z_2^{\mathcal T}`
operations; EquivariantX supplies the local
:math:`O(2)\times\mathbb Z_2^{\mathcal T}` operations. EquivariantX is bundled
with TACE, and time-reversal support is detected automatically.


Training
--------

Data and commands
~~~~~~~~~~~~~~~~~

Provide ``initial_noncollinear_magmoms`` with shape ``(num_atoms, 3)``.
Collinear moments can use the same vector field, for example
:math:`\mathbf m_i=(0,0,m_i)`. Magnetic-force labels, when available, use
``noncollinear_magnetic_forces`` with the same shape and convention

.. math::

   \mathbf F_i^{\mathrm{mag}}
   =-\frac{\partial E}{\partial\mathbf m_i}.

Start from ``example/train/soc_o2_mtace.yaml`` or
``example/train/nonsoc_o2_mtace.yaml``. Keep their base configurations alongside
them as described in :doc:`details/defaults`. Update the dataset paths, field mapping,
LMDB shard directories, and magnetic cutoff (scale) settings for your data. 
The examples include magnetic-force losses and validation metrics; remove these entries
if those labels are unavailable.

Run the appropriate configuration from ``example/train``:

.. code-block:: bash

   cd example/data
   python download_FeDeepSpin.py
   cd ../train
   tace-train -cn soc_o2_mtace.yaml

   # For non-SOC data, use instead:
   # cd ../data
   # python download_CrN.py
   # cd ../train
   # tace-train -cn nonsoc_o2_mtace.yaml

.. list-table:: Architecture selection
   :header-rows: 1
   :widths: 20 80

   * - ``use_spin_orbit_coupling``
     - Symmetry
   * - ``true``
     - Coupled space--spin
       :math:`O(3)\times\mathbb Z_2^{\mathcal T}`, using time-reversal e3nn.
   * - ``false``
     - Independent
       :math:`O(3)_{\mathrm{space}}\times SO(3)_{\mathrm{spin}}
       \times\mathbb Z_2^{\mathcal T}`.

Neither configuration needs symmetry augmentation for its matching data.
If an SOC architecture is deliberately trained on non-SOC data,
``dataset.augmentation: [spin_rotation]`` samples independent spin rotations.
Without time-reversal e3nn, add ``time_reversal`` to learn that symmetry as
well. Augmentation does not make either missing symmetry exact; see
:doc:`details/dataset`. Do not apply independent spin rotations to SOC data.

Magnetic cutoff and readout
---------------------------

Cutoff and radial features
~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration field ``model.config.fidelity[].magnetic_scale`` supplies
the magnetic cutoff :math:`m_{\mathrm{cut},Z}`. It accepts a positive scalar,
an atomic-number-to-cutoff mapping, or ``null``. Explicit values are used
directly. With ``null``, the training statistics give

.. math::

   m_{\mathrm{cut},Z}
   =1.2\max_{i\in\mathcal D_Z}\|\mathbf m_i\|+0.1,

where :math:`\mathcal D_Z` contains training moments of element :math:`Z`
within the corresponding fidelity. Each fidelity uses its own cutoff.
The cutoff acts only on the radial magnitude map:

.. math::

   u_i=\frac{\|\mathbf m_i\|}{m_{\mathrm{cut},Z_i}},\qquad
   \widetilde m_i=1-2\min(u_i^2,1),\qquad
   T(\widetilde m_i)
   =\left(T_1(\widetilde m_i),\ldots,T_{N_{\mathrm{mag}}}(\widetilde m_i)\right).

The squared-magnitude dependence is smooth at zero. For
:math:`\|\mathbf m_i\|\ge m_{\mathrm{cut},Z_i}`, the radial features are
constant, with a derivative discontinuity at the cutoff. The angular solid
harmonics are not clipped and grow as :math:`\|\mathbf m_i\|^l`.

One-body magnetic energy
~~~~~~~~~~~~~~~~~~~~~~~~

``readout_emlp.use_one_body_magmoms`` enables the separate magnitude-only
readout, which restores :math:`T_0=1`:

.. math::

   E_{\mathrm{1b}}=\sum_i\left(
   W_{Z_i0}+\sum_{n=1}^{N_{\mathrm{mag}}}W_{Z_i n}T_n(\widetilde m_i)
   \right).

Magnetic TACE
=============

mTACE supports collinear and noncollinear magnetic systems, with or without
spin--orbit coupling (SOC).

Installation
------------

For SOC models with explicit time-reversal symmetry, install time-reversal e3nn:

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"
   python -c "from e3nn import o3; print(o3.Irrep('1eo'))"

``o2_mag`` and ``w6j_mag`` detect time-reversal support automatically.
Non-SOC models also support standard e3nn. Disable incompatible backends for
time-reversal SOC models:

.. code-block:: bash

   export TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0

Training
--------

Provide ``initial_noncollinear_magmoms`` with shape ``(num_atoms, 3)``.
For collinear data, use vectors such as :math:`(0,0,m_i)`.
``noncollinear_magnetic_forces`` has the same shape and follows
:math:`\mathbf F_i^{\mathrm{mag}}=-\partial E/\partial\mathbf m_i`.
Remove magnetic-force losses and metrics if these labels are unavailable.

Adjust dataset paths, field mappings, cache directories and magnetic cutoffs.
Keep the ``TACE`` and ``mTACE`` subdirectories together under ``example/train``;
see :doc:`details/defaults`.
Run the following commands from the repository root.

SOC: local O(2)
~~~~~~~~~~~~~~~

.. code-block:: bash

   python example/data/download_FeDeepSpin.py
   cd example/train/mTACE
   tace-train -cn soc_o2_mtace.yaml

.. literalinclude:: ../../../example/train/mTACE/soc_o2_mtace.yaml
   :language: yaml

Non-SOC: local O(2)
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   python example/data/download_CrN.py
   cd example/train/mTACE
   tace-train -cn nosoc_o2_mtace.yaml

.. literalinclude:: ../../../example/train/mTACE/nosoc_o2_mtace.yaml
   :language: yaml

SOC: Wigner-6j
~~~~~~~~~~~~~~

This example inherits the SOC dataset and training settings. ``w6j_mag``
requires ``angular_basis.use_spin_orbit_coupling: true``.

.. code-block:: bash

   python example/data/download_FeDeepSpin.py
   cd example/train/mTACE
   tace-train -cn soc_w6j_mtace.yaml

.. literalinclude:: ../../../example/train/mTACE/soc_w6j_mtace.yaml
   :language: yaml

SOC couples spatial and spin rotations; non-SOC allows them independently.
Do not apply independent spin-rotation augmentation to SOC data.

Magnetic cutoff and readout
---------------------------

.. warning::

   Explicitly set ``model.config.fidelity[].magnetic_scale`` for each element
   and fidelity. TACE's automatic estimate is only a reference. Keep training
   and intended magnetic-relaxation ranges strictly below the cutoff;
   derivatives can jump at the boundary. This is a magnetic-moment cutoff,
   not the spatial neighbor cutoff.

Include isolated-atom spin-energy scans in training to constrain the one-body
magnetic readout and improve extrapolation.

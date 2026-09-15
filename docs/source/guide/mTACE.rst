Magnetic TACE
=============

mTACE learns magnetic potential energy surfaces with or without spin--orbit coupling (SOC) 
for both collinear and noncollinear magnetic systems.

Installation
------------

Standard e3nn supports full :math:`O(3)` equivariance but does not account for time-reversal symmetry.
For an SOC model with explicit time-reversal symmetry, install time-reversal e3nn:

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"
   python -c "from e3nn import o3; print(o3.Irrep('1eo'))"

Time-reversal e3nn supplies :math:`O(3)\times\mathbb Z_2^{\mathcal T}`
operations; EquivariantX supplies the local
:math:`O(2)\times\mathbb Z_2^{\mathcal T}` operations. EquivariantX is bundled
with TACE. For ``o2_mag``, time-reversal support is detected automatically;
there is no separate YAML switch. The explicit non-SOC construction also
preserves time reversal with standard e3nn, as explained below.

Disable backends that do not support time-odd irreps when constructing a
time-reversal SOC model:

.. code-block:: bash

   export TACE_USE_OEQ=0 TACE_USE_CUE=0 TACE_USE_EQT=0

This does not disable PyTorch compilation; see :doc:`acceleration`.


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
LMDB shard directories, and magnetic cutoff settings for your data.
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

Symmetry selection
------------------

.. code-block:: yaml

   defaults:
     - soc_o2_mtace@_global_
     - _self_

   model:
     config:
       parity: true
       Lmax: 2
       lmax: 2
       mmax: 2
       num_channel: 64
       angular_basis:
         magnetic_Lmax: 2
         use_spin_orbit_coupling: true
       atomic_basis:
         type: o2_mag
         edge_nonlinear: gate
         scalar_act: [silu, tanh]
         tensor_act: sigmoid
         use_radial_rotary_attention: false
         num_head: 2
       radial_basis:
         apply_cutoff: false
         num_mag_radial_basis: 10
         hidden: [64, 64]
       node_update:
         magnetic_type: element
       readout_emlp:
         use_one_body_magmoms: true
       fidelity:
         - name: LCAO
         - # We recommend using ground-state isolated-atom energies provided by the user.
           atomic_energy: null 
           # Illustrative Fe cutoff; replace after checking your data
           magnetic_scale:
             26: 5.0

SOC uses the same rotation for positions and moments; non-SOC allows spatial
and spin rotations independently. Spatial inversion reverses positions but
not axial magnetic moments. Time reversal reverses all moments together but
leaves positions unchanged.

.. list-table:: Guaranteed physical symmetries
   :header-rows: 1
   :widths: 45 15 15 25

   * - Symmetry
     - ``parity``
     - ``use_spin_orbit_coupling``
     - e3nn installation
   * - :math:`SO(3)`
     - ``false``
     - ``true``
     - Standard
   * - :math:`O(3)`
     - ``true``
     - ``true``
     - Standard
   * - :math:`O(3)\times\mathbb Z_2^{\mathcal T}`
     - ``true``
     - ``true``
     - Time-reversal
   * - :math:`O(3)_{\mathrm{space}}\times SO(3)_{\mathrm{spin}}
       \times\mathbb Z_2^{\mathcal T}`
     - ``true``
     - ``false``
     - Standard or time-reversal

For SOC data, prefer ``parity: true`` with time-reversal e3nn. For non-SOC
data, prefer the explicit non-SOC configuration; neither requires symmetry
augmentation for its matching data. If the ``parity: true`` SOC architecture
is deliberately trained on non-SOC data, set
``dataset.augmentation: [spin_rotation]``. With standard e3nn, use
``[spin_rotation, time_reversal]`` instead; on SOC data, use only
``[time_reversal]``. Augmentation samples missing symmetries but does not make
them exact. Do not apply independent spin rotations to SOC data. See
:doc:`details/dataset`.

Magnetic cutoff and readout
---------------------------

Cutoff and radial features
~~~~~~~~~~~~~~~~~~~~~~~~~~

The configuration field ``model.config.fidelity[].magnetic_scale`` supplies
the magnetic cutoff :math:`m_{\mathrm{cut},Z,f}` for element :math:`Z` and
fidelity :math:`f`.
an atomic-number-to-cutoff mapping, or ``null``. Explicit values are used
directly. With ``null``, the training statistics give

.. math::

   m_{\mathrm{cut},Z,f}
   =1.2\max_{i\in\mathcal D_{Z,f}}\|\mathbf m_i\|+0.1,

where :math:`\mathcal D_{Z,f}` contains the corresponding training moments.
The cutoff is fixed during training and uses the same units as the moments,
usually :math:`\mu_{\mathrm B}`. It is distinct from the spatial neighbor
cutoff ``model.config.cutoff``.

.. warning::

   You must review and explicitly set ``fidelity[].magnetic_scale`` for every
   fidelity before magnetic training. Although TACE can estimate it from
   ``max_noncollinear_magmoms_norm_by_element``, use that estimate only as a
   reference, not as a validated physical cutoff. Cover the training data,
   isolated-atom scans, and intended magnetic-relaxation range, including
   elements whose training moments are zero. Keep the intended range strictly
   below the cutoff: radial features saturate beyond it, and the derivative
   generally jumps at the boundary.


One-body magnetic energy
~~~~~~~~~~~~~~~~~~~~~~~~

To ensure physically reasonable extrapolation behavior, we recommend including 
spin-energy scans of isolated atoms in the training data.





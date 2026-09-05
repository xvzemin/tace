Magnetic TACE
=============

Magnetic TACE (mTACE) describes non-collinear magnetic moments and atomic
geometry in one equivariant model. It is intended for magnetic potential
energy surfaces with spin--orbit coupling (SOC), where spatial rotations of
the structure and magnetic moments are coupled. The model enforces
:math:`O(3)\times\mathbb Z_2^T`: proper rotations, spatial inversion, and
global time reversal.

The magnetic local-:math:`O(2)` interaction remains an experimental research
interface. Model and checkpoint compatibility may change while this interface
is under development.

Installation requirements
-------------------------

Install TACE first, either from PyPI or from a source checkout:

.. code-block:: bash

   pip install tace

To enforce time-reversal equivariance, replace the standard e3nn installation
with the ``time-reversal`` branch:

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"

The distribution and Python import are still named ``e3nn``. The additional
irrep label can be checked with:

.. code-block:: bash

   python -c "from e3nn import o3; print(o3.Irrep('1eo'))"

This e3nn variant is required for an mTACE model that strictly enforces time
reversal. With standard e3nn, TACE can still construct a spatially
:math:`O(3)`-equivariant model, but the time-reversal label is not represented.

Time-reversal models use the time-reversal implementation for irreps, solid
harmonics, linear maps, gates, and tensor products. Accelerated equivariant
kernels selected through EQT, CUEQ, OEQ, or EquivariantX do not carry the time
label and are therefore rejected for these models. EquivariantX itself is
bundled with TACE and does not need a separate installation.

Training tutorial
-----------------

Dataset fields
~~~~~~~~~~~~~~

Each atom must provide an initial non-collinear magnetic moment
``initial_noncollinear_magmoms`` with shape ``(num_atoms, 3)``. The vector is
an axial vector in the same Cartesian coordinate system as the positions.
Zero magnetic moments are valid inputs.

The usual energy, force, stress, and virial fields are supported. To train
magnetic forces, additionally provide ``noncollinear_magnetic_forces`` with
shape ``(num_atoms, 3)``. TACE uses the convention

.. math::

   \bm F_i^{\mathrm{mag}}=-\frac{\partial E}{\partial\bm m_i}.

Map the dataset names in the normal dataset configuration:

.. code-block:: yaml

   dataset:
     train_file: /path/to/train.xyz
     valid_file: /path/to/valid.xyz
     keys:
       energy_key: energy
       forces_key: forces
       initial_noncollinear_magmoms_key: initial_noncollinear_magmoms
       noncollinear_magnetic_forces_key: noncollinear_magnetic_forces

If magnetic-force labels are unavailable, omit their key and remove the
property from the loss. The predicted energy can still depend on the magnetic
input even when magnetic-force labels are not used for training.

Minimal configuration
~~~~~~~~~~~~~~~~~~~~~

Start from ``example/train/mtece.yaml`` or override the following fields in a
normal TACE configuration:

.. code-block:: yaml

   loss:
     loss_property:
       - energy
       - forces
       - noncollinear_magnetic_forces
     loss_function_name:
       - mse_energy_per_atom
       - mse_forces
       - mse_noncollinear_magnetic_forces
     loss_property_weights: [1.0, 1.0, 10.0]
     loss_function_kwargs: [{}, {}, {}]

   model:
     config:
       parity: true

       radial_basis:
         num_mag_radial_basis: 10
         magnetic_normalization: rational

       angular_basis:
         magnetic_basis:
           Lmax: 2
           normalization: element

       magnetic_edge_update:
         type: identity

       atomic_basis:
         type: o2_mag
         use_radial_rotary_attention: true

       scale_shift:
         magmoms_scale_type: rms_noncollinear_magmoms_norm_by_element

Run the configuration from its directory so Hydra can discover it:

.. code-block:: bash

   cd /path/to/configs
   tace-train -cn mtece.yaml

The repository example can be launched directly from ``example/train`` after
updating its dataset paths.

:math:`O(3)\times\mathbb Z_2^T` representations
------------------------------------------------

Irrep labels
~~~~~~~~~~~~

An irrep is labeled by :math:`(l,p,t)`, where :math:`l` is angular degree,
:math:`p\in\{+1,-1\}` is spatial-inversion parity, and
:math:`t\in\{+1,-1\}` is time-reversal parity. In the string notation, the
first ``e`` or ``o`` denotes :math:`p`, and the second denotes :math:`t`.

The principal quantities used by mTACE are:

.. list-table::
   :header-rows: 1
   :widths: 28 18 24 24

   * - Quantity
     - Irrep
     - Spatial inversion
     - Time reversal
   * - Relative position :math:`\bm r_{ij}`
     - ``1oe``
     - :math:`\bm r_{ij}\mapsto-\bm r_{ij}`
     - unchanged
   * - Magnetic moment :math:`\bm m_i`
     - ``1eo``
     - unchanged
     - :math:`\bm m_i\mapsto-\bm m_i`
   * - Energy :math:`E`
     - ``0ee``
     - unchanged
     - unchanged
   * - Atomic force :math:`\bm F_i`
     - ``1oe``
     - changes sign
     - unchanged
   * - Magnetic force :math:`\bm F_i^{\mathrm{mag}}`
     - ``1eo``
     - unchanged
     - changes sign

The spatial and time operations are independent. In particular, spatial
inversion does not reverse an axial magnetic moment, whereas time reversal
does. For an SOC model, a spatial rotation acts on positions and magnetic
moments together; an independent rotation of magnetic moments alone is not a
required symmetry.

Regular solid harmonics
~~~~~~~~~~~~~~~~~~~~~~~

mTACE uses regular solid harmonics rather than spherical harmonics of a unit
magnetic direction:

.. math::

   \mathcal R^{(l)}(\bm m)
   =\lVert\bm m\rVert^l\mathcal Y^{(l)}(\widehat{\bm m}).

They are homogeneous polynomials of degree :math:`l`. Consequently they are
well defined at :math:`\bm m=0`; every :math:`l>0` block vanishes smoothly
there. No operation divides by :math:`\lVert\bm m\rVert`.

Because a magnetic moment is ``1eo``, its successive solid-harmonic blocks are

.. math::

   \mathcal R^{(0)}:0ee,\qquad
   \mathcal R^{(1)}:1eo,\qquad
   \mathcal R^{(2)}:2ee,\qquad
   \mathcal R^{(3)}:3eo,\ldots

Spatial parity remains even for every degree because the input is axial. Time
parity alternates as :math:`(-1)^l`. By comparison, solid harmonics of the
polar relative position have ``0ee``, ``1oe``, ``2ee``, ``3oe``, and so on.

Tensor products and nonlinearities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a tensor-product path

.. math::

   (l_1,p_1,t_1)\otimes(l_2,p_2,t_2)
   \longrightarrow(l_3,p_3,t_3),

the usual angular-momentum rule and both parity rules must hold:

.. math::

   |l_1-l_2|\le l_3\le l_1+l_2,\qquad
   p_3=p_1p_2,\qquad t_3=t_1t_2.

Thus a single magnetic vector is time odd, but two magnetic factors may form a
time-even energy contribution. For example, ``1eo x 1eo -> 0ee`` contains the
scalar correlation :math:`\bm m_i\cdot\bm m_j`. Time reversal therefore does
not require removing all odd-:math:`l` magnetic features; it requires the
complete interaction path to end in the correct time parity.

Equivariant linear maps only mix copies with identical :math:`(l,p,t)`. Gates
and tensor products multiply both spatial and time parities. Biases, ordinary
radial weights, and the predicted energy are restricted to ``0ee``.

Global :math:`O(3)` and local :math:`O(2)` interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each spatial edge, mTACE rotates global :math:`O(3)` node features into a
bond-aligned local frame. Restricting an :math:`O(3)` irrep to local
:math:`O(2)` preserves its time label while separating its magnetic orders.
For example,

.. math::

   1eo\downarrow O(2)=0oo\oplus1mo.

The local interaction applies linear maps, gates, and tensor products with the
same time-parity selection rules, then rotates and scatters the result back to
global node features. The local representation changes the computational
basis; it does not weaken global :math:`O(3)\times\mathbb Z_2^T` equivariance.

For the magnetic edge input, mTACE first constructs solid harmonics for the two
endpoint moments and couples them with an unweighted tensor product. The
resulting magnetic edge irreps are projected to the model channels using
weights generated from endpoint magnetic radial bases. These projected edge
attributes enter the local-:math:`O(2)` interaction together with the spatial
edge frame. This is where geometry and magnetic moments interact in the SOC
model.

Magnetic normalization
----------------------

Element scales and statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``scale_shift.magmoms_scale_type`` selects a per-element statistic :math:`s_Z`.
The two common choices are:

``max_noncollinear_magmoms_norm_by_element``
   Maximum :math:`\lVert\bm m_i\rVert` for each element. This avoids mapping
   ordinary training samples far outside the characteristic range, but it is
   sensitive to outliers.

``rms_noncollinear_magmoms_norm_by_element``
   RMS of magnetic-vector magnitudes for each element. This is the default and
   describes the typical magnitude without being controlled by a single
   outlier. Values above the characteristic range remain smooth because the
   default radial map is rational rather than clamped.

The characteristic range used by the magnetic basis is

.. math::

   M_Z=1.2s_Z+0.1.

For multiple fidelities, TACE uses the largest selected value for each element
to obtain one common model scale.

The statistics follow the same convention as force statistics:

.. math::

   \operatorname{RMS}_{\mathrm{components}}
   =\sqrt{\frac{1}{3N}\sum_i\lVert\bm m_i\rVert^2},\qquad
   \operatorname{RMS}_{\mathrm{norm}}
   =\sqrt{\frac{1}{N}\sum_i\lVert\bm m_i\rVert^2}.

They are stored as ``rms_noncollinear_magmoms`` and
``rms_noncollinear_magmoms_norm``, respectively, with corresponding
``*_by_element`` forms. The norm RMS is :math:`\sqrt{3}` times the component
RMS.

Radial normalization
~~~~~~~~~~~~~~~~~~~~

The magnetic Chebyshev basis is evaluated on a coordinate in
:math:`[-1,1]`. ``radial_basis.magnetic_normalization`` controls the mapping.

``clamp``
   The baseline mapping is

   .. math::

      u_i=\frac{\lVert\bm m_i\rVert}{M_{Z_i}},\qquad
      x_i=1-2\min(u_i,1)^2.

   It is smooth at zero because it depends quadratically on the magnitude. It
   saturates at :math:`x=-1` and has a derivative discontinuity at
   :math:`u=1`.

``rational``
   The globally smooth alternative is

   .. math::

      x_i=\frac{1-u_i^2}{1+u_i^2}.

   It uses only :math:`\lVert\bm m_i\rVert^2`, is smooth in the Cartesian
   components at zero, requires no clipping, and approaches :math:`-1`
   continuously for large moments.

Angular normalization
~~~~~~~~~~~~~~~~~~~~~

``angular_basis.magnetic_basis.normalization`` accepts three modes:

``integral``
   Uses integral-normalized regular solid harmonics. This is the numerical
   baseline.

``component``
   Uses component-normalized regular solid harmonics. It changes the fixed
   normalization constants but leaves the physical transformation rules
   unchanged.

``element``
   Uses the integral convention and evaluates the solid harmonics at the
   dimensionless element-scaled vector:

   .. math::

      \widetilde{\mathcal R}^{(l)}(\bm m_i)
      =\mathcal R^{(l)}\!\left(\frac{\bm m_i}{M_{Z_i}}\right)
      =M_{Z_i}^{-l}\mathcal R^{(l)}(\bm m_i).

   This controls the growth of high-degree blocks without normalizing the
   magnetic direction, so smoothness at :math:`\bm m=0` and all symmetry labels
   are preserved.

Model configuration reference
-----------------------------

The complete magnetic part of the architecture configuration is:

.. code-block:: yaml

   model:
     config:
       # Full spatial inversion is required by o2_mag.
       parity: true

       # Global node, spatial edge, and local magnetic-order truncations.
       Lmax: 2
       lmax: 3
       mmax: 3

       radial_basis:
         # The constant term is constructed internally and is not passed to
         # magnetic_edge_update.
         num_mag_radial_basis: 10
         magnetic_normalization: rational  # [clamp, rational]

       angular_basis:
         magnetic_basis:
           # Must not exceed the global Lmax when o2_mag is used.
           Lmax: 2
           normalization: element  # [integral, component, element]

       scale_shift:
         # [max_noncollinear_magmoms_norm_by_element,
         #  rms_noncollinear_magmoms_norm_by_element]
         magmoms_scale_type: rms_noncollinear_magmoms_norm_by_element

       magnetic_edge_update:
         # identity: gather the two endpoint radial bases directly
         # element: one shared element-dependent projection
         # element2: independent source and target element projections
         type: identity  # [identity, element, element2]

       atomic_basis:
         type: o2_mag
         use_radial_rotary_attention: true

       readout_emlp:
         # Add the magnetic radial basis directly to the one-body energy term.
         use_one_body_magmoms: true

``Lmax`` under ``magnetic_basis`` truncates both the node solid harmonics and
the coupled magnetic edge irreps. ``num_mag_radial_basis`` controls the number
of magnetic Chebyshev functions. The magnetic edge update and its radial MLP
are instantiated independently for each interaction layer.

The default uses the per-element RMS magnitude together with
``normalization: element`` and ``magnetic_normalization: rational`` for
dimensionless, smooth magnetic features. To reproduce the previous numerical
baseline, use ``normalization: integral`` and
``magnetic_normalization: clamp``.

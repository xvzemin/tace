Magnetic TACE
=============

``Magnetic TACE (mTACE)`` describes non-collinear magnetic moments and atomic
geometry in one equivariant model. It is intended for magnetic potential
energy surfaces ``with/without spin--orbit coupling (SOC)``. The model enforces
:math:`O(3)\times\mathbb Z_2^T`: proper rotations, spatial inversion, and
global time reversal.


:math:`O(3)\times\mathbb Z_2^T` representations throught Local O(2) Frame
-------------------------------------------------------------------------

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
   * - Relative position :math:`\mathbf{r}_{ij}`
     - ``1oe``
     - :math:`\mathbf{r}_{ij}\mapsto-\mathbf{r}_{ij}`
     - unchanged
   * - Magnetic moment :math:`\mathbf{m}_i`
     - ``1eo``
     - unchanged
     - :math:`\mathbf{m}_i\mapsto-\mathbf{m}_i`
   * - Energy :math:`E`
     - ``0ee``
     - unchanged
     - unchanged
   * - Atomic forces :math:`\mathbf{F}_i`
     - ``1oe``
     - changes sign
     - unchanged
   * - Magnetic forces :math:`\mathbf{F}_i^{\mathrm{mag}}`
     - ``1eo``
     - unchanged
     - changes sign

The spatial and time operations are independent. In particular, spatial
inversion does not reverse an axial magnetic moment, whereas time reversal
does. For an SOC model, a spatial rotation acts on positions and magnetic
moments together; an independent rotation of magnetic moments alone is not a
required symmetry. In contrast, for a non-SOC model, the spatial and magnetic
moments may be rotated independently.

Regular solid harmonics
~~~~~~~~~~~~~~~~~~~~~~~

mTACE uses regular solid harmonics rather than spherical harmonics of a unit
magnetic direction:

.. math::

   \mathcal R^{(l)}(\mathbf{m})
   =\lVert\mathbf{m}\rVert^l
   \mathcal Y^{(l)}(\widehat{\mathbf{m}}).

They are homogeneous polynomials of degree :math:`l`. Consequently they are
well defined at :math:`\mathbf{m}=0`; every :math:`l>0` block vanishes smoothly
there.

Because a magnetic moment is ``1eo``, its successive solid-harmonic blocks are

.. math::

   \mathcal R^{(0)}:0ee,\qquad
   \mathcal R^{(1)}:1eo,\qquad
   \mathcal R^{(2)}:2ee,\qquad
   \mathcal R^{(3)}:3eo,\ldots


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
scalar correlation :math:`\mathbf{m}_i\cdot\mathbf{m}_j`.

Equivariant linear maps only mix copies with identical :math:`(l,p,t)`. Gates
and tensor products multiply both spatial and time parities. The predicted
energy are restricted to ``0ee``.

Global :math:`O(3)` and local :math:`O(2)` interaction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For each spatial edge, mTACE rotates global :math:`O(3)` node features into a
edge-aligned local frame. Restricting an :math:`O(3)` irrep to local
:math:`O(2)` preserves its time label while separating its magnetic orders.
For the magnetic solid-harmonic basis truncated at :math:`L_{\max}=2`,

.. math::

   \left(
   \mathrm{0ee}\oplus\mathrm{1eo}\oplus\mathrm{2ee}
   \right)
   \downarrow_{O(2)\times\mathbb Z_2^T}
   =2\,\mathrm{0ee}\oplus\mathrm{0oo}
   \oplus\mathrm{1me}\oplus\mathrm{1mo}\oplus\mathrm{2me}.

The two local ``0ee`` blocks come from the global :math:`l=0` block and the
:math:`m=0` component of the global :math:`l=2` block, respectively.
For the magnetic edge input, mTACE first constructs solid harmonics for the two
endpoint moments and couples them with an unweighted tensor product. The
resulting magnetic edge irreps are projected to the model channels using
weights generated from endpoint magnetic radial bases. These projected edge
attributes enter the local-:math:`O(2)` interaction together with the spatial
edge frame. This is where geometry and magnetic moments interact in the model.


Installation requirements
-------------------------

Install TACE from Github first:

.. code-block:: bash

   pip install git+https://github.com/xvzemin/tace.git@main

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
kernels selected through EQT, CUEQ, OEQ, or EQX do not carry the time
label and are therefore rejected for these models. The resulting
models remain fully compatible with ``AOTI`` compilation.


Training tutorial
-----------------

Dataset fields
~~~~~~~~~~~~~~

Each atom must provide an initial non-collinear magnetic moment
``initial_noncollinear_magmoms`` with shape ``(num_atoms, 3)``.

The usual energy, force, stress, and virial fields are supported. To train
magnetic forces, additionally provide ``noncollinear_magnetic_forces`` with
shape ``(num_atoms, 3)``. TACE uses the convention

.. math::

   \mathbf{F}_i^{\mathrm{mag}}
   =-\frac{\partial E}{\partial\mathbf{m}_i}.

Minimal configuration
~~~~~~~~~~~~~~~~~~~~~

Start from ``example/train/mtece.yaml`` or override the following fields in a
normal TACE configuration: The repository example can be launched directly 
from ``example/train`` after updating its dataset paths.

.. code-block:: yaml

   dataset:
     train_file: /path/to/train.xyz
     valid_file: /path/to/valid.xyz

Magnetic normalization
----------------------

Element scales and statistics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``scale_shift.magmoms_scale_type`` selects a per-element statistic :math:`s_Z`.
The two common choices are:

``max_noncollinear_magmoms_norm_by_element``
   Maximum :math:`\lVert\mathbf{m}_i\rVert` for each element. This avoids mapping
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

Radial and Angular normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default uses the per-element RMS magnitude together with
``normalization: element`` and ``magnetic_normalization: rational`` for
dimensionless, smooth, bounded learned magnetic features. The positive
quartic tail supplies the separate large-moment stability guarantee.

The magnetic Chebyshev basis is evaluated on a coordinate in
:math:`[-1,1]`. ``radial_basis.magnetic_normalization`` controls the mapping.

``rational``
   The globally smooth alternative is

   .. math::

      x_i=\frac{1-u_i^2}{1+u_i^2}.

   It uses only :math:`\lVert\mathbf{m}_i\rVert^2`, is smooth in the Cartesian
   components at zero, requires no clipping, and approaches :math:`-1`
   continuously for large moments. This compactification is used only for the
   bounded learned correction; it is not responsible for stabilizing the
   large-moment energy.

``clamp``

   .. math::

      u_i=\frac{\lVert\mathbf{m}_i\rVert}{M_{Z_i}},\qquad
      x_i=1-2\min(u_i,1)^2.

   It is smooth at zero because it depends quadratically on the magnitude. It
   saturates at :math:`x=-1` and has a derivative discontinuity at
   :math:`u=1`.


``angular_basis.magnetic_basis.normalization`` accepts three modes:

``integral``
   Uses e3nn integral-normalized regular solid harmonics.

``component``
   Uses e3nn component-normalized regular solid harmonics.

``element``
   Uses the integral convention and the dimensionless element-scaled vector.

For every mode, the learned angular representation is the bounded rational
solid harmonic

.. math::

   \widetilde{\mathcal R}^{(l)}(\mathbf{q}_i)
   =\frac{\mathcal R^{(l)}(\mathbf{q}_i)}
   {(1+\lVert\mathbf{q}_i\rVert^2)^{l/2}},

where :math:`\mathbf{q}_i=\mathbf{m}_i/M_{Z_i}` for ``element`` and
:math:`\mathbf{q}_i=\mathbf{m}_i` otherwise. Near zero it has the same leading
behavior as the regular solid harmonic, while it remains bounded for
arbitrarily large moments.

Magnetic one-body energy
~~~~~~~~~~~~~~~~~~~~~~~~

The learned one-body basis omits the constant Chebyshev mode and is anchored
at zero moment:

.. math::

   \widetilde T_n(x_i)=T_n(x_i)-T_n(1)=T_n(x_i)-1.

Consequently, the magnetic one-body energy is zero at
:math:`\mathbf{m}_i=0` and cannot absorb an arbitrary element-dependent atomic
reference energy. The learned term uses the same energy scale as the main
readout, without receiving its additive shift.

The bounded learned correction is supplemented by

.. math::

   E_{\mathrm{conf}}
   =\sum_i \kappa_{Z_i}
   \left(\frac{\lVert\mathbf{m}_i\rVert^2}{M_{Z_i}^2}\right)^2,
   \qquad
   \kappa_Z=\kappa_{\min}+\operatorname{softplus}(\theta_Z)>0.

This element-dependent quartic tail is in energy units and guarantees that the
energy tends to positive infinity when any magnetic moment diverges. Its
coefficient is initialized from the configuration and may be trained while
remaining strictly positive.

When this strict confinement is enabled, the raw noncollinear moment cannot
also be injected through ``universal_embedding`` because that unbounded path
would invalidate the guarantee. Magnetic moments instead enter the learned
interaction through ``MagneticBasis``.

Magnetic TACE
=============

``Magnetic TACE (mTACE)`` describes collinear/noncollinear magnetic moments and atomic
geometry in one equivariant model. It provides constructions for
potential-energy surfaces with and without spin--orbit coupling (SOC). The two
constructions do not have the same symmetry group and should not be
interchanged.

Symmetry groups
---------------

SOC symmetry
~~~~~~~~~~~~

With SOC, spatial rotations act jointly on the structure and axial magnetic
moments. In the absence of an external time-reversal-breaking field, the group
is

.. math::

   G_{\mathrm{SOC}}=O(3)\times\mathbb Z_2^{\mathcal T}.

For :math:`Q\in O(3)` and :math:`\tau\in\{+1,-1\}`, the inputs transform as

.. math::

   \mathbf r_{ij}\mapsto Q\mathbf r_{ij},\qquad
   \mathbf m_i\mapsto\tau\det(Q)Q\mathbf m_i.

The magnetic orientation relative to the lattice is physical. Consequently,
SOC mTACE may represent magnetocrystalline anisotropy and other lattice-locked
spin interactions.

Non-SOC symmetry
~~~~~~~~~~~~~~~~

Without SOC, coordinate and spin rotations are independent:

.. math::

   G_{\mathrm{nonSOC}}
   =O(3)_{\mathrm{space}}\times SO(3)_{\mathrm{spin}}
   \times\mathbb Z_2^{\mathcal T}.

For independent :math:`Q_r\in O(3)` and :math:`Q_s\in SO(3)`, the energy obeys

.. math::

   E(\{Q_r\mathbf r_{ij}\},\{Q_s\mathbf m_i\})
   =E(\{\mathbf r_{ij}\},\{\mathbf m_i\}),
   \qquad
   E(\{\mathbf r_{ij}\},\{-\mathbf m_i\})
   =E(\{\mathbf r_{ij}\},\{\mathbf m_i\}).


Representations and local frames
--------------------------------

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
energy is restricted to ``0ee``.

SOC architecture
~~~~~~~~~~~~~~~~

For each atom, the SOC construction first evaluates the magnetic solid
harmonics

.. math::

   \mathcal M_i=\bigoplus_{l=0}^{L_{\mathrm{mag}}}
   \widetilde{\mathcal R}^{(l)}(\mathbf m_i).

The two endpoint representations are coupled with an unweighted tensor
product, retaining all symmetry-allowed magnetic edge irreps up to
:math:`L_{\mathrm{mag}}`. A bias-free equivariant linear map then projects
these edge attributes to the model channels. Its edge-dependent weights are
generated from the source and target magnetic radial bases.

Node features and the projected magnetic edge tensors are rotated with the
bond frame into local :math:`O(2)`. Restricting an :math:`O(3)` irrep preserves
its time label while separating its magnetic orders. For
:math:`L_{\mathrm{mag}}=2`,

.. math::

   \left(
   \mathrm{0ee}\oplus\mathrm{1eo}\oplus\mathrm{2ee}
   \right)
   \downarrow_{O(2)\times\mathbb Z_2^T}
   =2\,\mathrm{0ee}\oplus\mathrm{0oo}
   \oplus\mathrm{1me}\oplus\mathrm{1mo}\oplus\mathrm{2me}.

The two local ``0ee`` blocks come from the global :math:`l=0` block and the
:math:`m=0` component of the global :math:`l=2` block, respectively.
Because non-scalar magnetic edge tensors share the bond frame with the spatial
features, this branch permits the magnetic orientation relative to the lattice
to affect the message.

This is the default mode, corresponding to
``angular_basis.magnetic_use_soc: true``.

Non-SOC architecture
~~~~~~~~~~~~~~~~~~~~

The implemented non-SOC branch follows the rank-zero spin construction. It
uses the same smooth magnetic node basis, but the endpoint tensor product
retains only the independent paths

.. math::

   \chi_{ij}^{(l)}=
   \left[
   \widetilde{\mathcal R}^{(l)}(\mathbf m_i)\otimes
   \widetilde{\mathcal R}^{(l)}(\mathbf m_j)
   \right]_{0ee},
   \qquad 0\le l\le L_{\mathrm{mag}}.

Hence the magnetic edge representation is

.. math::

   (L_{\mathrm{mag}}+1)\times\mathrm{0ee}.

For example, :math:`L_{\mathrm{mag}}=2` produces three distinct ``0ee``
paths, rather than one scalar obtained by summing them. They contain the
constant, bilinear, and quadrupolar angular correlations. The same
radial-conditioned linear map mixes these paths into channel-wise magnetic
scalars. They enter the ordinary spatial local-:math:`O(2)` interaction and
are unchanged by the bond-frame rotation. No spin tensor index is therefore
contracted with a spatial index.

Set the architecture with:

.. code-block:: yaml

   model:
     config:
       atomic_basis:
         type: o2_mag
       angular_basis:
         magnetic_use_soc: false

The rank-zero construction represents exchange and higher polynomial
functions of pairwise spin correlations, while excluding lattice-locked SOC
terms. A fully general product-group network could retain :math:`l_s>0`
intermediate spin irreps, but it would require separate spatial and spin
indices and substantially greater computational cost.

Shared local update
~~~~~~~~~~~~~~~~~~~

Both branches gather the source and target node features, rotate their spatial
irreps into the bond frame, and apply path weights followed by
``O2Linear -> O2Gate -> O2Linear``. Radial rotary attention is optional. The
message is then rotated back, scattered to the target node, and passed to the
TACE product basis for the explicit many-body expansion. The only difference
is the magnetic input: the SOC branch rotates complete magnetic edge tensors,
whereas the non-SOC branch inserts bond-frame-independent ``0ee`` scalars.


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

This e3nn variant is required for the SOC branch to propagate time parity
through its non-scalar magnetic intermediate features. With standard e3nn,
TACE can still construct a spatially :math:`O(3)`-equivariant model, but the
time-reversal label is not represented. The non-SOC rank-zero magnetic edge
features are themselves time even by construction; the time-reversal e3nn
variant is nevertheless recommended for a consistent magnetic workflow and is
required as soon as time-odd intermediate features, inputs, or outputs are
used.

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

Radial and angular normalization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default uses the per-element RMS magnitude together with
``angular_basis.magnetic_normalization: element`` and
``radial_basis.magnetic_normalization: rational`` for dimensionless, smooth,
bounded learned magnetic features.

The magnetic Chebyshev basis is evaluated on a coordinate in
:math:`[-1,1]`. ``radial_basis.magnetic_normalization`` controls the mapping.
``radial_basis.num_mag_radial_basis`` is the number of returned non-constant
basis functions; the constant Chebyshev mode is not included.

``rational``

   .. math::

      x_i=\frac{1-u_i^2}{1+u_i^2}.

   It uses only :math:`\lVert\mathbf{m}_i\rVert^2`, is smooth in the Cartesian
   components at zero, requires no clipping, and approaches :math:`-1`
   continuously for large moments.

``clamp``

   .. math::

      u_i=\frac{\lVert\mathbf{m}_i\rVert}{M_{Z_i}},\qquad
      x_i=1-2\min(u_i,1)^2.

   It is smooth at zero because it depends quadratically on the magnitude. It
   saturates at :math:`x=-1` and has a derivative discontinuity at
   :math:`u=1`.


``angular_basis.magnetic_Lmax`` selects the maximum magnetic solid-harmonic
degree. ``angular_basis.magnetic_normalization`` accepts three modes:

``integral``
   Uses e3nn integral-normalized regular solid harmonics.

``component``
   Uses e3nn component-normalized regular solid harmonics.

``element``
   Uses the component convention and the dimensionless element-scaled vector.
   Element scaling already controls the species-dependent magnitude; an
   additional integral factor would uniformly reduce every component variance
   by :math:`4\pi` without adding a physical constraint.

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

The magnetic interaction basis contains :math:`T_1,\ldots,T_{N_{\mathrm{mag}}}`.
The one-body path prepends :math:`T_0=1` and passes the complete basis directly
to the element-dependent linear readout:

.. math::

   E_{\mathrm{1b}}
   =\sum_i\sum_{n=0}^{N_{\mathrm{mag}}}
   W_{Z_i n}\,T_n(x_i).

Here :math:`W_{Zn}` are the trainable one-body readout weights. No bias,
zero-moment correction, or separately parameterized magnetic term is added.
Since :math:`T_n(1)=1`, the value at zero moment is
:math:`\sum_n W_{Zn}` and is learned together with the element-dependent atomic
reference energy. The one-body contribution is added after the main energy
transformation and therefore receives neither its multiplicative scale nor its
additive shift.

Because both the rational coordinate and the Chebyshev expansion are bounded,
the one-body contribution approaches a finite value at arbitrarily large
moment. It therefore does not impose a coercive large-moment prior; such
behavior can only be fitted over the magnetic-moment range represented in the
training data, while the asymptote remains finite.

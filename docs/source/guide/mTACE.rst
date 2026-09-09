Magnetic TACE
=============

mTACE learns magnetic potential-energy surfaces with or without spin--orbit
coupling (SOC). This tutorial uses ``atomic_basis.type: o2_mag`` and assumes
no external time-reversal-breaking field.

Installation
------------

Install TACE following :doc:`../install/install`. For SOC models with explicit
time-reversal symmetry, install the ``time-reversal`` branch of e3nn:

.. code-block:: bash

   pip install --force-reinstall --no-deps \
     "e3nn @ git+https://github.com/xvzemin/e3nn.git@time-reversal"
   python -c "from e3nn import o3; print(o3.Irrep('1eo'))"

Time-reversal e3nn supplies :math:`O(3)\times\mathbb Z_2^{\mathcal T}`
operations; EquivariantX supplies the local
:math:`O(2)\times\mathbb Z_2^{\mathcal T}` operations. EquivariantX is bundled
with TACE, and time-reversal support is detected automatically.

Standard e3nn is sufficient for the explicit non-SOC construction below,
whose magnetic inputs to the spatial interaction are already time-even
scalars. Optional accelerated :math:`O(3)` kernels do not support time-odd
paths; keep them disabled for this tutorial. This restriction does not apply
to the native ``eqx.o2`` operators used by ``o2_mag``.

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

Start from ``example/train/soc_mtece.yaml`` or
``example/train/nonsoc_mtece.yaml``. Update the dataset paths, field mapping,
LMDB shard directories, and fidelity settings for your data. The examples
include magnetic-force losses and validation metrics; remove these entries
if those labels are unavailable.

Run the appropriate configuration from ``example/train``:

.. code-block:: bash

   cd example/train
   tace-train -cn soc_mtece.yaml
   # For non-SOC data, use instead:
   # tace-train -cn nonsoc_mtece.yaml

Model settings
~~~~~~~~~~~~~~

Both examples inherit ``tace.yaml``. The core magnetic settings are:

.. code-block:: yaml

   dataset:
     augmentation: []

   model:
     config:
       num_channel: 64
       Lmax: 2
       parity: true
       atomic_basis:
         type: o2_mag
         use_radial_rotary_attention: false
       angular_basis:
         magnetic_Lmax: 2
         use_spin_orbit_coupling: true  # false for non-SOC
       radial_basis:
         num_mag_radial_basis: 10
       node_update:
         magnetic_type: element
       fidelity:
         - name: PBE
           atomic_energy: null
           magnetic_scale: null

``magnetic_Lmax`` controls both input and output magnetic angular degrees;
it must be positive and no larger than ``model.config.Lmax``.
``parity: true`` retains physical spatial parity, including axial magnetic
moments.

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

Magnetic edge attributes
------------------------

For :math:`Q\in O(3)` and time reversal :math:`\tau\in\{+1,-1\}`,

.. math::

   \mathbf r_{ij}\mapsto Q\mathbf r_{ij},\qquad
   \mathbf m_i\mapsto\tau\det(Q)Q\mathbf m_i.

Thus positions, magnetic moments, and energy carry ``1oe``, ``1eo``,
and ``0ee``, respectively. The two parity suffixes denote spatial
inversion and time reversal. Time reversal flips all moments together, not
each atom independently, so exchange terms such as
:math:`\mathbf m_i\cdot\mathbf m_j` remain allowed.

Unlike positions, magnetic moments do not require a relative-vector
construction. On each node, evaluate the regular solid harmonics

.. math::

   \mathcal M_i=\bigoplus_{l=0}^{L_{\mathrm{mag}}}
   \mathcal R^{(l)}(\mathbf m_i),\qquad
   \mathcal R^{(l)}(\mathbf m_i)
   =\|\mathbf m_i\|^l\mathcal Y^{(l)}(\widehat{\mathbf m}_i).

The implementation evaluates these as Cartesian polynomials, without dividing
by the moment norm. They are smooth at zero; every :math:`l>0` block vanishes
there. The angular basis uses integral normalization on the original,
unscaled magnetic vector.

Gather both endpoints and form an unweighted tensor product:

.. math::

   \mathcal M_{ij}^{\mathrm{SOC}}
   =\bigoplus_{\substack{0\le l_i,l_j,L\le L_{\mathrm{mag}}\\
                         |l_i-l_j|\le L\le l_i+l_j}}
   [\mathcal M_i^{(l_i)}\otimes\mathcal M_j^{(l_j)}]_L,

   \mathcal M_{ij}^{\mathrm{nonSOC}}
   =\bigoplus_{l=0}^{L_{\mathrm{mag}}}
   [\mathcal M_i^{(l)}\otimes\mathcal M_j^{(l)}]_{L=0}.

Coupling paths remain separate. Non-SOC retains only :math:`L=0`, which
requires :math:`l_i=l_j` and yields :math:`L_{\mathrm{mag}}+1` copies of
``0ee``. For ``magnetic_Lmax: 2``, these are three independent scalar
channels containing constant, bilinear, and quadrupolar spin correlations.
SOC also retains non-scalar channels that couple magnetic orientation to the
lattice.

The shared construction allows compatible non-SOC scalar paths to be reused
in an SOC model. Such transfer requires matched magnetic cutoffs, basis
normalization, and channel layouts; it is not a direct checkpoint load with
only the SOC flag changed. See :doc:`finetune` for the supported fine-tuning
workflow.

Magnetic local O(2) convolution
-------------------------------

Let :math:`\widetilde m_i` be the transformed scalar magnitude defined below,
and :math:`T(\widetilde m_i)` its magnetic radial feature vector. With
``node_update.magnetic_type: element``, the paper's edge-conditioned form is

.. math::

   W_{ij}^{(t)}=\operatorname{MLP}^{(t)}
   \left(W_{Z_i}^{(t)}T(\widetilde m_i)
   \oplus W_{Z_j}^{(t)}T(\widetilde m_j)\right),\qquad
   \widetilde{\mathcal M}_{ij}^{(t)}
   =\operatorname{Linear}\left(\mathcal M_{ij};W_{ij}^{(t)}\right).

The bias-free linear map projects the raw magnetic edge attributes to the
model channels. Together with the unweighted product, it realizes a weighted
``uuw`` coupling. Raw angular attributes are shared across layers; the
element embeddings and MLP parameters are layer-specific.

``O2MagneticInteraction.magnetic_info_type`` selects ``edge`` (the form above)
or ``node``. The current ``node`` setting applies independent endpoint MLPs
before gathering and multiplies their gathered outputs to obtain
:math:`W_{ij}^{(t)}`. This is a class-level setting, not a YAML option.

``magnetic_type: identity`` uses the radial features directly.
``element`` uses one element-dependent map shared by both endpoints;
``element2`` uses independent endpoint maps. Both element-based options
project to ``num_channel``.

Gather the endpoint node features, rotate all three inputs into the same
edge-aligned frame, and concatenate matching local irrep channels:

.. math::

   x_{ij}^{(t)}
   =D_{ij}h_i^{(t)}\oplus D_{ij}h_j^{(t)}
   \oplus D_{ij}\widetilde{\mathcal M}_{ij}^{(t)}.

The restriction preserves time parity. For a magnetic node basis with
:math:`L_{\mathrm{mag}}=2`,

.. math::

   (\mathrm{0ee}\oplus\mathrm{1eo}\oplus\mathrm{2ee})
   \downarrow_{O(2)\times\mathbb Z_2^{\mathcal T}}
   =2\,\mathrm{0ee}\oplus\mathrm{0oo}
   \oplus\mathrm{1me}\oplus\mathrm{1mo}\oplus\mathrm{2me}.

Here ``m`` denotes a positive-order two-dimensional local irrep. Non-SOC
magnetic edge attributes are scalars and are unchanged by the frame rotation.

The local update is

.. math::

   x_{ij}^{(t)}
   \xrightarrow{\;w_{ij}^{(t)}\;}
   \operatorname{O(2)Linear}_{\mathrm{up}}
   \longrightarrow\operatorname{O(2)Gate}
   \longrightarrow\operatorname{O(2)Linear}_{\mathrm{down}}.

Spatial radial features generate the path-wise convolution weights
:math:`w_{ij}^{(t)}`; magnetic magnitudes condition
:math:`\widetilde{\mathcal M}_{ij}^{(t)}` separately.
``atomic_basis.use_radial_rotary_attention: true`` adds optional Radial
Rotary Complex Attention (RRA). Channel-wise ``uu`` linear maps and edge
cluster expansions are further design possibilities, not additional
``o2_mag`` configuration switches.

Finally, rotate the message back, scatter it to the target node, and apply the
TACE product basis for the many-body expansion. The energy readout retains
only the invariant ``0ee`` scalar.

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

where :math:`\mathcal D_Z` contains training moments of element :math:`Z`.
For multiple fidelities, the model uses the largest resulting cutoff for
each element.

The cutoff acts only on the radial magnitude map:

.. math::

   u_i=\frac{\|\mathbf m_i\|}{m_{\mathrm{cut},Z_i}},\qquad
   \widetilde m_i=1-2\min(u_i^2,1),\qquad
   T(\widetilde m_i)
   =\left(T_1(\widetilde m_i),\ldots,T_{N_{\mathrm{mag}}}(\widetilde m_i)\right).

:math:`T_n` is the first-kind Chebyshev polynomial.
``radial_basis.num_mag_radial_basis`` is :math:`N_{\mathrm{mag}}`, excluding
the constant :math:`T_0`. The radial map is fixed to ``clamp`` and the angular
normalization to ``integral``; neither is a model configuration option.

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

This element-dependent term is added after the interaction-energy scale and
shift, so it receives neither transformation. Its zero-moment value is
learned, not forced to zero. Additional mMACE-style constraints on the
physical interpretation of the readout are not imposed here.

Beyond the cutoff, the one-body term is constant; it does not confine large
moments. The total model therefore has no guaranteed lower bound at
arbitrarily large moment. For magnetic relaxation, validate the model over
the moment range to be explored; see :doc:`magnetic_relax`.

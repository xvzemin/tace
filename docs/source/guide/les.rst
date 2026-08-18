TACE-LES
========

.. note::

   The complete TACE-LES functionality is adapted from
   `MACE <https://github.com/ACEsuit/mace>`_.

.. note::

   TACE-LES is currently being refactored. For now, please avoid using LES 
   in versions ``>= v0.2.1``. We recommend using ``v0.2.0`` instead.

TACE-LES augments the short-range TACE energy with long-range interactions
computed by `Latent Ewald Summation
<https://github.com/ChengUCB/les>`_ (LES).  TACE predicts environment-dependent
latent sources from its equivariant node descriptors and the external LES
backend evaluates their global interaction:

.. math::

   E(\mathcal X)
   = E_{\mathrm{TACE}}(\mathcal X)
   + E_{\mathrm{LES}}\!\left(
       \{q_i,\boldsymbol{u}_i,\boldsymbol{Q}_i,\kappa_i,
       \boldsymbol{\alpha}_i,\boldsymbol{r}_i\}
     \right).

The latent sources are learned from the configured energy-derived objectives;
atomic charge, dipole, or polarizability labels are not required by the basic
LES energy/force workflow.  Because ``les_energy`` is added to the total
energy before differentiation, conservative forces and stress include both
the short- and long-range contributions.

Supported latent sources
------------------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Source
     - TACE representation
     - Role in LES
   * - ``latent_charges`` :math:`q_i`
     - ``0e``
     - Permanent scalar source used by every TACE-LES model.
   * - ``latent_dipoles`` :math:`\boldsymbol{u}_i`
     - ``1o``
     - Permanent latent atomic dipole.
   * - ``latent_quadrupoles`` :math:`\boldsymbol{Q}_i`
     - ``2e`` and vector dyadics
     - Symmetric traceless atomic quadrupole.
   * - ``latent_kappas`` :math:`\kappa_i`
     - ``0e``
     - Local charge-response coefficient used to generate induced charges.
   * - ``latent_polarizabilities`` :math:`\boldsymbol{\alpha}_i`
     - ``0e``, ``2e``, and vector dyadics
     - Isotropic scalar or anisotropic symmetric polarizability used to
       generate induced dipoles.

TACE automatically adds the irreps required by the enabled LES sources to the
last descriptor layer.  Charge, induced charge, and isotropic polarizability
require ``0e``; permanent dipoles require ``1o``; quadrupoles request ``1o``
and ``2e``.  An anisotropic polarizability requests the entries listed in
``alpha_irreps``.  Selecting ``1e`` requires a full-O(3) TACE model with
``parity: true``.

Tensor readouts
---------------

The direct ``2e`` path is converted to a Cartesian symmetric tensor.  The
vector path constructs a Cartesian tensor from equivariant vector channels.
For the linear LES tensor readout,

.. math::

   \boldsymbol{A}_i
   = \sum_c w_c(s_i)\,\boldsymbol{v}_{ic}
     \boldsymbol{v}_{ic}^{\mathsf T},

where :math:`s_i` denotes ``0e`` features and
:math:`\boldsymbol{v}_{ic}` denotes
``1o`` or ``1e`` channels.  ``alpha_1o_linear_w_pos: true`` squares the
``1o`` channel weights before forming the dyadics.

The nonlinear variant uses a scalar-conditioned, symmetric channel-mixing
matrix,

.. math::

   \boldsymbol{A}_i
   = \sum_{c,d} M_{cd}(s_i)\,
     \boldsymbol{v}_{ic}\boldsymbol{v}_{id}^{\mathsf T}.

It includes cross-channel dyadics and is therefore more expressive, but it is
more expensive and currently uses ``1o`` channels only.  Quadrupoles are made
traceless after all enabled ``2e`` and dyadic paths are summed.

Installation
------------

LES remains an optional external dependency.  Install the current upstream
implementation in the environment used to run TACE:

.. code-block:: bash

  pip install git+https://github.com/ChengUCB/les.git@v0.2.0

TACE imports LES only when ``long_range.les.enable`` is true, so ordinary TACE
models do not require this package.

Minimal latent-charge model
---------------------------

The smallest TACE-LES model predicts only one scalar latent charge per atom:

.. code-block:: yaml

   model:
     config:
       long_range:
         les:
           enable: true
           les_arguments:
             sigma: 1.0
             dl: 2.0
             remove_self_interaction: true
             remove_mean: true
             use_atomwise: false
             output_scale: 0.1

``use_atomwise`` should remain false.  TACE always supplies
``latent_charges`` from its own equivariant readout; consequently the external
LES ``Atomwise`` options ``n_layers``, ``n_hidden``, ``add_linear_nn``,
``output_scaling_factor``, and ``use_atomwise`` are not part of the normal
TACE-LES data path.  ``output_scale`` is the TACE-side scale applied to the
learned charge, dipole, and quadrupole sources.

The training ``loss_property`` must explicitly include ``energy``.  TACE
rejects an LES configuration whose model target list has no energy; listing
only ``forces`` is not sufficient even though the forces are obtained by
differentiating the total energy.

Complete multipole configuration
--------------------------------

All extended source terms are opt-in.  The following block exposes the full
configuration with conservative positivity constraints for the induced
response coefficients:

.. code-block:: yaml

   model:
     config:
       parity: false

       long_range:
         les:
           enable: true
           les_arguments:
             # External Ewald backend
             sigma: 1.0
             dl: 2.0
             remove_self_interaction: true
             remove_mean: true
             epsilon_factor: 1.0
             use_fixed_atomic_charges: false
             fixed_atomic_charges_scaling_factor: 0.5
             use_atomic_alpha: false
             use_epsilon_r_scaling: false
             use_atomwise: false

             # Permanent multipoles
             use_dipole: true
             use_quad: true

             # Induced response
             use_induced_charge: true
             use_induced_dipole: true
             use_anisotropic_polarizability: true
             alpha_irreps: 0e+1o+2e

             # Tensor readout
             alpha_1o_nonlinear_readout: false
             alpha_1o_linear_w_pos: true
             make_alpha_positive: true
             make_kappa_positive: true

             # TACE readout scales
             output_scale: 0.1
             kappa_scale: 0.01
             alpha_scale: 0.01

             # Electrical response output
             compute_bec: false
             bec_output_index: null

``make_kappa_positive`` squares the scaled :math:`\kappa_i` values.
``make_alpha_positive`` squares a scalar polarizability, or maps an
anisotropic matrix :math:`A_i` to :math:`A_iA_i^{\mathsf T}`.  These options
are useful when the induced-response solver requires non-negative scalar
coefficients or positive-semidefinite polarizabilities.

The upstream options ``use_fixed_atomic_charges`` and ``use_atomic_alpha`` add
the corresponding element-dependent baselines inside LES.  TACE passes atomic
numbers to the backend, so these options can be enabled without changing the
TACE descriptor.

Born effective charges and external fields
------------------------------------------

Set ``compute_bec: true`` to ask LES for its Born effective charge response.
TACE enables coordinate autograd before constructing the representation, so
the result includes both explicit position dependence and the environment
dependence of the latent sources.  ``bec_output_index`` can be ``0``, ``1``,
or ``2`` to request only one polarization component.

When ``electric_field`` is present, TACE passes it to LES.  A batch may contain
different fields for different structures: the long-range response is
evaluated per graph and the BEC is then assembled on the complete coordinate
graph.  Without an electric-field input, LES evaluates the zero-field model.

Outputs
-------

An enabled model adds the following entries to the normal TACE output
dictionary:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Key
     - Description
   * - ``les_energy``
     - Per-graph long-range energy added to ``energy``.
   * - ``les_latent_charges``
     - Backend charge after optional fixed and induced charge contributions.
   * - ``les_latent_dipoles``
     - Permanent plus optional induced dipole.
   * - ``les_latent_quadrupoles``
     - Symmetric traceless quadrupole supplied to LES.
   * - ``les_latent_polarizabilities``
     - Scalar or Cartesian polarizability supplied to LES.
   * - ``les_latent_kappas``
     - TACE-predicted induced-charge coefficient.
   * - ``les_born_effective_charges``
     - LES BEC output when ``compute_bec`` is enabled; otherwise ``None``.

``energy`` contains ``les_energy``.  ``node_energy`` intentionally remains the
short-range atomic decomposition because a global Ewald energy does not have
a unique local atomic partition in this interface.

Multiple fidelities and layers
------------------------------

Every LES readout produces one head per TACE fidelity and selects the head
specified by ``fidelity_idx`` for each atom.  With
``readout_emlp.use_alllayer: true``, each descriptor layer has an LES readout
and the corresponding source contributions are summed before calling the
global backend.  With the default ``false``, only the final descriptor is
used.

Compatibility and limitations
------------------------------

* ``TACE_USE_COMPILE`` and current AOTI export paths do not support LES because
  the external global solver is not part of the flattened TACE graph.
* LES parameters should be selected together with the physical boundary
  conditions and the upstream LES recommendations, especially ``sigma``,
  ``dl``, and ``remove_self_interaction``.

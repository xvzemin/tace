TACE-LES
========

Using LES in TACE
-----------------

TACE-LES augments the short-range TACE energy with long-range interactions
computed by `Latent Ewald Summation <https://github.com/ChengUCB/les>`_ :

.. math::

   E(\mathcal X)
   =E_{\mathrm{TACE}}^{\mathrm{sr}}(\mathcal X)
   +E_{\mathrm{LES}}^{\mathrm{lr}}\!\left(
       \{q_i,\boldsymbol u_i,\boldsymbol Q_i,\kappa_i,
       \boldsymbol\alpha_i,\boldsymbol r_i\}
     \right).


Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install git+https://github.com/ChengUCB/les.git@v0.2.0

Supported latent sources
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 20 50

   * - TACE output
     - Representation
     - LES role
   * - ``les_latent_charges``
     - ``0e``
     - Permanent charge plus an optional induced-charge contribution.
   * - ``les_latent_dipoles``
     - ``1o``
     - Permanent dipole plus an optional induced-dipole contribution.
   * - ``les_latent_quadrupoles``
     - ``2e``
     - Permanent symmetric traceless Cartesian quadrupole.
   * - ``les_latent_kappas``
     - ``0e``
     - Local induced-charge response coefficient.
   * - ``les_latent_polarizabilities``
     - ``0e`` or ``0e+2e``
     - Per-atom scalar or symmetric Cartesian polarizability used to produce
       the induced dipole.


Complete configuration
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: yaml

   loss:
     loss_property: [energy, forces]

   model:
     config:
       # Natural parity already contains the 1o and 2e needed by LES.
       parity: false

       long_range:
         les:
           enable: true
           les_arguments:
             # Scale applied by TACE to permanent q, u, and Q readouts.
             output_scaling_factor: 0.1

             # Ewald backend.
             sigma: 1.0
             dl: 2.0
             remove_self_interaction: true
             is_periodic: true # change if needed
             N_max: 10
             use_epsilon_r_scaling: false

             # BEC construction.
             remove_mean: true
             epsilon_factor: 1.0

             # Optional element-dependent upstream baselines.
             use_fixed_atomic_charges: false
             fixed_atomic_charges_scaling_factor: 0.5
             use_atomic_alpha: false

             # Permanent latent multipoles predicted by TACE.
             use_dipole: false
             use_quad: false

             # Non-self-consistent induced response predicted by TACE.
             use_induced_charge: false
             use_induced_dipole: false
             use_anisotropic_polarizability: false

             # Constraints and scales applied by the TACE readouts.
             make_kappa_positive: false
             make_alpha_positive: false
             kappa_scale: 0.01
             alpha_scale: 0.01

Born effective charges
~~~~~~~~~~~~~~~~~~~~~~

BEC is an evaluation-time derived response and is not normally used as a
training target. Load a normal eager model, enable BEC on the registered
upstream LES instance, and then run the usual TACE forward pass:

.. code-block:: python

   from tace.lightning import load_tace

   model = load_tace("model.pt", device="cuda", dtype="float64")
   model.eval()

   upstream_les = model.readout_fn.les.les
   upstream_les.compute_bec = True
   upstream_les.bec_output_index = None

   output = model(batch)
   bec = output["les_born_effective_charges"]


Outputs
~~~~~~~

An enabled TACE-LES model adds these entries to the normal output dictionary:

.. list-table::
   :header-rows: 1
   :widths: 38 62

   * - Key
     - Description
   * - ``les_energy``
     - Per-graph long-range energy. This is included in the returned total
       ``energy``.
   * - ``les_latent_charges``
     - Per-atom permanent plus optional induced latent charge.
   * - ``les_latent_dipoles``
     - Per-atom permanent plus optional induced latent dipole.
   * - ``les_latent_quadrupoles``
     - Per-atom Cartesian quadrupole supplied to LES.
   * - ``les_latent_polarizabilities``
     - Per-atom scalar or Cartesian polarizability supplied to LES.
   * - ``les_latent_kappas``
     - Per-atom induced-charge coefficient supplied to LES.
   * - ``les_born_effective_charges``
     - BEC when runtime ``compute_bec`` is enabled; otherwise ``None``.

``node_energy`` remains only the short-range atomic contribution.
A global Ewald energy has no unique atomic partition in this interface.


Compatibility in TACE
---------------------

* ``TACE_USE_COMPILE`` and current AOTI export paths do not support LES because
  the external global solver is not part of the flattened TACE graph.
* Select ``sigma``, ``dl``, ``N_max``, boundary conditions, and self-interaction
  treatment consistently with the upstream LES recommendations and the
  intended physical system.
* ``is_periodic=true`` or ``false`` fixes one boundary condition for the entire
  vectorized LES instance. Use ``null`` for batches that mix periodic and
  non-periodic structures.


How LES works
-------------

The theory overview below is reproduced from the upstream
`How LES works <https://les.readthedocs.io/en/latest/theory.html>`_
guide.

The problem
~~~~~~~~~~~

A short-range MLIP writes the energy as a sum of local atomic contributions
inside a cutoff,

.. math::

   E^{\mathrm{sr}}=\sum_{i=1}^{N}E_i,

with :math:`r_c` typically around 5 to 6 Angstrom. Electrostatics does not fit
in that form: a Coulomb interaction decays as :math:`1/r`.

The idea
~~~~~~~~

LES splits the energy in two,

.. math::

   E=E^{\mathrm{sr}}+E^{\mathrm{lr}},

and builds :math:`E^{\mathrm{lr}}` from a **latent atomic charge**
:math:`q_i^{\mathrm{les}}` or other latent quantities, predicted by a small
network from the same local invariant features the model already uses for
:math:`E_i`.

Two design principles make this work, and they are the whole method:

#. **Use a Coulomb functional form with environment-dependent charges.** The
   physics is in the functional form, so the asymptotics are right by
   construction and the charge stays interpretable.
#. **Do not train on DFT partial charges.** There is no unique mapping from an
   electron density to atomic charges, so fitting one choice fixes an
   arbitrary electrostatic description and is not necessary.

Nothing supervises :math:`q_i^{\mathrm{les}}`. The only training signal is the
total energy and the forces, and the charges are whatever makes those come
out right. This is why they are called *latent*.

They are not arbitrary, either. Because :math:`E^{\mathrm{lr}}` depends on
them only through a physical functional form--the Ewald sum of a smeared
charge distribution--fitting energies and forces constrains them to behave
like real charges. In practice they can recover physical response quantities
that never appeared as training labels, including molecular dipoles, Born
effective charges, IR spectra, and ionic conductivities.

Coulomb interactions between atomic charges are screened by the fast
electronic background, described as a homogeneous dielectric medium with
relative permittivity :math:`\varepsilon_e`. Learning from forces absorbs that
screening automatically, so the learned quantities are *scaled* physical
ones,

.. math::

   q_i^{\mathrm{les}}=\frac{q_i}{\sqrt{\varepsilon_e}},\qquad
   \boldsymbol u_i^{\mathrm{les}}
   =\frac{\boldsymbol u_i}{\sqrt{\varepsilon_e}},\qquad
   \boldsymbol Q_i^{\mathrm{les}}
   =\frac{\boldsymbol Q_i}{\sqrt{\varepsilon_e}}.

Only the vacuum permittivity enters the Ewald sum. Consequently,
:math:`\varepsilon_e` does not have to be considered while training or
predicting energies and forces. It appears only when recovering unscaled
physical quantities such as BECs and polarizabilities:

* :math:`\varepsilon_e=1` for a system in vacuum, such as an isolated
  molecule;
* :math:`\varepsilon_e=\varepsilon_\infty` for a homogeneous bulk system with
  no induced-dipole term;
* :math:`\varepsilon_e=\varepsilon_\infty/(1+\chi^{\mathrm{les}})` when induced
  dipoles are included, since they contribute their own susceptibility
  :math:`\chi^{\mathrm{les}}=\sum_i\alpha_i^{\mathrm{les}}/(\varepsilon_0V)`.

The Ewald sum
~~~~~~~~~~~~~

Each latent charge is smeared into a Gaussian of width :math:`\sigma`, usually
about 1 Angstrom. That single choice makes both calculation paths finite.

**Isolated systems** are summed pairwise in real space, where the smeared
charges replace the bare :math:`1/r` with an error function:

.. math::

   E^{\mathrm{lr}}
   =\frac{1}{2}\frac{1}{4\pi\varepsilon_0}
    \sum_{i=1}^{N}\sum_{j=1}^{N}
    \left[1-\varphi(r_{ij})\right]
    \frac{q_i^{\mathrm{les}}q_j^{\mathrm{les}}}{r_{ij}},
   \qquad
   \varphi(r)=\operatorname{erfc}\!\left(
      \frac{r}{\sqrt{2}\sigma}
   \right).

At short range the kernel tends to a constant rather than diverging: Gaussian
overlap removes the singularity. The long-range term is therefore smooth and
weak exactly where the short-range model is already accurate, so the two do
not compete for the same physics.

**Periodic systems** are summed in reciprocal space:

.. math::

   E^{\mathrm{lr}}
   =\frac{1}{2\varepsilon_0V}
    \sum_{0<k<k_c}
    \frac{e^{-\sigma^2k^2/2}}{k^2}
    \left|S(\boldsymbol k)\right|^2,
   \qquad
   S(\boldsymbol k)
   =\sum_{i=1}^{N}q_i^{\mathrm{les}}
    e^{i\boldsymbol k\cdot\boldsymbol r_i}.

The Gaussian factor :math:`e^{-\sigma^2k^2/2}` suppresses large
:math:`k`. A finite number of terms can therefore reach any desired numerical
tolerance. ``dl`` sets :math:`k_c=2\pi/\mathtt{dl}`; the default
``dl: 2.0`` Angstrom corresponds to :math:`k_c=\pi` inverse Angstrom.

Charge neutrality is not imposed. For a neutral system the learned charges
usually sum close to zero, and a residual is absorbed as a uniform background
under the tinfoil boundary condition implicit in the reciprocal-space sum.

``remove_self_interaction`` subtracts each charge's interaction with its own
Gaussian, which is an artefact of the smearing rather than physical
interaction energy.

Multipoles and polarization response
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A single scalar per atom is the leading term of a multipole expansion of the
atomic charge density. The extension adds the next orders and a linear
response, all still learned from energies and forces alone.

**Fixed multipoles.** Latent dipoles :math:`\boldsymbol u_i^{\mathrm{les}}`
(an equivariant ``1o`` vector) and traceless quadrupoles
:math:`\boldsymbol Q_i^{\mathrm{les}}` (``2e``) enter the same Ewald machinery
through the structure factor:

.. math::

   S(\boldsymbol k)
   =\sum_{i=1}^{N}\left(
       q_i^{\mathrm{les}}
       +i\boldsymbol k\cdot\boldsymbol u_i^{\mathrm{les}}
       -\frac{1}{2}\boldsymbol k\cdot
        \boldsymbol Q_i^{\mathrm{les}}\cdot\boldsymbol k
     \right)e^{i\boldsymbol k\cdot\boldsymbol r_i}.

Each successive order decays with one additional factor of :math:`1/r`, so
the expansion is truncated at the dipole or quadrupole level.

**Induced response.** Instead of a global charge-equilibration solve, the
residual non-local effects are captured by *non-self-consistent* linear
response. The induced terms respond once to the field of the fixed
multipoles, not to each other. With inverse hardness :math:`\kappa_i` and
polarizability :math:`\boldsymbol\alpha_i`,

.. math::

   \Delta q_i=-\kappa_i\Phi(\boldsymbol r_i),\qquad
   U_i^{\mathrm{iq}}
   =-\frac{1}{2}\kappa_i\Phi^2(\boldsymbol r_i),

.. math::

   \Delta\boldsymbol u_i
   =\boldsymbol\alpha_i\cdot\boldsymbol E(\boldsymbol r_i),\qquad
   U_i^{\mathrm{iu}}
   =-\frac{1}{2}\boldsymbol E(\boldsymbol r_i)\cdot
     \boldsymbol\alpha_i\cdot\boldsymbol E(\boldsymbol r_i).

The total energy is assembled as

.. math::

   U=U^{\mathrm{sr}}+U^{\mathrm{elec}}
     +\sum_iU_i^{\mathrm{iq}}+\sum_iU_i^{\mathrm{iu}}.

Born effective charges: theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The polarization of a configuration follows from the latent variables,

.. math::

   \boldsymbol P
   =\sum_i(q_i+\Delta q_i)\boldsymbol r_i
    +\sum_i(\boldsymbol u_i+\Delta\boldsymbol u_i),

and the Born effective charge tensor is its derivative with respect to an
atomic position:

.. math::

   Z^*_{i\alpha\beta}
   =\frac{\partial P_\alpha}{\partial r_{i\beta}}.

Since the latent variables are differentiable functions of every position,
autograd gives this directly without finite differences or an additional
training target. For a homogeneous periodic system the charge part is taken
in the :math:`k\rightarrow0` limit,

.. math::

   Z^*_{i\alpha\beta}
   =\frac{\partial P^u_\alpha}{\partial r_{i\beta}}
    +\lim_{k\rightarrow0}\operatorname{Re}\left[
       e^{-ikr_{i\alpha}}
       \frac{\partial P^q_\alpha(k)}{\partial r_{i\beta}}
     \right],
   \qquad
   P^q_\alpha(k)
   =\sum_i\frac{\sqrt{\varepsilon_\infty}
                    q_i^{\mathrm{les}}}{ik}
             e^{ikr_{i\alpha}}.

Here :math:`\varepsilon_\infty` is set by ``epsilon_factor``. It is
:math:`\varepsilon_\infty`, not :math:`\varepsilon_e`: without induced
dipoles the two coincide; with induced dipoles,
:math:`\varepsilon_e` follows from the susceptibility relation above.

.. warning::

   The periodic expression requires one high-frequency permittivity
   :math:`\varepsilon_\infty` for a homogeneous bulk material. For a
   heterogeneous interface containing materials with different
   :math:`\varepsilon_\infty`, choosing one value is not well defined. Extending
   LES-based BEC extraction to such systems remains an open problem.

Cost and limitations
~~~~~~~~~~~~~~~~~~~~

At typical upstream settings the long-range reciprocal sum is designed to be
a small fraction of the short-range message-passing cost.

Two limitations are important:

* The charges come from **local** features, so there is no explicit mechanism
  for genuinely long-range charge transfer through mobile carriers, such as
  induced surface charge on a macroscopic metal electrode. Coupling to an
  explicit metallic-boundary model is the current remedy.
* Forces are autograd derivatives of a **global** energy. Distributing force
  evaluation across GPUs or MPI ranks is therefore not straightforward.


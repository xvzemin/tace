Magnetic TACE
=============

Magnetic TACE provides three ways to incorporate non-collinear magnetic
moments.  All three methods consider spin--orbit coupling (SOC): the magnetic
moments and atomic geometry are coupled in a common full-:math:`O(3)` model.

The three methods principally differ in where the magnetic information enters
the model.  The first method is currently recommended for general use.  The
Wigner-:math:`6j` and local-:math:`O(2)` methods are experimental research
interfaces; please wait for the corresponding papers before using them as the
default choice.

Common configuration
--------------------

The examples below are minimal overrides for the standard ``tace.yaml``.  A
non-collinear magnetic moment is represented as a full-:math:`O(3)` ``1e``
vector, so ``parity`` must be enabled.  Map the dataset array containing the
initial moments with ``initial_noncollinear_magmoms_key``:

.. code-block:: yaml

   dataset:
     keys:
       initial_noncollinear_magmoms_key: initial_noncollinear_magmoms
       # Only needed when magnetic-force labels are used:
       noncollinear_magnetic_forces_key: noncollinear_magnetic_forces

   model:
     config:
       parity: true
       mag_Lmax: 1

If non-collinear magnetic forces are trained, include
``noncollinear_magnetic_forces`` in ``loss.loss_property`` together with its
loss function and weight, following the normal TACE loss configuration.

Universal equivariant embedding (recommended)
----------------------------------------------

The universal equivariant embedding projects each non-collinear magnetic
moment with an element-dependent equivariant linear map and adds it directly
to the atomic basis.  The remaining interaction and product-basis workflow is
unchanged.  This is the simplest and most mature magnetic TACE configuration.
The construction is described in the
`Cartesian TACE paper <https://arxiv.org/abs/2509.14961>`_.

.. code-block:: yaml

   model:
     config:
       parity: true

       atomic_basis:
         type: cgtp

       universal_embedding:
         initial_noncollinear_magmoms:
           enable: true
           normalizer: 1.0

``normalizer`` controls the fixed scale applied to the input magnetic moments.
This method is the current recommendation.

Whenever noncollinear magnetic moments are model inputs, TACE automatically
computes the element-wise ``max(|m|)`` from the training set and stores it as
``magmoms_norm_by_element`` in ``statistics_*.yaml``. Inside the model, the
magnetic-moment normalizer uses ``1.2 * magmoms_norm_by_element + 0.1`` as the
effective scale like mMACE.

Wigner-6j recoupling (experimental)
-----------------------------------

The Wigner-:math:`6j` interaction changes the coupling tree so that magnetic
information modulates the edge-level interaction at low cost.  
Select it through ``atomic_basis.type`` and leave the
universal magnetic embedding disabled when testing this method in isolation:

.. code-block:: yaml

   model:
     config:
       parity: true
       mag_Lmax: 1

       radial_basis:
         num_mag_radial_basis: 8

       atomic_basis:
         type: o3_w6j_mag

       universal_embedding:
         initial_noncollinear_magmoms:
           enable: false

This path is experimental; its detailed
design and recommended hyperparameters will accompany the formal paper.

``mag_Lmax`` selects ``0e + 1e + ... + mag_Lmax e`` solid harmonics of the
axial magnetic-moment input. Its default value is ``1`` and it must satisfy
``1 <= mag_Lmax <= Lmax``.

Local O(2) frame (experimental)
-------------------------------

The local-:math:`O(2)` interaction evaluates each edge in an edge-aligned local
frame, uses the complete local :math:`O(2)` representation to modulate the
magnetic interaction, and then returns the result to the global
:math:`O(3)` frame.  The angular cutoffs must currently satisfy
``mmax = Lmax = lmax``:

.. code-block:: yaml

   model:
     config:
       parity: true
       mmax: 3
       Lmax: 3
       lmax: 3
       mag_Lmax: 1

       radial_basis:
         num_mag_radial_basis: 8

       atomic_basis:
         type: o2_mag
         scalar_act: silu

       universal_embedding:
         initial_noncollinear_magmoms:
           enable: false

This path is also experimental; its theoretical and implementation
details will be documented with the formal paper.
For ``o2_mag``, the existing ``mmax = Lmax = lmax`` requirement ensures that
every restricted ``0e``, ``0o``, and positive-order ``m`` block through
``mag_Lmax`` is retained in the local frame.

Choosing a method
-----------------

Use the universal equivariant embedding for current production experiments.
It has the simplest configuration and is already described in the Cartesian TACE
work.  The Wigner-:math:`6j` and local-:math:`O(2)` paths expose the two newer
edge-level magnetic interactions for research and comparison, but they should
not yet be treated as the recommended defaults.

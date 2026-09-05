Magnetic TACE (Experimental)
============================

Magnetic TACE provides two ways to incorporate non-collinear magnetic
moments. Both methods consider spin--orbit coupling (SOC): the magnetic
moments and atomic geometry are coupled in a common full-:math:`O(3)` model.

The two methods principally differ in where the magnetic information enters
the model. The first method is currently recommended for general use. The
local-:math:`O(2)` method is an experimental research interface; please wait
for the corresponding paper before using it as the default choice.

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

Universal equivariant embedding
-------------------------------

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

Local O(2) frame (experimental)
-------------------------------

The local-:math:`O(2)` interaction converts global :math:`O(3)` irreps to local
:math:`O(2)` irreps, uses the local representation to modulate the magnetic
interaction, and then converts the result back to global :math:`O(3)` irreps.
Time parity is retained during this restriction. In particular,
``1eo -> 0oo + 1mo`` for a non-collinear magnetic moment. Local linear maps
preserve time parity, while gates multiply the time parities of their inputs.
Consequently, pairs of time-odd magnetic features contribute to time-even
energy features. Biases and radial weights remain ``0ee``.

.. code-block:: yaml

   model:
     config:
       parity: true
       mmax: 3
       Lmax: 3
       lmax: 3
       mag_Lmax: 1

       radial_basis:
         num_mag_radial_basis: 10

       atomic_basis:
         type: o2_mag
         use_radial_rotary_attention: true

       universal_embedding:
         initial_noncollinear_magmoms:
           enable: false

The O(2) path applies a local linear, gate, and output linear. Radial rotary
attention uses ``atomic_basis.num_head``, separate O(2) query and key
projections, and a zero-initialized real radial scale-and-shift projection.
The scale uses a sigmoid and the shift is additive.

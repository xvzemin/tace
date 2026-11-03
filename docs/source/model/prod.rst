Product
=======

The following modules implement many-body expansions using different strategies.
We recommend using ``CgtpACE`` in practice. For other ACE variants, interested
users are encouraged to refer to the source code for further exploration.

.. autoclass:: tace.models.spherical.prod.CgtpACE
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models.spherical.prod.MaceACE
   :no-members:
   :show-inheritance:

Select MACE symmetric contractions with the existing product configuration:

.. code-block:: yaml

   product_basis:
     type: mace
     correlation: 3
     l1l2: null

The block applies ``Linear up → symmetric contraction → Linear`` and then the
configured residual connection. ``Linear up`` is omitted when the input and
contraction channel counts match. Select fused contractions with ``TACE_USE_EQX=1``
or ``TACE_USE_CUE=1``. EQX takes priority when both are enabled. With neither
enabled, or on CPU, the contraction uses PyTorch. The weight layout is unchanged.

Explicit selection of MACE symmetric contraction paths is not currently
supported. Set ``product_basis.l1l2: null``; any other value raises
``NotImplementedError``. Standard MACE paths and their element-dependent
coefficients are retained. CGTP path controls are unchanged.

Nonlinear products
------------------

Set ``product_basis.type`` to ``nonlinear_cgtp``.
For ``tace.models.CartesianTACE``, use ``nonlinear_ictd``. The Cartesian block uses
ICTD products with the same paths and trainable parameters; ``convert_basis``
converts between the two representations.
It always applies Linear up to produce contraction inputs and invariant scalars.
The scalars and a learned element embedding are concatenated, passed through an
ordinary Linear and scaled SiLU, and used as per-path, per-channel coefficients.
Coefficient and output Linear maps are element-independent.
Linear up groups equivalent irreps into channel blocks before separating the
contraction factors. Recursive products retain every path; their adjacent
equal-irrep blocks are merged before the coefficient Linear, as in ``CgtpACE``.

The class attribute ``asymmetric=True`` selects independent inputs; set it to
``False`` before constructing a model to reuse one input at every order.
Both modes retain only paths that are nonzero for symmetric products. There
are no additional configuration options. Expert mixtures and
``product_basis.nonlinear`` are not used; leave the latter at ``null``.

.. autoclass:: tace.models.spherical.prod.NonlinearCgtpACE
   :no-members:

.. autoclass:: tace.models.cartesian.prod.NonlinearIctdACE
   :no-members:

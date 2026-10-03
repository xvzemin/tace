.. _equivariantx-api-models:

Model integration
=================

Install the consuming model package separately. EQX provides ``mace``,
``nequip``, and ``sevennet`` extras. Follow each package's dependency constraints
when choosing an environment.

Converters accept instantiated models and preserve their forward interfaces.
``implementation="o3"`` is the default; ``"o2"`` selects the equivalent
harmonic convolution, not a different model symmetry. Load weights and set
the device and dtype before conversion, then construct the optimizer.
Converted state dictionaries require the same converted architecture.

Converters
----------

.. autofunction:: eqx.models.mace.convert_mace_to_eqx

.. autofunction:: eqx.models.nequip.convert_nequip_to_eqx

.. autofunction:: eqx.models.sevennet.convert_sevennet_to_eqx

.. autofunction:: eqx.models.prophet.convert_prophet_to_eqx

.. autofunction:: eqx.models.equflash.convert_equflash_to_eqx

Allegro's ``uuu`` environment products are not supported by the current
``uvu`` convolution adapters. Distributed ghost-atom exchange is not covered.

TACE operators
--------------

.. autoclass:: eqx.models.tace.tece_oam_rra.BilinearACE
   :members: forward

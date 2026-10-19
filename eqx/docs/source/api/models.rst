.. _equivariantx-api-models:

Model integration
=================

Converters preserve forward interfaces. Install the model package separately,
load its weights and set dtype and device before conversion. Construct the
optimizer afterwards. Converted state dictionaries require the converted architecture.

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

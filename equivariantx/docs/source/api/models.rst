.. _equivariantx-api-models:

Model adapters
==============

Install the model package separately. Load weights before conversion and
construct the optimizer afterwards. Each converter documents its supported
architectures and calculator interface.

.. autofunction:: equivariantx.models.mace.convert_mace_to_eqx

.. autofunction:: equivariantx.models.nequip.convert_nequip_to_eqx

.. autofunction:: equivariantx.models.sevennet.convert_sevennet_to_eqx

.. autofunction:: equivariantx.models.prophet.convert_prophet_to_eqx

.. autofunction:: equivariantx.models.equflash.convert_equflash_to_eqx

Allegro's ``uuu`` environment products are not supported by the current
``uvu`` convolution adapters. Distributed ghost-atom exchange is not covered.

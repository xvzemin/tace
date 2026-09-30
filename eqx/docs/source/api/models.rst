.. _equivariantx-api-models:

Model integration
=================

Converters operate on instantiated models. See :doc:`../models` for usage,
optional dependencies, and supported interactions.

.. autofunction:: eqx.models.mace.convert_mace_to_eqx

.. autofunction:: eqx.models.nequip.convert_nequip_to_eqx

.. autofunction:: eqx.models.sevennet.convert_sevennet_to_eqx

.. autofunction:: eqx.models.prophet.convert_prophet_to_eqx

.. autofunction:: eqx.models.equflash.convert_equflash_to_eqx

TACE operators
--------------

.. autoclass:: eqx.models.tace.tece_oam_rra.BilinearACE
   :members: forward

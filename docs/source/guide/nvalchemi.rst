NValCHEMI Wrapper
=================

NValCHEMI provides common data, neighbor-list, model, and execution interfaces
for GPU atomistic simulation workflows. The TACE wrapper adapts an
energy-based TACE model to the NValCHEMI model contract and exposes energy,
conservative forces, and stress.

Install both optional dependencies as described in
:ref:`nvalchemi-installation`, then import the existing wrapper with:

.. code-block:: python

   from tace.interface.nvalchemi import TACEWrapper
   model = TACEWrapper.from_checkpoint(
       "model.pt",
       device="cuda",
   )

Neighbor lists
--------------

NValCHEMI constructs neighbors with ``nvalchemiops``. ``TACEWrapper`` requests
a full COO list at the model cutoff through
``model.model_config.neighbor_config`` and has no ``neighborlist_backend``
argument. For an existing NValCHEMI ``batch``:

>>> from nvalchemi.neighbors import compute_neighbors
>>> compute_neighbors(batch, config=model.model_config.neighbor_config)

See :doc:`neighbor_list` for training and other inference interfaces.

Upstream projects:

- `nvalchemi-toolkit <https://github.com/NVIDIA/nvalchemi-toolkit>`_
- `nvalchemi-toolkit-ops <https://github.com/NVIDIA/nvalchemi-toolkit-ops>`_

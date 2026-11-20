NValCHEMI
=========

Install the optional dependency with ``pip install "tace[nvalchemi]"``.
For a batched NValCHEMI structure on the model device:

.. code-block:: python

    from nvalchemi.neighbors import compute_neighbors
    from tace.interface.nvalchemi import TACEWrapper

    model = TACEWrapper.from_checkpoint("TACE-OAM-L", device="cuda")
    compute_neighbors(batch, config=model.model_config.neighbor_config)
    results = model(batch)

The neighbor configuration uses the model cutoff and a full COO neighbor list.
See :ref:`nvalchemi-installation` and :doc:`neighbor_list`.

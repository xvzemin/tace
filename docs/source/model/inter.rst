Interaction
===========

We currently recommend using ``O3CgtpInteraction``. It supports operator fusion
via OpenEquivariance or CuEquivariance, which can significantly reduce memory
usage and improve computational efficiency.

Although the O(2)/SO(2) interaction variants are theoretically more advantageous 
at large angular momentum, they currently have fewer operator-fusion options and
are therefore not the default recommendation.


O(3) Cgtp
---------

.. autoclass:: tace.models._e3nn.inter.O3CgtpInteraction
   :no-members:
   :show-inheritance:

O(2) CGTP
---------

``atomic_basis.type: o2_cgtp`` evaluates the same CGTP as ``cgtp`` through
an edge-aligned frame. Spherical harmonics then have only an order-zero
entry, so the contraction uses only the nonzero coefficients in that CG
slice. Coupling paths, radial weights, and normalization are unchanged.
All required orders are retained, independently of ``mmax``.

Existing CGTP models can be converted without retraining. Values and
derivatives agree up to floating-point roundoff. The conversion returns a
new model and leaves the original unchanged:

.. code-block:: python

   from tace.lightning import convert_cgtp, export_tace, load_tace

   model = load_tace("TACE-OAM-7M.pt", dtype="float64")
   local_model = convert_cgtp(model)  # cgtp -> o2_cgtp
   global_model = convert_cgtp(local_model)  # o2_cgtp -> cgtp
   export_tace(local_model, "TACE-OAM-7M-o2-cgtp.pt")

This is an equivalent implementation of CGTP, not the different
``o2`` Linear--Gate--Linear architecture. By default it uses PyTorch sparse
reductions. ``TACE_USE_EQX=1`` selects the streamed CUDA implementation for
``o2_cgtp``, without changing its learned parameters or model-loading behavior.
See :ref:`eqx-streaming` for training support and requirements.

.. autoclass:: tace.models._e3nn.inter.O2CgtpInteraction
   :no-members:
   :show-inheritance:

.. autofunction:: tace.lightning.convert_cgtp

O(2) Linear
-----------

.. autoclass:: tace.models._e3nn.inter.O2Interaction

.. autoclass:: tace.models._e3nn.inter.O2MagneticInteraction
   :no-members:
   :show-inheritance:

SO(2) Linear
------------

.. autoclass:: tace.models._e3nn.inter.uvSO2Interaction

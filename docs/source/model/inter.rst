Interaction
===========

We currently recommend using ``CgtpInteraction``. It supports operator fusion
via OpenEquivariance or CuEquivariance, which can significantly reduce memory
usage and improve computational efficiency.

Although the SO(2) interaction variants are theoretically more advantageous at
large angular momentum, they currently have fewer operator-fusion options and
are therefore not the default recommendation.

.. autoclass:: tace.models._e3nn.inter.CgtpInteraction
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.inter.uuSO2Interaction
   :no-members:
   :show-inheritance:

.. autoclass:: tace.models._e3nn.inter.uvSO2Interaction

.. autoclass:: tace.models._e3nn.inter.O2Interaction

.. autoclass:: tace.models._e3nn.inter.O2MagneticInteraction
   :no-members:
   :show-inheritance:

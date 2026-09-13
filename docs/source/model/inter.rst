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

O(2) Linear
-----------

.. autoclass:: tace.models._e3nn.inter.O2Interaction

.. autoclass:: tace.models._e3nn.inter.O2MagneticInteraction
   :no-members:
   :show-inheritance:

SO(2) Linear
------------

.. autoclass:: tace.models._e3nn.inter.uvSO2Interaction

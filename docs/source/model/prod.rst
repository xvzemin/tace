Product
=======

The following modules implement many-body expansions using different strategies.
We recommend using ``CgtpACE`` in practice. For other ACE variants, interested
users are encouraged to refer to the source code for further exploration.

.. autoclass:: tace.models._e3nn.prod.CgtpACE
   :no-members:
   :show-inheritance:

Matrix products
---------------

``product_basis.type: mtp`` replaces the tensor products in ``CgtpACE``
with channel-wise matrix products. The hidden-channel projection, linear
map after each correlation order, sum over orders, output Linear and
residual addition are unchanged. Matrix products combine coupling paths
into one output per irrep and hidden channel.

Coefficients are element-dependent unless ``agnostic: true``. Shared
maps use ``eqx.o3.Linear`` and element-dependent maps use
``eqx.o3.ElementLinear`` when accelerated. ``TACE_USE_EQX=1`` enables
CUDA matrix products with the same weights as the PyTorch reference.
As in ``CgtpACE``, there are no mixture-of-experts or bilinear gates.
Independent CG path selection is not supported by matrix products.

.. autoclass:: tace.models._e3nn.prod.MtpACE
   :no-members:
   :show-inheritance:

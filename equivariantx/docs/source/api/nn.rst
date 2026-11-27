.. _equivariantx-api-nn:

Neural network modules
======================

.. automodule:: equivariantx.nn

.. autoclass:: equivariantx.nn.PolynomialActivation
   :members: forward

.. autoclass:: equivariantx.nn.EquivariantLayerNorm
   :members: forward

.. autoclass:: equivariantx.nn.SoftCapRmsNorm

.. autoclass:: equivariantx.nn.MergeRmsNorm
   :members: forward

.. autoclass:: equivariantx.nn.MergeLayerNorm
   :members: forward

.. autoclass:: equivariantx.nn.GraphSoftmax
   :members: forward

.. autofunction:: equivariantx.nn.graph_softmax

.. autoclass:: equivariantx.nn.GroupedRadialMLP

.. autofunction:: equivariantx.nn.fourier_radial

.. autofunction:: equivariantx.nn.uniform_cubic_spline

Graph attention
---------------

.. autoclass:: equivariantx.nn.StreamingGraphAttention
   :members: forward

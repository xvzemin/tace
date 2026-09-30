.. _equivariantx-spherical-o3:

Spherical O(3)
==============

``eqx.o3`` provides element-dependent linear maps and gated activations.
Representations use ``e3nn.o3.Irreps``. Each entry is stored as
``(..., mul, 2 * l + 1)`` and flattened into the final feature axis
(``mul_ir``).

Element-dependent linear maps
-----------------------------

``ElementLinear`` takes an unbiased ``e3nn.o3.Linear`` to define its paths
and normalization. Weights and integer element indices are passed to
``forward``. The operator does not own the weights.

.. code-block:: python

   import torch
   from eqx import o3
   from e3nn import o3 as spherical

   reference = spherical.Linear(
       "4x0e + 2x1o", "2x0e + 3x1o",
       internal_weights=False, shared_weights=False,
   )
   linear = o3.ElementLinear(reference, backend="torch")
   features = reference.irreps_in.randn(8, -1)
   weights = torch.randn(3, reference.weight_numel)
   node_type = torch.randint(3, (8,))
   output = linear(features, weights, node_type)
   torch.testing.assert_close(output, reference(features, weights[node_type]))

``MoEElementLinear`` applies an independent map to each expert and keeps
their outputs separate. Biases and expert routing are applied by the caller.

Gated activations
-----------------

``Gate`` applies normalized scalar activations and multiplies tensor
channels by scalar gates. Its representation labels determine the allowed
activation parities.

.. code-block:: python

   gate = o3.Gate(
       "2x0e", [torch.nn.functional.silu],
       "3x0e", [torch.sigmoid], "3x1o", backend="torch",
   )
   output = gate(gate.irreps_in.randn(8, -1))

See :ref:`equivariantx-api-o3` for parameters,
:ref:`equivariantx-api-tools` for basis conversions and tensor decomposition,
and :ref:`equivariantx-convolutions` for fused O(3) tensor products.

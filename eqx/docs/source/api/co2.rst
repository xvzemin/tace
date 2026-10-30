.. _equivariantx-api-co2:

Cartesian O(2)
==============

Features use flattened ``mul_ir`` storage with ``2**m`` entries per channel.
Positive-order tensors lie in the symmetric traceless subspace.

Cartesian restriction
---------------------

.. list-table:: Three evaluation routes
   :header-rows: 1

   * - Method
     - Intermediate storage
     - Local output
   * - ``direct`` (default)
     - Shared longitudinal and rectangular contractions
     - ``2**m``
   * - ``rotation``
     - Full rotated ``3**l`` tensor
     - ``2**m``
   * - ``projection``
     - Ambient transverse ``3**m`` tensors
     - ``2**m``

All routes preserve the same normalization, parity conventions, and weights.
``2**m`` denotes full planar Cartesian storage, not two independent coefficients.
Keeping every order permits reconstruction of the input STF tensor.

.. code-block:: python

   frame = co2.LocalFrame("32x0e + 32x1o + 32x2e", method="direct")
   local = frame(features, rotation)
   restored = frame.to_global(local, rotation)

``rotation`` maps the edge direction to the positive y axis.
``Restriction`` exposes the basis-free ``3**m`` tensors directly.

.. autoclass:: eqx.co2.LocalFrame
   :members: forward, to_local, to_global, rotate

.. autoclass:: eqx.co2.Restriction
   :members: forward, inverse

.. autoclass:: eqx.co2.TransverseProjector
   :members: forward

Representations
---------------

.. autoclass:: eqx.co2.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.co2.Irreps
   :members:

Harmonics
---------

.. autoclass:: eqx.co2.CartesianHarmonics
   :members: forward

Linear maps
-----------

.. autoclass:: eqx.co2.Linear
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co2.UuLinear
   :members: forward

Activations
-----------

.. autoclass:: eqx.co2.Activation
   :members: forward

.. autoclass:: eqx.co2.Gate
   :members: forward

Tensor products
---------------

.. autoclass:: eqx.co2.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co2.FullyConnectedTensorProduct

.. autoclass:: eqx.co2.ElementwiseTensorProduct

Projection
----------

``TensorProduct(project=False)`` returns raw Cartesian outputs. Projection may
follow compatible channel maps or sums in the same planar coordinates, but must
precede another operation requiring STF inputs. Difference-order contractions
already produce STF tensors.

``Projector(method="matrix")`` uses the analytic highest-order ICTD projector.
``method="factorized"`` evaluates the same projector through its path matrix.
The complete decomposition of a reducible planar tensor is ``eqx.o2.ICTD``.

.. autoclass:: eqx.co2.Projector
   :members: forward

.. autofunction:: eqx.co2.projector_matrix

.. autoclass:: eqx.co2.PlaneProjector
   :members: forward

.. autoclass:: eqx.co2.PlanarDetracer
   :members: forward, matrix

Harmonic O(3) coupling
----------------------

.. autoclass:: eqx.co2.O3TensorProduct
   :members: forward, forward_scatter, from_tensor_product, weight_views

.. autoclass:: eqx.co2.SphericalCoupling
   :members: forward

.. _equivariantx-api-co3:

Cartesian O(3)
==============

Features use flattened ``mul_ir`` storage with ``3**l`` entries per channel.
Tensor-product inputs lie in the symmetric traceless subspace.

Representations
---------------

.. autoclass:: equivariantx.co3.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: equivariantx.co3.Irreps
   :members:

Harmonics
---------

.. autoclass:: equivariantx.co3.CartesianHarmonics
   :members: forward

Linear maps
-----------

.. autoclass:: equivariantx.co3.Linear
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: equivariantx.co3.ElementLinear
   :members: forward

.. autoclass:: equivariantx.co3.MoEElementLinear
   :members: forward

Activations
-----------

.. autoclass:: equivariantx.co3.Activation
   :members: forward

.. autoclass:: equivariantx.co3.Gate
   :members: forward

Tensor products
---------------

``co3.TensorProduct`` uses ``co3.ictd.TensorProduct``. The alternative
``co3.ictp.TensorProduct`` uses analytic symmetrization and trace subtraction.
Both accept symmetric traceless (STF) inputs and share instruction weights,
normalization and allowed angular paths, including Levi-Civita couplings.

.. autoclass:: equivariantx.co3.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: equivariantx.co3.FullyConnectedTensorProduct

.. autoclass:: equivariantx.co3.ElementwiseTensorProduct

.. autoclass:: equivariantx.co3.ictp.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. code-block:: python

   from equivariantx import co3

   args = ("4x1o", "4x1o", "4x0e+4x1e+4x2e")
   paths = [(0, 0, k, "uuu", True) for k in range(3)]
   ictd = co3.ictd.TensorProduct(*args, paths)
   ictp = co3.ictp.TensorProduct(*args, paths)
   ictp.load_state_dict(ictd.state_dict())

With ``irrep_normalization="component"``, unit variance refers to independent
orthonormal irreducible coordinates, not to all ``3**l`` Cartesian entries.
For an isotropic rank-``l`` feature with unit covariance in these coordinates,
the mean squared Cartesian entry is ``(2*l+1)/3**l``. Self-products and correlated
paths do not in general retain unit variance.

Both algorithms require STF inputs. Their derivatives agree along the STF
subspace, not necessarily along perturbations that violate the input constraint.

``co3.w3j`` is :func:`equivariantx.o3.w3j`. Pass a path matrix on each input
or output index that uses Cartesian coordinates.

Storage and composition
-----------------------

The flattened ``mul_ir`` layout can be viewed as ``(..., mul, 3, ..., 3)``
without copying.

``co3.ictd.TensorProduct(project=False)`` defers projection until after graph
reduction and channel mixing. Both tensor products can be passed to
:class:`equivariantx.co3.CartesianTensorProductConv`; the CUDA backend projects
after aggregation.

In recursive products, an intermediate consumed by another tensor product must
be STF. Deferring the final projection is valid only across linear operations.

Layouts
-------

.. autoclass:: equivariantx.co3.TransposeIrrepsLayout
   :members: forward, inverse

Convolutions
------------

.. autoclass:: equivariantx.co3.CartesianTensorProductConv
   :members: forward

.. autoclass:: equivariantx.co3.conv.Linear
   :members: forward

Many-body contractions
----------------------

.. autoclass:: equivariantx.co3.SymmetricContraction
   :members: forward

.. autoclass:: equivariantx.co3.AsymmetricContraction
   :members: forward

.. autoclass:: equivariantx.co3.IctdUuuFullyConnectedSymmetricContraction
   :members: forward

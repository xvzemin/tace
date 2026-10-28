.. _equivariantx-api-co3:

Cartesian O(3)
==============

Features use flattened ``mul_ir`` storage with ``3**l`` entries per channel.
Tensor-product inputs lie in the symmetric traceless subspace.

Representations
---------------

.. autoclass:: eqx.co3.Irrep
   :members:
   :special-members: __mul__

.. autoclass:: eqx.co3.Irreps
   :members:

Harmonics
---------

.. autoclass:: eqx.co3.CartesianHarmonics
   :members: forward

Linear maps
-----------

.. autoclass:: eqx.co3.Linear
   :members: forward, weight_view_for_instruction, weight_views

Activations
-----------

.. autoclass:: eqx.co3.Activation
   :members: forward

.. autoclass:: eqx.co3.Gate
   :members: forward

Tensor products
---------------

``co3.ictp_ictc`` contracts Cartesian tensors and then projects the output.
``co3.ictp`` constructs the STF output by analytic symmetrization and trace
subtraction. Both retain every allowed angular path, including Levi-Civita
couplings, and use identical instruction weights and normalization.
``co3.TensorProduct`` continues to select ``co3.ictp_ictc.TensorProduct``.
The projection route skips outputs already known to be STF and batches the
remaining projections by rank. A mixed output entry is projected if any
incoming path requires projection; independent path weights are unchanged.
Both routes use elementwise products for outer products and native cross
products for uncontracted Levi-Civita couplings.

.. autoclass:: eqx.co3.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. autoclass:: eqx.co3.FullyConnectedTensorProduct

.. autoclass:: eqx.co3.ElementwiseTensorProduct

.. autoclass:: eqx.co3.ictp.TensorProduct
   :members: forward, weight_view_for_instruction, weight_views

.. code-block:: python

   from eqx import co3

   args = ("4x1o", "4x1o", "4x0e+4x1e+4x2e")
   paths = [(0, 0, k, "uuu", True) for k in range(3)]
   projected = co3.ictp_ictc.TensorProduct(*args, paths)
   analytic = co3.ictp.TensorProduct(*args, paths)
   analytic.load_state_dict(projected.state_dict())

With ``irrep_normalization="component"``, unit variance refers to independent
orthonormal irreducible coordinates, not to all ``3**l`` Cartesian entries.
For an isotropic rank-``l`` feature with unit covariance in these coordinates,
the mean squared Cartesian entry is ``(2*l+1)/3**l``. Self-products and correlated
paths do not in general retain unit variance.

Both algorithms require STF inputs. Their derivatives agree along the STF
subspace, not necessarily along perturbations that violate the input constraint.

Storage and composition
-----------------------

The public layout is flattened ``mul_ir``. A block can be viewed as
``(..., mul, 3, ..., 3)`` without copying; splitting the Cartesian axes does
not change storage or arithmetic. Internal symmetric coordinates store
``(l+1)*(l+2)//2`` permutation classes where useful.
The analytic route combines fixed symmetrization, delta insertion and output
scaling for small tensors. Larger tensors retain factored transformations to
bound the size of constant matrices.

For convolutions, ``co3.ictp_ictc.TensorProduct(project=False)`` allows a single
projection after graph reduction and channel mixing. The analytic algorithm
produces an STF tensor on each edge. Both can be passed to
:class:`eqx.conv.CartesianTensorProductConv`; its PyTorch backend follows the
supplied product, while CUDA uses the equivalent contraction with node-level
projection. There is no separate analytic ICTP CUDA kernel.

In recursive products, an intermediate consumed by another tensor product must
be STF. Deferring the final projection is valid only across linear operations.

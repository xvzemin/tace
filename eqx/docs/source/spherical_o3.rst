.. _equivariantx-spherical-o3:

Spherical O(3)
==============

``eqx.o3`` provides element-dependent linear maps, gated activations, and
quadrature transforms.
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

Sphere grid
-----------

``S2Grid`` transforms spherical-harmonic coefficients on the unit sphere.
Each channel has :math:`(L+1)^2` coefficients in degree order. Channels
occupy a leading axis, separate from the final coefficient axis. The
normalization is ``component``, ``norm``, or ``integral``.

.. code-block:: python

   from eqx.nn import PolynomialActivation

   act = PolynomialActivation("silu", degree=8, dtype=torch.float64)
   sphere = o3.S2Grid(3, resolution=(act.degree + 1) * 3, dtype=torch.float64)
   features = torch.randn(8, 64, sphere.dim, dtype=torch.float64)
   values = sphere.to_grid(features)
   output = sphere.from_grid(act(values))

``quadrature="lebedev"`` is the default. It selects the smallest supported
algebraic degree at least ``resolution``; available degrees extend to 131.
``"gauss_legendre"`` combines Gauss-Legendre polar nodes with uniform
azimuthal samples and permits higher degrees. ``"equiangular"`` uses the
spherical grid of ``e3nn.o3.ToS2Grid`` with float64 quadrature weights.
All rules integrate the normalized measure, so weights sum to one.

Here ``resolution`` is the quadrature exactness degree, not the number of
points. The default is ``max(1, 3*lmax)``. Linear reconstruction requires
degree :math:`2L`; a degree-:math:`d` pointwise polynomial followed by
projection requires :math:`(d+1)L`. The default therefore integrates
quadratic activations exactly up to floating-point error. Increase the
resolution for higher-degree polynomials, as in the example.
``PolynomialActivation`` is a smooth polynomial, not the original SiLU;
its approximation interval controls fidelity to SiLU, not equivariance.
See :ref:`equivariantx-api-nn` for the available activations.

Constants are constructed in float64; changing a grid from float32 to
float64 restores these constants rather than promoting rounded values.
Forward and inverse transforms use PyTorch matrix products and autograd,
including higher derivatives. ``S2Grid`` represents scalar functions with
natural spatial parity. Arbitrary reflection or time-parity assignments
are not supplied by a sphere grid alone.

The quadrature choices follow `SO(3) quadratures in angular-momentum
projection <https://arxiv.org/abs/2205.04119>`_ and
`Gauss-Legendre Sampling on the Rotation Group
<https://arxiv.org/abs/1508.03353>`_. The tabulated Lebedev rules are
described in `SciPy's quadrature reference
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.lebedev_rule.html>`_.

See :ref:`equivariantx-api-o3` for parameters,
:ref:`equivariantx-api-tools` for basis conversions and tensor decomposition,
and :ref:`equivariantx-convolutions` for fused O(3) tensor products.

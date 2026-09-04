.. _equivariantx-tutorials:

Tutorials
=========

EquivariantX (``eqx``) provides operators for
:math:`O(2)` and Cartesian :math:`O(3)`. 

.. note::

   **EquivariantX is under active development.**

   APIs, module names, and behaviors may change without notice, and backward
   compatibility is not guaranteed at this stage.

   A stable release will be published as a separate package.

``eqx.o2``
   General real :math:`O(2)\times\mathbb{Z}_2^T` irreps and operations, plus the specialized
   conversion between global spherical :math:`O(3)` features and local
   :math:`O(2)` frames.

``eqx.co3``
   Cartesian :math:`O(3)` irreps, polar and pseudotensor coupling,
   Cartesian harmonics, and projection onto symmetric traceless tensors.

Quick start
-----------

.. code-block:: python

   import torch

   from eqx import co3, o2

   irreps_o2 = o2.Irreps("8x0ee + 4x0oe + 6x1me + 3x2me")
   irreps_co3 = co3.Irreps("4x0e + 2x0o + 3x1o + 3x1e")

Common tensor conventions
-------------------------

The :mod:`eqx.o2` and :mod:`eqx.co3` operators use one flattened feature axis:

.. math::

   (\ldots,\ D_{\mathrm{irreps}}).

Inside every ``(ir, mul)`` entry, values are ordered as ``ir_mul``. The entry
can therefore be viewed directly as

.. math::

   (\ldots,\ d_{\mathrm{irrep}},\ \mathrm{mul}).

Linear, activation, gate, tensor-product, and local-frame modules use this
flattened ``ir_mul`` representation. Circular and Cartesian harmonics produce
the same layout. In :mod:`eqx.co3`, ``ir.dim`` is the ambient Cartesian size
:math:`3^l`; each entry is therefore viewed as
``(..., 3**l, multiplicity)``. Asymmetric contraction accepts a sequence of
independent flattened inputs, one for each correlation order. Inputs, internal
parameters, and external weights use real floating-point dtypes unless an API
states otherwise.

Real O(2) with time reversal
-----------------------------

Representations
~~~~~~~~~~~~~~~

The real irreps of :math:`O(2)\times\mathbb{Z}_2^T` carry a spatial
reflection parity and an independent time-reversal parity:

``0ee``, ``0eo``
   One-dimensional scalars even under reflection. The final letter is the
   time parity.

``0oe``, ``0oo``
   One-dimensional pseudoscalars odd under reflection.

``1me``, ``1mo``, ``2me``, ``2mo``, ...
   Two-dimensional real irreps. The components are stored as the cosine-like
   and sine-like pair for positive order :math:`m`. ``m`` denotes the spatial
   representation and the final letter denotes time parity.

The legacy names ``0e``, ``0o``, ``1m``, ``2m``, and so on remain accepted
and denote their time-even counterparts.

For a rotation by :math:`\theta`, a positive-order block transforms as

.. math::

   D_m(\theta)=
   \begin{pmatrix}
   \cos(m\theta)&-\sin(m\theta)\\
   \sin(m\theta)& \cos(m\theta)
   \end{pmatrix}.

Reflection distinguishes ``0ee`` and ``0oe`` and acts on the two-dimensional
blocks through the reflection component of :math:`O(2)`. Time reversal
multiplies an irrep by its time parity :math:`t=\pm1` without changing its
spatial components.

The tensor-product rules are

.. list-table:: Real O(2) tensor products
   :header-rows: 1
   :widths: 30 70

   * - Inputs
     - Outputs
   * - ``0ee x a``
     - ``a``
   * - ``0oo x 0oo``
     - ``0ee``
   * - ``0oo x 1mo``
     - ``1me``
   * - ``1mo x 2me``
     - ``1mo + 3mo``
   * - ``2mo x 2mo``
     - ``0ee + 0oe + 4me``

For every product, time parity follows

.. math::

   t_{\mathrm{out}}=t_1t_2.

Linear, Gate, and TensorProduct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.Linear` connects only identical irreps and uses a dense
input-output multiplicity matrix for every instruction. Missing output irreps
are returned as differentiable zeros; only ``0ee`` can receive a bias.

:class:`eqx.o2.Gate` applies an arbitrary scalar activation to ``0ee``. An
activation acting on a scalar that is odd under reflection or time reversal
must itself be even or odd. Gate products multiply both reflection and time
parities.

:class:`eqx.o2.TensorProduct` supports three channel contracts:

.. list-table:: Tensor-product path modes
   :header-rows: 1
   :widths: 18 32 50

   * - Mode
     - Channel constraint
     - Weight layout per path
   * - ``u1u``
     - :math:`C_2=1`, :math:`C_3=C_1`
     - One weight per output channel
   * - ``uuu``
     - :math:`C_1=C_2=C_3`
     - One weight per matched channel
   * - ``uvw``
     - No equality constraint
     - Dense :math:`C_1\times C_2\times C_3` weights

A minimal nonlinear block is

.. code-block:: python

   irreps = o2.Irreps("4x0ee + 2x0oe + 3x1me + 2x2me")
   irreps_scalars = o2.Irreps("4x0ee + 2x0oe")
   irreps_gated = o2.Irreps("3x1me + 2x2me")
   irreps_gates = o2.Irreps("5x0ee")

   nonlinearity = o2.Gate(
       irreps_scalars,
       [torch.nn.SiLU(), torch.nn.Tanh()],
       irreps_gates,
       [torch.nn.Sigmoid()],
       irreps_gated,
   )
   linear_up = o2.Linear(
       irreps,
       nonlinearity.irreps_in,
       biases=True,
   )
   linear_down = o2.Linear(
       nonlinearity.irreps_out,
       irreps,
       biases=False,
   )

   node_feats = torch.randn(32, irreps.dim)
   node_feats = linear_down(nonlinearity(linear_up(node_feats)))

Circular harmonics
~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.CircularHarmonics` constructs native two-dimensional angular
features. With ``normalize=True`` the output depends only on direction. With
``normalize=False``, order :math:`m` is homogeneous of degree :math:`m` in the
input vector. ``time_reversal=True`` declares a time-odd input and assigns
time parity :math:`(-1)^m` to order :math:`m`.

.. code-block:: python

   vectors_2d = torch.randn(128, 2)
   harmonics = o2.CircularHarmonics(mmax=3, normalize=True)
   edge_attrs = harmonics(vectors_2d)

   assert harmonics.irreps_out == o2.Irreps("0ee + 1me + 2me + 3me")
   assert edge_attrs.shape == (128, harmonics.irreps_out.dim)

For a time-odd two-dimensional vector:

.. code-block:: python

   magnetic_harmonics = o2.CircularHarmonics(
       mmax=3,
       normalize=True,
       time_reversal=True,
   )
   assert magnetic_harmonics.irreps_out == o2.Irreps(
       "0ee + 1mo + 2me + 3mo"
   )

Global O(3) to local O(2)
~~~~~~~~~~~~~~~~~~~~~~~~~

A directed three-dimensional vector defines a local axis. Restricting an
:math:`O(3)\times\mathbb{Z}_2^T` irrep ``(l, p, t)`` to the
:math:`O(2)\times\mathbb{Z}_2^T` isotropy subgroup gives

.. math::

   (l,p,t)\downarrow
   =\left(0,p(-1)^l,t\right)
   \oplus\bigoplus_{m=1}^{\min(l,m_{\max})}(m,0,t).

Time parity is retained by every local entry. For example, an axial,
time-odd vector restricts as ``1eo -> 0oo + 1mo``.

:class:`eqx.o2.WignerD` constructs the global-to-local and local-to-global
matrices from three-dimensional vectors. :class:`eqx.o2.LocalFrame` applies
those matrices. Its global input and local output both use flattened
``ir_mul`` layout. ``mmax`` may truncate local positive orders while inverse
rescaling preserves the intended variance.

.. code-block:: python

   channels = 64
   lmax = 3
   mmax = 2
   irreps_co3 = (
       "64x0e + 64x0o + 64x1o + 64x1e + "
       "64x2e + 64x2o + 64x3o + 64x3e"
   )
   edge_index = torch.randint(0, 16, (2, 48))
   edge_vectors = torch.randn(48, 3)

   wigner = o2.WignerD(lmax=lmax, mmax=mmax)
   D, D_inv = wigner.get_wigner(edge_vectors)
   frame = o2.LocalFrame(
       irreps_co3,
       lmax=lmax,
       mmax=mmax,
   )
   node_feats = torch.randn(16, frame.global_irreps.dim)

   local_features = frame.to_local(node_feats[edge_index[0]], D)
   global_messages = frame.to_global(local_features, D_inv)

``node_feats`` must already use flattened ``ir_mul`` order inside every O(3)
entry. ``local_features`` follows ``frame.irreps_out`` in the same flattened
order.

Asymmetric contraction
~~~~~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.AsymmetricContraction` contracts independent input features up
to a requested correlation order. All weights are supplied externally.
``algorithm="edge"`` recursively evaluates paths and minimizes coefficient
storage. ``algorithm="node"`` stores generalized CG tensors and evaluates
larger dense contractions, trading memory for node-level speed. Both
algorithms enumerate the same paths. ``path_mode="sum"`` is the default and
accumulates paths with the same output irrep using variance-preserving
normalization. ``path_mode="expand"`` retains every path as an output
multiplicity so that a following :class:`eqx.o2.Linear` performs the
compression.

Cartesian O(3)
--------------

Representations and projection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

An :class:`eqx.co3.Irrep` with degree :math:`l` is stored in the ambient
rank-:math:`l` Cartesian space of dimension :math:`3^l`. Its symmetric
traceless irreducible subspace has :math:`2l+1` independent components. Parity
is explicit, so ``2e`` and ``2o`` are different representations even though
both use nine stored Cartesian components.

For :math:`Q\in O(3)`, define

.. math::

   \eta_{l,p}=\frac{1-p(-1)^l}{2}.

The Cartesian action is

.. math::

   T'=[\det(Q)]^{\eta_{l,p}}Q^{\otimes l}T.

Natural-parity tensors have :math:`p=(-1)^l`; the complementary branch carries
one determinant factor and represents pseudotensors. :func:`eqx.co3.project`
maps an ambient tensor onto the symmetric traceless subspace using an
orthonormal Cartesian-to-spherical basis.

An :class:`eqx.co3.Irreps` entry is written as ``(ir, mul)``. Its flattened
segment has length ``ir.dim * mul`` and can be viewed without a permutation as
``(..., ir.dim, mul)``. Distinct entries remain distinct even when they carry
the same irrep, so explicit linear instructions can address them separately.

Linear and Gate
~~~~~~~~~~~~~~~

:class:`eqx.co3.Linear` connects equal ``(l, p)`` types and mixes their
multiplicity axes with dense matrices. Compatible input/output entries are
connected by default; ``instructions`` can select entry-level paths. The
``element`` normalization accounts for all input multiplicities feeding an
output, whereas ``path`` gives each incoming path equal variance. Bias is
available only for ``0e`` entries. Weights can be stored internally or supplied
as a flattened tensor to :meth:`eqx.co3.Linear.forward`.

:class:`eqx.co3.Gate` takes scalar entries, scalar gate entries, and gated
entries. Scalar and gate activations are normalized to preserve second
moments. Each activated gate multiplies one multiplicity channel of a gated
entry. An odd scalar gate flips the parity of the gated irrep; an even scalar
gate preserves it. Both input and output remain flattened ``ir_mul`` tensors.

Cartesian harmonics
~~~~~~~~~~~~~~~~~~~

:class:`eqx.co3.CartesianHarmonics` constructs symmetric traceless tensor
powers through ``lmax``. A polar ``1o`` input produces
``0e + 1o + 2e + ...``. An axial ``1e`` input produces even-parity outputs at
every degree. With ``normalize=True``, only the input direction is retained.
``normalization="component"`` gives every independent component unit second
moment over uniformly distributed directions; ``"norm"`` gives each degree
unit norm, and ``"integral"`` uses unit-sphere integral normalization.

.. code-block:: python

   vectors = torch.randn(128, 3)
   harmonics = co3.CartesianHarmonics(
       lmax=3,
       irreps_in="1o",
       normalize=True,
   )
   edge_attrs = harmonics(vectors)

   assert harmonics.irreps_out == co3.Irreps("0e + 1o + 2e + 3o")
   assert edge_attrs.shape == (128, 40)

Cartesian tensor product
~~~~~~~~~~~~~~~~~~~~~~~~

Cartesian coupling contains a Kronecker-delta branch,

.. math::

   (A\otimes_k^\delta B)_{\boldsymbol i\boldsymbol j}
   =3^{-k/2}\sum_{\boldsymbol a}
   A_{\boldsymbol i\boldsymbol a}B_{\boldsymbol a\boldsymbol j},
   \qquad l_3=l_1+l_2-2k,

and a Levi-Civita branch,

.. math::

   (A\otimes_k^\epsilon B)_{\boldsymbol i w\boldsymbol j}
   =(2\,3^k)^{-1/2}\sum_{\boldsymbol a,u,v}
   A_{\boldsymbol i u\boldsymbol a}\epsilon_{wuv}
   B_{\boldsymbol a v\boldsymbol j},
   \qquad l_3=l_1+l_2-2k-1.

Both branches satisfy :math:`p_3=p_1p_2`. The connection modes ``u1u``,
``uuu``, and ``uvw`` follow their stated multiplicity rules. The ``project``
argument is required. With ``project=True``, every output entry is projected
after its paths are summed. With ``project=False``, the ambient result is
retained so that a linear map can compress paths before
:func:`eqx.co3.project_irreps` is applied.

With ``simplify=True``, consecutive equal path entries are packed directly
into one ``ir_mul`` multiplicity axis. The tensor product reads similarly
packed inputs through path offsets and produces the simplified output layout
without a separate permutation.

Cartesian convolution pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The following convolution uses ``u1u`` paths. Node channels occupy the
multiplicity axis, while each Cartesian harmonic occurs once. The tensor
product therefore preserves the node multiplicity on every compatible path:

.. code-block:: python

   num_nodes = 16
   num_edges = 48
   channels = 64
   lmax = 2
   irreps_node = co3.Irreps(
       [(co3.Irrep(name), channels) for name in ("0e", "1o", "2e")]
   )
   harmonics = co3.CartesianHarmonics(lmax, irreps_in="1o")
   irreps_edge = harmonics.irreps_out

   node_parts = []
   for ir, mul in irreps_node:
       part = co3.project(torch.randn(num_nodes, ir.dim, mul), ir.l)
       node_parts.append(part.reshape(num_nodes, ir.dim * mul))
   node_feats = torch.cat(node_parts, dim=-1)
   edge_index = torch.randint(0, num_nodes, (2, num_edges))
   edge_attrs = harmonics(torch.randn(num_edges, 3))

   irreps_out = irreps_node
   instructions = [
       (i, j, k, "u1u", True)
       for i, (ir1, _) in enumerate(irreps_node)
       for j, (ir2, _) in enumerate(irreps_edge)
       for k, (ir_out, _) in enumerate(irreps_out)
       if ir_out in ir1 * ir2
   ]

   tensor_product = co3.TensorProduct(
       irreps_node,
       irreps_edge,
       irreps_out,
       instructions,
       project=False,
   )
   source, target = edge_index
   edge_messages = tensor_product(node_feats[source], edge_attrs)
   node_messages = edge_messages.new_zeros(num_nodes, irreps_out.dim)
   node_messages.index_add_(0, target, edge_messages)
   node_messages = co3.project_irreps(node_messages, irreps_out)

Cartesian many-body pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A node-level many-body step uses :class:`eqx.co3.TensorProduct` with ``uuu`` paths when
corresponding input and output entries have the same multiplicity. Paths with
the same input and output degrees are contracted together over a path axis:

.. code-block:: python

   instructions = [
       (i, j, k, "uuu", True)
       for i, (ir1, mul1) in enumerate(irreps_node)
       for j, (ir2, mul2) in enumerate(irreps_node)
       for k, (ir_out, mul_out) in enumerate(irreps_node)
       if ir_out in ir1 * ir2 and mul1 == mul2 == mul_out
   ]
   node_tensor_product = co3.TensorProduct(
       irreps_node,
       irreps_node,
       irreps_node,
       instructions,
       project=True,
   )
   node_product = node_tensor_product(node_feats, node_feats)

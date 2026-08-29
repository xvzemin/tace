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
   General real :math:`O(2)` irreps and operations, plus the specialized
   conversion between global spherical :math:`O(3)` features and local
   :math:`O(2)` frames.

``eqx.co3``
   Complete Cartesian :math:`O(3)` irreps, polar and pseudotensor coupling,
   Cartesian harmonics, and projection onto symmetric traceless tensors.

Quick start
-----------

Import the symmetry namespace explicitly:

.. code-block:: python

   import torch

   from eqx import co3, o2

   irreps_o2 = o2.Irreps("8x0e + 4x0o + 6x1m + 3x2m")
   irreps_co3 = co3.Irreps("4x0e + 2x0o + 3x1o + 3x1e")

Common tensor conventions
-------------------------

EquivariantX keeps the representation and channel axes separate. Unless an
API explicitly says otherwise, a dense feature tensor has shape

.. math::

   (\ldots,\ D_{\mathrm{irreps}},\ C),

where the leading axes are arbitrary broadcastable batch axes,
``irreps.dim`` is the representation axis, and :math:`C` is the channel
count. Multiplicity belongs to :class:`Irreps`; channels belong to the layer.

The grouped layout used by local :math:`O(2)` operations stores one tensor per
irrep group:

.. math::

   (\ldots,\ d_{\mathrm{irrep}},\ \mathrm{multiplicity}\times C).

The order of groups and components always follows the corresponding
``Irreps`` object. Inputs, internal parameters, and external weights must use
real floating-point dtypes. Modules and inputs should be moved to the same
device and dtype in the usual PyTorch way.

Complete real O(2)
------------------

Representations
~~~~~~~~~~~~~~~

The complete real irreps of :math:`O(2)` are

``0e``
   One-dimensional scalar, even under reflection.

``0o``
   One-dimensional pseudoscalar, odd under reflection.

``1m``, ``2m``, ...
   Two-dimensional real irreps. The components are stored as the cosine-like
   and sine-like pair for positive order :math:`m`.

For a rotation by :math:`\theta`, a positive-order block transforms as

.. math::

   D_m(\theta)=
   \begin{pmatrix}
   \cos(m\theta)&-\sin(m\theta)\\
   \sin(m\theta)& \cos(m\theta)
   \end{pmatrix}.

Reflection distinguishes ``0e`` and ``0o`` and acts on the two-dimensional
blocks through the reflection component of :math:`O(2)`.

The tensor-product rules are

.. list-table:: Real O(2) tensor products
   :header-rows: 1
   :widths: 30 70

   * - Inputs
     - Outputs
   * - ``0e x a``
     - ``a``
   * - ``0o x 0o``
     - ``0e``
   * - ``0o x mm``
     - ``mm``
   * - ``m1m x m2m``, :math:`m_1\ne m_2`
     - ``abs(m1-m2)m + (m1+m2)m``
   * - ``mm x mm``
     - ``0e + 0o + (2m)m``

Linear, Gate, and TensorProduct
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.Linear` connects only identical irreps. ``path_mode="uv"``
uses a dense input-output channel matrix for every path. ``path_mode="uu"``
requires equal channel counts and applies one channel-wise weight per path.
Missing output irreps are returned as differentiable zeros; only ``0e`` can
receive a bias.

:class:`eqx.o2.Gate` applies an arbitrary scalar activation to ``0e``. A
direct activation on ``0o`` must be odd. If no ``0o`` activation is supplied,
``0o`` is gated by an auxiliary ``0e`` scalar, just like every positive-order
block.

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

   channels = 64
   irreps = o2.Irreps("4x0e + 2x0o + 3x1m + 2x2m")

   nonlinearity = o2.Gate(
       irreps,
       act_0e=torch.nn.SiLU(),
       act_0o=torch.nn.Tanh(),
       act_lm=torch.nn.Sigmoid(),
   )
   linear_up = o2.Linear(
       irreps,
       nonlinearity.irreps_in,
       channels,
       channels,
   )
   linear_down = o2.Linear(
       nonlinearity.irreps_out,
       irreps,
       channels,
       channels,
       bias=False,
   )

   node_feats = torch.randn(32, irreps.dim, channels)
   node_feats = linear_down(nonlinearity(linear_up(node_feats)))

Circular harmonics
~~~~~~~~~~~~~~~~~~

:class:`eqx.o2.CircularHarmonics` constructs native two-dimensional angular
features. With ``normalize=True`` the output depends only on direction. With
``normalize=False``, order :math:`m` is homogeneous of degree :math:`m` in the
input vector.

.. code-block:: python

   vectors_2d = torch.randn(128, 2)
   harmonics = o2.CircularHarmonics(m_max=3, normalize=True)
   edge_attrs = harmonics(vectors_2d)

   assert harmonics.irreps_out == o2.Irreps("0e + 1m + 2m + 3m")
   assert edge_attrs.shape == (128, harmonics.irreps_out.dim)

Global O(3) to local O(2)
~~~~~~~~~~~~~~~~~~~~~~~~~

A directed three-dimensional vector defines a local axis. Restricting an
:math:`O(3)` irrep ``(l, p)`` to the :math:`O(2)` isotropy subgroup gives one
order-zero block and positive orders up to :math:`l`. The order-zero block is
``0e`` if :math:`p(-1)^l=1` and ``0o`` otherwise. This distinction retains
complete global :math:`O(3)` parity.

:class:`eqx.o2.WignerD` constructs the global-to-local and local-to-global
matrices from three-dimensional vectors. :class:`eqx.o2.LocalFrame` applies
those matrices and converts between flattened global features and grouped
local blocks. ``mmax`` may truncate local positive orders while inverse
rescaling preserves the intended variance.

.. code-block:: python

   channels = 64
   lmax = 3
   mmax = 2
   irreps_o3 = (
       "64x0e + 64x0o + 64x1o + 64x1e + "
       "64x2e + 64x2o + 64x3o + 64x3e"
   )
   edge_index = torch.randint(0, 16, (2, 48))
   edge_vectors = torch.randn(48, 3)

   wigner = o2.WignerD(lmax=lmax, mmax=mmax)
   D, D_inv = wigner.get_wigner(edge_vectors)
   frame = o2.LocalFrame(
       irreps_o3,
       lmax=lmax,
       mmax=mmax,
       layout="mul_ir",
   )
   node_feats = torch.randn(16, frame.global_irreps.dim)

   local_blocks = frame.to_local(node_feats[edge_index[0]], D)
   global_messages = frame.to_global(local_blocks, D_inv)

``layout="mul_ir"`` and ``layout="ir_mul"`` are both flattened global
layouts. Local O(2) features are returned as grouped blocks with explicit
representation and combined multiplicity-channel axes.

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

:class:`eqx.co3.Layout` converts between dense
``(batch, irreps.dim, channels)`` data and grouped
``(batch, 3**l, multiplicity * channels)`` blocks. It changes storage only;
it does not rotate or project features.

Linear and Gate
~~~~~~~~~~~~~~~

:class:`eqx.co3.Linear` provides only dense UV channel mixing. It connects
equal ``(l, p)`` types, zero-pads absent outputs, and permits bias only on
``0e``. Internal weights have unit normal initialization and are multiplied by
the fixed scale :math:`C_{\mathrm{in}}^{-1/2}`. Default path normalization
adds :math:`N_{\mathrm{path}}^{-1/2}`, keeping variance stable as channels and
path counts change.

:class:`eqx.co3.Gate` follows the same scalar rules as the O(2) gate. ``0e``
uses ``act_0e``; a direct ``0o`` activation must be odd; all non-scalars and
an indirectly activated ``0o`` use invariant ``0e`` gates passed through
``act_tensor``.

Cartesian harmonics
~~~~~~~~~~~~~~~~~~~

:class:`eqx.co3.CartesianHarmonics` constructs symmetric traceless tensor
powers through ``lmax``. A polar ``1o`` input produces
``0e + 1o + 2e + ...``. An axial ``1e`` input produces even-parity outputs at
every degree. With ``normalize=True``, only the input direction is retained.

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

Complete Cartesian coupling contains a Kronecker-delta branch,

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

Both branches satisfy :math:`p_3=p_1p_2`. The channel modes ``u1u``, ``uuu``,
and ``uvw`` have the same meaning as in :class:`eqx.o2.TensorProduct`.

The ``project`` argument is mandatory:

``project=True``
   Project the output immediately. Use this for ordinary tensor products and
   node-level many-body expansions.

``project=False``
   Return a generic Cartesian tensor in the same ambient shape. Linear
   operations may then be moved before projection. In particular, edge
   aggregation and path compression can precede one node-level projection.

Projection is linear, hence

.. math::

   \mathcal P\!\left(\sum_j M_{ij}\right)
   =\sum_j\mathcal P(M_{ij}).

The unprojected result must be projected before it is treated as an
irreducible feature by a nonlinear operation.

Cartesian convolution pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The intended Cartesian convolution keeps paths explicit on edges, aggregates
without projection, compresses paths with :class:`eqx.co3.Linear`, and then
projects once:

.. code-block:: python

   num_nodes = 16
   num_edges = 48
   channels = 64
   lmax = 2
   irreps_node = co3.Irreps("0e + 0o + 1o + 1e + 2e + 2o")
   harmonics = co3.CartesianHarmonics(lmax, irreps_in="1o")
   irreps_edge = harmonics.irreps_out

   node_feats = torch.cat(
       [
           co3.project(torch.randn(num_nodes, ir.dim, channels), ir.l)
           for ir in irreps_node.expanded()
       ],
       dim=1,
   )
   edge_index = torch.randint(0, num_nodes, (2, num_edges))
   edge_attrs = harmonics(torch.randn(num_edges, 3)).unsqueeze(-1)

   irreps_paths = []
   paths = []
   for i, irrep1 in enumerate(irreps_node.expanded()):
       for j, irrep2 in enumerate(irreps_edge.expanded()):
           for irrep_out in irrep1 * irrep2:
               if irrep_out.l <= lmax:
                   output_index = len(irreps_paths)
                   irreps_paths.append(irrep_out)
                   paths.append((output_index, i, j))
   irreps_paths = co3.Irreps(irreps_paths)

   tensor_product = co3.TensorProduct(
       irreps_node,
       irreps_edge,
       irreps_paths,
       channels_in1=channels,
       channels_in2=1,
       channels_out=channels,
       project=False,
       path_mode="u1u",
       path=paths,
   )
   source, target = edge_index
   raw_edges = tensor_product(node_feats[source], edge_attrs)
   raw_nodes = raw_edges.new_zeros(num_nodes, irreps_paths.dim, channels)
   raw_nodes.index_add_(0, target, raw_edges)

   linear_down = co3.Linear(
       irreps_paths,
       irreps_node,
       channels,
       channels,
       bias=False,
   )
   node_messages = linear_down(raw_nodes)
   node_messages = torch.cat(
       [
           co3.project(node_messages[..., block, :], ir.l)
           for ir, block in zip(
               irreps_node.expanded(),
               irreps_node.expanded_slices(),
           )
       ],
       dim=-2,
   )

Cartesian many-body pattern
~~~~~~~~~~~~~~~~~~~~~~~~~~~

A node-level many-body step uses immediate projection:

.. code-block:: python

   node_tensor_product = co3.TensorProduct(
       irreps_node,
       irreps_node,
       irreps_node,
       channels_in1=channels,
       channels_in2=channels,
       channels_out=channels,
       project=True,
       path_mode="uuu",
   )
   node_product = node_tensor_product(node_feats, node_feats)

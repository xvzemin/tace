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

SPACE radial weight MLP
~~~~~~~~~~~~~~~~~~~~~~~

``radial_basis.mlp_type: space`` selects a separate radial network for each
edge angular degree :math:`l_2` in each O(3) CGTP interaction. Install its
optional dependency with ``pip install -e '.[space,oeq]'``. The implementation
uses the eigenfunctions from ``physical-basis==1.0.0``, following
`SPACE <https://arxiv.org/html/2610.10830v1>`_ and
`metatrain's radial implementation <https://github.com/metatensor/metatrain/tree/911683a32eaccc1d99af9a3683b1bd016facc544/src/metatrain/experimental/space/modules>`_.

For each degree, :math:`n_l` is the number of physical eigenfunctions with
:math:`E_{nl}\leq E_{\max}`. Distances are transformed using the fixed ASE
covalent radii :math:`R_Z`:

.. math::

   x_{ij} = \frac{r_{ij}}{0.1 + s(R_{Z_i}+R_{Z_j})}, \qquad
   b^{(l)}_{ij,n} = \sqrt{r_{\mathrm{cut}}^3/3}\,
                   R_{nl}(x_{ij})\,f_{\mathrm{SPACE}}(r_{ij}).

The basis is zero for :math:`x\geq10`. As in SPACE, values and first
derivatives of the original numerical eigenfunctions are tabulated on an
adaptively refined grid and evaluated with cubic Hermite interpolation.
``spline_accuracy`` is a mean absolute-or-relative midpoint criterion, not
a maximum value or derivative error bound. The fixed tables are model
buffers; forward evaluation and autograd use only PyTorch. Tables are
evaluated once per graph and reused across interaction layers.

Each independent SiLU MLP uses the hidden-layer widths in
``radial_basis.hidden`` and the bias setting in ``radial_basis.bias``:

.. math::

   n_l \longrightarrow h_1 \longrightarrow \cdots \longrightarrow h_k
       \longrightarrow W_l.

All degrees use the same list :math:`[h_1,\ldots,h_k]`, with independent
parameters. Only input and output widths depend on the degree. For example,
``hidden: [64, 64, 64, 1024]`` gives exactly those four hidden layers for every
:math:`l_2`; neither ``num_channel`` nor the number of radial functions rescales
them. ``hidden: []`` gives a single linear map. Bias is enabled or disabled on
every linear layer, including the output layer. The old
``space.mlp_depth`` and ``space.mlp_expansion_ratio`` options are no longer used
and must be removed from configurations.

Linear layers use SPACE's normal initialization and :math:`1/\sqrt{d_{in}}`
scaling. The final layer emits
:math:`W_l=\sum_{p:\,l_2(p)=l}\lvert w_p\rvert` values. Outputs are reordered
into the original e3nn instruction order. Thus the CGTP paths, per-path
channel weights, normalization, and total ``weight_numel`` are retained;
the number of trainable MLP parameters can change. This is a new training
architecture, not an exact conversion of pretrained radial weights.

.. code-block:: yaml

   misc:
     env:
       TACE_USE_EQX: "0"
       TACE_USE_OEQ: "1"
       TACE_USE_CUE: "0"
   model:
     config:
       lmax: 5
       edge_embedding:
         type: identity
       edge_update:
         type: identity
       atomic_basis:
         type: cgtp
       radial_basis:
         mlp_type: space
         hidden: [64, 64, 64, 1024]
         bias: false
         space:
           max_eigenvalue: 100.0
           element_scale: 0.7
           cutoff_width: 1.0
           spline_accuracy: 1.0e-8

At ``max_eigenvalue: 100``, degrees 0 through 5 have
``n_max_l = [5, 5, 4, 3, 2, 2]``. ``lmax`` continues to specify the TACE
edge degrees; too small an eigenvalue threshold raises an error instead of
removing any CGTP paths. Degrees above ``lmax`` are not used. For example,
``lmax: 3`` with ``max_eigenvalue: 50`` uses ``[3, 3, 2, 1]``.

SPACE weights consume these physical functions directly. This version
requires identity edge embedding/update and spherical ``cgtp`` interactions.
It supports OEQ and native e3nn; EQX streaming has not been adapted.
The SPACE bump envelope is applied to the physical basis before the MLP.
With ``bias: false``, this retains the original SPACE cutoff behavior. With
``bias: true``, the same envelope also multiplies the MLP output so that bias
cannot leave a nonzero message at the neighbor cutoff. No additional TACE
envelope multiplies these convolution weights. The neighbor list keeps TACE's
fixed cutoff; SPACE's adaptive neighbor selection is not enabled.

The existing ``num_radial_basis`` and radial-function options continue to
serve other modules, such as spherical node embeddings and density
normalization. SPACE's input counts are set by ``space.max_eigenvalue``;
its MLP hidden layers and bias are set by the shared ``radial_basis`` options.
Existing models keep ``mlp_type: dense`` by default.
A training example is available in
``example/train/benchmark_configs/3bpa_space.yaml``.

.. autoclass:: tace.models.space.SpaceRadialBasis
   :no-members:

.. autoclass:: tace.models.space.SpaceRadialMLP
   :no-members:

O(2) CGTP
---------

``atomic_basis.type: o2_cgtp`` evaluates the same CGTP as ``cgtp`` through
an edge-aligned frame. Spherical harmonics then have only an order-zero
entry, so the contraction uses only the nonzero coefficients in that CG
slice. Coupling paths, radial weights, and normalization are unchanged.
All required orders are retained, independently of ``mmax``.

Existing CGTP models can be converted without retraining. Values and
derivatives agree up to floating-point roundoff. The conversion returns a
new model and leaves the original unchanged:

.. code-block:: python

   from tace.lightning import convert_cgtp, export_tace, load_tace

   model = load_tace("TACE-OAM-7M.pt", dtype="float64")
   local_model = convert_cgtp(model)  # cgtp -> o2_cgtp
   global_model = convert_cgtp(local_model)  # o2_cgtp -> cgtp
   export_tace(local_model, "TACE-OAM-7M-o2-cgtp.pt")

The same automatic conversion is available from the command line:

.. code-block:: bash

   tace-convert-cgtp -m TACE-OAM-7M.pt --dtype float64 --device cpu

This writes ``TACE-OAM-7M-converted.pt`` beside the original model.

This is an equivalent implementation of CGTP, not the different
``o2`` Linear--Gate--Linear architecture. By default it uses PyTorch sparse
reductions. ``TACE_USE_EQX=1`` selects the streamed CUDA implementation for
``o2_cgtp``, without changing its learned parameters or model-loading behavior.
See :ref:`eqx-streaming` for training support and requirements.

.. autoclass:: tace.models._e3nn.inter.O2CgtpInteraction
   :no-members:
   :show-inheritance:

.. autofunction:: tace.lightning.convert_cgtp

Cartesian O(3) ICTP/ICTC
------------------------

``atomic_basis.type: co3`` selects
``O3CartesianIctpIctcInteraction`` and automatically constructs the
``co3_angular_basis`` using ``eqx.co3.CartesianHarmonics``. ``lmax`` still
controls the edge angular degrees; no separate basis selector is needed.
Node embeddings that require spherical harmonics retain their own spherical
inputs. Cartesian and spherical interaction layers can be mixed.

The interaction preserves the CGTP paths, radial weights, channel-linear
weights, and normalization. Its computation is:

.. code-block:: text

   spherical nodes -> linear up -> Cartesian basis -> gather
   Cartesian edge harmonics ------------------------> ICTP/ICTC x radial weights
     -> scatter -> Cartesian linear down -> transposed path matrix
     -> spherical nonlinearity and product basis

Raw Cartesian edge tensors are not projected before aggregation. Channel
compression precedes the output projection, which uses rectangular path
matrices. Both delta and epsilon contractions are retained, including
unnatural-parity paths. The product basis, readouts, and residual connections
remain spherical.

.. code-block:: python

   from tace.lightning import convert_cgtp, export_tace, load_tace
   from tace.interface.ase import TACEAseCalc

   model = load_tace("TACE-OAM-7M.pt", device="cuda", dtype="float64")
   cartesian = convert_cgtp(model, implementation="co3")
   spherical = convert_cgtp(cartesian, implementation="o3")
   export_tace(cartesian, "TACE-OAM-7M-co3.pt")
   calculator = TACEAseCalc(cartesian, device="cuda", dtype="float64")

Conversion preserves all learned parameter names, shapes, and values.
Predictions and derivatives agree up to floating-point roundoff; recreate
the optimizer when continuing training with the returned model. New models
can use ``example/train/benchmark_configs/3bpa_co3.yaml``.

This interaction uses native PyTorch, not the fused spherical CUDA kernel.
Cartesian storage grows as ``3**l`` and can cost more memory at high degree.
See :ref:`equivariantx-api-co3` for layout and normalization conventions.

.. autoclass:: tace.models._e3nn.inter.O3CartesianIctpIctcInteraction
   :no-members:
   :show-inheritance:

O(2) Linear
-----------

``TACE_USE_EQX=1`` enables CUDA fusion for ``o2`` and ``o2_mag`` on float32
and float64 inputs. Both use ``eqx.conv.UvO2TensorProductConv``: gather,
frame rotation, basis changes and radial multiplication are fused, as are
gates, attention scores and inverse rotation with scatter. Channel maps remain
batched GEMMs. Radial weights and local GEMM operands are still materialized;
global edge messages are not. Magnetic edge preparation, node operations and
checkpoint parameters are unchanged. CPU and unsupported activations retain
the native PyTorch implementation. Derivatives include force training and
recursive higher orders.

.. autoclass:: tace.models._e3nn.inter.UvO2Interaction

.. autoclass:: tace.models._e3nn.inter.O2MagneticInteraction
   :no-members:
   :show-inheritance:

O(2) Channelwise Linear
------------------------

``atomic_basis.type: uu_o2`` uses an externally weighted
``eqx.o2.UuLinear`` in the local frame. Only source node features are
gathered and rotated; target features are not concatenated. Compatible
representation copies mix within each channel. The edge MLP supplies a
separate weight for each path and channel. These weights are used directly,
without an additional activation or a separate radial multiplication.

The rejector contains a single UuLinear, with no Gate or channel-mixing Linear.
Radial rotary attention is not supported: set
``use_radial_rotary_attention: false``. The node-level linear maps, frame
transformations, cutoff, and scatter retain their usual behavior.
``edge_nonlinear`` is not used by this interaction. As with ``o2``, keep
``radial_basis.apply_cutoff: false`` so the cutoff is applied to messages.

.. code-block:: yaml

   atomic_basis:
     type: uu_o2
     edge_nonlinear: null
     use_radial_rotary_attention: false
   radial_basis:
     apply_cutoff: false

A 3BPA configuration is available as
``example/train/benchmark_configs/3bpa_uu_o2.yaml``.

Set ``TACE_USE_EQX=1`` to use ``eqx.conv.UuO2TensorProductConv``. It fuses
gather, both frame rotations, UuLinear, cutoff and scatter. The last radial
MLP projection uses reusable bounded workspaces instead of a full edge-weight
tensor. The preceding MLP layers and node-level linear maps remain unchanged.
The same weights can be used with or without fusion. CUDA supports force
training, recursive higher derivatives and ``torch.compile``.

.. autoclass:: tace.models._e3nn.inter.UuO2Interaction
   :no-members:
   :show-inheritance:

SO(2) Linear
------------

.. autoclass:: tace.models._e3nn.inter.UvSO2Interaction

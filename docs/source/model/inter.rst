Interaction
===========

We currently recommend using ``O3CgtpInteraction``. It supports operator fusion
via OpenEquivariance or CuEquivariance, which can significantly reduce memory
usage and improve computational efficiency.

Although the O(2)/SO(2) interaction variants are theoretically more advantageous 
at large angular momentum, they currently have fewer operator-fusion options and
are therefore not the default recommendation.

``atomic_basis.bias`` enables scalar biases in the interaction's linear maps
(default: ``true``). Bias-free maps remain bias-free. The radial MLP is
controlled separately by ``radial_basis.bias``.

.. _radial-mlp:

Scalar MLPs
-----------

The top-level model option ``mlp_type`` selects ``mlp_v1`` or ``mlp_v2`` for all
scalar MLPs:
radial weights, tensor node embeddings, pair interactions, density normalization,
magnetic weights and invariant property embeddings. Equivariant readouts and
Linear--Gate--Linear blocks retain their representation-aware operators.
The default is ``mlp_v1``. Hidden widths and normalization remain specific to
each module.

``radial_basis.mlp_norm`` controls radial MLP hidden layers. Use ``identity``
to skip normalization, ``layer_norm`` or ``rms_norm`` to select it explicitly,
or ``null`` (default) to retain each module's default behavior.

.. code-block:: yaml

   model:
     config:
       mlp_type: mlp_v2
       radial_basis:
         hidden: [64, 64]
         mlp_norm: null

Radial mode selection
---------------------

``max_eigenvalue`` retains modes with eigenvalue **less than or equal to**
the threshold. With both ``max_eigenvalue`` and ``num_radial_basis`` null,
the reference is eight degree-zero modes. The units of the threshold depend
on the basis:

* ``j0`` (alias ``bessel``), a fixed order such as ``j2``, and ``jn``:
  for cutoff :math:`R` in angstroms, the Dirichlet eigenvalues are
  :math:`(z_{\ell k}/R)^2` in inverse square angstroms.
  The default threshold is :math:`(8\pi/R)^2`. Changing :math:`R` at a
  fixed numerical threshold can change the number of modes. Using the default
  keeps their number unchanged. For trainable fixed-order Bessel functions,
  the threshold selects the initial frequencies only. If a distance
  transform is used, the spectrum refers to the uncomposed basis.
* ``spb`` (SPACE): the threshold applies to the tabulated dimensionless
  eigenvalues in the coordinate :math:`x=r/r_0`, where :math:`r_0` depends
  on the two elements. The default is the eighth degree-zero eigenvalue,
  approximately 177.48726637. The model cutoff changes the envelope and
  normalization, not the selected modes.
* ``ejn``: the threshold applies to a shared dimensionless spectrum on
  :math:`[0,X]`, where :math:`X` is the numerical value of the graph cutoff
  in angstroms. The default is the eighth degree-zero eigenvalue on this
  interval. Changing the cutoff changes the spectrum; changing the element
  vocabulary does not.
* ``gaussian`` has no eigenvalue spectrum. Use ``num_radial_basis``
  (eight by default).

Alternatively, set ``num_radial_basis`` to an integer for a fixed order,
or a list of ``lmax + 1`` positive integers for a degree-specific basis.
Do not set both selectors. A threshold that leaves a requested degree empty
raises an error.

For a 6 Å graph cutoff, integer thresholds of 28 for ``j0``, 284 for
``ejn`` and 271 for ``spb`` each retain ten degree-zero modes.

Physical radial basis
---------------------

.. code-block:: yaml

   model:
     config:
       radial_basis:
         radial_basis: spb
         num_radial_basis: null
         max_eigenvalue: 36.0
         use_interpolation: false

Each weighted CGTP path uses the radial modes for its edge harmonic degree.
For ``spb``, the explicit threshold 36 selects ``[3, 2, 1, 1]`` at
``lmax: 3``, versus ``[8, 7, 6, 6]`` with the default threshold.
The default evaluation uses the full Fourier expansion.
For ``spb``, set ``use_interpolation: true`` for cubic Hermite interpolation. Both modes
use the same state dictionary.

With ``radial_basis: ejn``, the model solves the weighted eigenproblem

.. math::

   -\frac{d}{dx}\left(x^2 e^x \frac{du}{dx}\right)
   + \ell(\ell+1)e^x u = \lambda x^2 u, \qquad u(X)=0.

Both bases use :math:`r_0=0.1+0.7(r_{\rm cov,i}+r_{\rm cov,j})` in
angstroms and the same normalization in :math:`x=r/r_0`.
``ejn`` solves once per degree and stores the shared eigenvalues and
200-term Fourier coefficients as model buffers. The element pair enters
only when evaluating the shared modes at :math:`x=r/r_0`.
At :math:`X=10`, this reproduces the tabulated SPACE basis up to the
sign of each eigenfunction. Modes are zero for :math:`r \ge X r_0`.
If this boundary lies inside the graph cutoff, the basis is continuous there
but its first derivative need not be. The graph cutoff envelope still acts
on the physical distance :math:`r`.

Both bases support ``cgtp``, its Cartesian and edge-aligned equivalents,
and CGTP in ``tace-train-oamv2``. OAMv2 streams degree-specific radial heads
through the EQX convolution by default. Set ``TACE_MEMORY_FIRST=0`` to
evaluate them in batches instead.

Degree-specific spherical Bessel basis
--------------------------------------

.. code-block:: yaml

   model:
     config:
       lmax: 3
       radial_basis:
         radial_basis: jn
         max_eigenvalue: null

Each path uses :math:`j_\ell` modes with its edge harmonic degree
:math:`\ell`. At ``lmax: 3`` the default counts are ``[8, 7, 7, 6]``.
For cutoff :math:`R` and the
:math:`k`-th positive zero :math:`z_{\ell k}` of :math:`j_\ell`, the mode is

.. math::

   B_{\ell k}(r) =
   \frac{\sqrt{2/R^3}}{|j_{\ell+1}(z_{\ell k})|}
   j_\ell(z_{\ell k}r/R).

The modes are evaluated directly at the interatomic distance, without
interpolation. The grouped radial MLP and the
supported EQX and OAMv2 convolutions are the same as for ``spb`` and ``ejn``.
Selecting a fixed order such as ``j1`` instead uses that one order for all paths.

O(3) Cgtp
---------

.. autoclass:: tace.models.spherical.inter.O3CgtpInteraction
   :no-members:
   :show-inheritance:

Cutoff scatter normalization
----------------------------

.. code-block:: yaml

   atomic_basis:
     scatter_norm: cutoff
   radial_basis:
     apply_cutoff: false

For each incoming edge :math:`e`, replace its cutoff :math:`c_e` by
:math:`c_e / \sqrt{1 + \sum_{b\ne e,\,b\to i} c_b^2}`, where :math:`i`
is its target. A lone neighbor keeps its original cutoff. The weights are
computed once per graph and shared across interaction layers. CGTP applies
them to the radial MLP output; gated O(2) applies them to the completed edge
messages before summation. The SO(2) attention interaction retains its own
normalization.

Constant residual normalization
-------------------------------

Set ``layer_norm.pre_norm_type: const_norm`` to replace feature
normalization with :math:`h_{\rm skip} + \Delta h / \sqrt{B}` at each selected
residual addition. All irreps share one constant; no feature statistics are
computed. :math:`B` counts two additions per ``BAB`` layer or one per ``AB``
or ``BB`` layer. The first layer counts only when both ``use_first_resnet``
and ``use_first_pre_norm`` are true. Otherwise its update is unchanged.

``final_norm_type`` accepts ``null``, ``merge_layer_norm``, ``merge_rms_norm``
or ``soft_cap_rms_norm``. It does not scale the final features by the residual factor.

Soft-capped RMS normalization
-----------------------------

Use ``soft_cap_rms_norm`` for ``pre_norm_type``, ``final_norm_type`` or both,
including with ``tace-train-oamv2``. With the model defaults this maps
:math:`x\mapsto x/\sqrt{1+v}`, where :math:`v` is the merged mean square of
the irrep blocks in one sample. There is no learned scale or bias.

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
``uv_o2`` Linear--Gate--Linear architecture. By default it uses PyTorch sparse
reductions. ``TACE_USE_EQX=1`` selects the streamed CUDA implementation for
``o2_cgtp``, without changing its learned parameters or model-loading behavior.
See :ref:`eqx-streaming` for training support and requirements.

.. autoclass:: tace.models.spherical.inter.O2CgtpInteraction
   :no-members:
   :show-inheritance:

.. autofunction:: tace.lightning.convert_cgtp

Cartesian O(3) ICTD
-------------------

For spherical node storage, ``atomic_basis.type: co3`` selects
``O3IctdInteraction`` and automatically constructs the
``co3_angular_basis`` using ``equivariantx.co3.CartesianHarmonics``. ``lmax`` still
controls the edge angular degrees; no separate basis selector is needed.
Node embeddings that require spherical harmonics retain their own spherical
inputs. Cartesian and spherical interaction layers can be mixed.

The interaction preserves the CGTP paths, radial weights, channel-linear
weights, and normalization. Its computation is:

.. code-block:: text

   spherical nodes -> linear up -> Cartesian basis -> gather
   Cartesian edge harmonics ------------------------> ICTD x radial weights
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
can select ``atomic_basis.type: co3``.

This interaction uses native PyTorch, not the fused spherical CUDA kernel.
Cartesian storage grows as ``3**l`` and can cost more memory at high degree.
See :ref:`equivariantx-api-co3` for layout and normalization conventions.

.. autoclass:: tace.models.spherical.inter.O3IctdInteraction
   :no-members:
   :show-inheritance:

O(2) Linear
-----------

Set ``atomic_basis.type`` to select the UV interaction:

* ``uv_o2``: radial weighting followed by Linear--Gate--Linear.
* ``uv_o2_mag``: the magnetic counterpart, supporting
  SOC and non-SOC features.

``uv_o2`` and ``uv_o2_mag`` use
``atomic_basis.scatter_norm``. ``so2`` always enables radial rotary attention.

``TACE_USE_EQX=1`` enables CUDA fusion on float32 and float64 inputs.
``uv_o2`` and ``uv_o2_mag`` fuse the complete radial MLP, rotation, radial weighting,
Linear--Gate--Linear, inverse rotation and scatter in a tiled convolution.
CPU and unsupported activations use PyTorch. Both paths support force training
and recursive higher derivatives.

TACE uses one hidden layer. EQX accepts a sequence of Linear maps with a Gate
between consecutive maps through :class:`equivariantx.conv.UvO2TensorProductConv`.

.. autoclass:: tace.models.spherical.inter.UvO2Interaction

.. autoclass:: tace.models.spherical.mtace.O2MagneticInteraction
   :no-members:
   :show-inheritance:

O(2) Channelwise Linear
------------------------

``atomic_basis.type: uu_o2`` uses an externally weighted
``equivariantx.o2.UuLinear`` in the local frame. Only source node features are
gathered and rotated; target features are not concatenated. Compatible
representation copies mix within each channel. The edge MLP supplies a
separate weight for each path and channel. These weights are used directly,
without an additional activation or a separate radial multiplication.

The rejector contains a single UuLinear, with no Gate or channel-mixing Linear.
The node-level linear maps, frame
transformations, cutoff, and scatter retain their usual behavior.
``edge_nonlinear`` is not used by this interaction. As with ``uv_o2``, keep
``radial_basis.apply_cutoff: false`` so the cutoff is applied to messages.

.. code-block:: yaml

   atomic_basis:
     type: uu_o2
     edge_nonlinear: null
   radial_basis:
     apply_cutoff: false

A 3BPA configuration is available as
``example/train/benchmark/3BPA/uu_o2.yaml``.

Set ``TACE_USE_EQX=1`` to use ``equivariantx.conv.UuO2TensorProductConv``. It fuses
gather, both frame rotations, UuLinear, cutoff and scatter. The last radial
MLP projection uses reusable bounded workspaces instead of a full edge-weight
tensor. The preceding MLP layers and node-level linear maps remain unchanged.
The same weights can be used with or without fusion. CUDA supports force
training, recursive higher derivatives and ``torch.compile``.

.. autoclass:: tace.models.spherical.inter.UuO2Interaction
   :no-members:
   :show-inheritance:

SO(2) Linear
------------

.. autoclass:: tace.models.spherical.tece_oam_rra.UvSO2Interaction

Interaction
===========

We currently recommend using ``O3CgtpInteraction``. It supports operator fusion
via OpenEquivariance or CuEquivariance, which can significantly reduce memory
usage and improve computational efficiency.

Although the O(2)/SO(2) interaction variants are theoretically more advantageous 
at large angular momentum, they currently have fewer operator-fusion options and
are therefore not the default recommendation.


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

Multi-cutoff radial weights
--------------------------

``radial_basis.multi_cutoff`` defaults to ``null`` (one radial network).
For ``cgtp``, ``co3`` and ``o2_cgtp``, specify positive, strictly increasing
radii in angstroms ending at the model cutoff:

.. code-block:: yaml

   model:
     config:
       cutoff: 6.0
       radial_basis:
         multi_cutoff: [2.0, 4.0, 6.0]
         apply_cutoff: false

An explicit list requires ``apply_cutoff: false``; omitting it retains the
default ``true`` and raises a configuration error. All listed bases are stored
in ``multi_cutoff_radial_basis`` in the specified order. At the same cutoff
placement, ``[cutoff]`` and ``null`` use the same single radial MLP and give
the same outputs for matching weights.
Each interaction sums the outputs of independent radial MLPs before a single
tensor product. Each MLP uses the basis and cutoff envelope for its own radius.
For :math:`S` radii, TACE scales each MLP's final weight matrix by
:math:`1/\sqrt{S}` at initialization only. The neighbor list uses the largest
radius. The optional ``element`` or ``element2`` path correction is added once
with the largest-radius envelope.
With ``scatter_norm: cutoff``, each radial branch uses its own envelope in
the numerator and denominator. The element correction uses the normalized
envelope of the largest radius. Both EQX radial modes are supported.

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

``final_norm_type`` accepts only ``null``, ``merge_layer_norm`` or
``merge_rms_norm``. It does not scale the final features by the residual factor.

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

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
-----------------------

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
See :ref:`equivariantx-cartesian` for the layout and normalization conventions.

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

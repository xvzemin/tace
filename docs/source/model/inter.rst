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

Spectrally truncated spherical Bessel weights
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``radial_basis.radial_basis: jn`` selects all degrees from zero through the
edge ``lmax``, with an independent radial MLP for each degree in each CGTP
interaction. ``radial_basis.jn_max_eigenvalue`` accepts either a positive
spectral cutoff or an explicit list of basis counts. A scalar has inverse
squared distance units (angstrom\ :sup:`-2` when positions are in angstrom).
A list ``[n_0, ..., n_lmax]`` sets the counts directly: its length must be
exactly ``lmax + 1`` and every entry must be a positive integer. These manual
counts remain fixed when ``cutoff`` changes; the basis functions still scale
to the new radius. This option differs from ``j0``, ``j1``, etc., which retain
one fixed degree and ``num_radial_basis`` functions shared by all paths.

On a ball of radius :math:`r_c`, separation of
:math:`-\nabla^2[R(r)Y_{lm}(\hat r)]=\lambda R(r)Y_{lm}(\hat r)` gives

.. math::

   R''+\frac{2}{r}R'+\left(\lambda-\frac{l(l+1)}{r^2}\right)R=0.

The regular solution is :math:`j_l(\sqrt{\lambda}\,r)`; the Dirichlet condition
:math:`R(r_c)=0` selects the positive zeros :math:`z_{lk}` of :math:`j_l`.
Thus the eigenvalues, retained counts, and normalized radial inputs are

.. math::

   \lambda_{lk} &= \left(\frac{z_{lk}}{r_c}\right)^2, \qquad k=1,2,\ldots,\\
   n_l &= \#\{k: z_{lk}\leq r_c\sqrt{\lambda_{\max}}\},\\
   b_{lk}(r) &= \frac{\sqrt{2/r_c^3}}{|j_{l+1}(z_{lk})|}
                j_l\!\left(z_{lk}\frac{r}{r_c}\right).

The constant prefactor normalizes each radial function under the measure
:math:`r^2\,dr`. There is no exponential radial decay, distance transform,
element-dependent rescaling, or spline interpolation. See the
`spherical Bessel equation <https://dlmf.nist.gov/10.47>`_. For :math:`l=0`,
:math:`z_{0k}=k\pi`, so :math:`n_0=\lfloor r_c\sqrt{\lambda_{\max}}/\pi\rfloor`.
Higher-degree counts use numerically solved positive roots, not an asymptotic
estimate. At fixed threshold, changing ``cutoff`` changes the counts.
For ``jn_max_eigenvalue: 25.0`` and ``lmax: 5``:

.. list-table::
   :header-rows: 1

   * - cutoff
     - :math:`[n_0,n_1,n_2,n_3,n_4,n_5]`
   * - 4.0
     - [6, 5, 5, 4, 4, 4]
   * - 5.0
     - [7, 7, 6, 6, 6, 5]
   * - 6.0
     - [9, 9, 8, 8, 7, 7]

Each MLP has dimensions ``[n_l, *radial_basis.hidden, W_l]``, where
:math:`W_l` counts the existing CGTP weights belonging to edge degree :math:`l`.
It uses TACE's standard MLP initialization and ``radial_basis.bias``. The
outputs are reordered into the original tensor-product instruction order;
the CGTP paths and total weight count are unchanged. The MLP parameter count
can change. ``num_radial_basis`` does not control this mode. A threshold that
leaves any degree through ``lmax`` empty raises an error with the minimum
required value rather than removing paths.

Cutoff placement follows the existing ``radial_basis.apply_cutoff`` option.
Let :math:`\mathbf b_l(r)` collect the normalized Bessel functions of degree
:math:`l`. The two choices generate weights as

.. math::

   \mathbf w_l(r) =
   \begin{cases}
     \mathrm{MLP}_l\!\left[f(r/r_c)\mathbf b_l(r)\right],
       & \texttt{apply\_cutoff=true},\\
     f(r/r_c)\,\mathrm{MLP}_l\!\left[\mathbf b_l(r)\right],
       & \texttt{apply\_cutoff=false}.
   \end{cases}

The first applies the envelope before the MLP; the second applies it after.
These are different functions for a nonlinear MLP. With bias enabled, an
input envelope alone does not guarantee zero output at the cutoff, because
:math:`\mathrm{MLP}_l(0)` can be nonzero. An output envelope gates the bias
as well. No additional envelope is silently applied in either mode.
The default C2 polynomial uses
:math:`x=r/r_c` across the whole interval, without a ``cutoff_width``:

.. math::

   f(x) = 1 - \frac{(p+1)(p+2)}{2}x^p
            + p(p+2)x^{p+1} - \frac{p(p+1)}{2}x^{p+2},\quad 0\leq x<1.

It is zero for :math:`x\geq1`, decreases from 1 to 0, and has
:math:`f(1)=f'(1)=f''(1)=0`.
Frequencies are fixed: ``trainable: false`` and ``distance_transform: null``
are required.

.. code-block:: yaml

   model:
     config:
       cutoff: 5.0
       lmax: 3
       atomic_basis:
         type: cgtp
       edge_embedding:
         type: identity
       edge_update:
         type: identity
       radial_basis:
         radial_basis: jn
         jn_max_eigenvalue: 25.0
         # Alternatively: [7, 7, 6, 6] manually sets n_0, ..., n_3.
         apply_cutoff: false  # true: before MLP; false: after MLP
         hidden: [64, 64, 64, 1024]
         bias: false

This mode requires spherical ``cgtp`` and identity edge embedding/update.
EQX convolution is disabled locally for these interactions. ``TACE_USE_OEQ=1``
selects OEQ even if EQX is enabled for other operators; otherwise the usual
remaining backend selection applies, with native e3nn as the fallback.
Basis counts are Python tuples named ``num_basis_by_l``. The internal
``_weight_index`` is reconstructed from the TP instructions and registered
with ``persistent=False``: it follows device moves without entering
``state_dict``. The existing radial-basis constants retain their original
checkpoint representation.

Existing named-degree bases and their EQX convolution behavior are unchanged.
The training example
``example/train/benchmark_configs/3bpa_jn.yaml`` inherits 3BPA CGTP and changes
only the radial weight generator.

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

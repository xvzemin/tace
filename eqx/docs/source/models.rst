Model integration
=================

Converters replace interaction tensor products, aggregation and final radial
projections. They retain coupling paths, normalization, parameters and the
model's forward interface. Other operations remain in the consuming library.

All converters accept ``implementation="o3"`` (default) or
``implementation="o2"`` for the equivalent aligned tensor product.
``backend="cuda"`` uses fused kernels on CUDA tensors;
``backend="torch"`` selects the PyTorch reference. CPU tensors use PyTorch.
These names select the computation, not the symmetry of the original model.

Load original weights and set the model's device and dtype before conversion.
The default returns an independent copy; ``inplace=True`` preserves the supplied
model's parameter objects. Convert before constructing an optimizer or distributed
wrapper. Converted state dictionaries must be loaded into an identically
converted architecture, not an unmodified upstream model.

Supported models
----------------

.. list-table::
   :header-rows: 1
   :widths: 20 20 60

   * - Package
     - Tested version
     - Scope
   * - MACE
     - 0.3.17
     - Spatial RealAgnostic interactions, including density and attention variants
   * - NequIP
     - 0.19.1
     - Uncompiled InteractionBlock models with TensorProductScatter
   * - SevenNet
     - 0.13.0
     - Instantiated serial IrrepsConvolution models, including Omni-i12
   * - Prophet
     - 1.0.0
     - Spatial Prophet models loaded with use_kernel=False
   * - EquFlash
     - GGNN 0.1, cuEquivariance 0.6.0
     - Uniform-channel FullConv interactions in EquFlashV2, including the 45M OAM checkpoint

Install the model package separately. EQX also provides ``mace``, ``nequip``
and ``sevennet`` extras, for example ``pip install './tace/eqx[nequip,cuda]'``.
Prophet is available from https://github.com/kairosmaterial/prophet.
The adapters do not import optional model packages when EQX is imported.
Respect each model package's dependency constraints when choosing an environment.
For example, current MACE and NequIP releases require different e3nn versions.

The O2 implementation requires natural-parity spherical harmonics on the edge,
including degree one to recover the direction, or scalar-only edge attributes.
It does not replace a tensor product with arbitrary learned second inputs.
Magnetic variants, distributed ghost-atom execution and upstream deployment
commands are not covered by these converters.

MACE
----

.. code-block:: python

   from eqx.models.mace import convert_mace_to_eqx
   from mace.calculators import MACECalculator

   model = convert_mace_to_eqx(model.cuda(), implementation="o3")
   calculator = MACECalculator(models=model, device="cuda", default_dtype="float32")

``enable_cueq=True`` converts the remaining supported operations through MACE.
Do not request another backend conversion in the calculator. MACE force training
requires ``training=True`` in the forward call. Use separate training and ASE
instances because the calculator disables parameter gradients.

NequIP
------

.. code-block:: python

   from eqx.models.nequip import convert_nequip_to_eqx
   from nequip.data.transforms import ChemicalSpeciesToAtomTypeMapper, NeighborListTransform
   from nequip.integrations.ase import NequIPCalculator

   model = convert_nequip_to_eqx(model.cuda(), implementation="o3")
   calculator = NequIPCalculator(
       model.eval(), device="cuda",
       transforms=[
           ChemicalSpeciesToAtomTypeMapper(model_type_names=model.type_names),
           NeighborListTransform(r_max=float(model.metadata["r_max"])),
       ],
   )

Use the species order and cutoff of the loaded model when constructing transforms.
For training, call ``model.train()`` before evaluating force losses.
Packaged models are supported. For example, select ``["sole_model"]`` from
``ModelFromPackage("NequIP-OAM-L-0.1.nequip.zip")`` before conversion.

SevenNet
--------

.. code-block:: python

   from eqx.models.sevennet import convert_sevennet_to_eqx
   from sevenn.calculator import SevenNetCalculator
   from sevenn.util import load_checkpoint

   model = load_checkpoint("7net-omni-i12").build_model(
       enable_cueq=False, enable_flash=False, enable_oeq=False,
   ).cuda()
   model = convert_sevennet_to_eqx(model, implementation="o3")
   calculator = SevenNetCalculator(
       model=model.eval(), file_type="model_instance", device="cuda", modal="omat24",
   )

Choose the desired fidelity through the calculator's ``modal`` argument.
For training, retain the original batching and modal fields and call
``model.train()``. Convert before constructing the optimizer.
SevenNet's embedding uses float32; converting only the model to float64
does not change that upstream behavior.

Prophet
-------

.. code-block:: python

   from eqx.models.prophet import convert_prophet_to_eqx
   from prophet.calculator import KairosCalculator

   calculator = KairosCalculator(
       model_path="model.pt", use_kernel=False, use_compile=False, device="cuda",
   )
   calculator.model = convert_prophet_to_eqx(calculator.model, implementation="o3")

For training, load with ``prophet.model.load_model(..., use_kernel=False)``,
convert the returned model and call ``model.train()``. Cutoff branches, gates,
normalization and readouts are unchanged. Prophet-Spin is not included.
The upstream ASE graph builder produces float32 inputs, so use a float32 model
with ``KairosCalculator``.

EquFlash
--------

The ``equflash`` adapter supports the EquFlashV2 foundation model.

Install GGNN in a separate environment using its upstream requirements
(Python 3.12, PyTorch 2.9.1 and cuEquivariance 0.6.0 for the tested checkpoint).

.. code-block:: python

   from GGNN.common.calculator import UCalculator
   from eqx.models.equflash import convert_equflash_to_eqx

   calculator = UCalculator(checkpoint_path="equflashv2-oam.pt", cpu=False)
   calculator.trainer.model = convert_equflash_to_eqx(
       calculator.trainer.model, implementation="o3", inplace=True,
   )

Use ``inplace=True`` with UCalculator to retain its EMA parameter references.
For training, convert the uncompiled model after loading its checkpoint and
before constructing the optimizer. FullConv paths, CG normalization and radial
weight order are read from the cuEquivariance descriptor. Equivalent output
irreps are merged along channels after node aggregation. Other layers, including
bilinear gates and feed-forward blocks, are unchanged. EfficientConv and
distributed ghost-atom exchange are not supported.

Not yet supported
-----------------

Allegro contracts edge tensors with learned environment tensors using
channelwise ``uuu`` products. The current fused ``uvu`` convolution is not a
drop-in replacement, and the environment is not a single spherical harmonic
that can be reduced to order zero by alignment.

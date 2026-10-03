.. _equivariantx-api-models:

Model integration
=================

Install the consuming model package separately. EQX provides ``mace``,
``nequip``, and ``sevennet`` extras. Follow each package's dependency constraints
when choosing an environment.

Converters accept instantiated models and preserve their forward interfaces.
``implementation="o3"`` is the default; ``"o2"`` selects the equivalent
harmonic convolution, not a different model symmetry. Load weights and set
the device and dtype before conversion, then construct the optimizer.
Converted state dictionaries require the same converted architecture.

The adapters move the complete radial MLP into the convolution. Edge embedding
and radial basis evaluation are unchanged. Parameters retain their values and
remain trainable; only their module paths change.

Converters
----------

.. autofunction:: eqx.models.mace.convert_mace_to_eqx

.. autofunction:: eqx.models.nequip.convert_nequip_to_eqx

.. autofunction:: eqx.models.sevennet.convert_sevennet_to_eqx

.. autofunction:: eqx.models.prophet.convert_prophet_to_eqx

.. autofunction:: eqx.models.equflash.convert_equflash_to_eqx

Allegro's ``uuu`` environment products are not supported by the current
``uvu`` convolution adapters. Distributed ghost-atom exchange is not covered.

TACE operators
--------------

.. autofunction:: eqx.models.tace.convert_tace_to_eqx

Receiver-tiled execution retains layer-boundary node features and recomputes
radial networks, density normalization, gates, and ACE within each tile. The
source Linear is evaluated once per node; Linear down follows aggregation.
Frozen models retain compressed node messages by default, so their reverse
sweep calls convolution adjoints without replaying convolution forwards.
``cache_messages=False`` trades additional recomputation for lower storage.
Radial functions and edge embeddings are evaluated exactly, without tables
or interpolation. Energy readout, scale/shift, and ZBL retain their original
definitions.

This opt-in plan initially targets TACE-OAM-L and uses existing EQX CUDA
operators. It is a model-level storage schedule, not a single CUDA kernel.
Inference uses a bounded first-order reverse sweep. Force training and higher
derivatives use differentiable replay with larger memory requirements.
Use the unconverted model for compilation, AOTI, and distributed execution.

Frozen OAM inference
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: eqx.models.tace.OAM
   :members: from_checkpoint, __call__

.. autofunction:: eqx.models.tace.ase_calculator

The standalone evaluator supports TACE-OAM-7M and TACE-OAM-L energy,
forces, stress, and virials. It folds frozen element embeddings into radial
affine maps, packs equivariant linear weights, and streams receiver tiles
through convolution, density normalization, Gate, residuals, and ACE.
All checkpoint functions are evaluated without interpolation.

``storage="cuda"`` retains layer boundaries on the GPU. ``"cpu"`` and
``"disk"`` offload them explicitly; neither denotes GPU-resident inference.
``domain_size`` partitions an orthorhombic periodic system into owned atoms
and complete receptive-field halos. Receiver sets shrink at each layer;
only owned energies are read out, while forces include halo derivatives.
Domains do not store graph-wide hidden features, but repeat halo work.

The evaluator is inference-only. It does not preserve state-dict layout or
support training, AOTI, magnetic models, or multi-GPU communication. Small
systems may be slower than the ordinary EQX execution. This is a tiled
model schedule using fused operators, not a single whole-model CUDA kernel.

.. autoclass:: eqx.models.tace.tece_oam_rra.BilinearACE
   :members: forward

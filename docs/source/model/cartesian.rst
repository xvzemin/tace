Cartesian TACE
==============

Cartesian TACE is under refactoring. Set ``model.config._target_`` to
``tace.models.cartTACE`` with the same configuration as spherical
TACE. ``atomic_basis.type: cgtp`` uses Cartesian contractions based on the
2024 ICTD theory, with projection after aggregation and channel compression.
The node product block also uses Cartesian tensors. Weights, biases, and
coupling paths are unchanged.

``o2`` and ``uu_o2`` use direct Cartesian plane restriction and ``eqx.co2``
operators. Each local order stores ``2**m`` Cartesian entries per channel,
without first constructing ``3**m`` transverse tensors. UV attention evaluates
Frobenius inner products in the same Cartesian storage.
The existing ``so2`` architecture retains its compact RRA operations between
Cartesian restriction and reconstruction. ``o2_mag`` supports SOC and non-SOC
models, including time-reversal labels and magnetic forces. ``w6j_mag`` uses
Cartesian node recoupling followed by the edge convolution.

Convert a trained model
-----------------------

.. code-block:: python

   from tace.lightning import convert_basis, export_tace, load_tace
   from tace.interface.ase import TACEAseCalc

   spherical = load_tace("TACE-OAM-7M", device="cuda", dtype="float64")
   cartesian = convert_basis(spherical, basis="cartesian")
   spherical_copy = convert_basis(cartesian, basis="spherical")
   export_tace(cartesian, "cartesian.pt")
   calc = TACEAseCalc(cartesian, device="cuda", dtype="float64")

Conversion preserves predictions and derivatives up to floating-point rounding.
It copies parameters and physical buffers, and rebuilds angular constants and
layout indices. Recreate the optimizer for the returned model before training.

Acceleration
------------

``TACE_USE_EQX=1`` enables:

* Native Cartesian O(3) convolution with streamed radial MLP evaluation.
* Element and expert Linear contractions without per-node weight expansion.
* Fused scalar activations and tensor gates.
* ACE and gated Bilinear/MoE products with the final product and coefficient
  contraction fused. Final STF projection follows channel mixing; recursive
  intermediates are projected before the next product.
* The existing SO(2)/RRA adapter uses compact kernels with fixed basis maps.
  It is not a native Cartesian local CUDA implementation.

Local O(2) and magnetic O(2) use the ``2**m`` PyTorch path even when EQX is
enabled. They are not silently converted to the compact CUDA layout.
The full-rotation and ambient-projection algorithms remain available through
``eqx.co2.LocalFrame(method="rotation")`` and ``method="projection"``.
Weights, biases, multiplicities, and coupling paths are unchanged.
Flat and per-path matrix parameter layouts survive conversion and save/load.

``TACE_USE_COMPILE=1`` and ``tace-export`` use the ordinary compilation/export
interfaces. Cartesian storage grows as ``3**l``; conversion is not a guarantee
of lower memory use or higher speed.

Training example: ``example/train/benchmark/3BPA/3bpa_cartesian.yaml``.

Reducible tensors
-----------------

Set ``model.config.reducible: true`` with ``atomic_basis.type: cgtp`` and
``product_basis.type: cgtp``. The default is ``false``.

* ``tace.models.cartTACE`` gives ``rco3``: unprojected Cartesian rank blocks.
* ``tace.models.e3nnTACE`` gives ``ro3``: the complete coupled spherical basis
  of the same blocks, including repeated irreducible components.

Both store ``3**l`` values per rank and channel. Angular inputs are tensor
powers, not STF harmonics. Contractions, channel mixing, gates and residuals
retain the original path ordering and parameter shapes. Only at the readout
boundary is each rank block projected onto its original descriptor irrep.
The spherical contraction coefficients are transformed at construction;
hidden layers do not convert tensors to Cartesian storage at runtime.

``ro3`` restricts contractions to their reachable input and output subspaces
at construction. Symmetric angular moments need only ``(l+1)*(l+2)/2``
coordinates; the standard scalar node embedding uses this compact angular
storage. Convolutions gather packed node features and aggregate compact edge
messages before restoring node coordinates. ACE contractions propagate the
reachable spaces through each product. Original path weights and normalization
are unchanged. Node Linear, Gate and residual boundaries still use complete
rank blocks; this is not a globally minimal storage layout.

``convert_basis`` also converts trained ``ro3`` and ``rco3`` models without
changing predictions or weights. This equivalence is between the two
**reducible** models: removing intermediate projections changes the function
of an irreducible model, even when its weights are retained.

These products use PyTorch, including on CUDA. STF-specific convolution and
ACE kernels are not used; compatible channel Linear and Gate kernels remain
available. Local O(2), magnetic recoupling and MACE symmetric products are
not selectable in this mode.

The 3BPA examples ``3bpa_rco3.yaml`` and ``3bpa_ro3.yaml`` in
``example/train/benchmark/3BPA`` inherit ``3bpa_cgtp.yaml``, as does
``3bpa_cartesian.yaml``. All four configurations use the same parameter
names, shapes and count.

.. autofunction:: tace.lightning.convert_basis

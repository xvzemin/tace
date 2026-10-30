Cartesian TACE
==============

Cartesian TACE is under refactoring. Set ``model.config._target_`` to
``tace.models.cartTACE`` with the same configuration as spherical
TACE. ``atomic_basis.type: cgtp`` uses native ICTP--ICTC contractions, with
projection after aggregation and channel compression. The node product block
also uses Cartesian tensors. Weights, biases, and coupling paths are unchanged.

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

.. autofunction:: tace.lightning.convert_basis

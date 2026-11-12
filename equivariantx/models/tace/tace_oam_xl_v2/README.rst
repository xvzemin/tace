TACE-OAM-XL-v2
=============

Whole-model E/F/S training schedule for the TACE OAM architecture. Architecture,
parameter names, live weights and the caller's loss are preserved.

.. code-block:: python

   from equivariantx.models.tace.tace_oam_xl_v2 import TACEOAMXLV2

   # Construct and load the original TACE adapter first.
   model = TACEOAMXLV2.from_tace(
       source,
       radial_mode="materialize",
       angular_mode="cg",
       product_layout="ir_mul",
       edge_mode="scheduled",
       gradient_schedule="fused",
       product_mode="packed",
       compose_products=True,
       affine_radial=True,
   )
   predictions, loss = model.forward_loss(batch, loss_fn)
   loss.backward(inputs=tuple(model.parameters()))

Model tensors use FP32/FP64; FP32 GEMM may use TF32. Autocast and FP16/BF16
model intermediates are rejected. Upstream Muon precision is unchanged.
``extensions.register_graph_extension`` retains a versioned extension point;
it does not relax this precision policy.

Layout changes, mixed-gradient scheduling, specialized CG code generation and
per-program graph plans live in this package under the ``tace`` family.
Reusable EQX operators remain in their shared packages. ``EQX_XL_*`` settings
and the ``eqx_oam_xl`` operator namespace retain their existing identifiers.

The released ``TACE-OAM-L`` model keeps its name and download identifier. It
belongs to the first generation; no ``-v1`` rename is required. This package
defines ``TACE-OAM-XL-v2`` execution, not a new downloadable pretrained weight.

Importing the package and running its component tests does not require TACE.
Conversion consumes an already constructed compatible TACE adapter. Component
and optional TACE integration tests share ``tests/test_tace_oam_xl_v2.py``;
only the integration cases require TACE.

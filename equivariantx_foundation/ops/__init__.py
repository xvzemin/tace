"""Execution providers for model-specific operators."""

from equivariantx.ops import register_kernel

for operation in ("contraction", "interaction"):
    register_kernel(f"tece_{operation}", "cuda", f"{__name__}.cuda.tece:{operation}")
    register_kernel(
        f"tece_{operation}",
        "torch",
        f"{__name__}.torch.tece:{operation}",
        reference=f"{__name__}.torch.tece:{operation}_reference",
    )

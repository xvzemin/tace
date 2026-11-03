"""Register EQX operators for loading exported graphs."""

from importlib import import_module

for module in (
    "equivariantx.ops.ace",
    "equivariantx.ops.symmetric",
    "equivariantx.ops.attention",
    "equivariantx.ops.edge",
    "equivariantx.ops.o2_o3",
    "equivariantx.ops.geometry",
    "equivariantx.ops.o3",
    "equivariantx.ops.channel_product",
    "equivariantx.ops.frame",
    "equivariantx.ops.layout",
    "equivariantx.ops.quaternion",
    "equivariantx.ops.recompute",
    "equivariantx.ops.rotary",
    "equivariantx.ops.rotation",
    "equivariantx.ops.wigner",
    "equivariantx.ops.bilinear",
    "equivariantx.ops.tece",
    "equivariantx.ops.linear",
):
    import_module(module)

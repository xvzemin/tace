"""Register EQX operators for loading exported graphs."""

from importlib import import_module

for module in (
    "equivariantx.ops.common.ace",
    "equivariantx.ops.common.symmetric",
    "equivariantx.ops.common.attention",
    "equivariantx.ops.common.edge",
    "equivariantx.ops.common.o2_o3",
    "equivariantx.ops.common.geometry",
    "equivariantx.ops.common.o3",
    "equivariantx.ops.common.channel_product",
    "equivariantx.ops.common.frame",
    "equivariantx.ops.common.layout",
    "equivariantx.ops.common.quaternion",
    "equivariantx.ops.common.recompute",
    "equivariantx.ops.common.rotary",
    "equivariantx.ops.common.rotation",
    "equivariantx.ops.common.wigner",
    "equivariantx.ops.common.bilinear",
    "equivariantx.ops.common.tece",
    "equivariantx.ops.common.linear",
):
    import_module(module)

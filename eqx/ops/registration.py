"""Register EQX operators for loading exported graphs."""

from importlib import import_module

for module in (
    "eqx.ops.ace",
    "eqx.ops.attention",
    "eqx.ops.edge",
    "eqx.ops.o2_o3",
    "eqx.ops.geometry",
    "eqx.ops.o3",
    "eqx.ops.channel_product",
    "eqx.ops.frame",
    "eqx.ops.layout",
    "eqx.ops.quaternion",
    "eqx.ops.recompute",
    "eqx.ops.rotary",
    "eqx.ops.rotation",
    "eqx.ops.wigner",
    "eqx.ops.bilinear",
    "eqx.ops.tece",
    "eqx.ops.linear",
):
    import_module(module)

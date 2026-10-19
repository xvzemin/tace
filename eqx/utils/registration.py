"""Register EQX operators for loading exported graphs."""

from importlib import import_module

for module in (
    "eqx.ace.contraction",
    "eqx.conv.attention",
    "eqx.conv.edge",
    "eqx.conv.o2_o3.convolution",
    "eqx.conv.o2_o3.geometry",
    "eqx.conv.o3.convolution",
    "eqx.kernels.channel_product",
    "eqx.kernels.frame",
    "eqx.kernels.layout",
    "eqx.kernels.quaternion",
    "eqx.kernels.recompute",
    "eqx.kernels.rotary",
    "eqx.kernels.rotation",
    "eqx.kernels.wigner",
    "eqx.models.tace.tece_oam_rra.bilinear_contraction",
    "eqx.models.tace.tece_oam_rra.interaction",
    "eqx.o3.contraction",
):
    import_module(module)

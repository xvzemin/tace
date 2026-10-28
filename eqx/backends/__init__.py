"""Execution providers for PyTorch tensors."""

from .registry import get_kernel, register_kernel

register_kernel("ace_contract", "cuda", "eqx.backends.cuda.ace:contract")
register_kernel("alignment_cuda", "cuda", "eqx.backends.cuda.wigner:alignment_cuda")
register_kernel(
    "bilinear_ace", "cuda", "eqx.backends.cuda.models.tace.bilinear:contract"
)
register_kernel(
    "contraction", "cuda", "eqx.backends.cuda.o2_o3.convolution:contraction"
)
register_kernel("edge_program", "cuda", "eqx.backends.cuda.edge:evaluate")
register_kernel("element_linear", "cuda", "eqx.backends.cuda.linear:contract")
register_kernel("frame_rotation", "cuda", "eqx.backends.cuda.rotation:contract")
register_kernel("gather_sum", "cuda", "eqx.backends.cuda.layout:gather_sum")
register_kernel(
    "graph_normalized_exp",
    "cuda",
    "eqx.backends.cuda.attention:normalized_exp",
)
register_kernel(
    "graph_normalized_exp_adjoint",
    "cuda",
    "eqx.backends.cuda.attention:evaluate_adjoint",
)
register_kernel("grouped_permute", "cuda", "eqx.backends.cuda.layout:grouped_permute")
register_kernel("indexed_frame", "cuda", "eqx.backends.cuda.frame:contract")
register_kernel("indexed_sum", "cuda", "eqx.backends.cuda.layout:indexed_sum")
register_kernel(
    "local_channel_product",
    "cuda",
    "eqx.backends.cuda.channel_product:channel_product",
)
register_kernel(
    "o3_contraction", "cuda", "eqx.backends.cuda.o3.convolution:contraction"
)
register_kernel("packed_wigner", "cuda", "eqx.backends.cuda.wigner:packed_wigner")
register_kernel(
    "quaternion_polynomial",
    "cuda",
    "eqx.backends.cuda.quaternion:quaternion_polynomial",
)
register_kernel("rotary_product", "cuda", "eqx.backends.cuda.rotary:contract")
register_kernel("rotation", "cuda", "eqx.backends.cuda.wigner:rotation")
register_kernel(
    "tece_contraction", "cuda", "eqx.backends.cuda.models.tace.interaction:contraction"
)
register_kernel(
    "tece_interaction", "cuda", "eqx.backends.cuda.models.tace.interaction:interaction"
)
register_kernel("edge_program", "torch", "eqx.backends.torch.edge:evaluate_torch")

__all__ = ["get_kernel", "register_kernel"]

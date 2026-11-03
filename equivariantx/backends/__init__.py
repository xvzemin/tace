"""Execution providers for PyTorch tensors."""

from .registry import get_kernel, register_kernel

register_kernel("ace_contract", "cuda", "equivariantx.backends.cuda.ace:contract")
register_kernel("symmetric_contract", "cuda", "equivariantx.backends.cuda.symmetric:contract")
register_kernel("alignment_cuda", "cuda", "equivariantx.backends.cuda.wigner:alignment_cuda")
register_kernel("bilinear_ace", "cuda", "equivariantx.backends.cuda.bilinear:contract")
register_kernel(
    "contraction", "cuda", "equivariantx.backends.cuda.o2_o3.convolution:contraction"
)
register_kernel("edge_program", "cuda", "equivariantx.backends.cuda.edge:evaluate")
register_kernel("element_linear", "cuda", "equivariantx.backends.cuda.linear:contract")
register_kernel("frame_rotation", "cuda", "equivariantx.backends.cuda.rotation:contract")
register_kernel("gather_sum", "cuda", "equivariantx.backends.cuda.layout:gather_sum")
register_kernel(
    "graph_normalized_exp",
    "cuda",
    "equivariantx.backends.cuda.attention:normalized_exp",
)
register_kernel(
    "graph_normalized_exp_adjoint",
    "cuda",
    "equivariantx.backends.cuda.attention:evaluate_adjoint",
)
register_kernel("grouped_permute", "cuda", "equivariantx.backends.cuda.layout:grouped_permute")
register_kernel("indexed_frame", "cuda", "equivariantx.backends.cuda.frame:contract")
register_kernel("indexed_sum", "cuda", "equivariantx.backends.cuda.layout:indexed_sum")
register_kernel(
    "local_channel_product",
    "cuda",
    "equivariantx.backends.cuda.channel_product:channel_product",
)
register_kernel(
    "o3_contraction", "cuda", "equivariantx.backends.cuda.o3.convolution:contraction"
)
register_kernel("packed_wigner", "cuda", "equivariantx.backends.cuda.wigner:packed_wigner")
register_kernel(
    "quaternion_polynomial",
    "cuda",
    "equivariantx.backends.cuda.quaternion:quaternion_polynomial",
)
register_kernel("rotary_product", "cuda", "equivariantx.backends.cuda.rotary:contract")
register_kernel("rotation", "cuda", "equivariantx.backends.cuda.wigner:rotation")
register_kernel(
    "tece_contraction", "cuda", "equivariantx.backends.cuda.models.tace.interaction:contraction"
)
register_kernel(
    "tece_interaction", "cuda", "equivariantx.backends.cuda.models.tace.interaction:interaction"
)
register_kernel("edge_program", "torch", "equivariantx.backends.torch.edge:evaluate_torch")

__all__ = ["get_kernel", "register_kernel"]

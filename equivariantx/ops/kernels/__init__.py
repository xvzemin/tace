"""PyTorch and CUDA execution providers."""

from ..registry import get_kernel, register_kernel

register_kernel(
    "cartesian_product",
    "torch",
    "equivariantx.ops.kernels.torch.cartesian:cartesian_product",
)
register_kernel(
    "cartesian_product",
    "cuda",
    "equivariantx.ops.kernels.cuda.cartesian:cartesian_product",
)

register_kernel("ace_contract", "cuda", "equivariantx.ops.kernels.cuda.ace:contract")
register_kernel(
    "ace_contract", "torch", "equivariantx.ops.kernels.torch.ace:ace_contract"
)
register_kernel(
    "symmetric_contract", "cuda", "equivariantx.ops.kernels.cuda.symmetric:contract"
)
register_kernel(
    "symmetric_contract",
    "torch",
    "equivariantx.ops.kernels.torch.symmetric:contract",
    reference="equivariantx.ops.kernels.torch.symmetric:evaluate_torch",
)
register_kernel(
    "alignment_cuda", "cuda", "equivariantx.ops.kernels.cuda.wigner:alignment_cuda"
)
register_kernel(
    "alignment_cuda", "torch", "equivariantx.ops.kernels.torch.wigner:alignment"
)
register_kernel(
    "bilinear_ace", "cuda", "equivariantx.ops.kernels.cuda.bilinear:contract"
)
register_kernel(
    "bilinear_ace", "torch", "equivariantx.ops.kernels.torch.ace:bilinear_contract"
)
register_kernel(
    "contraction", "cuda", "equivariantx.ops.kernels.cuda.o2_o3.convolution:contraction"
)
register_kernel(
    "contraction", "torch", "equivariantx.ops.kernels.torch.convolution:contraction"
)
register_kernel(
    "direction_contraction",
    "cuda",
    "equivariantx.ops.kernels.cuda.o2_o3.convolution:direction_contraction",
)
register_kernel(
    "direction_contraction",
    "torch",
    "equivariantx.ops.kernels.torch.geometry:direction_contraction",
)
register_kernel("edge_program", "cuda", "equivariantx.ops.kernels.cuda.edge:evaluate")
register_kernel(
    "element_linear", "cuda", "equivariantx.ops.kernels.cuda.linear:contract"
)
register_kernel(
    "element_linear", "torch", "equivariantx.ops.kernels.torch.linear:contract"
)
register_kernel(
    "frame_rotation", "cuda", "equivariantx.ops.kernels.cuda.rotation:contract"
)
register_kernel(
    "frame_rotation", "torch", "equivariantx.ops.kernels.torch.frame:frame_rotation"
)
register_kernel("gather_sum", "cuda", "equivariantx.ops.kernels.cuda.layout:gather_sum")
register_kernel(
    "gather_sum", "torch", "equivariantx.ops.kernels.torch.layout:gather_sum"
)
register_kernel(
    "graph_normalized_exp",
    "cuda",
    "equivariantx.ops.kernels.cuda.attention:normalized_exp",
)
register_kernel(
    "graph_normalized_exp",
    "torch",
    "equivariantx.ops.kernels.torch.attention:normalized_exp",
)
register_kernel(
    "graph_normalized_exp_adjoint",
    "cuda",
    "equivariantx.ops.kernels.cuda.attention:evaluate_adjoint",
)
register_kernel(
    "graph_normalized_exp_adjoint",
    "torch",
    "equivariantx.ops.kernels.torch.attention:evaluate_adjoint",
)
register_kernel(
    "grouped_permute", "cuda", "equivariantx.ops.kernels.cuda.layout:grouped_permute"
)
register_kernel(
    "grouped_permute", "torch", "equivariantx.ops.kernels.torch.layout:grouped_permute"
)
register_kernel("indexed_frame", "cuda", "equivariantx.ops.kernels.cuda.frame:contract")
register_kernel(
    "indexed_frame", "torch", "equivariantx.ops.kernels.torch.frame:indexed_frame"
)
register_kernel(
    "indexed_sum", "cuda", "equivariantx.ops.kernels.cuda.layout:indexed_sum"
)
register_kernel(
    "indexed_sum", "torch", "equivariantx.ops.kernels.torch.layout:indexed_sum"
)
register_kernel(
    "local_channel_product",
    "cuda",
    "equivariantx.ops.kernels.cuda.channel_product:channel_product",
)
register_kernel(
    "local_channel_product",
    "torch",
    "equivariantx.ops.kernels.torch.channel_product:channel_product",
    reference="equivariantx.ops.kernels.torch.channel_product:reference",
)
register_kernel(
    "o3_contraction", "cuda", "equivariantx.ops.kernels.cuda.o3.convolution:contraction"
)
register_kernel(
    "o3_contraction", "torch", "equivariantx.ops.kernels.torch.o3:contraction"
)
register_kernel(
    "packed_wigner", "cuda", "equivariantx.ops.kernels.cuda.wigner:packed_wigner"
)
register_kernel(
    "packed_wigner", "torch", "equivariantx.ops.kernels.torch.wigner:packed_wigner"
)
register_kernel(
    "quaternion_polynomial",
    "cuda",
    "equivariantx.ops.kernels.cuda.quaternion:quaternion_polynomial",
)
register_kernel(
    "quaternion_polynomial",
    "torch",
    "equivariantx.ops.kernels.torch.quaternion:quaternion_polynomial",
)
register_kernel(
    "rotary_product", "cuda", "equivariantx.ops.kernels.cuda.rotary:contract"
)
register_kernel(
    "rotary_product", "torch", "equivariantx.ops.kernels.torch.frame:rotary_product"
)
register_kernel("rotation", "cuda", "equivariantx.ops.kernels.cuda.wigner:rotation")
register_kernel("rotation", "torch", "equivariantx.ops.kernels.torch.wigner:rotation")
register_kernel(
    "tece_contraction",
    "cuda",
    "equivariantx.ops.kernels.cuda.models.tace.interaction:contraction",
)
register_kernel(
    "tece_interaction",
    "cuda",
    "equivariantx.ops.kernels.cuda.models.tace.interaction:interaction",
)
register_kernel(
    "edge_program",
    "torch",
    "equivariantx.ops.kernels.torch.edge:evaluate",
    reference="equivariantx.ops.kernels.torch.edge:evaluate_torch",
)

__all__ = ["get_kernel", "register_kernel"]

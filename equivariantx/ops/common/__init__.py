"""PyTorch and CUDA execution providers."""

from ..registry import get_kernel, register_kernel

register_kernel(
    "cartesian_product",
    "torch",
    "equivariantx.ops.common.torch.cartesian:cartesian_product",
)
register_kernel(
    "cartesian_product",
    "cuda",
    "equivariantx.ops.common.cuda.cartesian:cartesian_product",
)

register_kernel("ace_contract", "cuda", "equivariantx.ops.common.cuda.ace:contract")
register_kernel(
    "ace_contract",
    "torch",
    "equivariantx.ops.common.torch.ace:ace_contract",
    reference="equivariantx.ops.common.torch.ace:ace_reference",
)
register_kernel(
    "symmetric_contract", "cuda", "equivariantx.ops.common.cuda.symmetric:contract"
)
register_kernel(
    "symmetric_contract",
    "torch",
    "equivariantx.ops.common.torch.symmetric:contract",
    reference="equivariantx.ops.common.torch.symmetric:evaluate_torch",
)
register_kernel(
    "alignment_cuda", "cuda", "equivariantx.ops.common.cuda.wigner:alignment_cuda"
)
register_kernel(
    "alignment_cuda", "torch", "equivariantx.ops.common.torch.wigner:alignment"
)
register_kernel(
    "bilinear_ace", "cuda", "equivariantx.ops.common.cuda.bilinear:contract"
)
register_kernel(
    "bilinear_ace",
    "torch",
    "equivariantx.ops.common.torch.ace:bilinear_contract",
    reference="equivariantx.ops.common.torch.ace:bilinear_reference",
)
register_kernel(
    "contraction", "cuda", "equivariantx.ops.common.cuda.o2_o3.convolution:contraction"
)
register_kernel(
    "contraction",
    "torch",
    "equivariantx.ops.common.torch.convolution:contraction",
    reference="equivariantx.ops.common.torch.convolution:reference",
)
register_kernel(
    "direction_contraction",
    "cuda",
    "equivariantx.ops.common.cuda.o2_o3.convolution:direction_contraction",
)
register_kernel(
    "direction_contraction",
    "torch",
    "equivariantx.ops.common.torch.geometry:direction_contraction",
)
register_kernel("edge_program", "cuda", "equivariantx.ops.common.cuda.edge:evaluate")
register_kernel(
    "element_linear", "cuda", "equivariantx.ops.common.cuda.linear:contract"
)
register_kernel(
    "element_linear",
    "torch",
    "equivariantx.ops.common.torch.linear:contract",
    reference="equivariantx.ops.common.torch.linear:contract_reference",
)
register_kernel(
    "frame_rotation", "cuda", "equivariantx.ops.common.cuda.rotation:contract"
)
register_kernel(
    "frame_rotation", "torch", "equivariantx.ops.common.torch.frame:frame_rotation"
)
register_kernel("gather_sum", "cuda", "equivariantx.ops.common.cuda.layout:gather_sum")
register_kernel(
    "gather_sum", "torch", "equivariantx.ops.common.torch.layout:gather_sum"
)
register_kernel(
    "graph_normalized_exp",
    "cuda",
    "equivariantx.ops.common.cuda.attention:normalized_exp",
)
register_kernel(
    "graph_normalized_exp",
    "torch",
    "equivariantx.ops.common.torch.attention:normalized_exp",
    reference="equivariantx.ops.common.torch.attention:normalized_exp_reference",
)
register_kernel(
    "graph_normalized_exp_adjoint",
    "cuda",
    "equivariantx.ops.common.cuda.attention:evaluate_adjoint",
)
register_kernel(
    "graph_normalized_exp_adjoint",
    "torch",
    "equivariantx.ops.common.torch.attention:evaluate_adjoint",
)
register_kernel(
    "grouped_permute", "cuda", "equivariantx.ops.common.cuda.layout:grouped_permute"
)
register_kernel(
    "grouped_permute", "torch", "equivariantx.ops.common.torch.layout:grouped_permute"
)
register_kernel("indexed_frame", "cuda", "equivariantx.ops.common.cuda.frame:contract")
register_kernel(
    "indexed_frame", "torch", "equivariantx.ops.common.torch.frame:indexed_frame"
)
register_kernel(
    "indexed_sum", "cuda", "equivariantx.ops.common.cuda.layout:indexed_sum"
)
register_kernel(
    "indexed_sum", "torch", "equivariantx.ops.common.torch.layout:indexed_sum"
)
register_kernel(
    "local_channel_product",
    "cuda",
    "equivariantx.ops.common.cuda.channel_product:channel_product",
)
register_kernel(
    "local_channel_product",
    "torch",
    "equivariantx.ops.common.torch.channel_product:channel_product",
    reference="equivariantx.ops.common.torch.channel_product:reference",
)
register_kernel(
    "o3_contraction", "cuda", "equivariantx.ops.common.cuda.o3.convolution:contraction"
)
register_kernel(
    "o3_contraction",
    "torch",
    "equivariantx.ops.common.torch.o3:contraction",
    reference="equivariantx.ops.common.torch.o3:reference",
)
register_kernel(
    "packed_wigner", "cuda", "equivariantx.ops.common.cuda.wigner:packed_wigner"
)
register_kernel(
    "packed_wigner", "torch", "equivariantx.ops.common.torch.wigner:packed_wigner"
)
register_kernel(
    "quaternion_polynomial",
    "cuda",
    "equivariantx.ops.common.cuda.quaternion:quaternion_polynomial",
)
register_kernel(
    "quaternion_polynomial",
    "torch",
    "equivariantx.ops.common.torch.quaternion:quaternion_polynomial",
)
register_kernel(
    "rotary_product", "cuda", "equivariantx.ops.common.cuda.rotary:contract"
)
register_kernel(
    "rotary_product", "torch", "equivariantx.ops.common.torch.frame:rotary_product"
)
register_kernel("rotation", "cuda", "equivariantx.ops.common.cuda.wigner:rotation")
register_kernel("rotation", "torch", "equivariantx.ops.common.torch.wigner:rotation")
register_kernel(
    "tece_contraction",
    "cuda",
    "equivariantx.ops.common.cuda.models.tace.interaction:contraction",
)
register_kernel(
    "tece_interaction",
    "cuda",
    "equivariantx.ops.common.cuda.models.tace.interaction:interaction",
)
register_kernel(
    "edge_program",
    "torch",
    "equivariantx.ops.common.torch.edge:evaluate",
    reference="equivariantx.ops.common.torch.edge:evaluate_torch",
)

__all__ = ["get_kernel", "register_kernel"]

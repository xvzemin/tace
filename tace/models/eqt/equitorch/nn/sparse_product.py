from typing import Union

import torch

from tace.models.eqt import segment_csr
from tace.utils.torch_scatter import scatter_sum
from ..structs import SparseProductInfo


def indexed_mul_scale_gather_cpu(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, segment_index=None, gather_index=None,
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None):
    # Handle input indexing
    if index1 is not None:
        input1 = input1.index_select(-2, index1)
    if index2 is not None:
        input2 = input2.index_select(-2, index2)
    
    # Core multiplication
    inter = input1 * input2
    
    # Apply scaling if provided
    if scale is not None:
        inter = inter * scale.unsqueeze(-1)
    
    # Handle segmentation/gathering
    if seg is not None:
        if gather_index is not None:
            inter = inter.index_select(-2, gather_index)
        if segment_csr is not None:
            inter = segment_csr(inter, seg.unsqueeze(0), reduce="sum")
        else:
            if segment_index is None:
                raise RuntimeError("Missing segment indices for the EQT fallback.")
            inter = scatter_sum(
                inter,
                segment_index,
                dim=-2,
                dim_size=out_size,
            )
    
    # Handle output indexing
    if index_out is not None:
        inter = scatter_sum(inter, index_out, dim=-2, dim_size=out_size)
    
    # Handle accumulation
    if out_accumulated:
        inter = inter.sum(dim=0)
    
    return inter


def indexed_mul_scale_gather(
        input1, input2, 
        scale=None, index1=None, index2=None,
        seg=None, segment_index=None, gather_index=None,
        index_out=None, out=None,
        out_accumulated=False,
        out_size=None,
    ):

        return indexed_mul_scale_gather_cpu(
            input1, input2, scale, index1, index2,
            seg, segment_index, gather_index, index_out, out,
            out_accumulated, out_size)


class SparseMul(torch.autograd.Function):

    @staticmethod
    def forward(
        ctx, 
        input1: torch.Tensor, 
        input2: torch.Tensor,  
        info_fwd: SparseProductInfo, 
        info_bwd1: Union[SparseProductInfo, None] = None, 
        info_bwd2: Union[SparseProductInfo, None] = None, 
        out_accumulated: bool = False
    ) -> torch.Tensor:
        ret = indexed_mul_scale_gather(
            input1, input2,
            info_fwd.scale,
            info_fwd.index1,
            info_fwd.index2,
            info_fwd.seg_out,
            info_fwd.segment_index,
            info_fwd.gather_index,
            info_fwd.index_out,
            out_accumulated=out_accumulated,
            out_size=info_fwd.out_size,
            )
        ctx.save_for_backward(input1 if input2.requires_grad else None, input2 if input1.requires_grad else None)
        ctx.infos = (info_fwd, info_bwd1, info_bwd2)
        # Determine shared status based on input dimensions heuristic
        ctx.shared1 = input1.ndim < 3
        ctx.shared2 = input2.ndim < 3
        ctx.out_accumulated = out_accumulated # Save for backward logic
        return ret

    @staticmethod
    def backward(ctx, grad_output):
        if grad_output is None:
            return None, None, None, None, None, None
            
        grad = grad_output
        input1, input2 = ctx.saved_tensors
        info_fwd, info_bwd1, info_bwd2 = ctx.infos
        left_shared_fwd = ctx.shared1
        right_shared_fwd = ctx.shared2

        grad1, grad2 = None, None
        if ctx.needs_input_grad[0]:
            # gx = op_bx(y, gz) -> out_accumulated_bx = left_shared_fwd
            out_accumulated_bwd1 = left_shared_fwd
            grad1 = SparseMul.apply(input2, grad, info_bwd1, info_bwd2, info_fwd, out_accumulated_bwd1)

        if ctx.needs_input_grad[1]:
            # gy = op_by(gz, x) -> out_accumulated_by = right_shared_fwd
            out_accumulated_bwd2 = right_shared_fwd
            grad2 = SparseMul.apply(grad, input1, info_bwd2, info_fwd, info_bwd1, out_accumulated_bwd2)
        else:
            grad2 = None

        # Return grads corresponding to input1, input2, out_accumulated, info_fwd, info_bwd1, info_bwd2
        return grad1, grad2, None, None, None, None


def sparse_mul(
        input1: torch.Tensor, 
        input2: torch.Tensor, 
        info_fwd: SparseProductInfo, 
        info_bwd1: Union[SparseProductInfo, None] = None, 
        info_bwd2: Union[SparseProductInfo, None] = None, 
        out_accumulated: bool = False
    ) -> torch.Tensor:
    r"""
    Computes sparse element-wise multiplication using indexed operations.

    This function performs an element-wise product of two input tensors, ``input1`` and ``input2``,
    based on the indexing and scaling information provided in ``info_fwd``.
    The operation is "sparse" in the sense that it uses predefined indexing schemes
    to select elements for multiplication, rather than a dense matrix multiplication.

    The backward pass information can be optionally provided via ``info_bwd1`` and ``info_bwd2``
    for custom gradient calculations if needed.

    Args:
        input1 (torch.Tensor): The first input tensor.
        input2 (torch.Tensor): The second input tensor.
        info_fwd (SparseProductInfo): Contains scaling factors, indices for ``input1`` and ``input2``,
            output segmentation, gather indices, and output indices for the forward pass.
        info_bwd1 (SparseProductInfo, optional): Information for the backward pass with respect to ``input1``.
            Defaults to None.
        info_bwd2 (SparseProductInfo, optional): Information for the backward pass with respect to ``input2``.
            Defaults to None.
        out_accumulated (bool, optional): If ``True``, the output is accumulated into an existing tensor
            (not fully supported by all underlying ops, behavior might vary). Defaults to ``False``.

    Returns:
        torch.Tensor: The result of the sparse element-wise multiplication.
    """
    return SparseMul.apply(input1, input2, info_fwd, info_bwd1, info_bwd2, out_accumulated)

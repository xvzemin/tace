"""Fused signed permutations and sheet transforms for Fourier grids."""

import torch

from ....utils.autograd import forward_ad_active
from ...precision import validate_uniform
from ..torch import grid as torch_kernel
from .pointwise import evaluate as pointwise


@torch.library.custom_op("eqx::grid_sheet", mutates_args=(), device_types="cuda")
def contract(
    features: torch.Tensor,
    input_index: torch.Tensor,
    output_index: torch.Tensor,
    input_sign: torch.Tensor,
    output_sign: torch.Tensor,
    sheets: int,
    width: int,
    reverse: bool = False,
) -> torch.Tensor:
    """Fuse signed permutations with the two- or four-sheet transform."""
    validate_uniform((features,))
    size = features.shape[-1]
    tables = input_index, output_index, input_sign, output_sign
    if (
        sheets not in (2, 4)
        or width < 1
        or size % (sheets * width)
        or any(v.ndim != 1 or v.numel() != size for v in tables)
    ):
        raise ValueError("Sheet permutation dimensions do not match the features.")
    features = features.contiguous()
    tables = tuple(v.contiguous() for v in tables)
    result = torch.empty_like(features)
    if not result.numel():
        return result
    destination = "p2[local]" if reverse else "local"
    source = "position" if reverse else "p1[position]"
    sign = "p4[local]" if reverse else "p3[position]"
    body = f"""
        I row=i/{size},local=i%{size},dest={destination};
        I sheet=(dest/{width})%{sheets},base=dest/{sheets * width}*{sheets * width}+dest%{width};
        T value=0;
        for(int k=0;k<{sheets};++k) {{
            I position=base+k*{width}; int parity=(sheet&k); parity=(parity&1)^(parity>>1);
            value+=p0[row*{size}+{source}]*T({sign})*T(1-2*parity);
        }}
        out[i]=value*T({sheets**-0.5:.17g});
    """
    return pointwise(result, (features, *tables), body)


@contract.register_fake
def fake(
    features,
    input_index,
    output_index,
    input_sign,
    output_sign,
    sheets,
    width,
    reverse=False,
):
    return torch.empty_like(features, memory_format=torch.contiguous_format)


def setup_context(ctx, inputs, output):
    _, *tables, ctx.sheets, ctx.width, ctx.reverse = inputs
    ctx.save_for_backward(*tables)


def backward(ctx, gradient):
    return (
        contract(gradient, *ctx.saved_tensors, ctx.sheets, ctx.width, not ctx.reverse),
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


contract.register_autograd(backward, setup_context=setup_context)


def evaluate(
    features,
    input_index,
    output_index,
    input_sign,
    output_sign,
    sheets,
    width,
    reverse=False,
):
    """Retain the native kernel in joint graphs and exact Torch functional transforms."""
    if forward_ad_active():
        return torch_kernel.evaluate(
            features,
            input_index,
            output_index,
            input_sign,
            output_sign,
            sheets,
            width,
            reverse,
        )
    return contract(
        features,
        input_index,
        output_index,
        input_sign,
        output_sign,
        sheets,
        width,
        reverse,
    )


@torch.library.custom_op("eqx::grid_linear", mutates_args=(), device_types="cuda")
def linear_contract(features: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
    """Use the shared native GEMM while retaining a constant-basis adjoint."""
    from .matmul import matmul

    return matmul(features.unsqueeze(-2), matrix.T).squeeze(-2)


@linear_contract.register_fake
def linear_fake(features, matrix):
    return features.new_empty((*features.shape[:-1], matrix.shape[0]))


def linear_context(ctx, inputs, output):
    ctx.save_for_backward(inputs[1])


def linear_backward(ctx, gradient):
    return linear_contract(gradient, ctx.saved_tensors[0].T), None


linear_contract.register_autograd(linear_backward, setup_context=linear_context)


def linear(features, matrix):
    if matrix.requires_grad or forward_ad_active():
        return torch_kernel.linear(features, matrix)
    return linear_contract(features, matrix)

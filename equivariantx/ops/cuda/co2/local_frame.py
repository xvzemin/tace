"""Native Cartesian frame maps sharing the rectangular tensor-axis kernel."""

from functools import lru_cache

import torch

from ....utils.autograd import forward_ad_active
from ... import config
from ...common.cuda.pointwise import evaluate as pointwise
from ...common.cuda.tensor_axis import evaluate, inline, transform
from ...common.torch.tensor_axis import primal
from ...precision import validate_precision
from ...torch.co2 import local_frame as torch_kernel
from .frame_matrix import evaluate as matrix_kernel


@lru_cache(maxsize=64)
def row_program(codes, transpose):
    """Build a signed Cartesian gather or a disjoint reduction per output."""
    if not codes or len(codes) > 32 or 0 in codes:
        raise ValueError("A Cartesian frame needs 1 to 32 nonzero row indices.")
    count = max(abs(k) for k in codes)
    if count > len(codes) or set(map(abs, codes)) != set(range(1, count + 1)):
        raise ValueError("Cartesian row indices must cover every input row.")
    if not transpose:
        body = f"""
        constexpr int codes[{len(codes)}]={{{",".join(map(str, codes))}}};
        int k=codes[c]; T v=p0[row*{count}+(k<0?-k:k)-1]; out[i]=k<0?-v:v;
        """
        return count, len(codes), body
    cases = []
    for k in range(1, count + 1):
        terms = "".join(
            f"{'+' if code > 0 else '-'}p0[row*{len(codes)}+{col}]"
            for col, code in enumerate(codes)
            if abs(code) == k
        )
        cases.append(f"case {k - 1}: out[i]={terms}; break;")
    return len(codes), count, "switch(c){" + "".join(cases) + "}"


@torch.library.custom_op(
    "eqx::cartesian_frame_rows", mutates_args=(), device_types="cuda"
)
def repeated_rows(
    features: torch.Tensor, codes: list[int], transpose: bool
) -> torch.Tensor:
    """Apply fixed signed rows without expanding features or using atomics."""
    validate_precision((features,))
    width, outputs, body = row_program(tuple(codes), transpose)
    if features.ndim < 1 or features.shape[-1] != width:
        raise ValueError("Cartesian features do not match the frame row map.")
    features = features.contiguous()
    result = features.new_empty((*features.shape[:-1], outputs))
    return pointwise(
        result, (features,), f"I row=i/{outputs};int c=i%{outputs};" + body
    )


@repeated_rows.register_fake
def fake_rows(features, codes, transpose):
    outputs = max(map(abs, codes)) if transpose else len(codes)
    return features.new_empty((*features.shape[:-1], outputs))


def setup_rows(ctx, inputs, output):
    _, ctx.codes, ctx.transpose = inputs


def rows_backward(ctx, gradient):
    return repeated_rows(gradient, ctx.codes, not ctx.transpose), None, None


repeated_rows.register_autograd(rows_backward, setup_context=setup_rows)


def to_local(module, features, rotation):
    """Contract every retained planar order with native tensor axes."""
    return torch_kernel.to_local(
        module,
        features,
        rotation,
        primal if inline() else evaluate,
        matrix_kernel if config.CUDA_CARTESIAN_FRAME_MATRIX else None,
        None if forward_ad_active() else repeated_rows,
    )


def to_global(module, features, rotation):
    """Reconstruct global Cartesian features with native tensor axes."""
    return torch_kernel.to_global(
        module,
        features,
        rotation,
        primal if inline() else evaluate,
        matrix_kernel if config.CUDA_CARTESIAN_FRAME_MATRIX else None,
        None if forward_ad_active() else repeated_rows,
    )


def rotate(module, features, rotation):
    """Rotate every tensor index with the shared native rectangular map."""
    return torch_kernel.rotate(
        module, features, rotation, primal if inline() else evaluate
    )


def transform_tensor(tensor, matrix, rank):
    """Apply native rectangular maps to a flattened Cartesian tensor."""
    return transform(tensor, matrix, rank)

"""Compiler-visible schedules for selected native operator calls."""

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from ..utils.metadata import parse_metadata
from . import config

SYMMETRIC_INLINE_LIMIT = 65536


def torch_decompositions():
    """Return compiler-visible polynomial, geometry and edge-expression kernels.

    Returns
    -------
    dict
        Explicit make_fx lowerings before differentiation and AOTI export.
        O3 paths support explicit attributes, solid harmonics and transverse
        polynomials, including ordered Lie derivatives.
        Large expanded programs may use more memory than native kernels.
    """
    from .common.torch.attention import evaluate_adjoint, normalized_exp
    from .common.torch.convolution import contraction
    from .common.torch.frame import frame_rotation, indexed_frame, rotary_product
    from .common.torch.layout import gather_sum, grouped_permute, indexed_sum
    from .common.torch.quaternion import quaternion_polynomial
    from .common.torch.wigner import alignment, packed_wigner, rotation

    return {
        **polynomial_decompositions(),
        torch.ops.eqx.edge_program.default: edge_decomposition,
        torch.ops.eqx.alignment_cuda.default: alignment,
        torch.ops.eqx.packed_wigner.default: packed_wigner,
        torch.ops.eqx.rotation.default: rotation,
        torch.ops.eqx.quaternion_polynomial.default: quaternion_polynomial,
        torch.ops.eqx.graph_normalized_exp.default: normalized_exp,
        torch.ops.eqx.graph_normalized_exp_adjoint.default: evaluate_adjoint,
        torch.ops.eqx.gather_sum.default: gather_sum,
        torch.ops.eqx.grouped_permute.default: grouped_permute,
        torch.ops.eqx.indexed_sum.default: indexed_sum,
        torch.ops.eqx.indexed_frame.default: indexed_frame,
        torch.ops.eqx.frame_rotation.default: frame_rotation,
        torch.ops.eqx.rotary_product.default: rotary_product,
        torch.ops.eqx.contraction.default: contraction,
        torch.ops.eqx.o3_contraction.default: o3_decomposition,
    }


def o3_decomposition(metadata, program, source, target, operands, *, batch_paths=False):
    """Retain native calls for geometric programs without an exact Torch lowering."""
    if not o3_lowerable(metadata):
        return NotImplemented
    from .common.torch.o3 import contraction

    return contraction(
        metadata, program, source, target, operands, batch_paths=batch_paths
    )


def o3_lowerable(metadata):
    """Whether a native description has the same polynomial in both providers."""
    description = parse_metadata(metadata)
    if len(description) > 3 and (
        isinstance(description[3], tuple)
        and description[3][0] == "transverse"
        and (
            len(description[3]) >= 3
            and description[3][2] not in ("baseline", "generator", "recurrence")
        )
    ):
        return False
    return True


def edge_decomposition(metadata, inputs, source, target, num_nodes):
    """Expose radial networks and supported embedded convolution expressions."""
    from .common.torch.edge import evaluate_torch

    for op, _, _, data in parse_metadata(metadata)[0]:
        if op == "convolution" and (
            data[0] not in ("o3", "o2") or data[0] == "o3" and not o3_lowerable(data[1])
        ):
            return NotImplemented
    return evaluate_torch(
        metadata,
        inputs,
        source,
        target,
        num_nodes,
        optimized=not config.TORCH_REFERENCE,
    )


def polynomial_decompositions():
    """Return explicit Torch lowerings for native polynomial export.

    Returns
    -------
    dict
        Decomposition table for ``make_fx`` before export. Includes all
        transposed roles. Expanded large products
        can use more memory than the native schedule; ordinary execution
        retains its measured dispatch policy.
    """
    from . import registration  # noqa: F401 - Ensure the schemas exist.
    from .common.torch.ace import ace_contract, bilinear_contract
    from .common.torch.channel_product import channel_product
    from .common.torch.linear import contract
    from .common.torch.symmetric import evaluate_torch

    return {
        torch.ops.eqx.ace_contract.default: ace_contract,
        torch.ops.eqx.bilinear_ace.default: bilinear_contract,
        torch.ops.eqx.element_linear.default: contract,
        torch.ops.eqx.local_channel_product.default: channel_product,
        torch.ops.eqx.symmetric_contract.default: evaluate_torch,
    }


def symmetric_forward(metadata, program, node_type, operands):
    """Inline small primal polynomials; retain native large or adjoint kernels."""
    degree = parse_metadata(metadata)[3]
    terms = parse_metadata(program)
    if len(terms) != 1 or terms[0][1] != degree + 1 or terms[0][3]:
        return NotImplemented
    if not statically_known_true(
        node_type.shape[0] * operands[0].shape[-1] <= SYMMETRIC_INLINE_LIMIT
    ):
        return NotImplemented
    from .common.torch.symmetric import evaluate_torch

    return evaluate_torch(metadata, program, node_type, operands)


def register_symmetric_decomposition():
    """Expose supported CUDA primals to Inductor without replacing eager kernels."""
    from torch._inductor.decomposition import register_decomposition
    from torch._inductor.lowering import make_fallback

    operator = torch.ops.eqx.symmetric_contract.default
    register_decomposition(operator)(symmetric_forward)
    make_fallback(operator, warn=False, override_decomp=True)

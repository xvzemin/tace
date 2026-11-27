"""CUDA local updates with receiver-normalized rotary attention."""

from functools import lru_cache

import torch
from torch.fx.experimental.symbolic_shapes import statically_known_true

from equivariantx.ops import get_kernel, resolve_backend
from equivariantx.utils.autograd import forward_ad_active
from equivariantx.utils.metadata import parse_metadata, static_dimensions
from equivariantx_foundation.ops.tece import interaction

from .expressions import build, operand_specs


@lru_cache(maxsize=64)
@torch.compiler.assume_constant_result
def _radial_roots(
    metadata, radial, basis, angular_in, angular_out, packed, separate_target=False
):
    nodes, score, value, specs, *settings = build(
        metadata, radial, basis, angular_in, angular_out, packed, separate_target
    )
    roots = repr((nodes, ((score, 0, "edge"), (value, 1, "edge"))))
    return roots, specs, tuple(settings)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def _interaction_program(metadata, radial, specs, settings, count, target_slot=None):
    nodes, outputs = parse_metadata(radial)
    specs = operand_specs(nodes, outputs, specs + (None,) * (count - len(specs)))
    description = (metadata, (nodes, outputs[0][0], outputs[1][0], specs, *settings))
    return repr(description if target_slot is None else (*description, target_slot))


def stream(
    module,
    features,
    radial,
    projection,
    bias,
    edge_index,
    cutoff,
    wigner,
    wigner_inv,
    radial_basis,
    *,
    radial_network=None,
    backend=None,
    num_nodes=None,
):
    """Fuse the complete edge update without graph replay or Python callbacks.

    Edge tiles use matrix products for dense contractions and CUDA kernels
    for intervening expressions. Only attention scores are retained per edge.
    Other intermediates are recomputed in bounded tiles. CUDA frame arguments
    may contain dense matrices or packed degree blocks shared by both directions.
    With ``num_nodes``, only this prefix of the full source table receives
    messages and owns softmax statistics.
    """
    from equivariantx.ops.common.radial import cast_radial

    radial = cast_radial(radial, features.dtype)
    projection = projection.to(features)
    wigner = wigner.to(features)
    wigner_inv = wigner_inv.to(features)
    radial_basis = radial_basis.to(features)
    if bias is not None:
        bias = bias.to(features)
    if cutoff is not None:
        cutoff = cutoff.to(features)
    if bias is None:
        bias = projection.new_zeros(projection.shape[1])
    if cutoff is None:
        cutoff = features.new_ones((edge_index.shape[1], 1))
    if module._eqx_metadata is None:
        if radial_network is not None:
            from equivariantx.ops.common.radial import materialize

            radial = radial_network(materialize(radial, edge_index))
        result = module(
            features,
            radial @ projection + bias,
            edge_index,
            cutoff,
            wigner,
            wigner_inv,
            radial_basis,
        )
        return result if num_nodes is None else result[:num_nodes]
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("The fused local interaction requires float32 or float64.")
    if hasattr(module, "prepare_stream"):
        kernel_features, wigner, wigner_inv = module.prepare_stream(
            features, wigner, wigner_inv
        )
    else:
        kernel_features = module.reshape_in(features)
    inputs = [
        kernel_features,
        radial,
        projection,
        bias,
        cutoff,
        wigner,
        wigner_inv,
        radial_basis,
        *module.parameters(),
    ]
    metadata = module._eqx_metadata
    separate_target = num_nodes is not None and not statically_known_true(
        num_nodes == kernel_features.shape[0]
    )
    if radial_network is not None or separate_target:
        encode_roots = (
            _radial_roots.__wrapped__
            if torch.compiler.is_compiling()
            else _radial_roots
        )
        encode_interaction = (
            _interaction_program.__wrapped__
            if torch.compiler.is_compiling()
            else _interaction_program
        )
        roots, specs, settings = encode_roots(
            metadata,
            static_dimensions(projection, (0,))[0],
            static_dimensions(radial_basis, (1,))[0],
            static_dimensions(inputs[0], (1,))[0],
            (module.lmax + 1) ** 2
            if wigner.ndim == 2
            else static_dimensions(wigner_inv, (1,))[0],
            wigner.ndim == 2,
            separate_target,
        )
        target_slot = None
        if separate_target:
            target_slot = len(inputs)
            inputs.append(kernel_features.narrow(0, 0, num_nodes))
        if radial_network is not None:
            from equivariantx.ops.common.radial import compose_radial

            roots, inputs = compose_radial(
                roots,
                inputs,
                1,
                radial_network,
                num_targets=num_nodes if separate_target else None,
            )
        metadata = encode_interaction(
            metadata, roots, specs, settings, len(inputs), target_slot
        )
    backend = resolve_backend(backend, features.device)
    if backend == "cuda" and not features.is_cuda:
        raise ValueError("backend='cuda' requires CUDA inputs.")
    if backend == "cuda" and forward_ad_active():
        from equivariantx_foundation.ops.torch.tece import interaction as execute
    elif (
        backend == "cuda"
        and not separate_target
        and getattr(module, "save_force_primals", False)
    ):
        from .force import capped_program
        from .force import saved_interaction as execute

        if getattr(module, "primal_edge_limit", None) is not None:
            metadata = capped_program(metadata, module.primal_edge_limit)
    else:
        execute = (
            interaction
            if backend == "cuda"
            else get_kernel("tece_interaction", "torch")
        )
    result = execute(
        metadata,
        edge_index[0],
        edge_index[1],
        inputs,
    )[0]
    return (
        module.finish_stream(result)
        if hasattr(module, "finish_stream")
        else module.reshape_out.inverse(result)
    )

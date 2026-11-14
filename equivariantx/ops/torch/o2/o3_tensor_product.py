"""Aligned spherical products without magnetic copies of channel weights."""

import math

import torch
from e3nn import o3
from torch.fx.experimental.proxy_tensor import get_proxy_mode

from ... import config
from ...common.sparse_coefficients import bilinear_tables
from ...common.torch import sparse_bilinear, sparse_chain


def initialize(module, input_components, output_components):
    """Preserve one channel matrix per instruction and fixed output maps."""
    paths, entries = [], []
    weighted, unweighted, weight_harmonics = [], [], []
    channelwise = all(ins.connection_mode == "uvu" for ins in module.instructions)
    weight_offset = projected_offset = 0
    for ins in module.instructions:
        mul1, ir1 = module.irreps_in1[ins.i_in1]
        _, ir2 = module.irreps_in2[ins.i_in2]
        mul_out, ir_out = module.irreps_out[ins.i_out]
        size = math.prod(ins.path_shape)
        weight_start = weight_offset if ins.has_weight else -1
        weight_offset += size if ins.has_weight else 0
        if ins.has_weight:
            weight_harmonics.extend([ins.i_in2] * size)
        if not mul1 or not mul_out or not ins.path_weight:
            continue
        cg = o3.wigner_3j(ir1.l, ir2.l, ir_out.l, dtype=torch.float64, device="cpu")[
            :, ir2.l, :
        ]
        pole = 1.0 if module.normalization == "norm" else math.sqrt(2 * ir2.l + 1)
        if module.normalization == "integral":
            pole /= math.sqrt(4 * math.pi)
        indices = []
        count = 0
        for m_out in range(-min(ir1.l, ir_out.l), min(ir1.l, ir_out.l) + 1):
            for m_in in sorted({m_out, -m_out}):
                coefficient = (
                    float(cg[ir1.l + m_in, ir_out.l + m_out]) * pole * ins.path_weight
                )
                if not coefficient:
                    continue
                source = input_components[ins.i_in1][m_in]
                destination = output_components[ins.i_out][m_out]
                indices.extend(range(source, source + mul1))
                if channelwise:
                    target = weighted if ins.has_weight else unweighted
                    target.extend(
                        (
                            source + channel,
                            weight_start + channel if ins.has_weight else ins.i_in2,
                            destination + channel,
                            coefficient,
                        )
                        for channel in range(mul1)
                    )
                entries.extend(
                    (
                        projected_offset + count * mul_out + channel,
                        0,
                        destination + channel,
                        coefficient,
                    )
                    for channel in range(mul_out)
                )
                count += 1
        if not count:
            continue
        name = f"path_input_{len(paths)}"
        module.register_buffer(
            name,
            torch.tensor(indices, dtype=torch.long, device=module.weight.device),
            persistent=False,
        )
        paths.append(
            (
                ins.connection_mode,
                mul1,
                mul_out,
                count,
                weight_start,
                size,
                ins.i_in2,
                name,
            )
        )
        projected_offset += count * mul_out
    module.local_paths = tuple(paths)
    module.projected_dim = projected_offset
    module.mixing_size = max(
        (ci * co for mode, ci, co, *_ in paths if mode == "uvw"), default=0
    )
    module.coefficient_data = {}
    tables = bilinear_tables(
        (projected_offset, 1, module.local_output_dim), tuple(entries)
    )
    for suffix, value in zip(
        ("indices", "coefficients", "transpose_indices", "transpose_coefficients"),
        (tables[6], tables[8], tables[1], tables[2]),
    ):
        name = "readout_" + suffix
        module.register_buffer(
            name,
            value.to(
                device=module.weight.device,
                dtype=module.weight.dtype if value.is_floating_point() else value.dtype,
                copy=True,
            ),
            persistent=False,
        )
        if value.is_floating_point():
            module.coefficient_data[name] = value

    module.channelwise = channelwise
    module.direct_groups = ()
    if channelwise:
        module.register_buffer(
            "weight_harmonics",
            torch.tensor(
                weight_harmonics, dtype=torch.long, device=module.weight.device
            ),
            persistent=False,
        )
        groups = []
        for label, dimensions, data in (
            (
                "weighted",
                (
                    module.local_frame_in.output_dim,
                    module.weight_numel,
                    module.local_output_dim,
                ),
                weighted,
            ),
            (
                "unweighted",
                (
                    module.local_frame_in.output_dim,
                    module.num_harmonics,
                    module.local_output_dim,
                ),
                unweighted,
            ),
        ):
            if not data:
                continue
            groups.append(label)
            for index, value in enumerate(bilinear_tables(dimensions, tuple(data))):
                name = f"direct_{label}_{index}"
                module.register_buffer(
                    name,
                    value.to(
                        device=module.weight.device,
                        dtype=module.weight.dtype
                        if value.is_floating_point()
                        else value.dtype,
                        copy=True,
                    ),
                    persistent=False,
                )
                if value.is_floating_point():
                    module.coefficient_data[name] = value
        module.direct_groups = tuple(groups)


def reference(module, features, weight, harmonic_scale=None):
    leading_shape = torch.broadcast_shapes(
        features.shape[:-1],
        weight.shape[:-1],
        () if harmonic_scale is None else harmonic_scale.shape[:-1],
    )
    if module._has_unweighted:
        weight = torch.cat((weight, weight.new_ones((*weight.shape[:-1], 1))), dim=-1)
    output = None
    for index, (mode, mul1, mul_out) in enumerate(module._contractions):
        scale = getattr(module, f"scale_{index}")
        count = scale.numel()
        values = features.index_select(-1, getattr(module, f"input_{index}"))
        values = values.reshape(*features.shape[:-1], count, mul1)
        weights = weight.index_select(-1, getattr(module, f"weight_{index}"))
        if mode == "uvu":
            values = values * weights.reshape(*weight.shape[:-1], count, mul1)
        else:
            weights = weights.reshape(*weight.shape[:-1], count, mul1, mul_out)
            values = torch.matmul(values.unsqueeze(-2), weights).squeeze(-2)
        if harmonic_scale is not None:
            scale = scale * harmonic_scale.index_select(
                -1, getattr(module, f"harmonic_{index}")
            )
        values = (values * scale.unsqueeze(-1)).flatten(-2)
        contribution = values.new_zeros(
            (*values.shape[:-1], module.local_output_dim)
        ).index_add_(-1, getattr(module, f"output_{index}"), values)
        output = contribution if output is None else output + contribution
    if output is None:
        zero = features[..., :0].sum() + weight[..., :0].sum()
        if harmonic_scale is not None:
            zero = zero + harmonic_scale[..., :0].sum()
        return features.new_zeros((*leading_shape, module.local_output_dim)) + zero
    return output.expand(*leading_shape, module.local_output_dim)


def program(module, features, weight, harmonic_scale, readout):
    """Share each path GEMM across all nonzero magnetic components."""
    shape = torch.broadcast_shapes(
        features.shape[:-1],
        weight.shape[:-1],
        () if harmonic_scale is None else harmonic_scale.shape[:-1],
    )
    zero = features[..., :0].sum() + weight[..., :0].sum()
    if harmonic_scale is not None:
        zero = zero + harmonic_scale[..., :0].sum()
    if not module.local_paths:
        return features.new_zeros((*shape, module.local_output_dim)) + zero
    projected = []
    for mode, ci, co, count, start, size, harmonic, name in module.local_paths:
        value = features.index_select(-1, getattr(module, name)).unflatten(
            -1, (count, ci)
        )
        if start >= 0:
            w = weight[..., start : start + size]
            if mode == "uvu":
                value = value * w.unsqueeze(-2)
            else:
                value = value @ w.unflatten(-1, (ci, co))
        elif mode == "uvw":
            value = value.sum(-1, keepdim=True).expand(*value.shape[:-1], co)
        if harmonic_scale is not None:
            value = value * harmonic_scale[..., harmonic, None, None]
        projected.append(value.expand(*shape, count, co).flatten(-2))
    value = torch.cat(projected, dim=-1)
    tables = tuple(
        getattr(module, "readout_" + suffix)
        for suffix in (
            "indices",
            "coefficients",
            "transpose_indices",
            "transpose_coefficients",
        )
    )
    return readout(value, (1,), False, *tables) + zero


def direct(module, features, weight, harmonic_scale, contract):
    """Fuse channelwise paths without expanded feature or weight intermediates."""
    shape = torch.broadcast_shapes(
        features.shape[:-1],
        weight.shape[:-1],
        () if harmonic_scale is None else harmonic_scale.shape[:-1],
    )
    zero = features[..., :0].sum() + weight[..., :0].sum()
    if harmonic_scale is not None:
        zero = zero + harmonic_scale[..., :0].sum()
    result = None
    for label in module.direct_groups:
        if label == "weighted":
            operand = (
                weight
                if harmonic_scale is None
                else weight * harmonic_scale.index_select(-1, module.weight_harmonics)
            )
        else:
            operand = (
                features.new_ones((module.num_harmonics,))
                if harmonic_scale is None
                else harmonic_scale
            )
        tables = tuple(getattr(module, f"direct_{label}_{index}") for index in range(9))
        value = contract(features, operand, tables)
        result = value if result is None else result + value
    if result is None:
        return features.new_zeros((*shape, module.local_output_dim)) + zero
    return result.expand(*shape, module.local_output_dim) + zero


def handwritten(module, features, weight, harmonic_scale=None):
    """Apply shared channel matrices and exact fixed-map adjoints."""
    if getattr(module, "channelwise", False):
        return direct(
            module, features, weight, harmonic_scale, sparse_bilinear.evaluate
        )
    return program(module, features, weight, harmonic_scale, sparse_chain.evaluate)


def local(module, features, weight, harmonic_scale=None):
    if (
        config.TORCH_REFERENCE
        or not config.TORCH_ALIGNED_PRODUCT
        or not hasattr(module, "local_paths")
    ):
        return reference(module, features, weight, harmonic_scale)
    tracing = torch.compiler.is_compiling() or get_proxy_mode() is not None
    # Narrow eager paths cost less as one grouped contraction. Compilation
    # removes the channelwise program's Python and intermediate-tensor costs.
    if (module.channelwise and not tracing) or (
        not module.channelwise
        and features.device.type == "cpu"
        and module.mixing_size <= (16 if tracing else 64)
    ):
        return reference(module, features, weight, harmonic_scale)
    return handwritten(module, features, weight, harmonic_scale)


def forward(module, features, wigner, wigner_inv, weight=None, harmonic_scale=None):
    value = module.local_frame_in.to_local(features, wigner)
    value = module.forward_local(value, weight, harmonic_scale)
    value = module.local_frame_out.to_global(value, wigner_inv)
    return (
        value.index_select(-1, module.output_index)
        if module._simplify_output
        else value
    )

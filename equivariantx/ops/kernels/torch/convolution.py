"""Multilinear frame convolutions with ordinary Torch derivatives."""

import torch

from ....utils.metadata import kernel_plan, parse_metadata
from ....utils.outputs import contraction_outputs


def contraction(metadata, program, source, target, operands):
    """Evaluate frame contractions and every transposed factor role."""
    outputs = [
        torch.zeros_like(value) for value in contraction_outputs(program, operands)
    ]
    for mapping, weighted, pairs in parse_metadata(program):
        values = tuple(operands[index] for index in mapping)
        for role, slot in pairs:
            outputs[slot] = outputs[slot] + aligned_pairing(
                metadata, source, target, values, role, weighted
            )
    return outputs


def aligned_pairing(metadata, source, target, values, role, weighted_only):
    """Contract all factors except the selected output, then reduce its owners."""
    plan = kernel_plan(metadata)
    x, radial, projection, incoming, outgoing, amplitudes, output = values
    edges = source.numel()
    projected = projection.numel() != 0
    result = torch.zeros_like(values[role])
    for path_index, ((mode, path), sparse) in enumerate(
        zip(plan.path_data, plan.sparse_paths)
    ):
        (
            start,
            end,
            channels,
            channels_out,
            dim,
            dim_out,
            begin_in,
            begin_out,
            weight,
            harmonic,
        ) = path
        if (weight < 0 and (weighted_only or role in (1, 2))) or (
            role == 2 and not projected
        ):
            continue
        scalar = path_index in plan.scalar_paths
        if scalar and role in (3, 4):
            continue
        coefficients = [[0.0] * dim_out for _ in range(dim)]
        for a, b, coefficient in sparse:
            coefficients[a][b] += coefficient
        cg = x.new_tensor(coefficients)
        tensors = [
            None
            if role == 0
            else x[:, start : start + dim * channels]
            .index_select(0, source)
            .reshape(edges, dim, channels),
            None,
            None,
            None
            if role == 3
            else (
                torch.eye(dim, dtype=x.dtype, device=x.device).expand(edges, -1, -1)
                if scalar
                else incoming[:, begin_in : begin_in + dim * dim]
                .reshape(-1, dim, dim)
                .expand(edges, -1, -1)
            ),
            None
            if role == 4
            else (
                torch.eye(dim_out, dtype=x.dtype, device=x.device).expand(edges, -1, -1)
                if scalar
                else outgoing[:, begin_out : begin_out + dim_out * dim_out]
                .reshape(-1, dim_out, dim_out)
                .expand(edges, -1, -1)
            ),
            None if role == 5 else amplitudes[:, harmonic].expand(edges),
            None
            if role == 6
            else output[:, end : end + dim_out * channels_out]
            .index_select(0, target)
            .reshape(edges, dim_out, channels_out),
        ]
        tokens = ["eau", "eh", "huv", "ema", "enb", "e", "ebv"]
        if mode == "uvu":
            tokens[2], tokens[6] = "hu", "ebu"
            width, weight_shape = channels, (channels,)
        elif mode == "uvw":
            width, weight_shape = channels * channels_out, (channels, channels_out)
        else:
            raise NotImplementedError(f"Frame contractions do not support {mode!r}.")
        if weight >= 0:
            if projected:
                if role != 1:
                    tensors[1] = radial.expand(edges, -1)
                if role != 2:
                    tensors[2] = projection[:, weight : weight + width].reshape(
                        projection.shape[0], *weight_shape
                    )
            else:
                tokens[1] = "eu" if mode == "uvu" else "euv"
                if role != 1:
                    tensors[1] = (
                        radial[:, weight : weight + width]
                        .reshape(-1, *weight_shape)
                        .expand(edges, *weight_shape)
                    )
        elif mode == "uvw":
            tokens[2] = "uv"
            tensors[2] = x.new_ones(weight_shape)
        active = [
            (token, value) for token, value in zip(tokens, tensors) if value is not None
        ]
        term = torch.einsum(
            ",".join([token for token, _ in active] + ["mn"]) + "->" + tokens[role],
            *[value for _, value in active],
            cg,
        )
        if role in (0, 6):
            index = source if role == 0 else target
            begin, size = (
                (start, dim * channels) if role == 0 else (end, dim_out * channels_out)
            )
            term = term.reshape(edges, size)
            term = term.new_zeros((result.shape[0], size)).index_add(0, index, term)
        elif role == 2:
            begin, size = weight, width
            term = term.reshape(projection.shape[0], width)
        else:
            if role == 3:
                begin, size = begin_in, dim * dim
            elif role == 4:
                begin, size = begin_out, dim_out * dim_out
            elif role == 5:
                begin, size = harmonic, 1
            elif projected:
                begin, size = 0, radial.shape[1]
            else:
                begin, size = weight, width
            term = term.reshape(edges, size)
            if result.shape[0] == 1:
                term = term.sum(0, keepdim=True)
        result = result + torch.nn.functional.pad(
            term, (begin, result.shape[1] - begin - size)
        )
    return result + sum(value[..., :0].sum() for value in values)

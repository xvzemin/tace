"""PyTorch reference for direction-dependent contractions."""

from string import ascii_letters

import torch

from ...ops.geometry import angular_coefficients
from ...ops.o2_o3 import kernel_plan


def reference(metadata, rank, output, source, target, operands, result, weighted_only):
    """Reference contraction for directions and arbitrary derivative order."""
    plan = kernel_plan(metadata)
    x, radial, projection, din, dout, amplitudes, y, *vectors = operands
    edges = source.numel()
    projected = projection.numel() != 0
    labels = "".join(letter for letter in ascii_letters if letter not in "aebkmnuv")
    if rank > len(labels):
        raise ValueError("The reference einsum exhausted its angular index labels.")
    for index, (mode, path) in enumerate(plan.path_data):
        start, end, mul, mul_out, dim, dim_out, dstart, dend, weight, harmonic = path
        if ((weighted_only or output in (1, 2)) and weight < 0) or (
            output == 2 and not projected
        ):
            continue
        if rank and index in plan.scalar_paths:
            continue
        cg = angular_coefficients(metadata, rank)[index].to(x)
        if index in plan.scalar_paths:
            di = torch.eye(dim, dtype=x.dtype, device=x.device).expand(edges, -1, -1)
            do = di
        else:
            di = (
                din[:, dstart : dstart + dim * dim]
                .reshape(-1, dim, dim)
                .expand(edges, -1, -1)
            )
            do = (
                dout[:, dend : dend + dim_out * dim_out]
                .reshape(-1, dim_out, dim_out)
                .expand(edges, -1, -1)
            )
        tensors = [
            None
            if output == 0
            else x[source, start : start + mul * dim].reshape(edges, dim, mul),
            None,
            None,
            di,
            do,
            None if output == 5 else amplitudes[:, harmonic].expand(edges),
            None
            if output == 6
            else y[target, end : end + mul_out * dim_out].reshape(
                edges, dim_out, mul_out
            ),
        ]
        tokens = ["eau", "ek", "kuv", "ema", "enb", "e", "ebv"]
        width = mul if mode == "uvu" else mul * mul_out
        weight_shape = (mul,) if mode == "uvu" else (mul, mul_out)
        if mode == "uvu":
            tokens[2], tokens[6] = "ku", "ebu"
        if weight >= 0:
            if projected:
                if output != 1:
                    tensors[1] = radial.expand(edges, -1)
                if output != 2:
                    tensors[2] = projection[:, weight : weight + width].reshape(
                        projection.size(0), *weight_shape
                    )
            else:
                tokens[1] = "eu" if mode == "uvu" else "euv"
                if output != 1:
                    tensors[1] = (
                        radial[:, weight : weight + width]
                        .reshape(-1, *weight_shape)
                        .expand(edges, *weight_shape)
                    )
        if rank:
            rotation = din[:, 1:10].reshape(-1, 3, 3).expand(edges, -1, -1)
            for axis, vector in enumerate(vectors):
                tokens.append("e" + labels[axis])
                tensors.append(
                    None
                    if output == 7 + axis
                    else torch.einsum("eab,eb->ea", rotation, vector.expand(edges, -1))
                )
        active = [
            (token, value) for token, value in zip(tokens, tensors) if value is not None
        ]
        value = torch.einsum(
            ",".join([token for token, _ in active] + ["mn" + labels[:rank]])
            + "->"
            + tokens[output],
            *[value for _, value in active],
            cg,
        )
        if output in (0, 6):
            nodes = source if output == 0 else target
            begin, size = (
                (start, mul * dim) if output == 0 else (end, mul_out * dim_out)
            )
            result[:, begin : begin + size].index_add_(
                0, nodes, value.reshape(edges, size)
            )
        elif output >= 7:
            value = torch.einsum("eab,ea->eb", rotation, value)
            result.add_(value if result.size(0) != 1 else value.sum(0, keepdim=True))
        elif output == 5:
            result[:, harmonic].add_(
                value if result.size(0) != 1 else value.sum().reshape(1)
            )
        elif output == 2:
            result[:, weight : weight + width].add_(
                value.reshape(projection.size(0), width)
            )
        elif projected:
            result.add_(value if result.size(0) != 1 else value.sum(0, keepdim=True))
        else:
            value = value.reshape(edges, width)
            result[:, weight : weight + width].add_(
                value if result.size(0) != 1 else value.sum(0, keepdim=True)
            )

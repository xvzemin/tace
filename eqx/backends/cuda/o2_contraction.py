"""Streaming real O(2) polynomial contractions."""

import math

import torch

from ...o2.clebsch_gordan import coupling_entries
from ...ops.channel_product import channel_product


def contraction_metadata(
    irreps_in,
    paths_by_order,
    scales_by_order,
    input_slices,
    output_slices,
    input_dim,
    output_dim,
    channels,
):
    """Expand analytic coupling paths into scalar monomials with independent weights."""
    correlation = len(paths_by_order)
    rows, cache, weight_index = [], {}, 0
    for paths, scales in zip(paths_by_order, scales_by_order):
        for path, scale in zip(paths, scales):
            for order, leaf in enumerate(path.leaves):
                key = path.leaves[: order + 1], path.intermediates[: order + 1]
                if key in cache:
                    continue
                start = input_slices[leaf].start
                if not order:
                    cache[key] = tuple(
                        {(start + k,): 1.0} for k in range(irreps_in[leaf].dim)
                    )
                    continue
                previous = cache[path.leaves[:order], path.intermediates[:order]]
                result = [{} for _ in range(path.intermediates[order].dim)]
                for i, j, k, coefficient in coupling_entries(
                    path.intermediates[order - 1],
                    irreps_in[leaf],
                    path.intermediates[order],
                ):
                    for monomial, value in previous[i].items():
                        indices = (*monomial, start + j)
                        result[k][indices] = (
                            result[k].get(indices, 0.0) + value * coefficient
                        )
                cache[key] = result
            output = output_slices[path.output_index].start
            for k, polynomial in enumerate(cache[path.leaves, path.intermediates]):
                for indices, coefficient in polynomial.items():
                    if coefficient:
                        rows.append(
                            (
                                (
                                    *indices,
                                    *([-1] * (correlation - len(indices))),
                                    weight_index,
                                    output + k,
                                ),
                                coefficient * scale,
                            )
                        )
            weight_index += 1
    return repr(
        (
            channels,
            (*([input_dim] * correlation), weight_index, output_dim),
            tuple(rows),
        )
    )


def contract(metadata, inputs, weight, shape, output_dim):
    """Evaluate all paths using CUDA and recursive multilinear adjoints."""
    values = (*inputs, weight)
    if any(
        not x.is_cuda or x.dtype not in (torch.float32, torch.float64) for x in values
    ):
        raise ValueError(
            "The CUDA contraction requires float32 or float64 CUDA tensors."
        )
    if any(x.device != inputs[0].device or x.dtype != inputs[0].dtype for x in values):
        raise ValueError("Inputs and weights must have the same device and dtype.")
    rows = math.prod(shape)
    operands = [
        x.reshape(1, x.shape[-1])
        if i and math.prod(x.shape[:-1]) == 1
        else x.expand(*shape, x.shape[-1]).reshape(rows, x.shape[-1])
        for i, x in enumerate(values)
    ]
    operands.append(inputs[0].new_empty((0, output_dim)))
    role = len(values)
    program = repr(((tuple(range(role + 1)), role, 0, 0),))
    return channel_product(metadata, program, operands)[0].reshape(*shape, output_dim)

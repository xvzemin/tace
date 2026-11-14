"""Spherical CG convolution in angular-first storage."""

import torch

from e3nn import o3
from .... import config


def evaluate(
    module,
    features,
    edge_attrs,
    radial,
    projection,
    edge_index,
    num_nodes,
    vectors,
    amplitudes,
    radial_network,
):
    """Evaluate validated convolution operands."""
    if vectors is not None:
        sections, offset = ([], 0)
        for mul, degree in module.harmonic_entries:
            harmonic = o3.spherical_harmonics(
                degree, vectors, normalize=False, normalization=module.normalization
            )
            sections.append(
                (
                    harmonic.unsqueeze(-1) * amplitudes[:, None, offset : offset + mul]
                ).flatten(1)
            )
            offset += mul
        edge_attrs = torch.cat(sections, dim=-1)
    edge_attrs = edge_attrs.to(features)
    if config.TORCH_REFERENCE:
        result = module.reference(
            features, edge_attrs, radial, projection, edge_index, num_nodes
        )
    else:
        from equivariantx.ops.common.convolution import convolve

        result = convolve(
            module.kernel_metadata,
            features,
            edge_attrs,
            radial,
            projection,
            edge_index,
            num_nodes,
            module.output_dim,
            network=radial_network,
            recompute=module.radial_mode == "full",
        )
    return result if vectors is None else result + vectors.sum() * 0


def reference(module, features, edge_attrs, radial, projection, edge_index, num_nodes):
    """Evaluate validated convolution operands."""
    indexed = edge_index is not None
    edges = edge_index.size(1) if indexed else features.size(0)
    if indexed:
        source, target = edge_index
    weights = radial @ projection if projection.numel() else radial
    weights = weights.expand(edges, -1)
    attrs = edge_attrs.expand(edges, -1)
    zero = sum(
        (value.sum() * 0 for value in (features, edge_attrs, radial, projection))
    )
    if module.grouped_reference:
        message = module.reference_contraction(
            features[source] if indexed else features,
            attrs,
            weights,
            module.reference_scales.to(features),
        )
    else:
        inputs, angular, outputs = ({}, {}, {})
        for index, path in enumerate(module.paths):
            start, attr, end, mul1, mul2, dim1, dim2, dim_out, weight, factor, _ = path
            key = (start, mul1, dim1)
            if key not in inputs:
                x = features[:, start : start + dim1 * mul1]
                inputs[key] = (x[source] if indexed else x).reshape(edges, dim1, mul1)
            other = (attr, mul2, dim2)
            if other not in angular:
                angular[other] = attrs[:, attr : attr + dim2 * mul2].reshape(
                    edges, dim2, mul2
                )
            x, y = (inputs[key], angular[other])
            cg = getattr(module, f"cg_{index}").to(features)
            if weight >= 0:
                w = weights[:, weight : weight + mul1 * mul2].reshape(edges, mul1, mul2)
                angular_value = torch.bmm(y, w.transpose(1, 2)).transpose(1, 2)
            else:
                angular_value = y.sum(-1).unsqueeze(1)
            coupled = (x.transpose(1, 2) @ cg.flatten(1)).reshape(
                edges, mul1, dim2, dim_out
            )
            value = (coupled * angular_value.unsqueeze(-1)).sum(2).transpose(1, 2)
            value = (factor * value).flatten(1)
            outputs[end] = value if end not in outputs else outputs[end] + value
        blocks = [
            outputs[start] if start in outputs else features.new_zeros((edges, width))
            for start, width in module.output_sections
        ]
        message = torch.cat(blocks, -1) if blocks else features.new_zeros((edges, 0))
    return (
        features.new_zeros((num_nodes, module.output_dim)).index_add(0, target, message)
        if indexed
        else message
    ) + zero

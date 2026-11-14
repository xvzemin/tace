"""Aligned and frame-free contraction schedules."""

import torch
from e3nn import o3

from .....utils.autograd import forward_ad_active
from .... import config
from ...co2 import spherical as spherical_kernel


def evaluate(
    module,
    features,
    radial,
    projection,
    wigner,
    amplitudes,
    edge_index,
    num_nodes,
    vectors,
    radial_network,
):
    """Evaluate validated aligned-convolution operands."""
    if (
        module.radial_mode == "full"
        and not config.TORCH_REFERENCE
        and not forward_ad_active()
    ):
        from equivariantx.ops.common.convolution import convolve

        result = convolve(
            module.kernel_metadata,
            features,
            wigner,
            radial,
            projection,
            edge_index,
            num_nodes,
            module.output_dim,
            network=radial_network,
            recompute=True,
            amplitudes=amplitudes,
        )
    else:
        result = module.reference(
            features, radial, projection, wigner, amplitudes, edge_index, num_nodes
        )
    return result if vectors is None else result + vectors.sum() * 0


def transverse(
    module,
    features,
    radial,
    projection,
    amplitudes,
    edge_index,
    num_nodes,
    vectors,
    method,
    radial_network,
):
    """Evaluate validated aligned-convolution operands."""
    metadata = (
        module.transverse_metadata
        if method == "baseline"
        else getattr(module, f"{method}_metadata")
    )
    direction = vectors / vectors.square().sum(dim=-1, keepdim=True).sqrt()
    if (
        module.radial_mode == "full"
        and not config.TORCH_REFERENCE
        and not forward_ad_active()
    ):
        from equivariantx.ops.common.convolution import convolve

        if module.channel_mixing:
            return convolve(
                metadata,
                features,
                direction,
                radial,
                projection,
                edge_index,
                num_nodes,
                module.output_dim,
                network=radial_network,
                recompute=True,
                amplitudes=amplitudes,
                transverse=module.transverse_layout,
            )
        attributes, paths, offsets = ([], [], {})
        for path in module.transverse_paths:
            harmonic, dimension = (path[1], path[6])
            key = (harmonic, dimension)
            if key not in offsets:
                offsets[key] = sum((value.shape[-1] for value in attributes))
                value = o3.spherical_harmonics(
                    (dimension - 1) // 2,
                    direction,
                    normalize=False,
                    normalization=module.normalization,
                )
                attributes.append(value * amplitudes[:, harmonic : harmonic + 1])
            paths.append((path[0], offsets[key], *path[2:]))
        metadata = repr((tuple(paths), projection.shape[1], False))
        attributes = torch.cat(attributes, -1) if attributes else vectors[:, :0]
        return convolve(
            metadata,
            features,
            attributes,
            radial,
            projection,
            edge_index,
            num_nodes,
            module.output_dim,
            network=radial_network,
            recompute=True,
        )
    source, target = edge_index
    edges = source.numel()
    weights = radial @ projection if projection.numel() else radial
    zero = sum(
        (
            value[..., :0].sum()
            for value in (features, radial, projection, amplitudes, direction)
        )
    )
    inputs, harmonics, outputs = ({}, {}, {})
    for (mode, mul_out), path in zip(module.transverse_layout, module.transverse_paths):
        start, harmonic, end, mul, _, dim, harmonic_dim, dim_out, weight, factor, _ = (
            path
        )
        key = (start, dim, mul)
        if key not in inputs:
            inputs[key] = (
                features[source, start : start + dim * mul]
                .reshape(edges, dim, mul)
                .transpose(-1, -2)
            )
        x = inputs[key]
        name = f"{(dim - 1) // 2}_{(harmonic_dim - 1) // 2}_{(dim_out - 1) // 2}"
        if method == "cg":
            degree = str((harmonic_dim - 1) // 2)
            if degree not in harmonics:
                harmonics[degree] = module.harmonics[degree](direction)
            value = torch.einsum(
                "eua,eb,abc->ecu",
                x,
                harmonics[degree],
                getattr(module, f"full_cg_{name}"),
            )
        else:
            coupling = module.transverse_couplings[name]
            angular_method = "recurrence" if method == "recurrence" else "chebyshev"
            value = (
                spherical_kernel.reference(
                    coupling, x, direction.unsqueeze(-2), angular_method
                )
                if (module.backend == "torch" or not features.is_cuda)
                and config.TORCH_REFERENCE
                else coupling(x, direction.unsqueeze(-2), method=angular_method)
            ).transpose(-1, -2)
        if weight >= 0:
            width = mul if mode == "uvu" else mul * mul_out
            w = weights[:, weight : weight + width]
            value = (
                value * w.unsqueeze(-2)
                if mode == "uvu"
                else value @ w.reshape(-1, mul, mul_out)
            )
        elif mode == "uvw":
            value = value.sum(-1, keepdim=True).expand(-1, dim_out, mul_out)
        value = value * (factor * amplitudes[:, harmonic : harmonic + 1]).unsqueeze(-1)
        outputs[end] = value if end not in outputs else outputs[end] + value
    return module.scatter_outputs(outputs, features, target, num_nodes, zero)


def reference(
    module, features, radial, projection, wigner, amplitudes, edge_index, num_nodes
):
    """Evaluate validated aligned-convolution operands."""
    source, target = edge_index
    edges = source.numel()
    weights = radial @ projection if projection.numel() else radial
    zero = sum(
        (
            value[..., :0].sum()
            for value in (features, radial, projection, wigner, amplitudes)
        )
    )
    inputs, matrices, outputs, groups = ({}, {}, {}, {})
    for index, (mode, path) in enumerate(module.path_data):
        start, end, mul, mul_out, dim, dim_out, dstart, dend, weight, harmonic = path
        key = (start, dim, mul, dstart)
        if key not in inputs:
            x = features[source, start : start + dim * mul].reshape(edges, dim, mul)
            din = wigner[:, dstart : dstart + dim * dim].reshape(
                wigner.size(0), dim, dim
            )
            inputs[key] = din @ x
        value = getattr(module, f"cg_{index}").to(features).T @ inputs[key]
        if weight >= 0:
            if mode == "uvu":
                value = value * weights[:, None, weight : weight + mul]
            else:
                value = value @ weights[:, weight : weight + mul * mul_out].reshape(
                    weights.size(0), mul, mul_out
                )
        elif mode == "uvw":
            value = value.sum(-1, keepdim=True).expand(-1, dim_out, mul_out)
        value = value * amplitudes[:, harmonic, None, None]
        if end not in outputs:
            groups.setdefault((dend, dim_out), []).append((end, mul_out))
            matrices[dend] = wigner[:, dend : dend + dim_out * dim_out].reshape(
                wigner.size(0), dim_out, dim_out
            )
            outputs[end] = value
        else:
            outputs[end] = outputs[end] + value
    for (dend, _), entries in groups.items():
        values = [outputs[end] for end, _ in entries]
        values = values[0] if len(values) == 1 else torch.cat(values, -1)
        values = matrices[dend].transpose(-1, -2) @ values
        for (end, _), value in zip(
            entries, values.split([mul for _, mul in entries], -1)
        ):
            outputs[end] = value
    return module.scatter_outputs(outputs, features, target, num_nodes, zero)


def scatter_outputs(module, outputs, features, target, num_nodes, zero):
    """Evaluate validated aligned-convolution operands."""
    values = []
    for start, width in module.output_sections:
        value = features.new_zeros((num_nodes, width)) + zero
        if start in outputs:
            value = value.index_add(0, target, outputs[start].flatten(1))
        values.append(value)
    return (
        torch.cat(values, -1) if values else features.new_zeros((num_nodes, 0)) + zero
    )

"""Indexed restriction and lifting in spherical transverse storage."""

from .... import config
from ...co2 import spherical as spherical_kernel


def evaluate(module, features, direction, source, target, num_nodes, inverse):
    """Restrict or lift validated features and aggregate matching edges."""
    if inverse:
        features = features.index_select(1, module.inverse_permutation)
    width = module.input_dim if inverse else module.output_dim
    result = (
        features.new_zeros((num_nodes, width)) + (features.sum() + direction.sum()) * 0
    )
    for start, _, end, mul, _, dim, harmonic_dim, dim_out, _, scale, _ in module.paths[
        inverse
    ]:
        x = features[source, start : start + mul * dim].reshape(
            source.numel(), dim, mul
        )
        name = f"{(dim - 1) // 2}_{(harmonic_dim - 1) // 2}_{(dim_out - 1) // 2}"
        coupling = module.couplings[name]
        value = (
            spherical_kernel.reference(
                coupling, x.transpose(1, 2), direction[:, None, :], coupling.method
            )
            if config.TORCH_REFERENCE
            else coupling(x.transpose(1, 2), direction[:, None, :])
        )
        value = value.transpose(1, 2).flatten(1) * scale
        result[:, end : end + mul * dim_out] += value.new_zeros(
            (num_nodes, mul * dim_out)
        ).index_add(0, target, value)
    return result if inverse else result.index_select(1, module.permutation)

"""Cartesian layout, geometry and compact coefficient execution."""

import torch

from .....utils.layout import permute


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
    """Evaluate validated Cartesian operands."""
    from equivariantx.o3.cgtp.convolution import O3TensorProductConv

    fused_geometry = vectors is None or vectors.dtype == features.dtype
    packed_input = (
        module.input_basis == "spherical" and vectors is not None and fused_geometry
    )
    if module.input_basis == "spherical":
        if packed_input:
            if features.size(-1) != module.to_cartesian.input_dim:
                raise ValueError(
                    "The spherical feature dimension does not match irreps_in1."
                )
            values = []
            for mul, l, dim, section in module.to_cartesian.paths:
                value = features[:, section].reshape(features.size(0), mul, dim)
                value = value @ getattr(module, f"input_basis_{l}").T
                values.append(value.transpose(-1, -2).flatten(1))
            features = torch.cat(values, -1) if values else features[:, :0]
        else:
            features = module.to_cartesian(features)
    if module.symmetric_inputs and (vectors is None or not fused_geometry):
        values = features.new_zeros((features.size(0), module.symmetric_dim)).index_add(
            -1, module.symmetric_index, features
        )
        features = (values / module.symmetric_count).index_select(
            -1, module.symmetric_index
        )
    if vectors is not None and (not fused_geometry):
        vectors = (
            torch.nn.functional.normalize(vectors, dim=-1)
            if module.normalize
            else vectors
        )
        attrs, offset = ([], 0)
        for mul, ir in module.irreps_in2:
            harmonic = module.harmonics[str(ir.l)](vectors)
            amplitude = (
                features.new_ones((1, mul))
                if amplitudes is None
                else amplitudes.expand(amplitudes.size(0), module.amplitude_dim)[
                    :, offset : offset + mul
                ]
            )
            attrs.append((amplitude.unsqueeze(-1) * harmonic.unsqueeze(-2)).flatten(1))
            offset += mul
        edge_attrs = torch.cat(attrs, dim=-1) if attrs else vectors[:, :0]
        edge_attrs = edge_attrs + vectors.sum() * 0
        vectors = None
    if vectors is not None and module.pack_inputs and (not packed_input):
        from equivariantx.ops.common.layout import indexed_sum

        if features.size(-1) != module.input_dim:
            raise ValueError(
                "Feature dimensions do not match the tensor-product irreps."
            )
        features = indexed_sum(
            features,
            module.harmonic_input_index,
            module.harmonic_input_ptr,
            module.harmonic_input_transpose_index,
            module.harmonic_input_transpose_ptr,
            module.harmonic_input_count,
        )
    elif not packed_input and (not module.input_identity):
        features = permute(features, module.input_index, module.input_inverse)
    if vectors is None and (not module.attrs_identity):
        edge_attrs = permute(edge_attrs, module.attrs_index, module.attrs_inverse)
    output = O3TensorProductConv.forward(
        module,
        features,
        edge_attrs,
        radial,
        projection,
        edge_index,
        num_nodes,
        vectors=vectors,
        amplitudes=amplitudes,
        radial_network=radial_network,
    )
    if vectors is not None and module.compact_output:
        return output
    if vectors is not None:
        output = output.index_select(-1, module.harmonic_output_index)
    elif not module.output_identity:
        output = permute(output, module.output_inverse, module.output_index)
    if module.compact_output:
        return output.index_select(-1, module.output_pack_index)
    return module.projector(output)
